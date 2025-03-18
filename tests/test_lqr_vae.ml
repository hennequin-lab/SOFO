(* Test gradients in the lqr-vae function end-to-end with Matheron sampling *)
open Base
open Torch
open Forward_torch
module Arr = Owl.Arr
module Mat = Owl.Mat
module Linalg = Owl.Linalg.D

let print s = Stdio.printf "%s\n%!" (Base.Sexp.to_string_hum s)
let rel_tol = Alcotest.float 1e-4
let n_tests = 10

(* -----------------------------------------
   -- Control Problem / Data Generation ---
   ----------------------------------------- *)
module Dims = struct
  let a = 24
  let b = 10
  let o = 48
  let tmax = 10
  let m = 128
  let batch_const = true
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.Cpu
end

module Data = Lds_data.Make_LDS_Tensor (Dims)

let x0 = Tensor.zeros ~device:Dims.device ~kind:Dims.kind [ Dims.m; Dims.a ]

(* in the linear gaussian case, _Fx, _Fu, c, b and cov invariant across time *)
let f_list : Tensor.t Lds_data.f_params list =
  let _Fx = Data.sample_fx () in
  let _Fu = Data.sample_fu () in
  let c = Data.sample_c () in
  let b = Data.sample_b () in
  let cov = Data.sample_output_cov () in
  List.init (Dims.tmax + 1) ~f:(fun _ ->
    Lds_data.
      { _Fx_prod = _Fx
      ; _Fu_prod = _Fu
      ; _f = None
      ; _c = Some c
      ; _b = Some b
      ; _cov = Some cov
      })

(* -----------------------------------------
   ----- Utilitiy Functions ------
   ----------------------------------------- *)
let with_given_seed_torch seed f =
  (* generate a random key to later restore the state of the RNG *)
  let key = Random.int Int.max_value in
  (* now force the state of the RNG under which f will be evaluated *)
  Torch_core.Wrapper.manual_seed seed;
  let result = f () in
  (* restore the RGN using key *)
  Torch_core.Wrapper.manual_seed key;
  (* return the result *)
  result

let sample_data =
  (* generate ground truth params and data *)
  let u_list = Data.sample_u_list () in
  let x_list, o_list = Data.traj_rollout ~x0 ~f_list ~u_list in
  let o_list = List.map o_list ~f:(fun o -> Option.value_exn o) in
  u_list, x_list, o_list

let tmp_einsum a b =
  if Dims.batch_const
  then Maths.einsum [ a, "ma"; b, "ab" ] "mb"
  else Maths.einsum [ a, "ma"; b, "mab" ] "mb"

let inv_cov x =
  let x_sqr = Maths.sqr x in
  let tmp = Tensor.of_float0 1. |> Maths.const in
  Maths.(tmp / x_sqr)

(* -----------------------------------------
   -- Model setup ----
   ----------------------------------------- *)
module PP = struct
  type 'a p =
    { _Fx_prod : 'a (* generative model *)
    ; _Fu_prod : 'a
    ; _c : 'a
    ; _b : 'a
    ; _cov_o : 'a (* sqrt of the diagonal of covariance of emission noise *)
    ; _cov_u : 'a (* sqrt of the diagonal of covariance of prior over u *)
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

(* test loss function end to end *)
(* create params for lds from f; all parameters carry tangents *)
let params_from_f_diff ~(theta : P.M.t) ~x0 ~o_list
  : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
  =
  (* set o at time 0 as 0 *)
  let o_list_tmp =
    Maths.const (Tensor.zeros_like (Maths.primal (List.hd_exn o_list))) :: o_list
  in
  let _cov_u_inv =
    theta._cov_u |> inv_cov |> Maths.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
  in
  let c_trans = Maths.transpose theta._c ~dim0:1 ~dim1:0 in
  let _cov_o_inv = inv_cov theta._cov_o in
  let _Cxx =
    let tmp = Maths.(einsum [ theta._c, "ab"; _cov_o_inv, "b" ] "ab") in
    Maths.(tmp *@ c_trans)
  in
  let _cx_common =
    let tmp = Maths.(einsum [ theta._b, "ab"; _cov_o_inv, "b" ] "ab") in
    Maths.(tmp *@ c_trans)
  in
  Lqr.Params.
    { x0 = Some x0
    ; params =
        List.map o_list_tmp ~f:(fun o ->
          let _cx = Maths.(_cx_common - (o *@ c_trans)) in
          Lds_data.Temp.
            { _f = None
            ; _Fx_prod = theta._Fx_prod
            ; _Fu_prod = theta._Fu_prod
            ; _cx = Some _cx
            ; _cu = None
            ; _Cxx
            ; _Cxu = None
            ; _Cuu = _cov_u_inv
            })
    }

(* rollout x list under sampled u *)
let rollout_x ~u_list ~x0 (theta : P.M.t) =
  let x0_tan = Maths.const x0 in
  let _, x_list =
    List.fold u_list ~init:(x0_tan, [ x0_tan ]) ~f:(fun (x, x_list) u ->
      let new_x = Maths.(tmp_einsum x theta._Fx_prod + tmp_einsum u theta._Fu_prod) in
      new_x, new_x :: x_list)
  in
  List.rev x_list

(* optimal u determined from lqr; carry tangents *)
let pred_u ~data (theta : P.M.t) =
  let x0, o_list = data in
  let x0_tan = Maths.const x0 in
  (* use lqr to obtain the optimal u *)
  (* optimal u and sampled u do not carry tangents! *)
  let optimal_u_list =
    let p =
      params_from_f_diff ~x0:x0_tan ~theta ~o_list:(List.map o_list ~f:Maths.const)
      |> Lds_data.map_naive ~batch_const:Dims.batch_const
    in
    let sol, _ ,_= Lqr._solve ~batch_const:Dims.batch_const p in
    List.map sol ~f:(fun s -> s.u)
  in
  Stdlib.Gc.major ();
  let sample_gauss ~_mean ~_cov ~dim () =
    let eps =
      Tensor.randn ~device:Dims.device ~kind:Dims.kind [ Dims.m; dim ] |> Maths.const
    in
    Maths.(einsum [ eps, "ma"; _cov, "ab" ] "mb" + _mean)
  in
  (* sample u from their priors *)
  let sampled_u =
    let cov_u =
      theta._cov_u |> Maths.abs |> Maths.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
    in
    List.init Dims.tmax ~f:(fun _ ->
      with_given_seed_torch
        1972
        (sample_gauss ~_mean:(Maths.const (Tensor.f 0.)) ~_cov:cov_u ~dim:Dims.b))
  in
  (* sample o with obsrevation noise *)
  let o_sampled =
    (* propagate prior sampled u through dynamics *)
    let x_rolled_out = rollout_x ~u_list:sampled_u ~x0 theta |> List.tl_exn in
    let cov_o =
      theta._cov_o |> Maths.abs |> Maths.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
    in
    List.map x_rolled_out ~f:(fun x ->
      let _mean = Maths.((x *@ theta._c) + theta._b) in
      with_given_seed_torch 1972 (sample_gauss ~_mean ~_cov:cov_o ~dim:Dims.o))
  in
  (* lqr on (o - o_sampled) *)
  let sol_delta_o, _,_ =
    let delta_o_list =
      List.map2_exn o_list o_sampled ~f:(fun a b -> Maths.(const a - b))
    in
    let p =
      params_from_f_diff ~x0:x0_tan ~theta ~o_list:delta_o_list
      |> Lds_data.map_naive ~batch_const:Dims.batch_const
    in
    Lqr._solve ~batch_const:Dims.batch_const p
  in
  Stdlib.Gc.major ();
  (* final u samples *)
  let u_list =
    let optimal_u_list_delta_o = List.map sol_delta_o ~f:(fun s -> s.u) in
    List.map2_exn sampled_u optimal_u_list_delta_o ~f:(fun u delta_u ->
      Maths.(u + delta_u))
  in
  optimal_u_list, u_list

(* gaussian llh with diagonal covariance *)
let gaussian_llh ~g_mean ~g_cov ~x =
  let g_cov = Maths.diag_embed g_cov ~offset:0 ~dim1:(-2) ~dim2:(-1) in
  let error_term =
    let error = Maths.(x - g_mean) in
    let tmp = tmp_einsum error (Maths.inv_sqr g_cov) in
    Maths.einsum [ tmp, "ma"; error, "ma" ] "m" |> Maths.reshape ~shape:[ -1; 1 ]
  in
  let cov_term =
    Maths.(sum (log (diagonal g_cov ~offset:0))) |> Maths.reshape ~shape:[ 1; 1 ]
  in
  let const_term =
    let o = x |> Maths.primal |> Tensor.shape |> List.last_exn in
    Tensor.of_float0 ~device:Dims.device Float.(log (2. * pi) * of_int o)
    |> Tensor.reshape ~shape:[ 1; 1 ]
    |> Maths.const
  in
  Maths.(0.5 $* error_term + cov_term + const_term)
  |> Maths.(mean_dim ~keepdim:false ~dim:(Some [ 1 ]))
  |> Maths.neg

let llh ~x_o_list (theta : P.M.t) =
  (* calculate the likelihood term *)
  let llh =
    List.foldi x_o_list ~init:None ~f:(fun t accu (x, o) ->
      if t % 1 = 0 then Stdlib.Gc.major ();
      let increment =
        gaussian_llh
          ~g_mean:o
          ~g_cov:(Maths.sqr theta._cov_o)
          ~x:Maths.(tmp_einsum x theta._c + theta._b)
      in
      match accu with
      | None -> Some increment
      | Some accu -> Some Maths.(accu + increment))
    |> Option.value_exn
  in
  llh

(* calculate negative llh *)
let f_loss ~data (theta : P.M.t) =
  let x0, o_list = data in
  let _, u_list = pred_u ~data theta in
  let rolled_out_x_list = rollout_x ~u_list ~x0 theta in
  (* These lists go from 1 to T *)
  let o_except_first = List.map o_list ~f:(fun o -> Maths.const o) in
  let x_except_first = List.tl_exn rolled_out_x_list in
  let x_o_list = List.map2_exn x_except_first o_except_first ~f:(fun x o -> x, o) in
  let neg_joint_llh = Maths.(neg (llh ~x_o_list theta)) in
  neg_joint_llh

let theta_init =
  let gaussian_tensor_2d_normed ~device ~kind ~a ~b ~sigma =
    let normaliser = Float.(sigma / sqrt (of_int a)) in
    Tensor.mul_scalar_ (Tensor.randn ~kind ~device [ a; b ]) (Scalar.f normaliser)
  in
  let _Fx_prod =
    gaussian_tensor_2d_normed
      ~device:Dims.device
      ~kind:Dims.kind
      ~a:Dims.a
      ~b:Dims.a
      ~sigma:0.1
  in
  let _Fu_prod =
    gaussian_tensor_2d_normed
      ~device:Dims.device
      ~kind:Dims.kind
      ~a:Dims.b
      ~b:Dims.a
      ~sigma:0.1
  in
  let _c =
    gaussian_tensor_2d_normed
      ~device:Dims.device
      ~kind:Dims.kind
      ~a:Dims.a
      ~b:Dims.o
      ~sigma:0.1
  in
  let _b =
    gaussian_tensor_2d_normed
      ~device:Dims.device
      ~kind:Dims.kind
      ~a:1
      ~b:Dims.o
      ~sigma:0.1
  in
  let _cov_o = Tensor.(abs (randn ~device:Dims.device ~kind:Dims.kind [ Dims.o ])) in
  let _cov_u = Tensor.(abs (randn ~device:Dims.device ~kind:Dims.kind [ Dims.b ])) in
  let _cov_pos = Tensor.(abs (randn ~device:Dims.device ~kind:Dims.kind [ Dims.b ])) in
  PP.{ _Fx_prod; _Fu_prod; _c; _b; _cov_o; _cov_u }

let check_grad ~f x =
  let module Input = PP.Make (Prms.P) in
  let module F = Framework.Make (Input) (Prms.P) in
  F.run x ~f

let test_LQR () =
  let _, _, o_list = sample_data in
  let data = x0, o_list in
  let f = f_loss ~data in
  let _, _, e = check_grad ~f theta_init in
  Alcotest.(check @@ rel_tol) "LQR test" 0.0 e

let () =
  Alcotest.run
    "LQR tests"
    [ ( "lqr_vae"
      , List.init n_tests ~f:(fun _ ->
          Alcotest.test_case "linear gaussian model" `Quick test_LQR) )
    ]
