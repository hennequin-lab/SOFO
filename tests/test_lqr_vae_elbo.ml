(* Test gradients in the lqr-vae function end-to-end with elbo *)
open Base
open Torch
open Forward_torch
module Arr = Owl.Arr
module Mat = Owl.Mat
module Linalg = Owl.Linalg.D

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

let sqr_inv x =
  let x_sqr = Maths.sqr x in
  let tmp = Tensor.of_float0 1. |> Maths.const in
  Maths.(tmp / x_sqr)

(* list of length T of [m x b] to matrix of [m x b x T]*)
let concat_time u_list =
  List.map u_list ~f:(fun u -> Maths.unsqueeze ~dim:(-1) u) |> Maths.concat_list ~dim:2

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
    ; _cov_space : 'a
      (* recognition model; sqrt of the diagonal of covariance of space factor *)
    ; _cov_time : 'a (* sqrt of the diagonal of covariance of the time factor *)
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

(* test loss function end to end *)
(* create params for lds from f; all parameters carry tangents *)
let params_from_f ~(theta : P.M.t) ~x0 ~o_list
  : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
  =
  (* set o at time 0 as 0 *)
  let o_list_tmp = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
  let _cov_u_inv =
    theta._cov_u |> sqr_inv |> Maths.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
  in
  let c_trans = Maths.transpose theta._c ~dim0:1 ~dim1:0 in
  let _cov_o_inv = sqr_inv theta._cov_o in
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
          let _cx = Maths.(_cx_common - (const o *@ c_trans)) in
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

(* optimal u determined from lqr *)
let pred_u ~data (theta : P.M.t) =
  let x0, o_list = data in
  let x0_tan = Maths.const x0 in
  (* use lqr to obtain the optimal u *)
  let p =
    params_from_f ~x0:x0_tan ~theta ~o_list
    |> Lds_data.map_implicit ~batch_const:Dims.batch_const
  in
  let sol, _ = Lqr.solve ~batch_const:Dims.batch_const p in
  let optimal_u_list = List.map sol ~f:(fun s -> s.u) in
  (* sample u from the kronecker formation *)
  let u_list =
    let optimal_u = concat_time optimal_u_list in
    let xi =
      Tensor.randn ~device:Dims.device ~kind:Dims.kind [ Dims.m; Dims.b; Dims.tmax ]
      |> Maths.const
    in
    let _chol_space = Maths.abs theta._cov_space in
    let _chol_time = Maths.abs theta._cov_time in
    let xi_space = Maths.einsum [ xi, "mbt"; _chol_space, "b" ] "mbt" in
    let xi_time = Maths.einsum [ xi_space, "mat"; _chol_time, "t" ] "mat" in
    let meaned = Maths.(xi_time + optimal_u) in
    List.init Dims.tmax ~f:(fun i ->
      Maths.slice ~dim:2 ~start:(Some i) ~end_:(Some (i + 1)) ~step:1 meaned
      |> Maths.reshape ~shape:[ Dims.m; Dims.b ])
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

let elbo ~x_o_list ~u_list ~optimal_u_list ~sample (theta : P.M.t) =
  (* calculate the likelihood term *)
  let llh =
    List.foldi x_o_list ~init:None ~f:(fun t accu (x, o) ->
      if t % 1 = 0 then Stdlib.Gc.major ();
      let increment =
        gaussian_llh
          ~g_mean:o
          ~g_cov:theta._cov_o
          ~x:Maths.(tmp_einsum x theta._c + theta._b)
      in
      match accu with
      | None -> Some increment
      | Some accu -> Some Maths.(accu + increment))
    |> Option.value_exn
  in
  (* M1: calculate the kl term using samples *)
  let optimal_u = concat_time optimal_u_list in
  let kl =
    if sample
    then (
      let prior =
        List.foldi u_list ~init:None ~f:(fun t accu u ->
          if t % 1 = 0 then Stdlib.Gc.major ();
          let u_zeros = Tensor.zeros_like (Maths.primal u) |> Maths.const in
          let increment = gaussian_llh ~g_mean:u_zeros ~g_cov:theta._cov_u ~x:u in
          match accu with
          | None -> Some increment
          | Some accu -> Some Maths.(accu + increment))
        |> Option.value_exn
      in
      let entropy =
        let u = concat_time u_list |> Maths.reshape ~shape:[ Dims.m; -1 ] in
        let optimal_u = Maths.reshape optimal_u ~shape:[ Dims.m; -1 ] in
        let g_cov = Maths.kron theta._cov_space theta._cov_time in
        gaussian_llh ~g_mean:optimal_u ~g_cov ~x:u
      in
      Maths.(entropy - prior))
    else (
      (* M2: calculate the kl term analytically *)
      let cov2 = Maths.kron theta._cov_space theta._cov_time in
      let det1 = Maths.(2. $* sum (log (abs cov2))) in
      let det2 = Maths.(Float.(2. * of_int Dims.tmax) $* sum (log (abs theta._cov_u))) in
      let _const = Tensor.of_float0 (Float.of_int Dims.b) |> Maths.const in
      let tr =
        let tmp1 = theta._cov_u |> sqr_inv in
        let tmp2 = Maths.(tmp1 * sqr theta._cov_space) in
        let tmp3 = Maths.(kron tmp2 (sqr theta._cov_time)) in
        Maths.sum tmp3
      in
      let quad =
        let _cov_u = theta._cov_u |> sqr_inv in
        let tmp1 = Maths.einsum [ optimal_u, "mbt"; _cov_u, "b" ] "mbt" in
        Maths.einsum [ tmp1, "mbt"; optimal_u, "mbt" ] "m" |> Maths.unsqueeze ~dim:1
      in
      let tmp = Maths.(det2 - det1 - _const + tr) |> Maths.reshape ~shape:[ 1; 1 ] in
      Maths.(tmp + quad) |> Maths.squeeze ~dim:1)
  in
  Maths.(llh - kl)

(* calculate negative llh *)
let f_loss ~data (theta : P.M.t) =
  let x0, o_list = data in
  let optimal_u_list, u_list = pred_u ~data theta in
  let rolled_out_x_list = rollout_x ~u_list ~x0 theta in
  (* These lists go from 1 to T *)
  let o_except_first = List.map o_list ~f:(fun o -> Maths.const o) in
  let x_except_first = List.tl_exn rolled_out_x_list in
  let x_o_list = List.map2_exn x_except_first o_except_first ~f:(fun x o -> x, o) in
  let neg_joint_llh =
    Maths.(neg (elbo ~x_o_list ~u_list ~optimal_u_list theta ~sample:false))
  in
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
  let _cov_space = Tensor.(abs (randn ~device:Dims.device ~kind:Dims.kind [ Dims.b ])) in
  let _cov_time =
    Tensor.(abs (randn ~device:Dims.device ~kind:Dims.kind [ Dims.tmax ]))
  in
  PP.{ _Fx_prod; _Fu_prod; _c; _b; _cov_o; _cov_u; _cov_space; _cov_time }

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
    [ ( "lqr_vae_elbo"
      , List.init n_tests ~f:(fun _ ->
          Alcotest.test_case "linear gaussian model" `Quick test_LQR) )
    ]
