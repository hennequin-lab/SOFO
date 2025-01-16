(* Linear Gaussian Dynamics, with same state/control/cost parameters constant across trials and across time *)
open Printf
open Base
open Forward_torch
open Torch
open Sofo

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

(* -----------------------------------------
   -- Control Problem / Data Generation ---
   ----------------------------------------- *)
module Dims = struct
  let a = 24
  let b = 10
  let o = 48
  let tmax = 10
  let m = 512
  let batch_const = true
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.cuda_if_available ()
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

let sample_data () =
  (* generate ground truth params and data *)
  let u_list =
    let _std_u = Tensor.ones ~device:Dims.device ~kind:Dims.kind [ Dims.b ] in
    Data.sample_u_list ~std_u:_std_u
  in
  let x_list, o_list = Data.traj_rollout ~x0 ~f_list ~u_list in
  let o_list = List.map o_list ~f:(fun o -> Option.value_exn o) in
  u_list, x_list, o_list

(* -----------------------------------------
   ----- Utilitiy Functions ------
   ----------------------------------------- *)

let tmp_einsum a b =
  if Dims.batch_const
  then Maths.einsum [ a, "ma"; b, "ab" ] "mb"
  else Maths.einsum [ a, "ma"; b, "mab" ] "mb"

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)
let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run

let base =
  Optimizer.Config.Base.
    { default with kind = Torch_core.Kind.(T f64); ba_kind = Bigarray.float64 }

let max_iter = 2000

module LGS = struct
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

  type args = unit
  type data = Tensor.t * Tensor.t list

  let remove_tangent x = x |> Maths.primal |> Maths.const

  let sqr_inv x =
    let x_sqr = Maths.sqr x in
    let tmp = Tensor.of_float0 1. |> Maths.const in
    Maths.(tmp / x_sqr)

  (* special care to be taken when dealing with elbo loss *)
  module Joint_llh_loss = struct
    let vtgt_hessian_gv ~rolled_out_x_list (theta : P.M.t) =
      (* fold ggn across time *)
      let ggn_final ~o_list ~like_hess ~diagonal =
        let vtgt_hess_eqn = if diagonal then "kma,a->kma" else "kma,ab->kmb" in
        List.fold o_list ~init:(Tensor.f 0.) ~f:(fun accu o ->
          let vtgt = Maths.tangent o |> Option.value_exn in
          let vtgt_hess =
            Tensor.einsum ~equation:vtgt_hess_eqn [ vtgt; like_hess ] ~path:None
          in
          let increment =
            Tensor.einsum ~equation:"kma,jma->kj" [ vtgt_hess; vtgt ] ~path:None
          in
          Tensor.(accu + increment))
      in
      let llh_ggn =
        let like_hess = theta._cov_o |> sqr_inv |> Maths.primal in
        let c_x_list =
          List.map rolled_out_x_list ~f:(fun x ->
            Maths.(einsum [ x, "ma"; theta._c, "ab" ] "mb" + theta._b))
        in
        ggn_final ~o_list:c_x_list ~like_hess ~diagonal:true
      in
      llh_ggn
  end

  (* create params for lds from f; no parameters carry tangents *)
  let params_from_f ~(theta : P.M.t) ~x0 ~o_list
    : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
    =
    (* set o at time 0 as 0 *)
    let o_list_tmp = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
    let _cov_u_inv =
      theta._cov_u
      |> sqr_inv
      |> Maths.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
      |> remove_tangent
    in
    let c_trans = Maths.transpose theta._c ~dim0:1 ~dim1:0 |> remove_tangent in
    let _cov_o_inv = sqr_inv theta._cov_o in
    let _Cxx =
      let tmp = Maths.(einsum [ theta._c, "ab"; _cov_o_inv, "b" ] "ab") in
      Maths.(tmp *@ c_trans) |> remove_tangent
    in
    let _cx_common =
      let tmp = Maths.(einsum [ theta._b, "ab"; _cov_o_inv, "b" ] "ab") in
      Maths.(tmp *@ c_trans) |> remove_tangent
    in
    Lqr.Params.
      { x0 = Some x0
      ; params =
          List.map o_list_tmp ~f:(fun o ->
            let _cx =
              let tmp = Maths.(einsum [ const o, "ab"; _cov_o_inv, "b" ] "ab") in
              Maths.(_cx_common - (tmp *@ c_trans)) |> remove_tangent
            in
            Lds_data.Temp.
              { _f = None
              ; _Fx_prod = theta._Fx_prod |> remove_tangent
              ; _Fu_prod = theta._Fu_prod |> remove_tangent
              ; _cx = Some _cx
              ; _cu = None
              ; _Cxx
              ; _Cxu = None
              ; _Cuu = _cov_u_inv
              })
      }

  (* create params for lds from f; all parameters carry tangents *)
  let params_from_f_diff ~(theta : P.M.t) ~x0 ~o_list
    : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
    =
    (* set o at time 0 as 0 *)
    let o_list_tmp =
      Maths.const (Tensor.zeros_like (Maths.primal (List.hd_exn o_list))) :: o_list
    in
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
            let _cx =
              let tmp = Maths.(einsum [ o, "ab"; _cov_o_inv, "b" ] "ab") in
              Maths.(_cx_common - (tmp *@ c_trans))
            in
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
  (* let pred_u ~data (theta : P.M.t) =
    let x0, o_list = data in
    let x0_tan = Maths.const x0 in
    (* use lqr to obtain the optimal u *)
    (* optimal u and sampled u do not carry tangents! *)
    let optimal_u_list =
      let p =
        params_from_f ~x0:x0_tan ~theta ~o_list
        |> Lds_data.map_naive ~batch_const:Dims.batch_const
      in
      let sol, u_cov_list = Lqr._solve ~batch_const:Dims.batch_const p in
      List.map sol ~f:(fun s -> s.u)
    in
    Stdlib.Gc.major ();
    let sample_gauss ~_mean ~_std ~dim =
      let eps = Tensor.randn ~device:Dims.device ~kind:Dims.kind [ Dims.m; dim ] in
      Tensor.(einsum ~equation:"ma,ab->mb" [ eps; _std ] ~path:None + _mean)
    in
    (* sample u from their priors *)
    let sampled_u =
      let std_u =
        theta._cov_u
        |> Maths.primal
        |> Tensor.abs
        |> Tensor.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
      in
      List.init Dims.tmax ~f:(fun _ ->
        sample_gauss ~_mean:(Tensor.f 0.) ~_std:std_u ~dim:Dims.b)
    in
    (* sample o with obsrevation noise *)
    let o_sampled =
      (* propagate prior sampled u through dynamics *)
      let x_rolled_out =
        let sampled_u_tan = List.map sampled_u ~f:Maths.const in
        rollout_x ~u_list:sampled_u_tan ~x0 theta |> List.tl_exn
      in
      let std_o =
        theta._cov_o
        |> Maths.primal
        |> Tensor.abs
        |> Tensor.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
      in
      List.map x_rolled_out ~f:(fun x ->
        let _mean =
          Tensor.(matmul (Maths.primal x) (Maths.primal theta._c) + Maths.primal theta._b)
        in
        sample_gauss ~_mean ~_std:std_o ~dim:Dims.o)
    in
    (* lqr on (o - o_sampled) *)
    let sol_delta_o, u_cov_list =
      let delta_o_list = List.map2_exn o_list o_sampled ~f:(fun a b -> Tensor.(a - b)) in
      let p =
        params_from_f ~x0:x0_tan ~theta ~o_list:delta_o_list
        |> Lds_data.map_naive ~batch_const:Dims.batch_const
      in
      Lqr._solve ~batch_const:Dims.batch_const p
    in
    Stdlib.Gc.major ();
    (* final u samples *)
    let u_list =
      let optimal_u_list_delta_o =
        List.map sol_delta_o ~f:(fun s -> s.u |> Maths.primal)
      in
      List.map2_exn sampled_u optimal_u_list_delta_o ~f:(fun u delta_u ->
        Tensor.(u + delta_u) |> Maths.const)
    in
    optimal_u_list, u_list *)

  (* optimal u determined from lqr; carry tangents *)
  let pred_u ~data (theta : P.M.t) =
    let x0, o_list = data in
    let x0_tan = Maths.const x0 in
    (* use lqr to obtain the optimal u *)
    (* optimal u and sampled u do carry tangents! *)
    let optimal_u_list =
      let p =
        params_from_f_diff ~x0:x0_tan ~theta ~o_list:(List.map o_list ~f:Maths.const)
        |> Lds_data.map_naive ~batch_const:Dims.batch_const
      in
      let sol, _ = Lqr._solve ~batch_const:Dims.batch_const p in
      List.map sol ~f:(fun s -> s.u)
    in
    Stdlib.Gc.major ();
    let sample_gauss ~_mean ~_std ~dim =
      let eps =
        Tensor.randn ~device:Dims.device ~kind:Dims.kind [ Dims.m; dim ] |> Maths.const
      in
      Maths.(einsum [ eps, "ma"; _std, "ab" ] "mb" + _mean)
    in
    (* sample u from their priors *)
    let sampled_u =
      let std_u =
        theta._cov_u |> Maths.abs |> Maths.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
      in
      List.init Dims.tmax ~f:(fun _ ->
        sample_gauss ~_mean:(Maths.const (Tensor.f 0.)) ~_std:std_u ~dim:Dims.b)
    in
    (* sample o with obsrevation noise *)
    let o_sampled =
      (* propagate prior sampled u through dynamics *)
      let x_rolled_out = rollout_x ~u_list:sampled_u ~x0 theta |> List.tl_exn in
      let std_o =
        theta._cov_o |> Maths.abs |> Maths.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
      in
      List.map x_rolled_out ~f:(fun x ->
        let _mean = Maths.((x *@ theta._c) + theta._b) in
        sample_gauss ~_mean ~_std:std_o ~dim:Dims.o)
    in
    (* lqr on (o - o_sampled) *)
    let sol_delta_o, u_cov_list =
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

  (* calculate optimal u *)
  let f ~update ~data ~init ~args:_ (theta : P.M.t) =
    let module L = Joint_llh_loss in
    let x0, o_list = data in
    let _, u_list = pred_u ~data theta in
    let rolled_out_x_list = rollout_x ~u_list ~x0 theta in
    (* These lists go from 1 to T *)
    let o_except_first = List.map o_list ~f:(fun o -> Maths.const o) in
    let x_except_first = List.tl_exn rolled_out_x_list in
    let x_o_list = List.map2_exn x_except_first o_except_first ~f:(fun x o -> x, o) in
    let neg_joint_llh = Maths.(neg (llh ~x_o_list theta)) in
    match update with
    | `loss_only u -> u init (Some neg_joint_llh)
    | `loss_and_ggn u ->
      let ggn = L.vtgt_hessian_gv ~rolled_out_x_list:x_except_first theta in
      u init (Some (neg_joint_llh, Some ggn))

  let init : P.tagged =
    let _Fx_prod =
      Convenience.gaussian_tensor_2d_normed
        ~device:Dims.device
        ~kind:Dims.kind
        ~a:Dims.a
        ~b:Dims.a
        ~sigma:0.1
      |> Prms.free
    in
    let _Fu_prod =
      Convenience.gaussian_tensor_2d_normed
        ~device:Dims.device
        ~kind:Dims.kind
        ~a:Dims.b
        ~b:Dims.a
        ~sigma:0.1
      |> Prms.free
    in
    let _c =
      Convenience.gaussian_tensor_2d_normed
        ~device:Dims.device
        ~kind:Dims.kind
        ~a:Dims.a
        ~b:Dims.o
        ~sigma:1.
      |> Prms.free
    in
    let _b =
      Convenience.gaussian_tensor_2d_normed
        ~device:Dims.device
        ~kind:Dims.kind
        ~a:1
        ~b:Dims.o
        ~sigma:1.
      |> Prms.free
    in
    let _cov_o =
      Tensor.(
        mul_scalar (ones ~device:Dims.device ~kind:Dims.kind [ Dims.o ]) (Scalar.f 0.1))
      |> Prms.free
    in
    let _cov_u =
      Tensor.(
        mul_scalar (ones ~device:Dims.device ~kind:Dims.kind [ Dims.b ]) (Scalar.f 0.1))
      |> Prms.free
    in
    { _Fx_prod; _Fu_prod; _c; _b; _cov_o; _cov_u }

  let simulate ~theta ~data =
    (* infer the optimal u *)
    let optimal_u_list, _ = pred_u ~data theta in
    (* rollout to obtain x *)
    let rolled_out_x_list = rollout_x ~u_list:optimal_u_list ~x0 theta in
    (* noiseless observation *)
    let noiseless_o_list =
      List.map rolled_out_x_list ~f:(fun x -> Maths.((x *@ theta._c) + theta._b))
      |> List.tl_exn
    in
    let o_error =
      List.fold2_exn noiseless_o_list (snd data) ~init:0. ~f:(fun accu o1 o2 ->
        let error = Tensor.(norm (Maths.primal o1 - o2)) |> Tensor.to_float0_exn in
        accu +. error)
    in
    o_error
end

let config ~base_lr ~gamma ~iter:_ =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some base_lr
    ; n_tangents = 256
    ; rank_one = false
    ; damping = gamma
    ; momentum = Some 0.9
    ; lm = false
    ; perturb_thresh = None
    }

module O = Optimizer.SOFO (LGS)

(* let config ~base_lr ~gamma:_ ~iter:_ =
  Optimizer.Config.Adam.{ default with learning_rate = Some base_lr }

module O = Optimizer.Adam (LGS) *)

let optimise ~max_iter ~f_name config_f =
  let rec loop ~iter ~state ~time_elapsed running_avg =
    Stdlib.Gc.major ();
    let config = config_f ~iter in
    let data =
      let _, _, o_list = sample_data () in
      x0, o_list
    in
    let t0 = Unix.gettimeofday () in
    let loss, new_state = O.step ~config ~state ~data ~args:() in
    let t1 = Unix.gettimeofday () in
    let time_elapsed = Float.(time_elapsed + t1 - t0) in
    let running_avg =
      let loss_avg =
        match running_avg with
        | [] -> loss
        | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
      in
      (* save params *)
      if iter % 1 = 0
      then (
        (* simulation error *)
        let o_error =
          let _, _, o_list = sample_data () in
          let data = x0, o_list in
          LGS.simulate ~theta:(LGS.P.const (LGS.P.value (O.params new_state))) ~data
        in
        (* avg error *)
        Convenience.print [%message (iter : int) (loss_avg : float)];
        let t = iter in
        Owl.Mat.(
          save_txt
            ~append:true
            ~out:(in_dir f_name)
            (of_array [| Float.of_int t; time_elapsed; loss_avg; o_error |] 1 4));
        O.W.P.T.save
          (LGS.P.value (O.params new_state))
          ~kind:base.ba_kind
          ~out:(in_dir f_name ^ "_params"));
      []
    in
    if iter < max_iter
    then loop ~iter:(iter + 1) ~state:new_state ~time_elapsed (loss :: running_avg)
  in
  (* ~config:(config_f ~iter:0) *)
  loop ~iter:0 ~state:(O.init ~config:(config_f ~iter:0) LGS.(init)) ~time_elapsed:0. []

let lr_rates = [ 1e-5; 1e-6; 1e-7 ]
let damping_list = [ Some 0.1 ]
let meth = "sofo"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    List.iter damping_list ~f:(fun gamma ->
      let config_f = config ~base_lr:eta ~gamma in
      let gamma_name = Option.value_map gamma ~default:"none" ~f:Float.to_string in
      let f_name = sprintf "lgs_%s_lr_%s_damp_%s" meth (Float.to_string eta) gamma_name in
      Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
      Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
      optimise ~max_iter ~f_name config_f))

(* let lr_rates = [ 0.01 ]
let meth = "adam"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    let config_f = config ~base_lr:eta ~gamma:None in
    let f_name = sprintf "lgs_%s_lr_%s" meth (Float.to_string eta) in
    Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
    Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
    optimise ~max_iter ~f_name config_f) *)
