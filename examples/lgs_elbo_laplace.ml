(* Linear Gaussian Dynamics, with same state/control/cost parameters constant across trials and across time; use Laplace approximation in recognition model *)
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
  let o = 28
  let tmax = 10
  let m = 256
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
  (* spectral mixture model *)
  (* simple case - two components mixure*)
  module SM = struct
    type 'a p =
      { _w_1 : 'a
      ; _nu_1 : 'a
      ; _mu_1 : 'a
      ; _w_2 : 'a
      ; _nu_2 : 'a
      ; _mu_2 : 'a
      }
    [@@deriving prms]
  end

  (* generative model *)
  module Gen = struct
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

  module PP = struct
    type ('a, 'p) p =
      { gen : 'a
      ; sm : 'p
      }
    [@@deriving prms]
  end

  module P = PP.Make (Gen.Make (Prms.P)) (Prms.List (SM.Make (Prms.P)))

  type args = unit (* beta *)
  type data = Tensor.t * Tensor.t list

  let sqr_inv x =
    let x_sqr = Maths.sqr x in
    let tmp = Tensor.of_float0 1. |> Maths.const in
    Maths.(tmp / x_sqr)

  (* list of length T of [m x b] to matrix of [m x b x T]*)
  let concat_time u_list =
    List.map u_list ~f:(fun u -> Maths.unsqueeze ~dim:(-1) u) |> Maths.concat_list ~dim:2

  (* special care to be taken when dealing with elbo loss *)
  module Elbo_loss = struct
    let vtgt_hessian_gv ~rolled_out_x_list ~u_list ~_Phi_T ~_Phi_M_chol (theta : P.M.t) =
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
        let like_hess = theta.gen._cov_o |> sqr_inv |> Maths.primal in
        (* y = cx + b *)
        let y_list =
          List.map rolled_out_x_list ~f:(fun x ->
            Maths.(einsum [ x, "ma"; theta.gen._c, "ab" ] "mb" + theta.gen._b))
        in
        ggn_final ~o_list:y_list ~like_hess ~diagonal:true
      in
      let prior_ggn =
        let like_hess = theta.gen._cov_u |> sqr_inv |> Maths.primal in
        ggn_final ~o_list:u_list ~like_hess ~diagonal:true
      in
      let entropy_ggn =
        let cov2_inv =
          let _Phi_T_chol = Tensor.linalg_cholesky (Maths.primal _Phi_T) ~upper:false in
          let cov2_chol = Tensor.matmul (Maths.primal _Phi_M_chol) _Phi_T_chol in
          let cov2_chol_inv = Tensor.linalg_inv ~a:cov2_chol in
          Tensor.(matmul cov2_chol_inv (transpose cov2_chol_inv ~dim0:1 ~dim1:0))
        in
        let vtgt =
          let vtgt_list =
            List.map u_list ~f:(fun u ->
              let vtgt = Maths.tangent u |> Option.value_exn in
              Tensor.unsqueeze vtgt ~dim:(-1))
          in
          (* [k x m x b x T] *)
          Tensor.concat vtgt_list ~dim:(-1) (* [k x m x T x b] *)
          |> Tensor.transpose ~dim0:2 ~dim1:3
          |> Tensor.reshape ~shape:[ -1; Dims.m; Dims.tmax * Dims.b ]
        in
        let tmp1 = Tensor.einsum ~equation:"kmc,cd->kmd" [ vtgt; cov2_inv ] ~path:None in
        Tensor.einsum ~equation:"kmd,jmd->kj" [ tmp1; vtgt ] ~path:None |> Tensor.neg
      in
      let final = Tensor.(llh_ggn + prior_ggn + entropy_ggn) in
      final
  end

  (* create params for lds from f *)
  let params_from_f ~(theta : P.M.t) ~x0 ~o_list
    : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
    =
    (* set o at time 0 as 0 *)
    let o_list_tmp = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
    let _cov_u_inv =
      theta.gen._cov_u |> sqr_inv |> Maths.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
    in
    let c_trans = Maths.transpose theta.gen._c ~dim0:1 ~dim1:0 in
    let _cov_o_inv = sqr_inv theta.gen._cov_o in
    let _Cxx =
      let tmp = Maths.(einsum [ theta.gen._c, "ab"; _cov_o_inv, "b" ] "ab") in
      Maths.(tmp *@ c_trans)
    in
    let _cx_common =
      let tmp = Maths.(einsum [ theta.gen._b, "ab"; _cov_o_inv, "b" ] "ab") in
      Maths.(tmp *@ c_trans)
    in
    Lqr.Params.
      { x0 = Some x0
      ; params =
          List.map o_list_tmp ~f:(fun o ->
            let _cx =
              let tmp = Maths.(einsum [ const o, "ab"; _cov_o_inv, "b" ] "ab") in
              Maths.(_cx_common - (tmp *@ c_trans))
            in
            Lds_data.Temp.
              { _f = None
              ; _Fx_prod = theta.gen._Fx_prod
              ; _Fu_prod = theta.gen._Fu_prod
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
        let new_x =
          Maths.(tmp_einsum x theta.gen._Fx_prod + tmp_einsum u theta.gen._Fu_prod)
        in
        new_x, new_x :: x_list)
    in
    List.rev x_list

  (* b by b diagonal matrix *)
  let _K ~tau (theta : P.M.t) =
    let const1 =
      Float.(-2. * square Float.pi * square (of_int tau))
      |> Tensor.of_float0 ~device:Dims.device
      |> Tensor.to_kind ~kind:Dims.kind
      |> Maths.const
    in
    let const2 =
      Float.(2. * Float.pi * of_int tau)
      |> Tensor.of_float0 ~device:Dims.device
      |> Tensor.to_kind ~kind:Dims.kind
      |> Maths.const
    in
    let k_array =
      List.map theta.sm ~f:(fun sm ->
        let comp1 = Maths.(sm._w_1 * exp (const1 * sm._nu_1) * cos (const2 * sm._mu_1)) in
        let comp2 = Maths.(sm._w_2 * exp (const1 * sm._nu_2) * cos (const2 * sm._mu_2)) in
        Maths.(comp1 + comp2) |> Maths.reshape ~shape:[ 1; 1 ])
      |> Maths.concat_list ~dim:1
      |> Maths.squeeze ~dim:0
    in
    Maths.diag_embed k_array ~offset:0 ~dim1:(-2) ~dim2:(-1)

  (* build the correlation matrix *)
  let _Phi_T (theta : P.M.t) =
    let _K_array = Array.init Dims.tmax ~f:(fun tau -> _K ~tau theta) in
    let corr =
      List.init Dims.tmax ~f:(fun i ->
        (* the ith row of the correlation matrix *)
        List.init Dims.tmax ~f:(fun j ->
          let tmp = Array.get _K_array Int.(abs (j - i)) in
          tmp |> Maths.unsqueeze ~dim:0)
        |> Maths.concat_list ~dim:0
        |> Maths.unsqueeze ~dim:0)
      |> Maths.concat_list ~dim:0
      |> Maths.transpose ~dim0:1 ~dim1:2
      |> Maths.reshape ~shape:[ Dims.b * Dims.tmax; Dims.b * Dims.tmax ]
    in
    corr

  (* build the u covariance matrix. TODO: assume batch const so u_cov are 2 by 2 matrices (no batch dim in front); otherwise need special care to block diag them *)
  let _Phi_M u_cov_list = Maths.block_diag u_cov_list

  (* optimal u determined from lqr *)
  let pred_u ~data (theta : P.M.t) =
    let x0, o_list = data in
    let x0_tan = Maths.const x0 in
    (* use lqr to obtain the optimal u *)
    let p =
      params_from_f ~x0:x0_tan ~theta ~o_list
      |> Lds_data.map_naive ~batch_const:Dims.batch_const
    in
    let sol, u_cov_list = Lqr._solve ~laplace:true ~batch_const:Dims.batch_const p in
    let optimal_u_list = List.map sol ~f:(fun s -> s.u) in
    (* both _Phi_T and _Phi_M have shape [Tb x Tb ] *)
    let _Phi_T, _Phi_M = _Phi_T theta, _Phi_M (Option.value_exn u_cov_list) in
    let _Phi_M_chol = Maths.cholesky _Phi_M in
    (* sample u from the laplace covariance *)
    let u_list =
      let optimal_u = concat_time optimal_u_list in
      let _Phi_T_chol = Maths.cholesky _Phi_T in
      let xi =
        Tensor.randn ~device:Dims.device ~kind:Dims.kind [ Dims.m; Dims.tmax * Dims.b ]
        |> Maths.const
      in
      let xi_time = Maths.einsum [ xi, "mt"; _Phi_T_chol, "tb" ] "mb" in
      let xi_time_space = Maths.einsum [ xi_time, "mb"; _Phi_M_chol, "bt" ] "mt" in
      let xi_reshaped =
        Maths.reshape xi_time_space ~shape:[ Dims.m; Dims.tmax; Dims.b ]
        |> Maths.transpose ~dim0:1 ~dim1:2
      in
      let meaned = Maths.(xi_reshaped + optimal_u) in
      List.init Dims.tmax ~f:(fun i ->
        Maths.slice ~dim:2 ~start:(Some i) ~end_:(Some (i + 1)) ~step:1 meaned
        |> Maths.reshape ~shape:[ Dims.m; Dims.b ])
    in
    optimal_u_list, u_list, _Phi_T, _Phi_M_chol

  (* gaussian llh with diagonal covariance *)
  let gaussian_llh ~g_mean ~g_cov ~x =
    let g_cov_inv = sqr_inv g_cov in
    let error_term =
      let error = Maths.(x - g_mean) in
      let tmp = Maths.einsum [ error, "ma"; g_cov_inv, "a" ] "ma" in
      Maths.einsum [ tmp, "ma"; error, "ma" ] "m" |> Maths.reshape ~shape:[ -1; 1 ]
    in
    let cov_term = Maths.(2. $* sum (log (abs g_cov))) |> Maths.reshape ~shape:[ 1; 1 ] in
    let const_term =
      let o = x |> Maths.primal |> Tensor.shape |> List.last_exn in
      Tensor.of_float0 ~device:Dims.device Float.(log (2. * pi) * of_int o)
      |> Tensor.reshape ~shape:[ 1; 1 ]
      |> Maths.const
    in
    Maths.(0.5 $* error_term + cov_term + const_term)
    |> Maths.(mean_dim ~keepdim:false ~dim:(Some [ 1 ]))
    |> Maths.neg

  let elbo ~x_o_list ~optimal_u_list ~_Phi_T ~_Phi_M_chol ~sample (theta : P.M.t) =
    (* calculate the likelihood term *)
    let llh =
      List.foldi x_o_list ~init:None ~f:(fun t accu (x, o) ->
        if t % 1 = 0 then Stdlib.Gc.major ();
        let increment =
          gaussian_llh
            ~g_mean:o
            ~g_cov:theta.gen._cov_o
            ~x:Maths.(tmp_einsum x theta.gen._c + theta.gen._b)
        in
        match accu with
        | None -> Some increment
        | Some accu -> Some Maths.(accu + increment))
      |> Option.value_exn
    in
    let optimal_u = concat_time optimal_u_list in
    (* calculate kl term analytically *)
    let kl =
      let cov2 = Maths.(_Phi_M_chol *@ _Phi_T *@ transpose _Phi_M_chol ~dim0:1 ~dim1:0) in
      (* determinant of a positive symmetric matrix is the sum of diagonals of its cholesky *)
      let det1 =
        let cov2_chol = Maths.cholesky cov2 in
        Maths.(diagonal cov2_chol ~offset:0) |> Maths.sum |> Maths.log
      in
      let det2 =
        Maths.(Float.(2. * of_int Dims.tmax) $* sum (log (abs theta.gen._cov_u)))
      in
      let _const = Tensor.of_float0 (Float.of_int Dims.b) |> Maths.const in
      let tr =
        let id =
          Tensor.eye ~n:Dims.tmax ~options:(Dims.kind, Dims.device) |> Maths.const
        in
        let _cov_u_embedded =
          Maths.diag_embed theta.gen._cov_u ~offset:0 ~dim2:(-1) ~dim1:(-2)
        in
        let sigma2 = Maths.(kron _cov_u_embedded id) in
        let sigma2_inv = Maths.inv_sqr sigma2 in
        Maths.(sigma2_inv *@ cov2) |> Maths.diagonal ~offset:0 |> Maths.sum
      in
      let quad =
        let _cov_u = theta.gen._cov_u |> sqr_inv in
        let tmp1 = Maths.einsum [ optimal_u, "mbt"; _cov_u, "b" ] "mbt" in
        Maths.einsum [ tmp1, "mbt"; optimal_u, "mbt" ] "m" |> Maths.unsqueeze ~dim:1
      in
      let tmp = Maths.(det2 - det1 - _const + tr) |> Maths.reshape ~shape:[ 1; 1 ] in
      Maths.(tmp + quad) |> Maths.squeeze ~dim:1
    in
    Maths.(llh - kl)

  (* calculate optimal u *)
  let f ~update ~data ~init ~args:_ (theta : P.M.t) =
    let module L = Elbo_loss in
    let x0, o_list = data in
    let optimal_u_list, u_list, _Phi_T, _Phi_M_chol = pred_u ~data theta in
    let rolled_out_x_list = rollout_x ~u_list ~x0 theta in
    (* These lists go from 1 to T *)
    let o_except_first = List.map o_list ~f:(fun o -> Maths.const o) in
    let x_except_first = List.tl_exn rolled_out_x_list in
    let x_o_list = List.map2_exn x_except_first o_except_first ~f:(fun x o -> x, o) in
    let neg_elbo =
      Maths.(
        neg (elbo ~x_o_list ~optimal_u_list theta ~sample:false ~_Phi_T ~_Phi_M_chol))
    in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let ggn =
        L.vtgt_hessian_gv
          ~rolled_out_x_list:x_except_first
          ~u_list
          ~_Phi_T
          ~_Phi_M_chol
          theta
      in
      u init (Some (neg_elbo, Some ggn))

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
        ~sigma:0.1
      |> Prms.free
    in
    let _b =
      Convenience.gaussian_tensor_2d_normed
        ~device:Dims.device
        ~kind:Dims.kind
        ~a:1
        ~b:Dims.o
        ~sigma:0.1
      |> Prms.free
    in
    let _cov_o =
      Tensor.(
        mul_scalar (ones ~device:Dims.device ~kind:Dims.kind [ Dims.o ]) (Scalar.f 1.))
      |> Prms.free
    in
    let _cov_u =
      Tensor.(
        mul_scalar (ones ~device:Dims.device ~kind:Dims.kind [ Dims.b ]) (Scalar.f 1.))
      |> Prms.free
    in
    let one = Tensor.of_float0 1. |> Tensor.to_type ~type_:Dims.kind |> Prms.free in
    PP.
      { gen = { _Fx_prod; _Fu_prod; _c; _b; _cov_o; _cov_u }
      ; sm =
          List.init Dims.b ~f:(fun _ ->
            SM.
              { _w_1 = one
              ; _nu_1 = one
              ; _mu_1 = one
              ; _w_2 = one
              ; _nu_2 = one
              ; _mu_2 = one
              })
      }

  let simulate ~theta ~data =
    (* infer the optimal u *)
    let optimal_u_list, _, _, _ = pred_u ~data theta in
    (* rollout to obtain x *)
    let rolled_out_x_list = rollout_x ~u_list:optimal_u_list ~x0 theta in
    (* noiseless observation *)
    let noiseless_o_list =
      List.map rolled_out_x_list ~f:(fun x -> Maths.((x *@ theta.gen._c) + theta.gen._b))
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
    ; n_tangents = 256 * 2
    ; rank_one = false
    ; damping = gamma
    ; momentum = None
    ; lm = false
    ; perturb_thresh = None
    ; sqrt = false
    }

module O = Optimizer.SOFO (LGS)

(* let config ~base_lr ~gamma:_ ~iter:_ =
  Optimizer.Config.Adam.{ default with learning_rate = Some base_lr }

module O = Optimizer.Adam (LGS) *)

let optimise ~max_iter ~f_name ~init config_f =
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
  loop ~iter:0 ~state:(O.init ~config:(config_f ~iter:0) init) ~time_elapsed:0. []

(* let checkpoint_name = Some "lgs_elbo_sofo_lr_0.01_damp_0.1" *)

let checkpoint_name = None
let lr_rates = [ 0.01 ]
let damping_list = [ Some 1e-1 ]
let meth = "sofo"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    List.iter damping_list ~f:(fun gamma ->
      let config_f = config ~base_lr:eta ~gamma in
      let gamma_name = Option.value_map gamma ~default:"none" ~f:Float.to_string in
      let init, f_name =
        match checkpoint_name with
        | None ->
          ( LGS.(init)
          , sprintf
              "lgs_elbo_laplace_%s_lr_%s_damp_%s"
              meth
              (Float.to_string eta)
              gamma_name )
        | Some checkpoint_name ->
          let params_ba =
            O.W.P.T.load ~device:Dims.device (in_dir checkpoint_name ^ "_params")
          in
          ( LGS.P.map params_ba ~f:(fun x ->
              (* randomly perturb to escape local minimum *)
              let x_perturbed = Tensor.(x + mul_scalar (rand_like x) (Scalar.f 0.01)) in
              Prms.free x_perturbed)
          , sprintf
              "lgs_elbo_laplace_%s_lr_%s_damp_%s_%s"
              meth
              (Float.to_string eta)
              gamma_name
              checkpoint_name )
      in
      Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
      Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
      optimise ~max_iter ~f_name ~init config_f))

(* let lr_rates = [ 0.01 ]
let meth = "adam"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    let config_f = config ~base_lr:eta ~gamma:None in
    let init, f_name =
      match checkpoint_name with
      | None -> LGS.(init), sprintf "lgs_elbo_laplace_%s_lr_%s" meth (Float.to_string eta)
      | Some checkpoint_name ->
        let params_ba =
          O.W.P.T.load ~device:Dims.device (in_dir checkpoint_name ^ "_params")
        in
        ( LGS.P.map params_ba ~f:(fun x -> Prms.free x)
        , sprintf "lgs_elbo_laplace_%s_lr_%s_%s" meth (Float.to_string eta) checkpoint_name )
    in
    Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
    Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
    optimise ~max_iter ~f_name ~init config_f) *)
