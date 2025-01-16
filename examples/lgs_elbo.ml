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
  let a = 8
  let b = 8
  let o = 8
  let tmax = 10
  let m = 512
  let batch_const = true
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.cuda_if_available ()
end

module Data = Lds_data.Make_LDS_Tensor (Dims)

let x0 = Tensor.zeros ~device:Dims.device ~kind:Dims.kind [ Dims.m; Dims.a ]

(* in the linear gaussian case, _Fx, _Fu, c, b and cov invariant across time *)
(* TODO: noise cov as identity as the simplest case *)
let _std_o = Tensor.eye ~options:(Dims.kind, Dims.device) ~n:Dims.o
let _std_u = Tensor.ones ~device:Dims.device ~kind:Dims.kind [ Dims.b ]
let _Fx = Data.sample_fx ()
let _Fu = Data.sample_fu ()
let c = Data.sample_c ()
let b = Data.sample_b ()

let f_list : Tensor.t Lds_data.f_params list =
  List.init (Dims.tmax + 1) ~f:(fun _ ->
    Lds_data.
      { _Fx_prod = _Fx
      ; _Fu_prod = _Fu
      ; _f = None
      ; _c = Some c
      ; _b = Some b
      ; _cov = Some Tensor.(matmul _std_o (transpose _std_o ~dim0:1 ~dim1:0))
      })

let sample_data () =
  (* generate ground truth params and data *)
  let u_list = Data.sample_u_list ~std_u:_std_u in
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
let laplace = false
let sample = false

module LGS = struct
  module PP = struct
    type 'a p =
      { _Fx_prod : 'a (* generative model *)
      ; _Fu_prod : 'a
      ; _c : 'a
      ; _b : 'a
      ; _std_o : 'a (* sqrt of the diagonal of covariance of emission noise *)
      ; _std_u : 'a (* sqrt of the diagonal of covariance of prior over u *)
      ; _std_space : 'a
        (* recognition model; sqrt of the diagonal of covariance of space factor *)
      ; _std_time : 'a (* sqrt of the diagonal of covariance of the time factor *)
      }
    [@@deriving prms]
  end

  module P = PP.Make (Prms.P)

  type args = unit (* beta *)
  type data = Tensor.t * Tensor.t list

  (* 1/ (x^2) *)
  let sqr_inv x = Maths.(1. $/ sqr x)

  (* list of length T of [m x b] to matrix of [m x b x T]*)
  let concat_time u_list =
    List.map u_list ~f:(fun u -> Maths.unsqueeze ~dim:(-1) u) |> Maths.concat_list ~dim:2

  (* special care to be taken when dealing with elbo loss *)
  module Elbo_loss = struct
    let vtgt_hessian_gv ~rolled_out_x_list ~u_list (theta : P.M.t) =
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
        let like_hess = theta._std_o |> sqr_inv |> Maths.primal in
        (* y = cx + b *)
        let y_list =
          List.map rolled_out_x_list ~f:(fun x ->
            Maths.(einsum [ x, "ma"; theta._c, "ab" ] "mb" + theta._b))
        in
        ggn_final ~o_list:y_list ~like_hess ~diagonal:true
      in
      let prior_ggn =
        let like_hess = theta._std_u |> sqr_inv |> Maths.primal in
        ggn_final ~o_list:u_list ~like_hess ~diagonal:true
      in
      let entropy_ggn =
        let _cov_space_inv = theta._std_space |> sqr_inv |> Maths.primal in
        let _cov_time_inv = theta._std_time |> sqr_inv |> Maths.primal in
        let vtgt =
          let vtgt_list =
            List.map u_list ~f:(fun u ->
              let vtgt = Maths.tangent u |> Option.value_exn in
              Tensor.unsqueeze vtgt ~dim:(-1))
          in
          Tensor.concat vtgt_list ~dim:(-1)
        in
        let tmp1 =
          Tensor.einsum ~equation:"kmbt,b->kmbt" [ vtgt; _cov_space_inv ] ~path:None
        in
        let tmp2 =
          Tensor.einsum ~equation:"t,kmbt->tbmk" [ _cov_time_inv; vtgt ] ~path:None
        in
        Tensor.einsum ~equation:"kmbt,tbmj->kj" [ tmp1; tmp2 ] ~path:None |> Tensor.neg
      in
      (* TODO: do not include entropy term for now *)
      let final = Tensor.(div_scalar llh_ggn (Scalar.f (Float.of_int Dims.tmax))) in
      let _, final_s, _ = Tensor.svd ~some:true ~compute_uv:false final in
      let final_s =
        final_s
        |> Tensor.reshape ~shape:[ -1; 1 ]
        |> Tensor.to_bigarray ~kind:base.ba_kind
      in
      Owl.Mat.save_txt ~out:(in_dir (sprintf "svals")) final_s;
      final
  end

  (* create params for lds from f *)
  let params_from_f ~(theta : P.M.t) ~x0 ~o_list
    : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
    =
    (* set o at time 0 as 0 *)
    let o_list_tmp = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
    let _cov_u_inv =
      theta._std_u |> sqr_inv |> Maths.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
    in
    let c_trans = Maths.transpose theta._c ~dim0:1 ~dim1:0 in
    let _cov_o_inv = sqr_inv theta._std_o in
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
              let tmp = Maths.(einsum [ const o, "ab"; _cov_o_inv, "b" ] "ab") in
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
  let pred_u ~data (theta : P.M.t) =
    let x0, o_list = data in
    (* use lqr to obtain the optimal u *)
    let p =
      params_from_f ~x0:(Maths.const x0) ~theta ~o_list
      |> Lds_data.map_naive ~batch_const:Dims.batch_const
    in
    let sol, _ = Lqr._solve ~laplace ~batch_const:Dims.batch_const p in
    let optimal_u_list = List.map sol ~f:(fun s -> s.u) in
    (* sample u from the kronecker formation *)
    let u_list =
      let optimal_u = concat_time optimal_u_list in
      let xi =
        Tensor.randn ~device:Dims.device ~kind:Dims.kind [ Dims.m; Dims.b; Dims.tmax ]
        |> Maths.const
      in
      let _chol_space = Maths.abs theta._std_space in
      let _chol_time = Maths.abs theta._std_time in
      let xi_space = Maths.einsum [ xi, "mbt"; _chol_space, "b" ] "mbt" in
      let xi_time = Maths.einsum [ xi_space, "mat"; _chol_time, "t" ] "mat" in
      let meaned = Maths.(xi_time + optimal_u) in
      List.init Dims.tmax ~f:(fun i ->
        Maths.slice ~dim:2 ~start:(Some i) ~end_:(Some (i + 1)) ~step:1 meaned
        |> Maths.reshape ~shape:[ Dims.m; Dims.b ])
    in
    optimal_u_list, u_list

  (* gaussian llh with diagonal covariance *)
  let gaussian_llh ~g_mean ~g_std ~x =
    let g_cov_inv = sqr_inv g_std in
    let error_term =
      let error = Maths.(x - g_mean) in
      let tmp = Maths.einsum [ error, "ma"; g_cov_inv, "a" ] "ma" in
      Maths.einsum [ tmp, "ma"; error, "ma" ] "m" |> Maths.reshape ~shape:[ -1; 1 ]
    in
    let cov_term = Maths.(2. $* sum (log (abs g_std))) |> Maths.reshape ~shape:[ 1; 1 ] in
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
            ~g_std:theta._std_o
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
            let increment = gaussian_llh ~g_mean:u_zeros ~g_std:theta._std_u ~x:u in
            match accu with
            | None -> Some increment
            | Some accu -> Some Maths.(accu + increment))
          |> Option.value_exn
        in
        let entropy =
          let u = concat_time u_list |> Maths.reshape ~shape:[ Dims.m; -1 ] in
          let optimal_u = Maths.reshape optimal_u ~shape:[ Dims.m; -1 ] in
          let g_std = Maths.kron theta._std_space theta._std_time in
          gaussian_llh ~g_mean:optimal_u ~g_std ~x:u
        in
        Maths.(entropy - prior))
      else (
        (* M2: calculate the kl term analytically *)
        let std2 = Maths.kron theta._std_space theta._std_time in
        let det1 = Maths.(2. $* sum (log (abs std2))) in
        let det2 =
          Maths.(Float.(2. * of_int Dims.tmax) $* sum (log (abs theta._std_u)))
        in
        let _const = Tensor.of_float0 (Float.of_int Dims.b) |> Maths.const in
        let tr =
          let tmp1 = theta._std_u |> sqr_inv in
          let tmp2 = Maths.(tmp1 * sqr theta._std_space) in
          let tmp3 = Maths.(kron tmp2 (sqr theta._std_time)) in
          Maths.sum tmp3
        in
        let quad =
          let _cov_u = theta._std_u |> sqr_inv in
          let tmp1 = Maths.einsum [ optimal_u, "mbt"; _cov_u, "b" ] "mbt" in
          Maths.einsum [ tmp1, "mbt"; optimal_u, "mbt" ] "m" |> Maths.unsqueeze ~dim:1
        in
        let tmp = Maths.(det2 - det1 - _const + tr) |> Maths.reshape ~shape:[ 1; 1 ] in
        Maths.(tmp + quad) |> Maths.squeeze ~dim:1)
    in
    Maths.((llh - kl) /$ Float.of_int Dims.tmax)

  let f ~update ~data ~init ~args:() (theta : P.M.t) =
    let module L = Elbo_loss in
    let x0, o_list = data in
    let optimal_u_list, u_list = pred_u ~data theta in
    let rolled_out_x_list = rollout_x ~u_list ~x0 theta in
    (* These lists go from 1 to T *)
    let o_except_first = List.map o_list ~f:(fun o -> Maths.const o) in
    let x_except_first = List.tl_exn rolled_out_x_list in
    let x_o_list = List.map2_exn x_except_first o_except_first ~f:(fun x o -> x, o) in
    let neg_elbo = Maths.(neg (elbo ~x_o_list ~u_list ~optimal_u_list theta ~sample)) in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let ggn = L.vtgt_hessian_gv ~rolled_out_x_list:x_except_first ~u_list theta in
      u init (Some (neg_elbo, Some ggn))

  (* TODO: here we only learn _Fx, _Fu, _c and _b *)
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
    let _std_o = Tensor.diag ~diagonal:0 _std_o |> Prms.const in
    (* Tensor.(
        f 0.1 * (ones ~device:Dims.device ~kind:Dims.kind [ Dims.o ]) )
      |> Prms.free
    in *)
    let _std_u = _std_u |> Prms.const in
    (* Tensor.(
         f 0.1 * (ones ~device:Dims.device ~kind:Dims.kind [ Dims.b ]))
      |> Prms.free
    in *)
    let _std_space =
      Tensor.(f 1. * ones ~device:Dims.device ~kind:Dims.kind [ Dims.b ]) |> Prms.free
    in
    let _std_time =
      Tensor.(f 1. * ones ~device:Dims.device ~kind:Dims.kind [ Dims.tmax ]) |> Prms.free
    in
    { _Fx_prod; _Fu_prod; _c; _b; _std_o; _std_u; _std_space; _std_time }

  (* calculate the error between latents *)
  let simulate ~data ~(theta : P.M.t) =
    (* rollout under the given u *)
    let x0, x_list, u_list = data in
    (* rollout to obtain x *)
    let rolled_out_x_list =
      rollout_x ~u_list:(List.map u_list ~f:Maths.const) ~x0 theta
    in
    let error =
      List.fold2_exn rolled_out_x_list x_list ~init:0. ~f:(fun accu x1 x2 ->
        let error = Tensor.(norm (Maths.primal x1 - x2)) |> Tensor.to_float0_exn in
        accu +. error)
    in
    Float.(error / of_int Dims.tmax)
end

let config ~base_lr ~gamma ~iter:_ =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some base_lr
    ; n_tangents = 128
    ; rank_one = false
    ; damping = gamma
    ; momentum = None
    ; lm = false
    ; perturb_thresh = None
    }

module O = Optimizer.SOFO (LGS)

(* let config ~base_lr ~gamma:_ ~iter:_ =
  Optimizer.Config.Adam.{ default with learning_rate = Some base_lr }

module O = Optimizer.Adam (LGS)  *)

let optimise ~max_iter ~f_name ~init config_f =
  let rec loop ~iter ~state ~time_elapsed running_avg =
    Stdlib.Gc.major ();
    let config = config_f ~iter in
    let u_list, x_list, o_list = sample_data () in
    let data = x0, o_list in
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
        (* ground truth elbo *)
        let elbo_true =
          let theta_true =
            let theta_curr = O.params new_state in
            let _Fx_prod = Maths.const _Fx in
            let _Fu_prod = Maths.const _Fu in
            let _c = Maths.const c in
            let _b = Maths.const b in
            let _std_o = Tensor.diag ~diagonal:0 _std_o |> Maths.const in
            let _std_u = _std_u |> Maths.const in
            let _std_space = theta_curr._std_space |> Prms.value |> Maths.const in
            let _std_time = theta_curr._std_time |> Prms.value |> Maths.const in
            LGS.PP.{ _Fx_prod; _Fu_prod; _c; _b; _std_o; _std_u; _std_space; _std_time }
          in
          let u_list = List.map u_list ~f:Maths.const in
          let x_o_list =
            let x_except_first = List.tl_exn x_list in
            List.map2_exn x_except_first o_list ~f:(fun x o ->
              Maths.const x, Maths.const o)
          in
          let elbo_tmp =
            LGS.elbo ~x_o_list ~u_list ~optimal_u_list:u_list ~sample theta_true
            |> Maths.primal
            |> Tensor.neg
            |> Tensor.mean
            |> Tensor.to_float0_exn
          in
          Float.(elbo_tmp / of_int Dims.tmax)
        in
        (* simulation error *)
        let o_error =
          let u_list, x_list, _ = sample_data () in
          let data = x0, x_list, u_list in
          LGS.simulate ~theta:(LGS.P.const (LGS.P.value (O.params new_state))) ~data
        in
        (* avg error *)
        Convenience.print [%message (iter : int) (loss_avg : float)];
        let t = iter in
        Owl.Mat.(
          save_txt
            ~append:true
            ~out:(in_dir f_name)
            (of_array
               [| Float.of_int t; time_elapsed; loss_avg; o_error; elbo_true |]
               1
               5));
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

let lr_rates = [ 1e-4 ]
let damping_list = [ Some 0.1 ]
let meth = "sofo"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    List.iter damping_list ~f:(fun gamma ->
      let config_f = config ~base_lr:eta ~gamma in
      let gamma_name = Option.value_map gamma ~default:"none" ~f:Float.to_string in
      let init, f_name =
        ( LGS.(init)
        , sprintf
            "lgs_elbo_%s_lr_%s_damp_%s_sample_%s"
            meth
            (Float.to_string eta)
            gamma_name
            (Bool.to_string sample) )
      in
      Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
      Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
      optimise ~max_iter ~f_name ~init config_f))

(* let lr_rates = [ 0.1 ]
let meth = "adam"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    let config_f = config ~base_lr:eta ~gamma:None in
    let init, f_name =
      LGS.(init), sprintf "lgs_elbo_%s_lr_%s_sample_%s" meth (Float.to_string eta) (Bool.to_string sample)
    in
    Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
    Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
    optimise ~max_iter ~f_name ~init config_f) *)
