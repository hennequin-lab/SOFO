(* Linear Gaussian Dynamics, with same state/control/cost parameters constant across trials and across time *)
open Printf
open Base
open Forward_torch
open Torch
open Sofo
module Mat = Owl.Dense.Matrix.D
module Arr = Owl.Dense.Ndarray.D
module Linalg = Owl.Linalg.D
module Z = Owl.Dense.Matrix.Z

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let n_fisher = 100

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f64)
    ; ba_kind = Bigarray.float64
    }

(* -----------------------------------------
   ----- Utilitiy Functions ------
   ----------------------------------------- *)

let in_dir = Cmdargs.in_dir "-d"
let tmp_einsum a b = Maths.einsum [ a, "ma"; b, "ab" ] "mb"

(* make sure fx is stable *)
let sample_stable ~a =
  let w =
    let tmp = Mat.gaussian a a in
    let r = tmp |> Linalg.eigvals |> Z.re |> Mat.max' in
    Mat.(Float.(0.8 / r) $* tmp)
  in
  let w_i = Mat.((w - eye a) *$ 0.1) in
  Owl.Linalg.Generic.expm w_i

let save_svals m =
  let _, s, _ = Tensor.svd ~some:true ~compute_uv:false m in
  s
  |> Tensor.reshape ~shape:[ -1; 1 ]
  |> Tensor.to_bigarray ~kind:base.ba_kind
  |> Owl.Mat.save_txt ~out:(in_dir (sprintf "svals"))

(* -----------------------------------------
   -- Control Problem / Data Generation ---
   ----------------------------------------- *)
let n = 24
let m = 10
let o = 40
let tmax = 10
let bs = 128
let batch_const = true
let x0 = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n ]

(* x_list goes from 0 to T and o list goes from 1 to T. *)
let traj_rollout ~x0 ~_Fx ~_Fu ~_c ~_std_o ~u_list =
  let tmp_einsum a b = Tensor.einsum ~equation:"ma,ab->mb" [ a; b ] ~path:None in
  let _, x_list, o_list =
    List.fold u_list ~init:(x0, [ x0 ], []) ~f:(fun (x, x_list, o_list) u ->
      let new_x = Tensor.(tmp_einsum x _Fx + tmp_einsum u _Fu) in
      let new_o =
        let noise =
          let eps = Tensor.randn ~device:base.device ~kind:base.kind [ bs; o ] in
          Tensor.einsum ~equation:"ma,a->ma" [ eps; _std_o ] ~path:None
        in
        Tensor.(noise + tmp_einsum new_x _c)
      in
      new_x, new_x :: x_list, new_o :: o_list)
  in
  List.rev x_list, List.rev o_list

(* in the linear gaussian case, _Fx, _Fu, c and cov invariant across time *)
let _std_o = Tensor.(ones ~device:base.device ~kind:base.kind [ o ])
let _std_o_log = Tensor.(log _std_o)
let _Fx = sample_stable ~a:n |> Tensor.of_bigarray ~device:base.device
let _Fu = Tensor.randn ~device:base.device ~kind:base.kind [ m; n ]
let _c = Tensor.randn ~device:base.device ~kind:base.kind [ n; o ]

let sample_data () =
  (* generate ground truth params and data *)
  let u_list =
    List.init tmax ~f:(fun _ ->
      Tensor.(randn ~device:base.device ~kind:base.kind [ bs; m ]))
  in
  let x_list, o_list = traj_rollout ~x0 ~_Fx ~_Fu ~_c ~_std_o ~u_list in
  u_list, x_list, o_list

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f64)
    ; ba_kind = Bigarray.float64
    }

let laplace = false
let step = 0.1
let std_o_weight = 10000.

module PP = struct
  (* note that all std live in log space *)
  type 'a p =
    { _S_params : 'a
    ; _L_params : 'a
      (* generative model; Fx = (1-step) I + step * W, W = I + (-I + S - S^T) LL^T *)
      (* _Fx_prod_params : 'a *)
    ; _Fu_prod_params : 'a
    ; _c_params : 'a
    ; _std_o_params : 'a (* sqrt of the diagonal of covariance of emission noise *)
    ; _std_space_params : 'a
      (* recognition model; sqrt of the diagonal of covariance of space factor *)
    ; _std_time_params : 'a (* sqrt of the diagonal of covariance of the time factor *)
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

module LGS = struct
  module P = P

  type args = unit
  type data = Tensor.t list

  let _Fx_prod (theta : P.M.t) =
    (* theta._Fx_prod_params *)
    let eye_tmp = Maths.const (Tensor.eye ~n ~options:(base.kind, base.device)) in
    let _W =
      Maths.(
        eye_tmp
        + ((theta._S_params - transpose theta._S_params ~dim0:1 ~dim1:0 - eye_tmp)
           *@ (theta._L_params *@ transpose theta._L_params ~dim0:1 ~dim1:0)))
    in
    let tmp1 = Maths.(f (1. -. step) * eye_tmp) in
    let tmp2 = Maths.(f step * _W) in
    Maths.(tmp1 + tmp2)

  let step ~_Fx_prod ~_Fu_prod ~u ~prev_x =
    Maths.(tmp_einsum prev_x _Fx_prod + tmp_einsum u _Fu_prod)

  (* 1/ (x^2) *)
  let sqr_inv x = Maths.(1. $/ sqr x)

  (* list of length T of [m x b] to matrix of [m x b x T]*)
  let concat_time u_list =
    List.map u_list ~f:(fun u -> Maths.unsqueeze ~dim:(-1) u) |> Maths.concat_list ~dim:2

  let gaussian_llh ?mu ?(fisher_batched = false) ~std x =
    let inv_std = Maths.(f 1. / std) in
    let error_term =
      if fisher_batched
      then (
        (* dimension l is number of fisher samples *)
        let error =
          match mu with
          | None -> Maths.(einsum [ x, "lma"; inv_std, "a" ] "lma")
          | Some mu -> Maths.(einsum [ x - mu, "lma"; inv_std, "a" ] "lma")
        in
        Maths.einsum [ error, "lma"; error, "lma" ] "lm")
      else (
        let error =
          match mu with
          | None -> Maths.(einsum [ x, "ma"; inv_std, "a" ] "ma")
          | Some mu -> Maths.(einsum [ x - mu, "ma"; inv_std, "a" ] "ma")
        in
        Maths.einsum [ error, "ma"; error, "ma" ] "m")
    in
    let cov_term =
      let cov_term_shape = if fisher_batched then [ 1; 1 ] else [ 1 ] in
      Maths.(sum (log (sqr std))) |> Maths.reshape ~shape:cov_term_shape
    in
    let const_term =
      let o = x |> Maths.primal |> Tensor.shape |> List.last_exn in
      Float.(log (2. * pi) * of_int o)
    in
    Maths.(0.5 $* (const_term $+ error_term + cov_term)) |> Maths.neg

  (* special care to be taken when dealing with elbo loss *)
  module Elbo_loss = struct
    let fisher ?(fisher_batched = false) ~n:_ lik_term =
      let neg_lik_t = Maths.(tangent lik_term) |> Option.value_exn in
      let n_tangents = List.hd_exn (Tensor.shape neg_lik_t) in
      let fisher =
        if fisher_batched
        then (
          let fisher_half =
            Tensor.reshape neg_lik_t ~shape:[ n_tangents; n_fisher; -1 ]
          in
          Tensor.einsum ~equation:"kla,jla->lkj" [ fisher_half; fisher_half ] ~path:None)
        else (
          let fisher_half = Tensor.reshape neg_lik_t ~shape:[ n_tangents; -1 ] in
          Tensor.(matmul fisher_half (transpose fisher_half ~dim0:0 ~dim1:1)))
      in
      fisher

    let llh_increment ~_std_o_vec ~_std_o_extended ~new_o =
      let noise =
        Tensor.(
          _std_o_extended
          * randn (n_fisher :: Maths.shape new_o) ~device:base.device ~kind:base.kind)
      in
      let new_o_unsqueezed =
        List.init n_fisher ~f:(fun _ -> Maths.unsqueeze new_o ~dim:0)
        |> Maths.concat_list ~dim:0
      in
      let o_samples_batched = Maths.(const Tensor.(Maths.primal new_o + noise)) in
      gaussian_llh
        ~mu:new_o_unsqueezed
        ~std:_std_o_vec
        ~fisher_batched:true
        o_samples_batched

    let true_fisher ~u_list (theta : P.M.t) =
      (* let _std_o_vec =
        Maths.(
          const (Tensor.ones ~device:base.device ~kind:base.kind [ o ])
          * theta._std_o_params)
        |> Maths.exp
      in *)
      let _std_o_vec = Maths.(exp theta._std_o_params) in
      let _std_o_extended =
        _std_o_vec |> Maths.primal |> Tensor.unsqueeze ~dim:0 |> Tensor.unsqueeze ~dim:0
      in
      let _Fx_prod = _Fx_prod theta in
      let _, llh_rollout =
        List.fold
          u_list
          ~init:(Maths.const x0, Maths.f 0.)
          ~f:(fun accu u ->
            let prev_x, llh_accu = accu in
            let new_x = step ~_Fx_prod ~_Fu_prod:theta._Fu_prod_params ~u ~prev_x in
            let new_o = tmp_einsum new_x theta._c_params in
            let increment = llh_increment ~_std_o_vec ~_std_o_extended ~new_o in
            Stdlib.Gc.major ();
            new_x, Maths.(increment + llh_accu))
      in
      let fisher_rollout =
        let fisher_batched = fisher ~n:o llh_rollout ~fisher_batched:true in
        Tensor.mean_dim fisher_batched ~dim:(Some [ 0 ]) ~keepdim:false ~dtype:base.kind
      in
      let final = Tensor.(fisher_rollout / f (Float.of_int tmax)) in
      save_svals final;
      final

    let ggn_increment ~new_o ~precision ~hess_y ~hess_sigma_o =
      let ggn_y_increment =
        let vtgt = Maths.tangent new_o |> Option.value_exn in
        let vtgt_hess =
          Tensor.einsum ~equation:"kma,a->kma" [ vtgt; hess_y ] ~path:None
        in
        Tensor.einsum ~equation:"kma,jma->kj" [ vtgt_hess; vtgt ] ~path:None
      in
      let ggn_sigma_increment =
        let vtgt = precision |> Maths.tangent |> Option.value_exn in
        (* let vtgt_h = Tensor.(hess_sigma_o * vtgt) in *)
        let vtgt_h =
          Tensor.einsum ~equation:"ka,a->ka" [ vtgt; hess_sigma_o ] ~path:None
        in
        Tensor.einsum ~equation:"ka,ja->kj" [ vtgt_h; vtgt ] ~path:None
      in
      Tensor.(ggn_y_increment + (f std_o_weight * ggn_sigma_increment))

    let ggn ~u_list (theta : P.M.t) =
      let _Fx_prod = _Fx_prod theta in
      let _std_o = Maths.exp theta._std_o_params in
      let _std_o_vec = _std_o in
      (* let _std_o_vec =
        Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]) * _std_o)
      in *)
      let precision = _std_o |> sqr_inv in
      let hess_y = _std_o_vec |> sqr_inv |> Maths.primal in
      let hess_sigma_o =
        Tensor.(f Float.(of_int bs / 2.) * square (square (Maths.primal _std_o)))
      in
      let _std_o_extended =
        _std_o_vec |> Maths.primal |> Tensor.unsqueeze ~dim:0 |> Tensor.unsqueeze ~dim:0
      in
      let _, ggn_rollout =
        List.fold
          u_list
          ~init:(Maths.const x0, Tensor.f 0.)
          ~f:(fun accu u ->
            let prev_x, ggn_accu = accu in
            let new_x =
              Maths.(tmp_einsum prev_x _Fx_prod + tmp_einsum u theta._Fu_prod_params)
            in
            let new_o = Maths.(tmp_einsum new_x theta._c_params) in
            let increment = ggn_increment ~new_o ~precision ~hess_y ~hess_sigma_o in
            Stdlib.Gc.major ();
            new_x, Tensor.(increment + ggn_accu))
      in
      let final = Tensor.(ggn_rollout / f (Float.of_int tmax)) in
      save_svals final;
      final

    let ggn_fisher ~u_list (theta : P.M.t) =
      let _Fx_prod = _Fx_prod theta in
      let _std_o = Maths.exp theta._std_o_params in
      let _std_o_vec = _std_o in
      (* let _std_o_vec =
        Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]) * _std_o)
      in *)
      let _std_o_extended =
        _std_o_vec |> Maths.primal |> Tensor.unsqueeze ~dim:0 |> Tensor.unsqueeze ~dim:0
      in
      let _, ggn_rollout, llh_rollout =
        List.fold
          u_list
          ~init:(Maths.const x0, Tensor.f 0., Maths.f 0.)
          ~f:(fun accu u ->
            let prev_x, ggn_accu, llh_accu = accu in
            let new_x =
              Maths.(tmp_einsum prev_x _Fx_prod + tmp_einsum u theta._Fu_prod_params)
            in
            let new_o = Maths.(tmp_einsum new_x theta._c_params) in
            (* ggn increment only operates on std_o *)
            let ggn_increment =
              let vtgt = _std_o |> sqr_inv |> Maths.tangent |> Option.value_exn in
              let hess_sigma_o =
                Tensor.(f Float.(of_int bs / 2.) * square (square (Maths.primal _std_o)))
              in
              let vtgt_h =
                Tensor.einsum ~equation:"ka,a->ka" [ vtgt; hess_sigma_o ] ~path:None
              in
              Tensor.einsum ~equation:"ka,ja->kj" [ vtgt_h; vtgt ] ~path:None
            in
            let llh_increment = llh_increment ~_std_o_vec ~_std_o_extended ~new_o in
            Stdlib.Gc.major ();
            new_x, Tensor.(ggn_increment + ggn_accu), Maths.(llh_increment + llh_accu))
      in
      let fisher_rollout =
        let fisher_batched = fisher ~n:o llh_rollout ~fisher_batched:true in
        Tensor.mean_dim fisher_batched ~dim:(Some [ 0 ]) ~keepdim:false ~dtype:base.kind
      in
      Tensor.((ggn_rollout + fisher_rollout) / f (Float.of_int tmax))
  end

  (* create params for lds from f *)
  let params_from_f ~(theta : P.M.t) ~x0 ~o_list
    : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
    =
    (* set o at time 0 as 0 *)
    let o_list_tmp = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
    let _cov_u_inv = Tensor.eye ~n:m ~options:(base.kind, base.device) |> Maths.const in
    let c_trans = Maths.transpose theta._c_params ~dim0:1 ~dim1:0 in
    (* let _std_o_vec =
      Maths.(
        const (Tensor.ones ~device:base.device ~kind:base.kind [ o ])
        * theta._std_o_params)
      |> Maths.exp
    in *)
    let _std_o = Maths.exp theta._std_o_params in
    let _std_o_vec = _std_o in
    let _cov_o_inv = _std_o_vec |> sqr_inv in
    let _Cxx =
      let tmp = Maths.(einsum [ theta._c_params, "ab"; _cov_o_inv, "b" ] "ab") in
      Maths.(tmp *@ c_trans)
    in
    let _Fx_prod = _Fx_prod theta in
    Lqr.Params.
      { x0 = Some x0
      ; params =
          List.map o_list_tmp ~f:(fun o ->
            let _cx =
              let tmp = Maths.(einsum [ const o, "ab"; _cov_o_inv, "b" ] "ab") in
              Maths.(neg (tmp *@ c_trans))
            in
            Lds_data.Temp.
              { _f = None
              ; _Fx_prod
              ; _Fu_prod = theta._Fu_prod_params
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
    let _Fx_prod = _Fx_prod theta in
    let _, x_list =
      List.fold u_list ~init:(x0_tan, [ x0_tan ]) ~f:(fun (x, x_list) u ->
        let new_x = step ~_Fx_prod ~_Fu_prod:theta._Fu_prod_params ~u ~prev_x:x in
        new_x, new_x :: x_list)
    in
    List.rev x_list

  (* optimal u determined from lqr *)
  let pred_u ~data:o_list (theta : P.M.t) =
    (* use lqr to obtain the optimal u *)
    let p =
      params_from_f ~x0:(Maths.const x0) ~theta ~o_list |> Lds_data.map_naive ~batch_const
    in
    let sol, _ = Lqr._solve ~laplace ~batch_const p in
    let optimal_u_list = List.map sol ~f:(fun s -> s.u) in
    (* sample u from the kronecker formation *)
    let u_list =
      let optimal_u = concat_time optimal_u_list in
      let xi =
        Tensor.randn ~device:base.device ~kind:base.kind [ bs; m; tmax ] |> Maths.const
      in
      let xi_space =
        Maths.einsum [ xi, "mbt"; Maths.exp theta._std_space_params, "b" ] "mbt"
      in
      let xi_time =
        Maths.einsum [ xi_space, "mat"; Maths.exp theta._std_time_params, "t" ] "mat"
      in
      let meaned = Maths.(xi_time + optimal_u) in
      List.init tmax ~f:(fun i ->
        Maths.slice ~dim:2 ~start:(Some i) ~end_:(Some (i + 1)) ~step:1 meaned
        |> Maths.reshape ~shape:[ bs; m ])
    in
    optimal_u_list, u_list

  let elbo ~o_list ~u_list ~optimal_u_list (theta : P.M.t) =
    (* calculate the likelihood term *)
    let u_o_list = List.map2_exn u_list o_list ~f:(fun u o -> u, o) in
    let _Fx_prod = _Fx_prod theta in
    (* let _std_o_vec =
      Maths.(
        const (Tensor.ones ~device:base.device ~kind:base.kind [ o ])
        * theta._std_o_params)
      |> Maths.exp
    in *)
    let _std_o_vec = Maths.exp theta._std_o_params in
    let llh =
      let _, llh =
        List.foldi
          u_o_list
          ~init:(Maths.const x0, None)
          ~f:(fun t accu (u, o) ->
            if t % 1 = 0 then Stdlib.Gc.major ();
            let prev_x, llh_summed = accu in
            let new_x = step ~_Fx_prod ~_Fu_prod:theta._Fu_prod_params ~u ~prev_x in
            let increment =
              gaussian_llh ~mu:o ~std:_std_o_vec (tmp_einsum new_x theta._c_params)
            in
            let new_llh_summed =
              match llh_summed with
              | None -> Some increment
              | Some accu -> Some Maths.(accu + increment)
            in
            Stdlib.Gc.major ();
            new_x, new_llh_summed)
      in
      Option.value_exn llh
    in
    let optimal_u = concat_time optimal_u_list in
    let kl =
      (* calculate the kl term analytically *)
      let std2 =
        Maths.(kron (exp theta._std_space_params) (exp theta._std_time_params))
      in
      let det1 = Maths.(2. $* sum (log std2)) in
      let _const = Float.of_int (m * tmax) in
      let tr =
        let tmp2 = Maths.(sqr (exp theta._std_space_params)) in
        let tmp3 = Maths.(kron tmp2 (sqr (exp theta._std_time_params))) in
        Maths.sum tmp3
      in
      let quad =
        Maths.einsum [ optimal_u, "mbt"; optimal_u, "mbt" ] "m" |> Maths.unsqueeze ~dim:1
      in
      let tmp = Maths.(tr - det1 -$ _const) |> Maths.reshape ~shape:[ 1; 1 ] in
      Maths.(tmp + quad) |> Maths.squeeze ~dim:1
    in
    Maths.((llh - kl) /$ Float.of_int tmax)

  let f ~update ~data ~init ~args:() (theta : P.M.t) =
    let module L = Elbo_loss in
    let optimal_u_list, u_list = pred_u ~data theta in
    let neg_elbo =
      Maths.(
        neg (elbo ~o_list:(List.map data ~f:Maths.const) ~u_list ~optimal_u_list theta))
    in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let ggn = L.ggn ~u_list theta in
      u init (Some (neg_elbo, Some ggn))

  let init : P.tagged =
    let _S_params =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:n
        ~b:n
        ~sigma:0.1
      |> Prms.free
    in
    let _L_params =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:n
        ~b:n
        ~sigma:0.1
      |> Prms.free
    in
    (* let _Fx_prod_params =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:n
        ~b:n
        ~sigma:0.1
      |> Prms.free 
    in *)
    let _Fu_prod_params =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:m
        ~b:n
        ~sigma:0.1
      |> Prms.free
    in
    let _c_params =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:n
        ~b:o
        ~sigma:1.
      |> Prms.free
    in
    let _std_o_params =
      Prms.create
        ~above:(Tensor.f (-5.))
        Tensor.(zeros ~device:base.device ~kind:base.kind [ o ])
      (* |> Prms.const *)
    in
    let _std_space_params =
      Prms.create
        ~above:(Tensor.f (-5.))
        Tensor.(zeros ~device:base.device ~kind:base.kind [ m ])
    in
    let _std_time_params =
      Prms.create
        ~above:(Tensor.f (-5.))
        Tensor.(zeros ~device:base.device ~kind:base.kind [ tmax ])
    in
    { _S_params
    ; _L_params (* _Fx_prod_params *)
    ; _Fu_prod_params
    ; _c_params
    ; _std_o_params
    ; _std_space_params
    ; _std_time_params
    }

  (* calculate the error between observations *)
  let simulate ~data ~(theta : P.M.t) =
    (* rollout under the given u *)
    let u_list, _, o_list = data in
    (* rollout to obtain x *)
    let rolled_out_x_list =
      rollout_x ~u_list:(List.map u_list ~f:Maths.const) ~x0 theta |> List.tl_exn
    in
    let rolled_out_o_list =
      List.map rolled_out_x_list ~f:(fun x -> tmp_einsum x theta._c_params)
    in
    let error =
      List.fold2_exn rolled_out_o_list o_list ~init:0. ~f:(fun accu x1 x2 ->
        let error = Tensor.(norm (Maths.primal x1 - x2)) |> Tensor.to_float0_exn in
        accu +. error)
    in
    Float.(error / of_int tmax)
end

(* --------------------------------
   -- Generic type of optimiser
   -------------------------------- *)

module type Do_with_T = sig
  module O :
    Optimizer.T
    with type 'a W.P.p = 'a LGS.P.p
     and type W.data = Tensor.t list
     and type W.args = unit

  val name : string
  val config_f : iter:int -> (float, Bigarray.float64_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state ~time_elapsed running_avg =
      Stdlib.Gc.major ();
      let _, _, o_list = sample_data () in
      let t0 = Unix.gettimeofday () in
      let config = config_f ~iter in
      let loss, new_state = O.step ~config ~state ~data:o_list ~args:() in
      let std_o_mean =
        let a = LGS.P.value (O.params state) in
        a._std_o_params |> Tensor.mean |> Tensor.to_float0_exn
      in
      let t1 = Unix.gettimeofday () in
      let time_elapsed = Float.(time_elapsed + t1 - t0) in
      let running_avg =
        let loss_avg =
          match running_avg with
          | [] -> loss
          | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
        in
        (* save params *)
        if iter % 10 = 0
        then (
          (* elbo when std_o_params is the true std_o *)
          let elbo_std_o_true =
            let theta_std_o_true =
              let theta_curr = O.params state in
              let _L_params = theta_curr._L_params |> Prms.value |> Maths.const in
              let _S_params = theta_curr._S_params |> Prms.value |> Maths.const in
              let _Fu_prod_params =
                theta_curr._Fu_prod_params |> Prms.value |> Maths.const
              in
              let _c_params = theta_curr._c_params |> Prms.value |> Maths.const in
              let _std_o_params = _std_o_log |> Maths.const in
              let _std_space_params =
                theta_curr._std_space_params |> Prms.value |> Maths.const
              in
              let _std_time_params =
                theta_curr._std_time_params |> Prms.value |> Maths.const
              in
              PP.
                { _S_params
                ; _L_params
                ; _Fu_prod_params
                ; _c_params
                ; _std_o_params
                ; _std_space_params
                ; _std_time_params
                }
            in
            let optimal_u_list, u_list = LGS.pred_u ~data:o_list theta_std_o_true in
            LGS.elbo
              ~o_list:(List.map o_list ~f:Maths.const)
              ~u_list
              ~optimal_u_list
              theta_std_o_true
            |> Maths.primal
            |> Tensor.neg
            |> Tensor.mean
            |> Tensor.to_float0_exn
          in
          (* simulation error *)
          let o_error =
            let data = sample_data () in
            LGS.simulate ~theta:(LGS.P.const (LGS.P.value (O.params new_state))) ~data
          in
          (* avg error *)
          Convenience.print [%message (iter : int) (loss_avg : float)];
          let t = iter in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array
                 [| Float.of_int t
                  ; time_elapsed
                  ; loss_avg
                  ; o_error
                  ; std_o_mean
                  ; elbo_std_o_true
                 |]
                 1
                 6));
          O.W.P.T.save
            (LGS.P.value (O.params new_state))
            ~kind:base.ba_kind
            ~out:(in_dir name ^ "_params"));
        []
      in
      if iter < max_iter
      then loop ~iter:(iter + 1) ~state:new_state ~time_elapsed (loss :: running_avg)
    in
    loop ~iter:0 ~state:init ~time_elapsed:0. []
end

(* --------------------------------
   -- SOFO
   -------------------------------- *)

module Do_with_SOFO : Do_with_T = struct
  module O = Optimizer.SOFO (LGS)

  let config_f ~iter =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some Float.(1. / (1. +. (0.0 * sqrt (of_int iter))))
      ; n_tangents = 128
      ; sqrt = false
      ; rank_one = false
      ; damping = None
      ; momentum = None
      ; lm = false
      ; perturb_thresh = None
      }

  let name =
    let init_config = config_f ~iter:0 in
    let gamma_name =
      Option.value_map init_config.damping ~default:"none" ~f:Float.to_string
    in
    sprintf
      "ggn_lr_%s_sqrt_%s_damp_%s_std_o_weight_%s"
      (Float.to_string (Option.value_exn init_config.learning_rate))
      (Bool.to_string init_config.sqrt)
      gamma_name
      (Float.to_string std_o_weight)

  let init = O.init ~config:(config_f ~iter:0) LGS.init
end

(* --------------------------------
     -- Adam
     -------------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (LGS)

  let config_f ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.01 }

  let init = O.init LGS.init
end

let _ =
  let max_iter = 10000 in
  let optimise =
    match Cmdargs.get_string "-m" with
    | Some "sofo" ->
      let module X = Make (Do_with_SOFO) in
      X.optimise
    | Some "adam" ->
      let module X = Make (Do_with_Adam) in
      X.optimise
    | _ -> failwith "-m [sofo | fgd | adam]"
  in
  optimise max_iter
