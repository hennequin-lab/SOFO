(* Lorenz attractor with no controls, with same state/control/cost parameters constant across trials and across time; use a mini-gru (ilqr-vae appendix c) as generative model. *)
open Printf
open Base
open Forward_torch
open Torch
open Sofo

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run

(* -----------------------------------------
   -- Generate Lorenz data            ------
   ----------------------------------------- *)

open Lorenz_common

(* state dim *)
let a = 3

(* control dim *)
let b = 1
let batch_size = 128
let num_epochs_to_run = 500
let tmax = 33
let train_data = Arr.load_npy (in_dir "lorenz_train")

(* let train_data = data Int.(tmax - 1) *)
let full_batch_size = Arr.(shape train_data).(1)
let train_data_batch = get_batch train_data
let epoch_of t = Convenience.epoch_of ~full_batch_size ~batch_size t

let num_train_loops =
  Convenience.num_train_loops ~full_batch_size ~batch_size num_epochs_to_run

let batch_const = true
let kind = Torch_core.Kind.(T f64)
let device = Torch.Device.cuda_if_available ()
let x0 = Tensor.zeros ~device ~kind [ batch_size; a ]

(* -----------------------------------------
   ----- Utilitiy Functions ------
   ----------------------------------------- *)
let ( +? ) a b =
  match a, b with
  | None, None -> None
  | Some a, None -> Some a
  | None, Some b -> Some b
  | Some a, Some b -> Some Maths.(a + b)

let tmp_einsum a b =
  if batch_const
  then Maths.einsum [ a, "ma"; b, "ab" ] "mb"
  else Maths.einsum [ a, "ma"; b, "mab" ] "mb"

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)

let base =
  Optimizer.Config.Base.
    { default with kind = Torch_core.Kind.(T f64); ba_kind = Bigarray.float64 }

let laplace = false
let conv_threshold = 0.01

module LGS = struct
  module PP = struct
    type 'a p =
      { _U_f : 'a (* generative model *)
      ; _U_h : 'a
      ; _b_f : 'a
      ; _b_h : 'a
      ; _W : 'a
      ; _c : 'a (* likelihood: o = N(x _c + _b, cov_o^2) *)
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
        let like_hess = theta._cov_o |> sqr_inv |> Maths.primal in
        (* y = cx + b *)
        let y_list =
          List.map rolled_out_x_list ~f:(fun x ->
            Maths.(einsum [ x, "ma"; theta._c, "ab" ] "mb" + theta._b))
        in
        ggn_final ~o_list:y_list ~like_hess ~diagonal:true
      in
      let prior_ggn =
        let like_hess = theta._cov_u |> sqr_inv |> Maths.primal in
        ggn_final ~o_list:u_list ~like_hess ~diagonal:true
      in
      let entropy_ggn =
        let _cov_space_inv = theta._cov_space |> sqr_inv |> Maths.primal in
        let _cov_time_inv = theta._cov_time |> sqr_inv |> Maths.primal in
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
      let final = Tensor.(llh_ggn + (prior_ggn + entropy_ggn)) in
      final
  end

  (* (1 + e^-x)^{-2} (e^-x)*)
  let d_sigmoid x =
    let tmp = Maths.(f 1. / sqr (f 1. + exp (neg x))) in
    Maths.(tmp * exp (neg x))

  (* 1/2 [1 + (x^2 + 4)^{-1/2} x]*)
  let d_soft_relu x =
    let tmp = Maths.(sqr x + f 4.) in
    let tmp2 = Maths.(f 1. / sqrt tmp) in
    Maths.(((tmp2 * x) + f 1.) /$ 2.)

  let _Fu ~x (theta : P.M.t) =
    match x with
    | Some x ->
      let pre_sig = Maths.((x *@ theta._U_f) + theta._b_f) in
      let f_t = Maths.sigmoid pre_sig in
      Maths.einsum [ f_t, "ma"; theta._W, "ba" ] "mba"
    | None -> Tensor.zeros ~device ~kind [ batch_size; b; a ] |> Maths.const

  let _Fx ~x ~u (theta : P.M.t) =
    match x, u with
    | Some x, Some u ->
      let pre_sig = Maths.((x *@ theta._U_f) + theta._b_f) in
      let f_t = Maths.sigmoid pre_sig in
      let pre_g = Maths.((f_t * x *@ theta._U_h) + theta._b_h) in
      let x_hat = Maths.(soft_relu pre_g + (u *@ theta._W)) in
      let tmp_einsum2 a b = Maths.einsum [ a, "ab"; b, "ma" ] "mba" in
      let term1 = Maths.diag_embed Maths.(f 1. - f_t) ~offset:0 ~dim1:(-2) ~dim2:(-1) in
      let term2 =
        let tmp = Maths.(d_sigmoid pre_sig * (x_hat - x)) in
        tmp_einsum2 theta._U_f tmp
      in
      let term3 =
        let tmp1 = Maths.diag_embed f_t ~offset:0 ~dim1:(-2) ~dim2:(-1) in
        let tmp2 = tmp_einsum2 theta._U_f (d_sigmoid pre_sig) in
        let tmp3 = tmp_einsum2 theta._U_h (d_soft_relu pre_g) in
        let tmp4 = Maths.einsum [ tmp3, "mab"; Maths.(tmp2 + tmp1), "mbc" ] "mac" in
        Maths.(tmp1 * tmp4)
      in
      let final = Maths.(term1 + term2 + term3) in
      final
    | _ -> Tensor.zeros ~device ~kind [ batch_size; a; a ] |> Maths.const

  (* rollout x list under sampled u *)
  let rollout_one_step ~x ~u (theta : P.M.t) =
    let pre_sig = Maths.((x *@ theta._U_f) + theta._b_f) in
    let f_t = Maths.sigmoid pre_sig in
    let pre_g = Maths.((f_t * x *@ theta._U_h) + theta._b_h) in
    let x_hat = Maths.(soft_relu pre_g + (u *@ theta._W)) in
    let new_x = Maths.(((f 1. - f_t) * x) + (f_t * x_hat)) in
    new_x

  let rollout_x ~u_list ~x0 (theta : P.M.t) =
    let x0_tan = Maths.const x0 in
    let _, x_list =
      List.fold u_list ~init:(x0_tan, [ x0_tan ]) ~f:(fun (x, x_list) u ->
        let new_x = rollout_one_step ~x ~u theta in
        new_x, new_x :: x_list)
    in
    List.rev x_list

  let rollout_sol ~u_list ~x0 (theta : P.M.t) =
    let x0_tan = Maths.const x0 in
    let _, x_list =
      List.fold u_list ~init:(x0_tan, []) ~f:(fun (x, accu) u ->
        let new_x = rollout_one_step ~x ~u theta in
        new_x, Lqr.Solution.{ u = Some u; x = Some new_x } :: accu)
    in
    List.rev x_list

  (* artificially add one to tau so it goes from 0 to T *)
  let extend_tau_list (tau : Maths.t option Lqr.Solution.p list) =
    let u_list = List.map tau ~f:(fun s -> s.u) in
    let x_list = List.map tau ~f:(fun s -> s.x) in
    let u_ext = u_list @ [ None ] in
    let x_ext = Some (Maths.const x0) :: x_list in
    List.map2_exn u_ext x_ext ~f:(fun u x -> Lqr.Solution.{ u; x })

  (* given a trajectory and parameters, calculate average cost across batch (summed over time) *)
  let cost_func
        ~batch_const
        (tau : Maths.t option Lqr.Solution.p list)
        (p :
          ( Maths.t option
            , (Maths.t, Maths.t -> Maths.t) Lqr.momentary_params list )
            Lqr.Params.p)
    =
    let tau_extended = extend_tau_list tau in
    let maybe_tmp_einsum_sqr ~batch_const a c b =
      match a, c, b with
      | Some a, Some c, Some b ->
        let c_eqn = if batch_const then "ab" else "mab" in
        Some (Maths.einsum [ a, "ma"; c, c_eqn; b, "mb" ] "m")
      | _ -> None
    in
    let maybe_tmp_einsum ~batch_const a b =
      match a, b with
      | Some a, Some b ->
        let b_eqn = if batch_const then "a" else "ma" in
        Some (Maths.einsum [ a, "ma"; b, b_eqn ] "m")
      | _ -> None
    in
    let cost =
      List.fold2_exn tau_extended p.params ~init:None ~f:(fun accu tau p ->
        let x_sqr_cost = maybe_tmp_einsum_sqr ~batch_const tau.x p.common._Cxx tau.x in
        let u_sqr_cost = maybe_tmp_einsum_sqr ~batch_const tau.u p.common._Cuu tau.u in
        let xu_cost =
          let tmp = maybe_tmp_einsum_sqr ~batch_const tau.x p.common._Cxu tau.u in
          match tmp with
          | None -> None
          | Some x -> Some Maths.(2. $* x)
        in
        let x_cost = maybe_tmp_einsum ~batch_const tau.x p._cx in
        let u_cost = maybe_tmp_einsum ~batch_const tau.u p._cu in
        accu +? (x_sqr_cost +? u_sqr_cost) +? (xu_cost +? (x_cost +? u_cost)))
      |> Option.value_exn
    in
    cost |> Maths.primal |> Tensor.mean |> Tensor.to_float0_exn

  (* optimal u determined from lqr *)
  let pred_u ~data (theta : P.M.t) =
    let x0, o_list = data in
    let x0_tan = Maths.const x0 in
    let params_func (tau : Maths.t option Lqr.Solution.p list)
      : ( Maths.t option
          , (Maths.t, Maths.t -> Maths.t) Lqr.momentary_params list )
          Lqr.Params.p
      =
      (* set o at time 0 as 0 *)
      let o_list_tmp = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
      let _cov_u_inv =
        theta._cov_u |> sqr_inv |> Maths.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
      in
      let _Cuu_batched =
        List.init batch_size ~f:(fun _ -> Maths.reshape _cov_u_inv ~shape:[ 1; b; b ])
        |> Maths.concat_list ~dim:0
      in
      let c_trans = Maths.transpose theta._c ~dim0:1 ~dim1:0 in
      let _cov_o_inv = sqr_inv theta._cov_o in
      let _Cxx =
        let tmp = Maths.(einsum [ theta._c, "ab"; _cov_o_inv, "b" ] "ab") in
        Maths.(tmp *@ c_trans)
      in
      let _Cxx_batched =
        List.init batch_size ~f:(fun _ -> Maths.reshape _Cxx ~shape:[ 1; a; a ])
        |> Maths.concat_list ~dim:0
      in
      let _cx_common =
        let tmp = Maths.(einsum [ theta._b, "ab"; _cov_o_inv, "b" ] "ab") in
        Maths.(tmp *@ c_trans)
      in
      let tau_extended = extend_tau_list tau in
      let tmp_list =
        Lqr.Params.
          { x0 = Some x0_tan
          ; params =
              List.map2_exn o_list_tmp tau_extended ~f:(fun o s ->
                let _cx = Maths.(_cx_common - (const o *@ c_trans)) in
                Lds_data.Temp.
                  { _f = None
                  ; _Fx_prod = _Fx ~x:s.x ~u:s.u theta
                  ; _Fu_prod = _Fu ~x:s.x theta
                  ; _cx = Some _cx
                  ; _cu = None
                  ; _Cxx = _Cxx_batched
                  ; _Cxu = None
                  ; _Cuu = _Cuu_batched
                  })
          }
      in
      Lds_data.map_naive tmp_list ~batch_const:false
    in
    let u_init =
      List.init tmax ~f:(fun _ ->
        let rand = Tensor.randn ~device ~kind [ batch_size; b ] in
        Maths.const rand)
    in
    let tau_init = rollout_sol ~u_list:u_init ~x0 theta in
    (* TODO: is there a more elegant way? Currently I need to set batch_const to false since _Fx and _Fu has batch dim. *)
    (* use lqr to obtain the optimal u *)
    let f_theta = rollout_one_step theta in
    let sol, _ =
      Ilqr._isolve
        ~laplace
        ~f_theta
        ~batch_const:false
        ~cost_func
        ~params_func
        ~conv_threshold
        ~tau_init
    in
    let optimal_u_list = List.map sol ~f:(fun s -> s.u |> Option.value_exn) in
    (* sample u from the kronecker formation *)
    let u_list =
      let optimal_u = concat_time optimal_u_list in
      let xi = Tensor.randn ~device ~kind [ batch_size; b; tmax ] |> Maths.const in
      let _chol_space = Maths.abs theta._cov_space in
      let _chol_time = Maths.abs theta._cov_time in
      let xi_space = Maths.einsum [ xi, "mbt"; _chol_space, "b" ] "mbt" in
      let xi_time = Maths.einsum [ xi_space, "mat"; _chol_time, "t" ] "mat" in
      let meaned = Maths.(xi_time + optimal_u) in
      List.init tmax ~f:(fun i ->
        Maths.slice ~dim:2 ~start:(Some i) ~end_:(Some (i + 1)) ~step:1 meaned
        |> Maths.reshape ~shape:[ batch_size; b ])
    in
    optimal_u_list, u_list

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
      Tensor.of_float0 ~device Float.(log (2. * pi) * of_int o)
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
          let u = concat_time u_list |> Maths.reshape ~shape:[ batch_size; -1 ] in
          let optimal_u = Maths.reshape optimal_u ~shape:[ batch_size; -1 ] in
          let g_cov = Maths.kron theta._cov_space theta._cov_time in
          gaussian_llh ~g_mean:optimal_u ~g_cov ~x:u
        in
        Maths.(entropy - prior))
      else (
        (* M2: calculate the kl term analytically *)
        let cov2 = Maths.kron theta._cov_space theta._cov_time in
        let det1 = Maths.(2. $* sum (log (abs cov2))) in
        let det2 = Maths.(Float.(2. * of_int tmax) $* sum (log (abs theta._cov_u))) in
        let _const = Tensor.of_float0 (Float.of_int b) |> Maths.const in
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

  (* calculate optimal u *)
  let f ~update ~data ~init ~args:() (theta : P.M.t) =
    let module L = Elbo_loss in
    let x0, o_list = data in
    let optimal_u_list, u_list = pred_u ~data theta in
    let rolled_out_x_list = rollout_x ~u_list ~x0 theta in
    (* These lists go from 1 to T *)
    let o_except_first = List.map o_list ~f:(fun o -> Maths.const o) in
    let x_except_first = List.tl_exn rolled_out_x_list in
    let x_o_list = List.map2_exn x_except_first o_except_first ~f:(fun x o -> x, o) in
    let neg_elbo =
      Maths.(neg (elbo ~x_o_list ~u_list ~optimal_u_list theta ~sample:false))
    in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let ggn = L.vtgt_hessian_gv ~rolled_out_x_list:x_except_first ~u_list theta in
      u init (Some (neg_elbo, Some ggn))

  let init : P.tagged =
    let _device = device in
    let _kind = kind in
    let _U_f =
      Convenience.gaussian_tensor_2d_normed ~device ~kind ~a ~b:a ~sigma:0.1 |> Prms.free
    in
    let _U_h =
      Convenience.gaussian_tensor_2d_normed ~device ~kind ~a ~b:a ~sigma:0.1 |> Prms.free
    in
    let _b_f =
      Convenience.gaussian_tensor_2d_normed ~device ~kind ~a:1 ~b:a ~sigma:0.1
      |> Prms.free
    in
    let _b_h =
      Convenience.gaussian_tensor_2d_normed ~device ~kind ~a:1 ~b:a ~sigma:0.1
      |> Prms.free
    in
    let _W =
      Convenience.gaussian_tensor_2d_normed ~device ~kind ~a:b ~b:a ~sigma:0.1
      |> Prms.free
    in
    let _c =
      Convenience.gaussian_tensor_2d_normed ~device ~kind ~a ~b:a ~sigma:1. |> Prms.free
    in
    let _b =
      Convenience.gaussian_tensor_2d_normed ~device ~kind ~a:1 ~b:a ~sigma:1. |> Prms.free
    in
    let _cov_o =
      Tensor.(mul_scalar (ones ~device:_device ~kind:_kind [ a ]) (Scalar.f 0.1))
      |> Prms.free
    in
    let _cov_u =
      Tensor.(mul_scalar (ones ~device:_device ~kind:_kind [ b ]) (Scalar.f 0.1))
      |> Prms.free
    in
    let _cov_space =
      Tensor.(mul_scalar (ones ~device:_device ~kind:_kind [ b ]) (Scalar.f 0.1))
      |> Prms.free
    in
    let _cov_time =
      Tensor.(mul_scalar (ones ~device:_device ~kind:_kind [ tmax ]) (Scalar.f 0.1))
      |> Prms.free
    in
    { _U_f; _U_h; _b_f; _b_h; _W; _c; _b; _cov_o; _cov_u; _cov_space; _cov_time }

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

module O = Optimizer.Adam (LGS) *)

let optimise ~f_name ~init config_f =
  let rec loop ~iter ~state ~time_elapsed running_avg =
    Stdlib.Gc.major ();
    let e = epoch_of iter in
    let config = config_f ~iter in
    let data =
      let trajectory = train_data_batch batch_size in
      ( x0
      , List.map trajectory ~f:(fun x ->
          let x =
            Tensor.of_bigarray ~device:base.device x |> Tensor.to_type ~type_:kind
          in
          x) )
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
          let data =
            let trajectory = train_data_batch batch_size in
            ( x0
            , List.map trajectory ~f:(fun x ->
                let x =
                  Tensor.of_bigarray ~device:base.device x |> Tensor.to_type ~type_:kind
                in
                x) )
          in
          LGS.simulate ~theta:(LGS.P.const (LGS.P.value (O.params new_state))) ~data
        in
        (* avg error *)
        Convenience.print [%message (e : float) (loss_avg : float)];
        Owl.Mat.(
          save_txt
            ~append:true
            ~out:(in_dir f_name)
            (of_array [| e; time_elapsed; loss_avg; o_error |] 1 4));
        O.W.P.T.save
          (LGS.P.value (O.params new_state))
          ~kind:base.ba_kind
          ~out:(in_dir f_name ^ "_params"));
      []
    in
    if iter < num_train_loops
    then loop ~iter:(iter + 1) ~state:new_state ~time_elapsed (loss :: running_avg)
  in
  (* ~config:(config_f ~iter:0) *)
  loop ~iter:0 ~state:(O.init ~config:(config_f ~iter:0) init) ~time_elapsed:0. []

let checkpoint_name = None
let lr_rates = [ 1e-5; 1e-4 ]
let damping_list = [ Some 1e-5 ]
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
          , sprintf "lorenz_elbo_%s_lr_%s_damp_%s" meth (Float.to_string eta) gamma_name )
        | Some checkpoint_name ->
          let params_ba = O.W.P.T.load ~device (in_dir checkpoint_name ^ "_params") in
          ( LGS.P.map params_ba ~f:(fun x ->
              (* randomly perturb to escape local minimum *)
              let x_perturbed = Tensor.(x + mul_scalar (rand_like x) (Scalar.f 0.01)) in
              Prms.free x_perturbed)
          , sprintf
              "lorenz_elbo_%s_lr_%s_damp_%s_%s"
              meth
              (Float.to_string eta)
              gamma_name
              checkpoint_name )
      in
      Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
      Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
      optimise ~f_name ~init config_f))

(* let checkpoint_name = None
let lr_rates = [ 0.001; 0.0005]
let meth = "adam"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    let config_f = config ~base_lr:eta ~gamma:None in
    let init, f_name =
      match checkpoint_name with
      | None -> LGS.(init), sprintf "lorenz_elbo_%s_lr_%s" meth (Float.to_string eta)
      | Some checkpoint_name ->
        let params_ba = O.W.P.T.load ~device (in_dir checkpoint_name ^ "_params") in
        ( LGS.P.map params_ba ~f:(fun x -> Prms.free x)
        , sprintf "lorenz_elbo_%s_lr_%s_%s" meth (Float.to_string eta) checkpoint_name )
    in
    Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
    Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
    optimise ~f_name ~init config_f) *)
