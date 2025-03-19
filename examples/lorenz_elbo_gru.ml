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
let n = 20

(* control dim *)
let m = 5
let o = 3
let bs = 64
let num_epochs_to_run = 200
let tmax = 33
let train_data = Arr.load_npy (in_dir "lorenz_train")

(* let train_data = data Int.(tmax - 1) *)
let full_batch_size = Arr.(shape train_data).(1)
let train_data_batch = get_batch train_data
let epoch_of t = Convenience.epoch_of ~full_batch_size ~batch_size:bs t

let num_train_loops =
  Convenience.num_train_loops ~full_batch_size ~batch_size:bs num_epochs_to_run

let batch_const = true
let base = Optimizer.Config.Base.default

let sample_data () =
  let trajectory = train_data_batch bs in
  List.map trajectory ~f:(fun x ->
    let x = Tensor.of_bigarray ~device:base.device x |> Tensor.to_type ~type_:base.kind in
    let noise = Tensor.(f 0.1 * rand_like x) in
    Tensor.(noise + x))

let x0 = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n ]
let eye_m = Maths.(const (Tensor.eye ~n:m ~options:(base.kind, base.device)))
let ones_u = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ m ]))
let ones_tmax = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ tmax ]))
let ones_o = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
let sample = false

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

let precision_of_log_var log_var = Maths.(exp (neg log_var))
let std_of_log_var log_var = Maths.(exp (f 0.5 * log_var))
let sqrt_precision_of_log_var log_var = Maths.(exp (f (-0.5) * log_var))

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)

let conv_threshold = 0.01
let max_iter_ilqr = 200

module GRU = struct
  module PP = struct
    type 'a p =
      { _U_f : 'a (* generative model *)
      ; _U_h : 'a
      ; _b_f : 'a
      ; _b_h : 'a
      ; _W : 'a
      ; _c : 'a (* likelihood: o = N(x _c + _b, std_o^2) *)
      ; _b : 'a (* all std params live in log space *)
      ; _log_obs_var : 'a (* log of the diagonal covariance of emission noise; *)
      ; _log_space_var : 'a
        (* recognition model; sqrt of the diagonal covariance of space factor *)
      ; _log_time_var : 'a (* sqrt of the diagonal covariance of the time factor *)
      }
    [@@deriving prms]
  end

  module P = PP.Make (Prms.P)

  type args = unit
  type data = Tensor.t list (* observations *)

  let sqr_inv x = Maths.(1. $/ sqr x)

  (* list of length T of [m x b] to matrix of [m x b x T]*)
  let concat_time u_list =
    List.map u_list ~f:(fun u -> Maths.unsqueeze ~dim:(-1) u) |> Maths.concat_list ~dim:2

  let gaussian_llh ?mu ~inv_std x =
    let d = x |> Maths.primal |> Tensor.shape |> List.last_exn in
    let error_term =
      let error =
        match mu with
        | None -> Maths.(einsum [ x, "ma"; inv_std, "a" ] "ma")
        | Some mu -> Maths.(einsum [ x - mu, "ma"; inv_std, "a" ] "ma")
      in
      Maths.einsum [ error, "ma"; error, "ma" ] "m"
    in
    let cov_term = Maths.(neg (sum (log (sqr inv_std))) |> reshape ~shape:[ 1 ]) in
    let const_term = Float.(log (2. * pi) * of_int d) in
    Maths.(0.5 $* (const_term $+ error_term + cov_term)) |> Maths.neg

  let q_of u =
    let open Maths in
    let ell = einsum [ u, "ik"; u, "jk" ] "ij" |> cholesky in
    linsolve_triangular ~left:true ~upper:false ell u

  let _Fx_reparam (q, d) =
    let q = q_of q in
    let open Maths in
    let d = exp d in
    let left_factor = sqrt d in
    let right_factor = f 1. / sqrt (f 1. + d) in
    einsum [ left_factor, "qi"; q, "ij"; right_factor, "qj" ] "ji"

  let pre_sig ~x (theta : P.M.t) = Maths.((x *@ theta._U_f) + theta._b_f)
  let pre_g ~f_t ~x (theta : P.M.t) = Maths.((f_t * x *@ theta._U_h) + theta._b_h)
  let x_hat ~pre_g ~u (theta : P.M.t) = Maths.(soft_relu pre_g + (u *@ theta._W))

  (* rollout x list under sampled u *)
  let rollout_one_step ~x ~u (theta : P.M.t) =
    let pre_sig = pre_sig ~x theta in
    let f_t = Maths.sigmoid pre_sig in
    let pre_g = pre_g ~f_t ~x theta in
    let x_hat = x_hat ~pre_g ~u theta in
    let new_x = Maths.(((f 1. - f_t) * x) + (f_t * x_hat)) in
    new_x

  (* (1 + e^-x)^{-2} (e^-x)*)
  let d_sigmoid x = Maths.(sigmoid x * (f 1. - sigmoid x))

  (* 1/2 [1 + (x^2 + 4)^{-1/2} x]*)
  let d_soft_relu x =
    let tmp = Maths.(sqr x + f 4.) in
    let tmp2 = Maths.(f 1. / sqrt tmp) in
    Maths.(((tmp2 * x) + f 1.) /$ 2.)

  let _Fu ~x (theta : P.M.t) =
    match x with
    | Some x ->
      let pre_sig = pre_sig ~x theta in
      let f_t = Maths.sigmoid pre_sig in
      Maths.einsum [ f_t, "ma"; theta._W, "ba" ] "mba"
    | None -> Tensor.zeros ~device:base.device ~kind:base.kind [ bs; m; n ] |> Maths.const

  let _Fx ~x ~u (theta : P.M.t) =
    match x, u with
    | Some x, Some u ->
      let pre_sig = pre_sig ~x theta in
      let f_t = Maths.sigmoid pre_sig in
      let pre_g = pre_g ~f_t ~x theta in
      let x_hat = x_hat ~pre_g ~u theta in
      let tmp_einsum2 a b = Maths.einsum [ a, "ab"; b, "mb" ] "mba" in
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
        Maths.(unsqueeze ~dim:2 f_t * tmp4)
      in
      let final = Maths.(term1 + term2 + term3) in
      final
    | _ -> Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n; n; n ] |> Maths.const

  let rollout_y ~u_list (theta : P.M.t) =
    let x0_tan = Maths.const x0 in
    let _, y_list =
      List.fold u_list ~init:(x0_tan, []) ~f:(fun (x, accu) u ->
        let new_x = rollout_one_step ~x ~u theta in
        let new_y = Maths.(tmp_einsum new_x theta._c + theta._b) in
        Stdlib.Gc.major ();
        new_x, new_y :: accu)
    in
    List.rev y_list

  let rollout_sol ~u_list (theta : P.M.t) =
    let x0_tan = Maths.const x0 in
    let _, x_list =
      List.fold u_list ~init:(x0_tan, []) ~f:(fun (x, accu) u ->
        let new_x = rollout_one_step ~x ~u theta in
        Stdlib.Gc.major ();
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
    let o_list = data in
    let params_func (tau : Maths.t option Lqr.Solution.p list)
      : ( Maths.t option
          , (Maths.t, Maths.t -> Maths.t) Lqr.momentary_params list )
          Lqr.Params.p
      =
      (* set o at time 0 as 0 *)
      let o_list_tmp = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
      let _Cuu_batched =
        List.init bs ~f:(fun _ -> Maths.reshape eye_m ~shape:[ 1; m; m ])
        |> Maths.concat_list ~dim:0
      in
      let c_trans = Maths.transpose theta._c ~dim0:1 ~dim1:0 in
      let _obs_var_inv = Maths.(exp (neg theta._log_obs_var) * ones_o) in
      let _Cxx =
        let tmp = Maths.(einsum [ theta._c, "ab"; _obs_var_inv, "b" ] "ab") in
        Maths.(tmp *@ c_trans)
      in
      let _Cxx_batched =
        List.init bs ~f:(fun _ -> Maths.reshape _Cxx ~shape:[ 1; n; n ])
        |> Maths.concat_list ~dim:0
      in
      let _cx_common =
        let tmp = Maths.(einsum [ theta._b, "ab"; _obs_var_inv, "b" ] "ab") in
        Maths.(tmp *@ c_trans)
      in
      let tau_extended = extend_tau_list tau in
      let tmp_list =
        Lqr.Params.
          { x0 = Some (Maths.const x0)
          ; params =
              List.map2_exn o_list_tmp tau_extended ~f:(fun o s ->
                let _cx =
                  let tmp = Maths.(einsum [ const o, "ab"; _obs_var_inv, "b" ] "ab") in
                  Maths.(_cx_common - (tmp *@ c_trans))
                in
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
        let rand = Tensor.randn ~device:base.device ~kind:base.kind [ bs; m ] in
        Maths.const rand)
    in
    let tau_init = rollout_sol ~u_list:u_init theta in
    (* TODO: is there a more elegant way? Currently I need to set batch_const to false since _Fx and _Fu has batch dim. *)
    (* use lqr to obtain the optimal u *)
    let f_theta = rollout_one_step theta in
    let sol, _ =
      Ilqr._isolve
        ~f_theta
        ~batch_const:false
        ~cost_func
        ~params_func
        ~conv_threshold
        ~tau_init
        ~max_iter:max_iter_ilqr
    in
    let optimal_u_list = List.map sol ~f:(fun s -> s.u |> Option.value_exn) in
    (* sample u from the kronecker formation *)
    let u_list =
      let optimal_u = concat_time optimal_u_list in
      let xi =
        Tensor.randn ~device:base.device ~kind:base.kind [ bs; m; tmax ] |> Maths.const
      in
      let xi_space =
        Maths.einsum [ xi, "mbt"; std_of_log_var theta._log_space_var, "b" ] "mbt"
      in
      let xi_time =
        Maths.einsum [ xi_space, "mat"; std_of_log_var theta._log_time_var, "t" ] "mat"
      in
      let meaned = Maths.(xi_time + optimal_u) in
      List.init tmax ~f:(fun i ->
        Maths.slice ~dim:2 ~start:(Some i) ~end_:(Some (i + 1)) ~step:1 meaned
        |> Maths.reshape ~shape:[ bs; m ])
    in
    optimal_u_list, u_list

  let elbo ~data ~sample (theta : P.M.t) =
    (* obtain u from lqr *)
    let optimal_u_list, u_sampled = pred_u ~data theta in
    (* calculate the likelihood term *)
    let y_pred = rollout_y ~u_list:u_sampled theta in
    let lik_term =
      let inv_sigma_o_expanded =
        Maths.(sqrt_precision_of_log_var theta._log_obs_var * ones_o)
      in
      List.fold2_exn
        data
        y_pred
        ~init:Maths.(f 0.)
        ~f:(fun accu o y_pred ->
          Maths.(
            accu + gaussian_llh ~mu:y_pred ~inv_std:inv_sigma_o_expanded (Maths.const o)))
    in
    (* M1: calculate the kl term using samples *)
    let optimal_u = concat_time optimal_u_list in
    let kl =
      if sample
      then (
        let prior =
          List.foldi u_sampled ~init:None ~f:(fun t accu u ->
            if t % 1 = 0 then Stdlib.Gc.major ();
            let increment =
              gaussian_llh
                ~inv_std:
                  (Tensor.ones [ m ] ~device:base.device ~kind:base.kind |> Maths.const)
                u
            in
            match accu with
            | None -> Some increment
            | Some accu -> Some Maths.(accu + increment))
          |> Option.value_exn
        in
        let neg_entropy =
          let u = concat_time u_sampled |> Maths.reshape ~shape:[ bs; -1 ] in
          let optimal_u = Maths.reshape optimal_u ~shape:[ bs; -1 ] in
          let inv_std =
            Maths.(
              f 1.
              / kron
                  (std_of_log_var theta._log_space_var)
                  (std_of_log_var theta._log_time_var))
          in
          gaussian_llh ~mu:optimal_u ~inv_std u
        in
        Maths.(neg_entropy - prior))
      else (
        (* M2: calculate the kl term analytically *)
        let std2 =
          Maths.(
            kron
              (std_of_log_var theta._log_space_var)
              (std_of_log_var theta._log_time_var))
        in
        let det1 = Maths.(2. $* sum (log std2)) in
        let _const = Float.of_int (m * tmax) in
        let tr =
          let tmp2 = Maths.(exp theta._log_space_var) in
          let tmp3 = Maths.(kron tmp2 (exp theta._log_time_var)) in
          Maths.sum tmp3
        in
        let quad =
          Maths.einsum [ optimal_u, "mbt"; optimal_u, "mbt" ] "m"
          |> Maths.unsqueeze ~dim:1
        in
        let tmp = Maths.(tr - det1 -$ _const) |> Maths.reshape ~shape:[ 1; 1 ] in
        Maths.(0.5 $* tmp + quad) |> Maths.squeeze ~dim:1)
    in
    Maths.(neg (lik_term - kl) / f Float.(of_int tmax * of_int o)), y_pred

  let ggn ~y_pred (theta : P.M.t) =
    let obs_precision = precision_of_log_var theta._log_obs_var in
    let obs_precision_p = Maths.(const (primal obs_precision)) in
    let sigma2_t =
      Maths.(tangent (exp theta._log_obs_var)) |> Option.value_exn |> Maths.const
    in
    List.fold y_pred ~init:(Maths.f 0.) ~f:(fun accu y_pred ->
      let mu_t = Maths.tangent y_pred |> Option.value_exn |> Maths.const in
      let ggn_part1 =
        Maths.(einsum [ mu_t, "kmo"; mu_t, "lmo" ] "kl" * obs_precision_p)
      in
      let ggn_part2 =
        Maths.(
          einsum
            [ f Float.(0.5 * of_int o * of_int bs) * sigma2_t * sqr obs_precision_p, "ky"
            ; sigma2_t, "ly"
            ]
            "kl")
      in
      Maths.(
        accu + const (primal ((ggn_part1 + ggn_part2) / f Float.(of_int o * of_int tmax)))))
    |> Maths.primal

  let f ~update ~data ~init ~args:() (theta : P.M.t) =
    let neg_elbo, y_pred = elbo ~data theta ~sample in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let ggn = ggn ~y_pred theta in
      let _ =
        let _, s, _ = Owl.Linalg.S.svd (Tensor.to_bigarray ~kind:base.ba_kind ggn) in
        Owl.Dense.Matrix.S.(save_txt ~out:(in_dir "svals") (transpose s))
      in
      u init (Some (neg_elbo, Some ggn))

  let init : P.tagged =
    let _U_f =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:n
        ~b:n
        ~sigma:1.
      |> Prms.free
    in
    let _U_h =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:n
        ~b:n
        ~sigma:1.
      |> Prms.free
    in
    let _b_f = Tensor.zeros ~device:base.device [ 1; n ] |> Prms.free in
    let _b_h = Tensor.zeros ~device:base.device [ 1; n ] |> Prms.free in
    let _W =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:m
        ~b:n
        ~sigma:1.
      |> Prms.free
    in
    let _c =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:n
        ~b:o
        ~sigma:1.
      |> Prms.free
    in
    let _b = Tensor.zeros ~device:base.device [ 1; o ] |> Prms.free in
    let _log_obs_var =
      Tensor.(log (f Float.(square 1.) * ones ~device:base.device ~kind:base.kind [ 1 ]))
      |> Prms.free
    in
    let _log_space_var =
      Tensor.(log (f Float.(square 1.) * ones ~device:base.device ~kind:base.kind [ m ]))
      |> Prms.free
    in
    let _log_time_var =
      Tensor.(
        log (f Float.(square 1.) * ones ~device:base.device ~kind:base.kind [ tmax ]))
      |> Prms.free
    in
    { _U_f; _U_h; _b_f; _b_h; _W; _c; _b; _log_obs_var; _log_space_var; _log_time_var }

  let simulate ~theta ~data =
    (* infer the optimal u *)
    let optimal_u_list, _ = pred_u ~data theta in
    let _, o_error =
      List.fold2_exn
        data
        optimal_u_list
        ~init:(Maths.const x0, 0.)
        ~f:(fun accu o1 u ->
          let x_prev, error_accu = accu in
          let x = rollout_one_step ~x:x_prev ~u theta in
          let o2 = Maths.((x *@ theta._c) + theta._b) in
          let error = Tensor.(norm (o1 - Maths.primal o2)) |> Tensor.to_float0_exn in
          Stdlib.Gc.major ();
          x, Float.(error_accu + error))
    in
    o_error
end

(* --------------------------------
   -- Generic type of optimiser
   -------------------------------- *)

module type Do_with_T = sig
  module O :
    Optimizer.T
    with type 'a W.P.p = 'a GRU.P.p
     and type W.data = Tensor.t list
     and type W.args = unit

  val name : string
  val config_f : iter:int -> (float, Bigarray.float32_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state ~time_elapsed running_avg =
      Stdlib.Gc.major ();
      let e = epoch_of iter in
      let data = sample_data () in
      let t0 = Unix.gettimeofday () in
      let config = config_f ~iter in
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
        if iter % 10 = 0
        then (
          (* simulation error *)
          let o_error =
            let data = sample_data () in
            GRU.simulate ~theta:(GRU.P.const (GRU.P.value (O.params new_state))) ~data
          in
          (* avg error *)
          Convenience.print [%message (e : float) (loss_avg : float)];
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array [| e; time_elapsed; loss_avg; o_error |] 1 4));
          O.W.P.T.save
            (GRU.P.value (O.params new_state))
            ~kind:base.ba_kind
            ~out:(in_dir name ^ ""));
        []
      in
      if iter < max_iter
      then loop ~iter:(iter + 1) ~state:new_state ~time_elapsed (loss :: running_avg)
    in
    (* ~config:(config_f ~iter:0) *)
    loop ~iter:0 ~state:init ~time_elapsed:0. []
end

(* --------------------------------
     -- SOFO
     -------------------------------- *)

module Do_with_SOFO : Do_with_T = struct
  module O = Optimizer.SOFO (GRU)

  let config_f ~iter =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some Float.(0.03 / (1. +. (0.0 * sqrt (of_int iter))))
      ; n_tangents = 60
      ; sqrt = false
      ; rank_one = false
      ; damping = Some 1e-5
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
      "true_fisher_lr_%s_sqrt_%s_damp_%s"
      (Float.to_string (Option.value_exn init_config.learning_rate))
      (Bool.to_string init_config.sqrt)
      gamma_name

  let init = O.init ~config:(config_f ~iter:0) GRU.init
end

(* --------------------------------
       -- Adam
       -------------------------------- *)

module Do_with_Adam : Do_with_T = struct
  module O = Optimizer.Adam (GRU)

  let config_f ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.0004 }

  let name =
    sprintf
      "adam_lr_%s"
      ((config_f ~iter:0).learning_rate |> Option.value_exn |> Float.to_string)

  let init = O.init GRU.init
end

let _ =
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
  optimise num_train_loops
