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

(* bottleneck size *)
let p = 40
let o = 3
let bs = 32
let num_epochs_to_run = 70
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
let ones_tmax = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ tmax ]))
let ones_o = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
let ones_u = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ m ]))
let sample = true
let gamma = 1.

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

let gaussian_llh_chol ?(batched_chol = false) ?mu ~precision_chol:ell x =
  let d = x |> Maths.primal |> Tensor.shape |> List.last_exn in
  let error =
    match mu with
    | None -> x
    | Some mu -> Maths.(x - mu)
  in
  let error_term =
    if batched_chol
    then Maths.einsum [ error, "ma"; ell, "mai"; ell, "mbi"; error, "mb" ] "m"
    else Maths.einsum [ error, "ma"; ell, "ai"; ell, "bi"; error, "mb" ] "m"
  in
  let cov_term =
    Maths.(neg (sum (log (sqr (diagonal ~offset:0 ell)))) |> reshape ~shape:[ 1 ])
  in
  let const_term = Float.(log (2. * pi) * of_int d) in
  Maths.(-0.5 $* (const_term $+ error_term + cov_term))

let precision_of_log_var log_var = Maths.(exp (neg log_var))
let std_of_log_var log_var = Maths.(exp (f 0.5 * log_var))
let sqrt_precision_of_log_var log_var = Maths.(exp (f (-0.5) * log_var))

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)

let conv_threshold = 0.01
let max_iter_ilqr = 200

module PP = struct
  type 'a p =
    { _W : 'a (* generative model *)
    ; _A : 'a
    ; _D : 'a
    ; _B : 'a
    ; _c : 'a (* likelihood: o = N(x _c + _b, std_o^2) *)
    ; _b : 'a (* all std params live in log space *)
    ; _log_obs_var : 'a (* log of the diagonal covariance of emission noise; *)
    ; _scaling_factor : 'a
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

module GRU = struct
  module P = P

  type args = unit
  type data = Tensor.t list (* observations *)

  (* list of length T of [m x b] to matrix of [m x b x T]*)
  let concat_time u_list =
    List.map u_list ~f:(fun u -> Maths.unsqueeze ~dim:(-1) u) |> Maths.concat_list ~dim:2

  (* 1/2 [1 + (x^2 + 4)^{-1/2} x]*)
  let d_soft_relu x =
    let tmp = Maths.(sqr x + f 4.) in
    let tmp2 = Maths.(f 1. / sqrt tmp) in
    Maths.(((tmp2 * x) + f 1.) /$ 2.)

  let pre_soft_relu ~x ~u (theta : P.M.t) =
    let bs = Maths.shape x |> List.hd_exn in
    let x =
      Maths.concat
        x
        (Maths.const (Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ]))
        ~dim:1
    in
    Maths.((x *@ theta._D) + (u *@ theta._B))

  (* rollout x list under sampled u *)
  let rollout_one_step ~x ~u (theta : P.M.t) =
    let pre_soft_relu = pre_soft_relu ~x ~u theta in
    Maths.((x *@ theta._A) + (soft_relu pre_soft_relu *@ theta._W))

  (* df/du *)
  let _Fu ~x ~u (theta : P.M.t) =
    match x, u with
    | Some x, Some u ->
      let d_soft_relu = d_soft_relu (pre_soft_relu ~x ~u theta) in
      Maths.einsum [ theta._B, "mp"; d_soft_relu, "bp"; theta._W, "pn" ] "bmn"
    | _ -> Tensor.zeros ~device:base.device ~kind:base.kind [ bs; m; n ] |> Maths.const

  (* df/dx *)
  let _Fx ~x ~u (theta : P.M.t) =
    match x, u with
    | Some x, Some u ->
      let d_soft_relu = d_soft_relu (pre_soft_relu ~x ~u theta) in
      let tmp1 =
        let _D = Maths.slice theta._D ~dim:0 ~start:(Some 0) ~end_:(Some n) ~step:1 in 
        Maths.einsum [ _D, "mp"; d_soft_relu, "bp"; theta._W, "pn" ] "bmn"
      in
      let tmp2 = Maths.unsqueeze theta._A ~dim:0 in
      Maths.(tmp1 + tmp2)
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
                  ; _Fu_prod = _Fu ~x:s.x ~u:s.u theta
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
    let sol, backward_info =
      Ilqr._isolve
        ~batch_const:false
        ~gamma
        ~f_theta
        ~cost_func
        ~params_func
        ~conv_threshold
        ~tau_init
        ~max_iter:max_iter_ilqr
    in
    List.map sol ~f:(fun s -> s.u |> Option.value_exn), backward_info

  (* approximate kalman filtered distribution *)
  let sample_and_kl ~(theta : P.M.t) ~optimal_u_list o_list =
    let open Maths in
    let o_list = List.map ~f:Maths.const o_list in
    let scaling_factor = reshape theta._scaling_factor ~shape:[ 1; 1; -1 ] in
    let z0 = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n ] |> Maths.const in
    let _, kl, us =
      List.fold2_exn
        optimal_u_list
        o_list
        ~init:(z0, f 0., [])
        ~f:(fun (z, kl, us) ustar o ->
          Stdlib.Gc.major ();
          let zpred = rollout_one_step ~x:z ~u:ustar theta in
          let ypred = Maths.((zpred *@ theta._c) + theta._b) in
          let delta = o - ypred in
          (* approximate z_{t+1} = f(z_t, u_opt) + df/du u. dimension is now [bs x m x n] *)
          let _b_prime = _Fu ~x:(Some z) ~u:(Some ustar) theta in
          (* BC / obs_var *)
          let btrinv =
            let tmp = einsum [ _b_prime, "mij"; theta._c, "jo" ] "mio" in
            tmp / reshape ~shape:[ -1; 1; 1 ] (exp theta._log_obs_var)
          in
          (* cholesky of posterior precision of filtered covariance of u *)
          let precision_chol =
            let tmp =
              unsqueeze ~dim:0 eye_m
              + einsum [ btrinv, "mio"; theta._c, "jo"; _b_prime, "mkj" ] "mik"
              |> cholesky
            in
            tmp * scaling_factor
          in
          (* cov = cov_chol cov_chol^T *)
          let cov_chol =
            let batched_eye_m =
              List.init bs ~f:(fun _ -> Maths.unsqueeze eye_m ~dim:0)
              |> concat_list ~dim:0
            in
            linsolve_triangular ~left:true ~upper:false precision_chol batched_eye_m
            |> transpose ~dim0:1 ~dim1:2
          in
          (* posterior mean of filtered u *)
          let mu =
            let tmp = einsum [ btrinv, "mio"; delta, "mo" ] "mi" in
            einsum [ tmp, "ma"; cov_chol, "mab"; cov_chol, "mcb" ] "mc"
          in
          (* sample from posterior filtered covariance of u. *)
          let u_diff_elbo =
            einsum
              [ const (Tensor.randn ~device:base.device ~kind:base.kind [ bs; m ]), "ma"
              ; cov_chol, "mba"
              ]
              "mb"
          in
          let u_sample = mu + u_diff_elbo in
          let u_final = ustar + u_sample in
          (* propagate that sample to update z *)
          let z = rollout_one_step ~x:z ~u:u_final theta in
          (* update the KL divergence *)
          let kl =
            let prior_term = gaussian_llh ~inv_std:ones_u u_final in
            let q_term =
              gaussian_llh_chol ~batched_chol:true ~precision_chol u_diff_elbo
            in
            kl + q_term - prior_term
          in
          z, kl, (ustar + u_sample) :: us)
    in
    kl, List.rev us

  let elbo_filter ~data (theta : P.M.t) =
    (* obtain u from lqr *)
    let optimal_u_list, _ = pred_u ~data theta in
    let kl, u_sampled = sample_and_kl ~theta:(theta : P.M.t) ~optimal_u_list data in
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
          Stdlib.Gc.major ();
          Maths.(
            accu + gaussian_llh ~mu:y_pred ~inv_std:inv_sigma_o_expanded (Maths.const o)))
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
    let neg_elbo, y_pred = elbo_filter ~data theta in
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
    let _W =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:p
        ~b:n
        ~sigma:1.
      |> Prms.free
    in
    let _A =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:n
        ~b:n
        ~sigma:1.
      |> Prms.free
    in
    let _D =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:(n+1)
        ~b:p
        ~sigma:1.
      |> Prms.free
    in
    let _B =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:m
        ~b:p
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
    let _scaling_factor =
      Prms.create ~above:(Tensor.f 0.1) (Tensor.ones [ 1 ] ~device:base.device)
    in
    { _W; _A; _D; _B; _c; _b; _log_obs_var; _scaling_factor }

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

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
let _K = 128

type param_name =
  | W
  | A
  | D
  | B
  | C
  | B_bias
  | Log_obs_var
  | Scaling_factor

let n_params_a = 20
let n_params_b = 20
let n_params_c = 20

let n_params_b_bias = 2
let n_params_log_obs_var = 1
let n_params_scaling_factor = 1
let n_params_w = Int.((_K - (n_params_a * 3) - 4) / 2)
let n_params_d = n_params_w

module GGN : Wrapper.Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { w_left : 'a
      ; w_right : 'a
      ; a_left : 'a
      ; a_right : 'a
      ; d_left : 'a
      ; d_right : 'a
      ; b_left : 'a
      ; b_right : 'a

      ; c_left : 'a
      ; c_right : 'a
      ; b_bias_left : 'a
      ; b_bias_right : 'a
      ; log_obs_var_left : 'a
      ; log_obs_var_right : 'a
      ; scaling_factor_left : 'a
      ; scaling_factor_right : 'a
      }
    [@@deriving prms]
  end

  module P = P
  module A = Make (Prms.P)

  type sampling_state = unit

  let init_sampling_state () = ()

  let zero_params ~shape _K =
    Tensor.zeros ~device:base.device ~kind:base.kind (_K :: shape)

  let random_params ~shape _K =
    Tensor.randn ~device:base.device ~kind:base.kind (_K :: shape)

  (* approximation defined implicitly via Gv products *)
  let g12v ~(lambda : A.M.t) (v : P.M.t) : P.M.t =
    let open Maths in
    let einsum_w ~left ~right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    let _W = einsum_w ~left:lambda.w_left ~right:lambda.w_right v._W in
    let _A = einsum_w ~left:lambda.a_left ~right:lambda.a_right v._A in
    let _D = einsum_w ~left:lambda.d_left ~right:lambda.d_right v._D in
    let _B = einsum_w ~left:lambda.b_left ~right:lambda.b_right v._B in

    let _c = einsum_w ~left:lambda.c_left ~right:lambda.c_right v._c in
    let _b = einsum_w ~left:lambda.b_bias_left ~right:lambda.b_bias_right v._b in
    let _log_obs_var =
      einsum_w
        ~left:lambda.log_obs_var_left
        ~right:lambda.log_obs_var_right
        (reshape v._log_obs_var ~shape:[ -1; 1; 1 ])
    in
    let _scaling_factor =
      einsum_w
        ~left:lambda.scaling_factor_left
        ~right:lambda.scaling_factor_right
        (reshape v._scaling_factor ~shape:[ -1; 1; 1 ])
    in
    { _W; _A; _D; _B;  _b; _c; _log_obs_var; _scaling_factor }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~local ~param_name ~n_per_param v =
    let sample = if local then zero_params else random_params in
    let _W = sample ~shape:[ p; n ] n_per_param in
    let _A = sample ~shape:[ n; n ] n_per_param in
    let _D = sample ~shape:[ n+1; p ] n_per_param in
    let _B = sample ~shape:[ m; p ] n_per_param in
    let _c = sample ~shape:[ n; o ] n_per_param in
    let _b = sample ~shape:[ 1; o ] n_per_param in
    let _log_obs_var = sample ~shape:[ 1 ] n_per_param in
    let _scaling_factor = sample ~shape:[ 1 ] n_per_param in
    let params_tmp = PP.{ _W; _A; _D; _B; _c; _b; _log_obs_var; _scaling_factor } in
    match param_name with
    | W -> { params_tmp with _W = v }
    | A -> { params_tmp with _A = v }
    | D -> { params_tmp with _D = v }
    | B -> { params_tmp with _B = v }
    | C -> { params_tmp with _c = v }
    | B_bias -> { params_tmp with _b = v }
    | Log_obs_var -> { params_tmp with _log_obs_var = v }
    | Scaling_factor -> { params_tmp with _scaling_factor = v }

  let random_localised_vs _K : P.T.t =
    { _W = random_params ~shape:[ p; n ] _K
    ; _A = random_params ~shape:[ n; n ] _K
    ; _D = random_params ~shape:[ n+1; p ] _K
    ; _B = random_params ~shape:[ m; p ] _K
    ; _c = random_params ~shape:[ n; o ] _K
    ; _b = random_params ~shape:[ 1; o ] _K
    ; _log_obs_var = random_params ~shape:[ 1 ] _K
    ; _scaling_factor = random_params ~shape:[ 1 ] _K
    }

  let eigenvectors_for_each_params ~local ~lambda ~param_name =
    let left, right, n_per_param =
      match param_name with
      | W -> lambda.w_left, lambda.w_right, n_params_w
      | A -> lambda.a_left, lambda.a_right, n_params_a
      | D -> lambda.d_left, lambda.d_right, n_params_d
      | B -> lambda.b_left, lambda.b_right, n_params_b
      | C -> lambda.c_left, lambda.c_right, n_params_c
      | B_bias -> lambda.b_bias_left, lambda.b_bias_right, n_params_b_bias
      | Log_obs_var ->
        lambda.log_obs_var_left, lambda.log_obs_var_right, n_params_log_obs_var
      | Scaling_factor ->
        lambda.scaling_factor_left, lambda.scaling_factor_right, n_params_scaling_factor
    in
    let u_left, s_left, _ = Tensor.svd ~some:true ~compute_uv:true Maths.(primal left) in
    let u_right, s_right, _ =
      Tensor.svd ~some:true ~compute_uv:true Maths.(primal right)
    in
    let s_left = Tensor.to_float1_exn s_left |> Array.to_list in
    let s_right = Tensor.to_float1_exn s_right |> Array.to_list in
    let s_all =
      List.mapi s_left ~f:(fun il sl ->
        List.mapi s_right ~f:(fun ir sr -> il, ir, Float.(sl * sr)))
      |> List.concat
      |> List.sort ~compare:(fun (_, _, a) (_, _, b) -> Float.compare b a)
      |> Array.of_list
    in
    (* randomly select the indices *)
    let n_params =
      Convenience.first_dim (Maths.primal left)
      * Convenience.first_dim (Maths.primal right)
    in
    let selection =
      List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    let selection = List.map selection ~f:(fun j -> s_all.(j)) in
    let local_vs =
      List.map selection ~f:(fun (il, ir, _) ->
        let u_left =
          Tensor.(
            squeeze_dim
              ~dim:1
              (slice u_left ~dim:1 ~start:(Some il) ~end_:(Some Int.(il + 1)) ~step:1))
        in
        let u_right =
          Tensor.(
            squeeze_dim
              ~dim:1
              (slice u_right ~dim:1 ~start:(Some ir) ~end_:(Some Int.(ir + 1)) ~step:1))
        in
        let tmp =
          match param_name with
          | Log_obs_var | Scaling_factor -> Tensor.(u_left * u_right)
          | _ -> Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_left; u_right ]
        in
        Tensor.unsqueeze tmp ~dim:0)
      |> Tensor.concatenate ~dim:0
    in
    local_vs |> localise ~local ~param_name ~n_per_param

  let eigenvectors ~(lambda : A.M.t) () (_K : int) =
    let param_names_list = [ W; A; D; B;  C; B_bias; Log_obs_var; Scaling_factor ] in
    let eigenvectors_each =
      List.map param_names_list ~f:(fun param_name ->
        eigenvectors_for_each_params ~local:true ~lambda ~param_name)
    in
    let vs =
      List.fold eigenvectors_each ~init:None ~f:(fun accu local_vs ->
        match accu with
        | None -> Some local_vs
        | Some a -> Some (P.map2 a local_vs ~f:(fun x y -> Tensor.concat ~dim:0 [ x; y ])))
    in
    Option.value_exn vs, ()

  let init () =
    let init_eye size =
      Mat.(0.1 $* eye size) |> Tensor.of_bigarray ~device:base.device |> Prms.free
    in
    { w_left = init_eye p
    ; w_right = init_eye n
    ; a_left = init_eye n
    ; a_right = init_eye n
    ; d_left = init_eye (n+1)
    ; d_right = init_eye p
    ; b_left = init_eye m
    ; b_right = init_eye p

    ; c_left = init_eye n
    ; c_right = init_eye o
    ; b_bias_left = init_eye 1
    ; b_bias_right = init_eye o
    ; log_obs_var_left = init_eye 1
    ; log_obs_var_right = init_eye 1
    ; scaling_factor_left = init_eye 1
    ; scaling_factor_right = init_eye 1
    }
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
      let loss, new_state = O.step ~config ~state ~data () in
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
            ~out:(in_dir name ^ "_params"));
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
  module O = Optimizer.SOFO (GRU) (GGN)

  let config_f ~iter =
    let aux =
      Optimizer.Config.SOFO.
        { (default_aux (in_dir "aux")) with
          config =
            Optimizer.Config.Adam.
              { default with base; learning_rate = Some 1e-3; eps = 1e-4 }
        ; steps = 5
        }
    in
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some Float.(0.1 / (1. +. (0.0 * sqrt (of_int iter))))
      ; n_tangents = _K
      ; rank_one = false
      ; damping = Some 1e-5
      ; aux = None
      }

  let name = "sofo"
  let init = O.init GRU.init
end

(* --------------------------------
       -- Adam
       -------------------------------- *)

module Do_with_Adam : Do_with_T = struct
  module O = Optimizer.Adam (GRU)

  let config_f ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 1e-3 }

  let name = "adam"
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
