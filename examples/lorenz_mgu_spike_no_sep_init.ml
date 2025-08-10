(* Lorenz attractor with no controls, with same state/control/cost parameters constant across trials and across time; 
  use a mini-MGU2 (ilqr-vae paper Appendix C.2) as generative model. use kroneckered posterior cov. 
  Generate spikes from Lorenz and use Poisson likelihood. *)
open Base
open Forward_torch
open Torch
open Sofo
open Lds_data

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run
let base = Optimizer.Config.Base.default

(* -----------------------------------------
   -- Generate Lorenz data            ------
   ----------------------------------------- *)

open Lorenz_common

(* state dim *)
let n = 20

(* control dim *)
let m = 5
let o = 200
let batch_size = 64
let dt = 0.01

(* tmax needs to be divisible by 8 *)
let tmax = 112
let tmax_simulate = 10000
let train_data = Lorenz_common.data tmax
let train_data_batch = get_batch train_data
let max_iter = 100000

(* map from Lorenz state to neural firing rate *)
let linear_map = Tensor.(randn [ 3; o ] ~device:base.device ~kind:base.kind)

(* let soft_relu_t x =
  let tmp = Tensor.(square x + f 4.) in
  let num = Tensor.(sqrt tmp + x) in
  Tensor.((num / f 2.) - f 1.) *)

(* list of Lorenz state * list of spike trains *)
let sample_data () =
  let trajectory = train_data_batch batch_size in
  let traj_no_init = List.tl_exn trajectory in
  let both =
    List.map traj_no_init ~f:(fun x ->
      let x =
        Tensor.of_bigarray ~device:base.device x |> Tensor.to_type ~type_:base.kind
      in
      (* TODO: how to ensure it is positive if using soft_relu? *)
      let mu = Tensor.(exp (matmul x linear_map)) in
      let o = Tensor.(poisson (f dt * mu)) in
      x, o)
  in
  List.map both ~f:fst, List.map both ~f:snd

let x0 = Maths.zeros ~device:base.device ~kind:base.kind [ batch_size; n ] |> Maths.any
let _W_0 = Maths.eye n ~device:base.device ~kind:base.kind |> Maths.any

(* -----------------------------------------
   ----- Utilitiy Functions ------
   ----------------------------------------- *)
(* inv_chol is the cholesky of the inverse of the covariance *)
let gaussian_llh ?mu ?(diagonal_inv_chol = true) ?(batched_inv_chol = false) ~inv_chol x =
  let d = x |> Maths.shape |> List.last_exn in
  let error_term =
    let error =
      let diff =
        match mu with
        | None -> x
        | Some mu -> Maths.(x - mu)
      in
      match diagonal_inv_chol, batched_inv_chol with
      | true, true -> Maths.(einsum [ diff, "ma"; inv_chol, "ma" ] "ma")
      | true, false -> Maths.(einsum [ diff, "ma"; inv_chol, "a" ] "ma")
      | false, true -> Maths.(einsum [ diff, "ma"; inv_chol, "mab" ] "mb")
      | false, false -> Maths.(einsum [ diff, "ma"; inv_chol, "ab" ] "mb")
    in
    Maths.einsum [ error, "ma"; error, "ma" ] "m"
  in
  let cov_term =
    let inv_chol =
      if diagonal_inv_chol then inv_chol else Maths.diagonal inv_chol ~offset:0
    in
    Maths.(neg (sum (log (sqr inv_chol))) |> reshape ~shape:[ 1 ])
  in
  let const_term = Float.(log (2. * pi) * of_int d) in
  Maths.(0.5 $* (const_term $+ error_term + cov_term)) |> Maths.neg

let precision_of_log_var log_var = Maths.(exp (neg log_var))
let sqrt_precision_of_log_var log_var = Maths.(exp (f (-0.5) * log_var))

(* list of length T of [m x b] to matrix of [m x b x T] *)
let concat_time (u_list : Maths.any Maths.t list) =
  List.map u_list ~f:(fun u -> Maths.unsqueeze ~dim:(-1) u) |> Maths.concat ~dim:2

(* concat a list of [m x 3] tensors to [T x m x 3] mat *)
let t_list_to_mat data =
  Tensor.concat (List.map data ~f:(Tensor.unsqueeze ~dim:0)) ~dim:0
  |> Tensor.to_bigarray ~kind:base.ba_kind

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)

let conv_threshold = 1e-4

module PP = struct
  type 'a p =
    { _U_f : 'a (* generative model *)
    ; _U_h : 'a
    ; _b_f : 'a
    ; _b_h : 'a
    ; _W : 'a
    ; _c : 'a
    ; _b : 'a
    ; _beta : 'a (* neuron-specific gain factor *)
    ; _log_prior_var : 'a (* log of the diagonal covariance of prior over u *)
    ; _space_var_lt : 'a
      (* the lower triangular part of the FULL covariance of space factor; *)
    ; _log_space_var_diag : 'a
      (* log of the diagonal part of the FULL covariance of space factor; *)
    ; _time_var_lt : 'a
      (* the lower triangular part of the FULL covariance of time factor; *)
    ; _log_time_var_diag : 'a
      (* log of the diagonal part of the FULL covariance of time factor; *)
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.Single)

module MGU = struct
  module P = P

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

  let pre_sig ~x (theta : _ Maths.some P.t) = Maths.((x *@ theta._U_f) + theta._b_f)

  let pre_g ~f_t ~x (theta : _ Maths.some P.t) =
    Maths.((f_t * x *@ theta._U_h) + theta._b_h)

  let x_hat ~pre_g ~u (theta : _ Maths.some P.t) =
    Maths.(soft_relu pre_g + (u *@ theta._W))

  (* rollout x list under sampled u *)
  let rollout_one_step ~x ~u (theta : _ Maths.some P.t) =
    let pre_sig = pre_sig ~x theta in
    let f_t = Maths.sigmoid pre_sig in
    let pre_g = pre_g ~f_t ~x theta in
    let x_hat = x_hat ~pre_g ~u theta in
    let new_x = Maths.(((f 1. - f_t) * x) + (f_t * x_hat)) in
    new_x

  (* (1 + e^-x)^{-2} (e^-x) *)
  let d_sigmoid x = Maths.(sigmoid x * (f 1. - sigmoid x))

  (* 1/2 [1 + (x^2 + 4)^{-1/2} x] *)
  let d_soft_relu x =
    let tmp = Maths.(sqr x + f 4.) in
    let tmp2 = Maths.(f 1. / sqrt tmp) in
    Maths.(((tmp2 * x) + f 1.) / f 2.)

  (*
     (* 2/ (x^2+4)^{3/2} *)
  let dd_soft_relu x =
    let tmp = Maths.(sqr x + f 4.) in
    let tmp2 = Maths.(sqrt tmp) in
    Maths.(f 2. / (tmp2 * tmp)) *)

  (* h = log(soft_relu(x)) *)
  (* let d_h x = Maths.(d_soft_relu x / soft_relu x)

  let dd_h x =
    let open Maths in
    let num = (soft_relu x * dd_soft_relu x) - sqr (d_soft_relu x) in
    let denom = sqr (d_soft_relu x) in
    num / denom *)

  (* _Fu and _Fx TESTED. *)
  let _Fu ~x (theta : _ Maths.some P.t) =
    let open Maths in
    match x with
    | Some x ->
      let pre_sig = pre_sig ~x theta in
      let f_t = sigmoid pre_sig in
      einsum [ f_t, "ma"; theta._W, "ba" ] "mba"
    | None -> any (zeros ~device:base.device ~kind:base.kind [ batch_size; m; n ])

  let _Fx ~x ~u (theta : _ Maths.some P.t) =
    let open Maths in
    match x, u with
    | Some x, Some u ->
      let pre_sig = pre_sig ~x theta in
      let f_t = sigmoid pre_sig in
      let pre_g = pre_g ~f_t ~x theta in
      let x_hat = x_hat ~pre_g ~u theta in
      let tmp_einsum2 a b = einsum [ a, "ba"; b, "ma" ] "mba" in
      let term1 = diag_embed (f 1. - f_t) ~offset:0 ~dim1:(-2) ~dim2:(-1) in
      let term2 =
        let tmp = d_sigmoid pre_sig * (x_hat - x) in
        tmp_einsum2 theta._U_f tmp
      in
      let term3 =
        let tmp1 = tmp_einsum2 theta._U_h (d_soft_relu pre_g) in
        let tmp2 = tmp_einsum2 theta._U_f (d_sigmoid pre_sig) in
        let tmp3 = diag_embed f_t ~offset:0 ~dim1:(-2) ~dim2:(-1) in
        let tmp4 = tmp2 + tmp3 in
        let tmp5 = einsum [ tmp4, "mab"; tmp1, "mbc" ] "mac" in
        unsqueeze ~dim:1 f_t * tmp5
      in
      let final = term1 + term2 + term3 in
      final
    | _ -> any (zeros ~device:base.device ~kind:base.kind [ batch_size; n; n ])

  let rollout_x ~u_list (theta : _ Maths.some P.t) =
    let _, x_list =
      List.foldi u_list ~init:(x0, []) ~f:(fun i (x, accu) u ->
        let new_x = rollout_one_step ~x ~u theta in
        new_x, new_x :: accu)
    in
    List.rev x_list

  let y_from_x x (theta : _ Maths.some P.t) =
    Maths.(einsum [ x, "ma"; theta._c, "ab" ] "mb" + theta._b)

  let rollout_y ~u_list (theta : _ Maths.some P.t) =
    let x_list = rollout_x ~u_list theta in
    List.map x_list ~f:(fun x -> y_from_x x theta)

  let mu ~y bs (theta : _ Maths.some P.t) =
    Maths.(f dt * exp y * broadcast_to theta._beta ~size:[ bs; o ])

  let rollout_mu ~u_list (theta : _ Maths.some P.t) =
    let bs = List.hd_exn (Maths.shape (List.hd_exn u_list)) in
    let y_list = rollout_y ~u_list theta in
    List.map y_list ~f:(fun y -> mu ~y bs theta)

  let rollout_sol ~u_list (theta : _ Maths.some P.t) =
    let x_list = rollout_x ~u_list theta in
    List.map2_exn u_list x_list ~f:(fun u x -> Lqr.Solution.{ u = Some u; x = Some x })

  (* artificially add one to tau so it goes from 0 to T *)
  let extend_tau_list (tau : Maths.any Maths.t option Lqr.Solution.p list) =
    let u_list = List.map tau ~f:(fun s -> s.u) in
    let x_list = List.map tau ~f:(fun s -> s.x) in
    let u_ext = u_list @ [ None ] in
    let x_ext = Some x0 :: x_list in
    List.map2_exn u_ext x_ext ~f:(fun u x -> Lqr.Solution.{ u; x })

  let any_to_const x = Maths.(any (const x))
  let opt_const_map x = Option.map x ~f:(fun x -> any_to_const x)

  let map_to_const (tau : Maths.any Maths.t option Lqr.Solution.p list) =
    List.map tau ~f:(fun tau ->
      Lqr.Solution.{ u = opt_const_map tau.u; x = opt_const_map tau.x })

  let logfact_tensor t = Tensor.lgamma Tensor.(t + f 1.)

  (* neg log-likelihood; -logp(o|x)*)
  let poisson_nll ~mu_list ~data =
    let open Maths in
    List.fold2_exn
      data
      mu_list
      ~init:(any (f 0.))
      ~f:(fun accu o mu ->
        let llh_t =
          let tmp1 = sum ~keepdim:false ~dim:[ 1 ] ((any (of_tensor o) * log mu) - mu) in
          let tmp2 =
            o
            |> logfact_tensor (* apply lgamma to each element: log(o_i!) *)
            |> Tensor.sum_dim_intlist
                 ~dim:(Some [ 1 ])
                 ~keepdim:false
                 ~dtype:base.kind (* sum over neurons *)
            |> Maths.of_tensor
          in
          neg (tmp1 + tmp2)
        in
        accu + llh_t)

  let poisson_nll_neg_jac ~data ~y (theta : _ Maths.some P.t) =
    let open Maths in
    let bs = List.hd_exn (Tensor.shape data) in
    let beta_reshaped = Maths.broadcast_to ~size:[ bs; o ] theta._beta in
    (* f is soft_relu, h is log(f) *)
    (* let _cx =
                  let tmp1 = any (of_tensor o) * d_h y_t in
                  let tmp2 = f dt * beta_reshaped * d_soft_relu y_t in
                  einsum [ tmp2 - tmp1, "ma"; theta._c, "ba" ] "mb"
                in *)
    (* f is exp, h (y)= y *)
    let tmp1 = any (of_tensor data) in
    let tmp2 = f dt * beta_reshaped * exp y in
    einsum [ tmp2 - tmp1, "ma"; theta._c, "ba" ] "mb"

  let poisson_nll_neg_hess ~data ~y (theta : _ Maths.some P.t) =
    let open Maths in
    let open Maths in
    let bs = List.hd_exn (Tensor.shape data) in
    let beta_reshaped = Maths.broadcast_to ~size:[ bs; o ] theta._beta in
    (* f is soft_relu, h is log(f) *)
    (*
       let _a =
        let tmp1 = any (of_tensor o) * dd_h y_t in
        let tmp2 = f dt * beta_reshaped * dd_soft_relu y_t in
        tmp2 - tmp1
      in
      einsum [ theta._c, "ik"; _a, "mk"; theta._c, "ij" ] "mij" *)
    (* f is exp, h (y)= y *)
    let _a =
      let tmp2 = f dt * beta_reshaped * exp y in
      tmp2
    in
    einsum [ theta._c, "ik"; _a, "mk"; theta._c, "jk" ] "mij"

  let gaussian_prior_nll ~u_list (theta : _ Maths.some P.t) =
    let open Maths in
    List.foldi
      u_list
      ~init:(any (f 0.))
      ~f:(fun i accu u ->
        accu
        + neg (gaussian_llh ~inv_chol:(sqrt_precision_of_log_var theta._log_prior_var) u))

  let gaussian_prior_neg_jac ~u (theta : _ Maths.some P.t) =
    Maths.(einsum [ u, "ma"; precision_of_log_var theta._log_prior_var, "a" ] "ma")

  let gaussian_prior_neg_hess ~bs ~m (theta : _ Maths.some P.t) =
    Maths.(
      diag_embed
        (precision_of_log_var theta._log_prior_var)
        ~offset:0
        ~dim1:(-2)
        ~dim2:(-1))
    |> Maths.unsqueeze ~dim:0
    |> Maths.broadcast_to ~size:[ bs; m; m ]

  (* optimal u determined from lqr *)
  let pred_u ~data (theta : _ Maths.some P.t) =
    let open Maths in
    let bs = List.hd_exn (Tensor.shape (List.hd_exn data)) in
    let _Cuu_batched = gaussian_prior_neg_hess ~bs ~m theta in
    let params_func ~no_tangents (tau : any t option Lqr.Solution.p list)
      : (any t option, (any t, any t -> any t) Lqr.momentary_params list) Lqr.Params.p
      =
      let o_tau_list =
        (* set o at time 0 as 0 *)
        let o_list_extended = Tensor.zeros_like (List.hd_exn data) :: data in
        let tau_extended =
          let tmp = extend_tau_list tau in
          if no_tangents then map_to_const tmp else tmp
        in
        List.map2_exn o_list_extended tau_extended ~f:(fun o tau -> o, tau)
      in
      let _Cuu_batched =
        if no_tangents then any_to_const _Cuu_batched else _Cuu_batched
      in
      let theta = if no_tangents then P.map theta ~f:any_to_const else theta in
      let tmp_list =
        Lqr.Params.
          { x0 = Some x0
          ; params =
              List.mapi o_tau_list ~f:(fun i (o, s) ->
                let y_t = y_from_x (Option.value_exn s.x) theta in
                let _cx = poisson_nll_neg_jac ~data:o ~y:y_t theta in
                let _Cxx = poisson_nll_neg_hess ~data:o ~y:y_t theta in
                let _cu =
                  match s.u with
                  | None -> None
                  | Some u -> Some (gaussian_prior_neg_jac ~u theta)
                in
                Lds_data.Temp.
                  { _f = None
                  ; _Fx_prod = _Fx ~x:s.x ~u:s.u theta
                  ; _Fu_prod = _Fu ~x:s.x theta
                  ; _cx = Some _cx
                  ; _cu
                  ; _Cxx
                  ; _Cxu = None
                  ; _Cuu = _Cuu_batched
                  })
          }
      in
      Lds_data.map_naive tmp_list ~batch_const:false
    in
    (* given a trajectory and parameters, calculate average cost across batch (summed over time) *)
    let cost_func (tau : any t option Lqr.Solution.p list) =
      let u_list = List.map tau ~f:(fun s -> s.u |> Option.value_exn) in
      let mu_list = rollout_mu ~u_list theta in
      let x_cost = poisson_nll ~mu_list ~data in
      let u_cost = gaussian_prior_nll ~u_list theta in
      x_cost + u_cost |> to_tensor
    in
    let u_init =
      List.init tmax ~f:(fun i ->
        any (zeros ~device:base.device ~kind:base.kind [ batch_size; m ]))
    in
    let tau_init = rollout_sol ~u_list:u_init theta in
    (* TODO: is there a more elegant way? Currently I need to set batch_const to false since _Fx and _Fu has batch dim. *)
    (* use iqr to obtain the optimal u *)
    let f_theta ~i:_ = rollout_one_step theta in
    let sol, backward_info =
      Ilqr._isolve
        ~linesearch:true
        ~linesearch_bs_avg:true
        ~expected_reduction:false
        ~batch_const:false
        ~gamma:0.5
        ~f_theta
        ~cost_func
        ~params_func
        ~conv_threshold
        ~tau_init
        2000
    in
    List.map sol ~f:(fun s -> s.u), backward_info

  let _space_cov (theta : _ Maths.some P.t) =
    let tri = Maths.(tril ~_diagonal:(-1) theta._space_var_lt) in
    Maths.(
      (tri *@ transpose tri)
      + diag_embed ~offset:0 ~dim1:0 ~dim2:1 (exp theta._log_space_var_diag))

  let _time_cov (theta : _ Maths.some P.t) =
    let tri = Maths.(tril ~_diagonal:(-1) theta._time_var_lt) in
    Maths.(
      (tri *@ transpose tri)
      + diag_embed ~offset:0 ~dim1:0 ~dim2:1 (exp theta._log_time_var_diag))

  let kronecker_sample ~optimal_u_list (theta : _ Maths.some P.t) =
    let open Maths in
    (* sample u_{0} to u_{T-1} from the kronecker formation *)
    let u_list =
      let xi =
        randn ~device:base.device ~kind:base.kind [ batch_size; m; Int.(tmax) ] |> any
      in
      let optimal_u = concat_time optimal_u_list in
      let _space_chol = cholesky (_space_cov theta)
      and _time_chol = cholesky (_time_cov theta) in
      (* shape [bx x m x T] *)
      let xi_space = einsum [ _space_chol, "ab"; xi, "mbt" ] "mat" in
      let xi_time = einsum [ xi_space, "mat"; _time_chol, "kt" ] "mak" in
      let meaned = xi_time + optimal_u in
      List.init
        Int.(tmax)
        ~f:(fun i ->
          slice ~dim:2 ~start:i ~end_:Int.(i + 1) ~step:1 meaned
          |> reshape ~shape:[ batch_size; m ])
    in
    u_list

  (* M1: use kroneckered posterior *)
  let neg_elbo ~data (theta : _ Maths.some P.t) =
    let open Maths in
    (* obtain u from lqr *)
    let optimal_u_list, _ = pred_u ~data theta in
    let u_sampled_total = kronecker_sample ~optimal_u_list theta in
    (* calculate the likelihood term *)
    let mu_list = rollout_mu ~u_list:u_sampled_total theta in
    let lik_term = neg (poisson_nll ~mu_list ~data) in
    (* calculate the kl term using samples *)
    let optimal_u_concated = concat_time u_sampled_total in
    let kl =
      let prior = neg (gaussian_prior_nll ~u_list:u_sampled_total theta) in
      let neg_entropy_u =
        let u = concat_time u_sampled_total |> reshape ~shape:[ batch_size; -1 ] in
        let optimal_u = reshape optimal_u_concated ~shape:[ batch_size; -1 ] in
        let inv_chol =
          let _space_var_inv = inv_sqr (_space_cov theta) |> Maths.contiguous in
          let _time_var_inv = inv_sqr (_time_cov theta) |> Maths.contiguous in
          cholesky (kron _space_var_inv _time_var_inv)
        in
        gaussian_llh ~diagonal_inv_chol:false ~mu:optimal_u ~inv_chol u
      in
      neg_entropy_u - prior
    in
    mean ~dim:[ 0 ] (neg (lik_term - kl) / f Float.(of_int tmax * of_int o)), mu_list

  let f ~data (theta : _ Maths.some P.t) =
    let neg_elbo, _ = neg_elbo ~data theta in
    neg_elbo, None

  let init : P.param =
    let to_param a = a |> Maths.of_tensor |> Prms.Single.free in
    let _U_f =
      Sofo.gaussian_tensor_normed ~device:base.device ~kind:base.kind ~sigma:1. [ n; n ]
      |> to_param
    in
    let _U_h =
      Sofo.gaussian_tensor_normed ~device:base.device ~kind:base.kind ~sigma:1. [ n; n ]
      |> to_param
    in
    let _b_f = Tensor.zeros ~device:base.device [ 1; n ] |> to_param in
    let _b_h = Tensor.zeros ~device:base.device [ 1; n ] |> to_param in
    let _W =
      Sofo.gaussian_tensor_normed ~device:base.device ~kind:base.kind ~sigma:1. [ m; n ]
      |> to_param
    in
    let _c =
      Sofo.gaussian_tensor_normed ~device:base.device ~kind:base.kind ~sigma:1. [ n; o ]
      |> to_param
    in
    let _b = Tensor.ones ~device:base.device [ 1; o ] |> to_param in
    let _beta = Tensor.ones ~device:base.device [ 1; o ] |> to_param in
    (* scale the prior over the initial condition w.r.t. 1/dt *)
    let _log_prior_var =
      Tensor.(log (f Float.(square 1.) * ones ~device:base.device ~kind:base.kind [ m ]))
      |> to_param
    in
    let _space_var_lt =
      Tensor.(f 1e-3 * ones ~device:base.device ~kind:base.kind [ m; m ]) |> to_param
    in
    let _log_space_var_diag =
      Tensor.(log (ones ~device:base.device ~kind:base.kind [ m ])) |> to_param
    in
    let _time_var_lt =
      Tensor.(
        f 1e-3 * ones ~device:base.device ~kind:base.kind [ Int.(tmax); Int.(tmax) ])
      |> to_param
    in
    let _log_time_var_diag =
      Tensor.(log (ones ~device:base.device ~kind:base.kind [ Int.(tmax) ])) |> to_param
    in
    { _U_f
    ; _U_h
    ; _b_f
    ; _b_h
    ; _W
    ; _c
    ; _b
    ; _beta
    ; _log_prior_var
    ; _log_space_var_diag
    ; _space_var_lt
    ; _log_time_var_diag
    ; _time_var_lt
    }

  let simulate ~theta ~data =
    (* infer the optimal u *)
    let optimal_u_list, _ = pred_u ~data theta in
    (* rollout firing rate and spike train *)
    let mu_list = rollout_mu ~u_list:optimal_u_list theta in
    let o_list =
      List.map mu_list ~f:(fun mu ->
        let mu_t = Maths.to_tensor mu in
        Tensor.(poisson mu_t))
    in
    optimal_u_list, o_list

  (* simulate the autonomous dynamics of Lorenz given initial condition. *)
  let simulate_auto ~theta =
    let init_cond =
      Maths.randn ~device:base.device ~kind:base.kind [ 1; n ] |> Maths.any
    in
    (* rollout y with init_cond *)
    let _, o_list_rev =
      List.fold
        (List.init tmax_simulate ~f:(fun i -> i))
        ~init:(init_cond, [])
        ~f:(fun (x, o_list) i ->
          let x =
            rollout_one_step
              ~x
              ~u:Maths.(any (zeros ~device:base.device ~kind:base.kind [ 1; m ]))
              theta
          in
          let y = y_from_x x theta in
          let mu = mu ~y 1 theta in
          let o = Tensor.(poisson (Maths.to_tensor mu)) in
          x, o :: o_list)
    in
    List.rev o_list_rev
end

module O = Optimizer.Adam (MGU.P)

let config ~t =
  Optimizer.Config.Adam.
    { base
    ; beta_1 = 0.9
    ; beta_2 = 0.99
    ; eps = 1e-4
    ; learning_rate = Some Float.(0.001 / (1. + sqrt (of_int t / 1.)))
    ; weight_decay = None
    }

let rec loop ~t ~out ~state running_avg =
  Stdlib.Gc.major ();
  let theta = O.params state in
  let theta_ = O.P.value theta in
  let theta_dual =
    O.P.map theta_ ~f:(fun x ->
      let x =
        x |> Maths.to_tensor |> Tensor.copy |> Tensor.to_device ~device:base.device
      in
      let x = Tensor.set_requires_grad x ~r:true in
      Tensor.zero_grad x;
      Maths.of_tensor x)
  in
  let data_hidden, data = sample_data () in
  let loss, true_g =
    let loss, _ = MGU.f ~data (P.map theta_dual ~f:Maths.any) in
    let loss = Maths.to_tensor loss in
    Tensor.backward loss;
    ( Tensor.to_float0_exn loss
    , O.P.map2 theta (O.P.to_tensor theta_dual) ~f:(fun tagged p ->
        match tagged with
        | Prms.Pinned _ -> Maths.(f 0.)
        | _ -> Maths.of_tensor (Tensor.grad p)) )
  in
  let new_state = O.step ~config:(config ~t) ~info:true_g state in
  let running_avg =
    let loss_avg =
      match running_avg with
      | [] -> loss
      | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
    in
    if t % 10 = 0
    then (
      let theta = MGU.P.value (O.params new_state) in
      (* simulate trajectory *)
      let u_list, o_list = MGU.simulate ~theta:(MGU.P.map theta ~f:Maths.any) ~data in
      let u_list_t = List.map u_list ~f:Maths.to_tensor in
      Arr.(save_npy ~out:(in_dir "o") (t_list_to_mat data));
      Arr.(save_npy ~out:(in_dir "h") (t_list_to_mat data_hidden));
      Arr.(save_npy ~out:(in_dir "u_inferred") (t_list_to_mat u_list_t));
      Arr.(save_npy ~out:(in_dir "o_gen") (t_list_to_mat o_list));
      (* let o_list_auto = MGU.simulate_auto ~theta:(MGU.P.map theta ~f:Maths.any) in
      Arr.(save_npy ~out:(in_dir "o_auto") (t_list_to_mat o_list_auto)); *)
      (* save params *)
      O.P.C.save theta ~kind:base.ba_kind ~out:(in_dir "adam_params");
      print [%message (t : int) (loss_avg : float)];
      Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss_avg |] 1 2)));
    []
  in
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state (loss :: running_avg)

(* Start the loop. *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init MGU.init) []

(* let _ =
  let theta = O.P.C.load ~device:base.device (in_dir "adam_params") in
  Sofo.print [%message "params loaded"];
  (* simulate trajectory *)
  let y_list_auto = MGU.simulate_auto ~theta:(MGU.P.map theta ~f:Maths.any) in
  let y_list_auto_t = List.map y_list_auto ~f:Maths.to_tensor in
  Arr.(save_npy ~out:(in_dir "y_auto") (t_list_to_mat y_list_auto_t)) *)
