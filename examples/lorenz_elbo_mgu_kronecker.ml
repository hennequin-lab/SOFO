(* Lorenz attractor with no controls, with same state/control/cost parameters constant across trials and across time; 
  use a mini-MGU2 (ilqr-vae paper Appendix C.2) as generative model. use kroneckered posterior cov. *)
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

(* -----------------------------------------
   -- Generate Lorenz data            ------
   ----------------------------------------- *)

open Lorenz_common

(* state dim *)
let n = 20

(* control dim *)
let m = 5
let o = 3
let batch_size = 64

(* tmax needs to be divisible by 8 *)
let tmax = 112
let tmax_simulate = 10000
let train_data = Lorenz_common.data tmax
let train_data_batch = get_batch train_data
let max_iter = 100000
let base = Optimizer.Config.Base.default

(* list of clean data * list of noisy data *)
let sample_data () =
  let trajectory = train_data_batch batch_size in
  let traj_no_init = List.tl_exn trajectory in
  let both =
    List.map traj_no_init ~f:(fun x ->
      let x =
        Tensor.of_bigarray ~device:base.device x |> Tensor.to_type ~type_:base.kind
      in
      let noise = Tensor.(f 0.1 * rand_like x) in
      x, Tensor.(noise + x))
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
let std_of_log_var log_var = Maths.(exp (f 0.5 * log_var))
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
    ; _log_prior_var_0 : 'a (* log of the diagonal covariance of prior over u_0 *)
    ; _log_prior_var : 'a (* log of the diagonal covariance of prior over u *)
    ; _log_obs_var : 'a (* log of the diagonal covariance of emission noise; *)
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

  let x_hat ~i ~pre_g ~u (theta : _ Maths.some P.t) =
    Maths.(soft_relu pre_g + (u *@ if i = 0 then _W_0 else theta._W))

  (* rollout x list under sampled u *)
  let rollout_one_step ~i ~x ~u (theta : _ Maths.some P.t) =
    let pre_sig = pre_sig ~x theta in
    let f_t = Maths.sigmoid pre_sig in
    let pre_g = pre_g ~f_t ~x theta in
    let x_hat = x_hat ~i ~pre_g ~u theta in
    let new_x = Maths.(((f 1. - f_t) * x) + (f_t * x_hat)) in
    new_x

  (* (1 + e^-x)^{-2} (e^-x) *)
  let d_sigmoid x = Maths.(sigmoid x * (f 1. - sigmoid x))

  (* 1/2 [1 + (x^2 + 4)^{-1/2} x] *)
  let d_soft_relu x =
    let tmp = Maths.(sqr x + f 4.) in
    let tmp2 = Maths.(f 1. / sqrt tmp) in
    Maths.(((tmp2 * x) + f 1.) / f 2.)

  (* _Fu and _Fx TESTED. *)
  let _Fu ~i ~x (theta : _ Maths.some P.t) =
    let open Maths in
    match x with
    | Some x ->
      let pre_sig = pre_sig ~x theta in
      let f_t = sigmoid pre_sig in
      einsum [ f_t, "ma"; (if i = 0 then _W_0 else theta._W), "ba" ] "mba"
    | None -> any (zeros ~device:base.device ~kind:base.kind [ batch_size; m; n ])

  let _Fx ~i ~x ~u (theta : _ Maths.some P.t) =
    let open Maths in
    match x, u with
    | Some x, Some u ->
      let pre_sig = pre_sig ~x theta in
      let f_t = sigmoid pre_sig in
      let pre_g = pre_g ~f_t ~x theta in
      let x_hat = x_hat ~i ~pre_g ~u theta in
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
        let new_x = rollout_one_step ~i ~x ~u theta in
        new_x, new_x :: accu)
    in
    List.rev x_list

  let rollout_y ~u_list (theta : _ Maths.some P.t) =
    let x_list = rollout_x ~u_list theta in
    List.map x_list ~f:(fun x ->
      Maths.(einsum [ x, "ma"; theta._c, "ab" ] "mb" + theta._b))

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

  (* optimal u determined from lqr *)
  let pred_u ~data (theta : _ Maths.some P.t) =
    let open Maths in
    let c_trans = transpose theta._c ~dims:[ 1; 0 ] in
    let _obs_var_inv = precision_of_log_var theta._log_obs_var in
    let _Cxx_batched =
      let _Cxx =
        Maths.(einsum [ theta._c, "ab"; _obs_var_inv, "b"; c_trans, "bc" ] "ac")
        |> Maths.unsqueeze ~dim:0
      in
      Maths.broadcast_to _Cxx ~size:[ batch_size; n; n ]
    in
    let _Cuu_0_batched =
      let _Cuu_0 =
        Maths.(diag_embed (exp theta._log_prior_var_0) ~offset:0 ~dim1:(-2) ~dim2:(-1))
        |> Maths.unsqueeze ~dim:0
      in
      Maths.broadcast_to _Cuu_0 ~size:[ batch_size; n; n ]
    in
    let _Cuu_batched =
      let _Cuu =
        Maths.(diag_embed (exp theta._log_prior_var) ~offset:0 ~dim1:(-2) ~dim2:(-1))
        |> Maths.unsqueeze ~dim:0
      in
      Maths.broadcast_to _Cuu ~size:[ batch_size; m; m ]
    in
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
      let _obs_var_inv =
        if no_tangents then Maths.(any (const _obs_var_inv)) else _obs_var_inv
      in
      let theta = if no_tangents then P.map theta ~f:any_to_const else theta in
      let _cx_common =
        let tmp = einsum [ theta._b, "ab"; _obs_var_inv, "b"; c_trans, "bc" ] "ac" in
        if no_tangents then any_to_const tmp else tmp
      in
      let tmp_list =
        Lqr.Params.
          { x0 = Some x0
          ; params =
              List.mapi o_tau_list ~f:(fun i (o, s) ->
                let _cx =
                  let tmp1 = einsum [ any (of_tensor o), "ab"; _obs_var_inv, "b" ] "ab" in
                  let tmp2 =
                    einsum
                      [ Option.value_exn s.x, "ab"; theta._c, "bc"; _obs_var_inv, "c" ]
                      "ac"
                  in
                  _cx_common + ((tmp2 - tmp1) *@ c_trans)
                in
                let _cu =
                  match s.u with
                  | None -> None
                  | Some u ->
                    Some
                      (einsum
                         [ u, "ma"
                         ; (if i = 0 then _Cuu_0_batched else _Cuu_batched), "mab"
                         ]
                         "mb")
                in
                Lds_data.Temp.
                  { _f = None
                  ; _Fx_prod = _Fx ~i ~x:s.x ~u:s.u theta
                  ; _Fu_prod = _Fu ~i ~x:s.x theta
                  ; _cx = Some _cx
                  ; _cu
                  ; _Cxx = _Cxx_batched
                  ; _Cxu = None
                  ; _Cuu = (if i = 0 then _Cuu_0_batched else _Cuu_batched)
                  })
          }
      in
      Lds_data.map_naive tmp_list ~batch_const:false
    in
    (* given a trajectory and parameters, calculate average cost across batch (summed over time) *)
    let cost_func (tau : any t option Lqr.Solution.p list) =
      let x_list = List.map tau ~f:(fun s -> s.x |> Option.value_exn) in
      let u_list = List.map tau ~f:(fun s -> s.u |> Option.value_exn) in
      let x_cost =
        let x_cost_list =
          List.map2_exn x_list data ~f:(fun x data ->
            let tmp1 =
              einsum
                [ x, "ma"; theta._c, "ab"; _obs_var_inv, "b"; c_trans, "bc"; x, "mc" ]
                "m"
            in
            let tmp2 =
              let diff =
                einsum
                  [ broadcast_to theta._b ~size:[ batch_size; o ] - of_tensor data, "mb"
                  ; _obs_var_inv, "b"
                  ; c_trans, "bc"
                  ]
                  "mc"
              in
              f 2. * einsum [ x, "ma"; diff, "ma" ] "m"
            in
            tmp1 + tmp2)
        in
        List.fold x_cost_list ~init:(any (f 0.)) ~f:(fun accu c -> accu + c)
      in
      let u_cost =
        List.foldi
          u_list
          ~init:(any (f 0.))
          ~f:(fun i accu u ->
            accu
            + einsum
                [ u, "ma"
                ; (if i = 0 then _Cuu_0_batched else _Cuu_batched), "mab"
                ; u, "mb"
                ]
                "m")
      in
      x_cost + u_cost |> to_tensor
    in
    let u_init =
      List.init tmax ~f:(fun i ->
        any
          (zeros
             ~device:base.device
             ~kind:base.kind
             [ batch_size; (if i = 0 then n else m) ]))
    in
    let tau_init = rollout_sol ~u_list:u_init theta in
    (* TODO: is there a more elegant way? Currently I need to set batch_const to false since _Fx and _Fu has batch dim. *)
    (* use iqr to obtain the optimal u *)
    let f_theta = rollout_one_step theta in
    let sol, backward_info =
      Ilqr._isolve
        ~linesearch:true
        ~linesearch_bs_avg:true
        ~expected_reduction:true
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
    (* sample u_{1} to u_{T-1} from the kronecker formation *)
    let u_list =
      let optimal_u = concat_time optimal_u_list in
      let xi =
        randn ~device:base.device ~kind:base.kind [ batch_size; m; Int.(tmax - 1) ] |> any
      in
      let _space_chol = cholesky (_space_cov theta)
      and _time_chol = cholesky (_time_cov theta) in
      let xi_space = einsum [ xi, "mbt"; _space_chol, "ba" ] "mat" in
      let xi_time = einsum [ xi_space, "mat"; _time_chol, "tk" ] "mak" in
      let meaned = xi_time + optimal_u in
      List.init
        Int.(tmax - 1)
        ~f:(fun i ->
          slice ~dim:2 ~start:i ~end_:Int.(i + 1) ~step:1 meaned
          |> reshape ~shape:[ batch_size; m ])
    in
    u_list

  let lik_term ~y_pred ~data (theta : _ Maths.some P.t) =
    let open Maths in
    List.fold2_exn
      data
      y_pred
      ~init:(any (f 0.))
      ~f:(fun accu o y_pred ->
        accu
        + gaussian_llh
            ~mu:y_pred
            ~inv_chol:(sqrt_precision_of_log_var theta._log_obs_var)
            (any (of_tensor o)))

  (* M1: use kroneckered posterior *)
  let neg_elbo ~data (theta : _ Maths.some P.t) =
    let open Maths in
    (* obtain u from lqr *)
    let optimal_u_list, bck_info = pred_u ~data theta in
    (* posterior covariance of u_0 = Quu_0^-1 *)
    let _Quu_0_chol = (List.hd_exn bck_info)._Quu_chol |> Option.value_exn in
    let _Quu_0_inv_chol =
      let _Quu_0_chol_inv = inv_sqr _Quu_0_chol in
      cholesky (einsum [ _Quu_0_chol_inv, "mba"; _Quu_0_chol_inv, "mbc" ] "mac")
    in
    let optimal_u_0 = List.hd_exn optimal_u_list
    and optimal_u_rest_list = List.tl_exn optimal_u_list in
    let u_0_sampled =
      let xi = randn ~device:base.device ~kind:base.kind [ batch_size; n ] |> any in
      optimal_u_0 + einsum [ xi, "ma"; _Quu_0_inv_chol, "mab" ] "mb"
    in
    let u_sampled_rest = kronecker_sample ~optimal_u_list:optimal_u_rest_list theta in
    let u_sampled_total = u_0_sampled :: u_sampled_rest in
    (* calculate the likelihood term *)
    let y_pred = rollout_y ~u_list:u_sampled_total theta in
    let lik_term = lik_term ~y_pred ~data theta in
    (* calculate the kl term using samples *)
    let optimal_u_rest_concated = concat_time optimal_u_rest_list in
    let kl =
      let prior =
        List.foldi u_sampled_total ~init:None ~f:(fun t accu u ->
          if t % 1 = 0 then Stdlib.Gc.major ();
          let increment =
            gaussian_llh
              ~inv_chol:
                (any
                   (sqrt_precision_of_log_var
                      (if t = 0 then theta._log_prior_var_0 else theta._log_prior_var)))
              u
          in
          match accu with
          | None -> Some increment
          | Some accu -> Some (accu + increment))
        |> Option.value_exn
      in
      let neg_entropy_u_0 =
        gaussian_llh
          ~diagonal_inv_chol:false
          ~batched_inv_chol:true
          ~mu:optimal_u_0
          ~inv_chol:_Quu_0_chol
          u_0_sampled
      in
      let neg_entropy_u_rest =
        let u = concat_time u_sampled_rest |> reshape ~shape:[ batch_size; -1 ] in
        let optimal_u = reshape optimal_u_rest_concated ~shape:[ batch_size; -1 ] in
        let inv_chol =
          let _space_var_inv = inv_sqr (_space_cov theta) |> Maths.contiguous in
          let _time_var_inv = inv_sqr (_time_cov theta) |> Maths.contiguous in
          cholesky (kron _space_var_inv _time_var_inv)
        in
        gaussian_llh ~diagonal_inv_chol:false ~mu:optimal_u ~inv_chol u
      in
      neg_entropy_u_0 + neg_entropy_u_rest - prior
    in
    mean ~dim:[ 0 ] (neg (lik_term - kl) / f Float.(of_int tmax * of_int o)), y_pred

  (* M2: use Kalman-filtering to estimate kl *)
  (* let sample_and_kl ~(theta : _ Maths.some P.t) ~optimal_u_list o_list =
    let open Maths in
    let o_list = List.map ~f:Maths.of_tensor o_list in
    let optimal_u_o_list = List.map2_exn optimal_u_list o_list ~f:(fun u o -> u, o) in
    let scaling_factor = reshape theta._scaling_factor ~shape:[ 1; 1; -1 ] in
    let z0 = Maths.(any (zeros ~device:base.device ~kind:base.kind [ batch_size; n ])) in
    let _Cuu_0_batched =
      let _Cuu_0 =
        Maths.(diag_embed (exp theta._log_prior_var_0) ~offset:0 ~dim1:(-2) ~dim2:(-1))
        |> Maths.unsqueeze ~dim:0
      in
      Maths.broadcast_to _Cuu_0 ~size:[ batch_size; n; n ]
    in
    let _Cuu_batched =
      let _Cuu =
        Maths.(diag_embed (exp theta._log_prior_var) ~offset:0 ~dim1:(-2) ~dim2:(-1))
        |> Maths.unsqueeze ~dim:0
      in
      Maths.broadcast_to _Cuu ~size:[ batch_size; m; m ]
    in
    let _, kl, us =
      List.foldi
        optimal_u_o_list
        ~init:(z0, any (f 0.), [])
        ~f:(fun i (z, kl, us) (ustar, o) ->
          Stdlib.Gc.major ();
          let zpred = rollout_one_step ~i ~x:z ~u:ustar theta in
          let ypred = Maths.((zpred *@ theta._c) + theta._b) in
          let delta = o - ypred in
          (* approximate z_{t+1} = f(z_t, u_opt) + df/du u. dimension is now [batch_size x m x n] *)
          let _b_prime = _Fu ~i ~x:(Some z) theta in
          (* BC / obs_var *)
          let btrinv =
            let tmp = einsum [ _b_prime, "mij"; theta._c, "jo" ] "mio" in
            tmp / reshape ~shape:[ -1; 1; 1 ] (exp theta._log_obs_var)
          in
          (* cholesky of posterior precision of filtered covariance of u *)
          let precision_chol =
            let tmp =
              (if i = 0 then _Cuu_0_batched else _Cuu_batched)
              + einsum [ btrinv, "mio"; theta._c, "jo"; _b_prime, "mkj" ] "mik"
              |> cholesky
            in
            tmp * scaling_factor
          in
          (* cov = cov_chol cov_chol^T *)
          let cov_chol =
            let batched_eye =
              let s = if i = 0 then n else m in
              broadcast_to
                (eye ~device:base.device ~kind:base.kind s)
                ~size:[ batch_size; s; s ]
            in
            linsolve_triangular ~left:true ~upper:false precision_chol batched_eye
            |> transpose ~dims:[ 0; 2; 1 ]
          in
          (* posterior mean of filtered u *)
          let mu =
            let tmp = einsum [ btrinv, "mio"; delta, "mo" ] "mi" in
            einsum [ tmp, "ma"; cov_chol, "mab"; cov_chol, "mcb" ] "mc"
          in
          (* sample from posterior filtered covariance of u. *)
          let u_diff_elbo =
            einsum
              [ ( any
                    (Maths.randn
                       ~device:base.device
                       ~kind:base.kind
                       [ batch_size; (if i = 0 then n else m) ])
                , "ma" )
              ; cov_chol, "mba"
              ]
              "mb"
          in
          let u_sample = mu + u_diff_elbo in
          let u_final = ustar + u_sample in
          (* propagate that sample to update z *)
          let z = rollout_one_step ~i ~x:z ~u:u_final theta in
          (* update the KL divergence *)
          let kl =
            let prior_term =
              gaussian_llh
                ~inv_chol:
                  (any
                     (std_of_log_var
                        (if i = 0 then theta._log_prior_var_0 else theta._log_prior_var)))
                u_final
            in
            let q_term =
              gaussian_llh
                ~diagonal_inv_chol:false
                ~batched_inv_chol:true
                ~inv_chol:precision_chol
                u_diff_elbo
            in
            kl + q_term - prior_term
          in
          z, kl, (ustar + u_sample) :: us)
    in
    kl, List.rev us

  let neg_elbo_filter ~data (theta : _ Maths.some P.t) =
    (* obtain u from lqr *)
    let optimal_u_list, _ = pred_u ~data theta in
    let kl, u_sampled =
      sample_and_kl ~theta:(theta : _ Maths.some P.t) ~optimal_u_list data
    in
    let y_pred = rollout_y ~u_list:u_sampled theta in
    let lik_term = lik_term ~y_pred ~data theta in
    let neg_elbo =
      Maths.(neg (mean ~dim:[ 0 ] (lik_term - kl)) / f Float.(of_int tmax * of_int o))
    in
    neg_elbo, y_pred *)

  let ggn ~y_pred (theta : _ Maths.some P.t) =
    let open Maths in
    let obs_precision = precision_of_log_var theta._log_obs_var in
    let obs_precision_p = of_tensor (to_tensor obs_precision) in
    let sigma2_t = tangent (exp theta._log_obs_var) |> Option.value_exn in
    List.fold y_pred ~init:(f 0.) ~f:(fun accu y_pred ->
      let mu_t = tangent y_pred |> Option.value_exn |> const in
      let ggn_part1 =
        C.(einsum [ mu_t, "kmo"; obs_precision_p, "o"; mu_t, "lmo" ] "kl")
      in
      let ggn_part2 =
        C.(
          einsum
            [ Float.(0.5 * of_int batch_size) $* sigma2_t * sqr obs_precision_p, "ky"
            ; sigma2_t, "ly"
            ]
            "kl")
      in
      C.(accu + (Float.(1. / (of_int o * of_int tmax)) $* ggn_part1 + ggn_part2)))

  let f ~data (theta : _ Maths.some P.t) =
    let neg_elbo, _ = neg_elbo ~data theta in
    neg_elbo, None

  let f_sofo ~data (theta : _ Maths.some P.t) =
    let neg_elbo, y_pred = neg_elbo ~data theta in
    let ggn = ggn ~y_pred theta in
    neg_elbo, ggn

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
    let _b = Tensor.zeros ~device:base.device [ 1; o ] |> to_param in
    (* scale the prior over the initial condition w.r.t. 1/dt *)
    let _log_prior_var_0 =
      Tensor.(log (f Float.(1. / 0.01) * ones ~device:base.device ~kind:base.kind [ n ]))
      |> to_param
    in
    let _log_prior_var =
      Tensor.(log (f Float.(square 1.) * ones ~device:base.device ~kind:base.kind [ m ]))
      |> to_param
    in
    let _log_obs_var =
      Tensor.(log (f Float.(square 1.) * ones ~device:base.device ~kind:base.kind [ o ]))
      |> to_param
    in
    let _space_var_lt =
      Tensor.zeros ~device:base.device ~kind:base.kind [ m; m ] |> to_param
    in
    let _log_space_var_diag =
      Tensor.(log (ones ~device:base.device ~kind:base.kind [ m ])) |> to_param
    in
    let _time_var_lt =
      Tensor.zeros ~device:base.device ~kind:base.kind [ tmax - 1; tmax - 1 ] |> to_param
    in
    let _log_time_var_diag =
      Tensor.(log (ones ~device:base.device ~kind:base.kind [ Int.(tmax - 1) ]))
      |> to_param
    in
    { _U_f
    ; _U_h
    ; _b_f
    ; _b_h
    ; _W
    ; _c
    ; _b
    ; _log_prior_var_0
    ; _log_prior_var
    ; _log_obs_var
    ; _log_space_var_diag
    ; _space_var_lt
    ; _log_time_var_diag
    ; _time_var_lt
    }

  let simulate ~theta ~data =
    (* infer the optimal u *)
    let optimal_u_list, _ = pred_u ~data theta in
    (* rollout x and y *)
    let y_list = rollout_y ~u_list:optimal_u_list theta in
    let o_list =
      List.map y_list ~f:(fun y ->
        Maths.(
          y
          + einsum
              [ ( any (Maths.randn ~device:base.device ~kind:base.kind [ batch_size; o ])
                , "ma" )
              ; std_of_log_var theta._log_obs_var, "a"
              ]
              "ma"))
    in
    List.hd_exn optimal_u_list, List.tl_exn optimal_u_list, y_list, o_list

  (* simulate the autonomous dynamics of Lorenz given initial condition. *)
  let simulate_auto ~theta =
    let init_cond =
      Maths.randn ~device:base.device ~kind:base.kind [ 1; n ] |> Maths.any
    in
    (* rollout y with init_cond *)
    let _, y_list_rev =
      List.fold
        (List.init tmax_simulate ~f:(fun i -> i))
        ~init:(init_cond, [])
        ~f:(fun (x, y_list) i ->
          let x =
            rollout_one_step
              ~i
              ~x
              ~u:
                Maths.(
                  any
                    (zeros
                       ~device:base.device
                       ~kind:base.kind
                       [ 1; (if i = 0 then n else m) ]))
              theta
          in
          let y = Maths.((x *@ theta._c) + theta._b) in
          x, y :: y_list)
    in
    List.rev y_list_rev
end

(* module O = Optimizer.SOFO (MGU.P)

let config ~t =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some Float.(1. / (1. + sqrt (of_int t / 100.)))
    ; n_tangents = 64
    ; damping = `relative_from_top 1e-5
    }

let rec loop ~t ~out ~state running_avg =
  Stdlib.Gc.major ();
  let data_unnoised, data = sample_data () in
  let theta, tangents = O.prepare ~config:(config ~t) state in
  let loss, ggn = MGU.f_sofo ~data (P.map theta ~f:Maths.any) in
  let new_state = O.step ~config:(config ~t) ~info:{ loss; ggn; tangents } state in
  let loss = Maths.to_float_exn (Maths.const loss) in
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
      let u_0, u_list, y_list, o_list =
        MGU.simulate ~theta:(MGU.P.map theta ~f:Maths.any) ~data
      in
      let y_list_auto = MGU.simulate_auto ~theta:(MGU.P.map theta ~f:Maths.any) in
      let u_0 = Maths.to_tensor u_0
      and u_list_t = List.map u_list ~f:Maths.to_tensor
      and y_list_t = List.map y_list ~f:Maths.to_tensor
      and o_list_t = List.map o_list ~f:Maths.to_tensor
      and y_list_auto_t = List.map y_list_auto ~f:Maths.to_tensor in
      Arr.(save_npy ~out:(in_dir "o") (t_list_to_mat data));
      Arr.(save_npy ~out:(in_dir "y") (t_list_to_mat data_unnoised));
      Arr.(save_npy ~out:(in_dir "u0_inferred") (t_list_to_mat [ u_0 ]));
      Arr.(save_npy ~out:(in_dir "u_inferred") (t_list_to_mat u_list_t));
      Arr.(save_npy ~out:(in_dir "y_inferred") (t_list_to_mat y_list_t));
      Arr.(save_npy ~out:(in_dir "y_auto") (t_list_to_mat y_list_auto_t));
      Arr.(save_npy ~out:(in_dir "o_gen") (t_list_to_mat o_list_t));
      (* save params *)
      O.P.C.save theta ~kind:base.ba_kind ~out:(in_dir "sofo_params");
      print [%message (t : int) (loss_avg : float)];
      Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss_avg |] 1 2)));
    []
  in
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state (loss :: running_avg) *)

module O = Optimizer.Adam (MGU.P)

let config ~t =
  Optimizer.Config.Adam.
    { base
    ; beta_1 = 0.9
    ; beta_2 = 0.99
    ; eps = 1e-4
    ; learning_rate = Some Float.(0.01 / (1. + sqrt (of_int t / 1.)))
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
  let data_unnoised, data = sample_data () in
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
      let u_0, u_list, y_list, o_list =
        MGU.simulate ~theta:(MGU.P.map theta ~f:Maths.any) ~data
      in
      let y_list_auto = MGU.simulate_auto ~theta:(MGU.P.map theta ~f:Maths.any) in
      let u_0 = Maths.to_tensor u_0
      and u_list_t = List.map u_list ~f:Maths.to_tensor
      and y_list_t = List.map y_list ~f:Maths.to_tensor
      and o_list_t = List.map o_list ~f:Maths.to_tensor
      and y_list_auto_t = List.map y_list_auto ~f:Maths.to_tensor in
      Arr.(save_npy ~out:(in_dir "o") (t_list_to_mat data));
      Arr.(save_npy ~out:(in_dir "y") (t_list_to_mat data_unnoised));
      Arr.(save_npy ~out:(in_dir "u0_inferred") (t_list_to_mat [ u_0 ]));
      Arr.(save_npy ~out:(in_dir "u_inferred") (t_list_to_mat u_list_t));
      Arr.(save_npy ~out:(in_dir "y_inferred") (t_list_to_mat y_list_t));
      Arr.(save_npy ~out:(in_dir "y_auto") (t_list_to_mat y_list_auto_t));
      Arr.(save_npy ~out:(in_dir "o_gen") (t_list_to_mat o_list_t));
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
  Sofo.print [%message ("params loaded")];
  (* simulate trajectory *)
  let y_list_auto = MGU.simulate_auto ~theta:(MGU.P.map theta ~f:Maths.any) in
  let y_list_auto_t = List.map y_list_auto ~f:Maths.to_tensor in
  Arr.(save_npy ~out:(in_dir "y_auto") (t_list_to_mat y_list_auto_t))    *)
