(* Linear Gaussian Dynamics, with same state/control/cost parameters constant across trials and across time *)
open Base
open Forward_torch
open Torch
open Sofo
module Mat = Owl.Dense.Matrix.D

let primal_detach (x, _) = Maths.const Tensor.(detach x)

let _ =
  Random.init 1996;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f64)
    ; ba_kind = Bigarray.float64
    }

let m = 5
let n = 10
let o = 20
let tmax = 56
let bs = 32
let eye_m = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye m)))
let eye_o = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye o)))
let eye_n = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye n)))
let ones_1 = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ 1 ]))
let ones_o = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
let ones_u = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ m ]))

let q_of u =
  let open Maths in
  let ell = einsum [ u, "ik"; u, "jk" ] "ij" |> cholesky in
  linsolve_triangular ~left:true ~upper:false ell u

let a_reparam (q, d) =
  let q = q_of q in
  let open Maths in
  let d = exp d in
  let left_factor = sqrt d in
  let right_factor = f 1. / sqrt (f 1. + d) in
  einsum [ left_factor, "qi"; q, "ij"; right_factor, "qj" ] "ji"

let make_a_prms target_sa =
  let a =
    let w =
      let w = Mat.(gaussian n n) in
      let sa = Owl.Linalg.D.eigvals w |> Owl.Dense.Matrix.Z.re |> Mat.max' in
      Mat.(Float.(target_sa / sa) $* w)
    in
    Owl.Linalg.D.expm Mat.((w - eye n) *$ 0.1)
  in
  let p = Owl.Linalg.D.discrete_lyapunov a Mat.(eye n) in
  let u, s, _ = Owl.Linalg.D.svd p in
  let z = Mat.(transpose u *@ a *@ u) in
  let d12 = Mat.(sqrt (s - ones 1 n)) in
  let s12 = Mat.(sqrt s) in
  let q =
    Mat.(transpose (reci d12) * z * s12)
    |> Tensor.of_bigarray ~device:base.device
    |> Maths.const
  in
  let d = Mat.(log (sqr d12)) |> Tensor.of_bigarray ~device:base.device |> Maths.const in
  q, d

let make_b () =
  Tensor.(
    f Float.(1. /. sqrt (of_int m)) * randn ~device:base.device ~kind:base.kind [ m; n ])
  |> Maths.const

let make_c () =
  Tensor.(
    f Float.(1. /. sqrt (of_int n)) * randn ~device:base.device ~kind:base.kind [ n; o ])
  |> Maths.const

module PP = struct
  type 'a p =
    { q : 'a
    ; d : 'a
    ; b : 'a
    ; c : 'a
    ; log_obs_var : 'a (* ; scaling_factor : 'a *)
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

let true_theta =
  let q, d = make_a_prms 0.8 in
  let b = make_b () in
  let c = make_c () in
  let log_obs_var =
    Tensor.(of_float0 ~device:base.device Float.(square 0.01)) |> Maths.const |> Maths.log
  in
  let scaling_factor = Maths.const (Tensor.of_float0 ~device:base.device 1.) in
  PP.{ q; d; b; c; log_obs_var }

let theta =
  let q, d =
    let tmp_q, tmp_d = make_a_prms 0.8 in
    Prms.free (Maths.primal tmp_q), Prms.free (Maths.primal tmp_d)
  in
  let b = Prms.const (Maths.primal (primal_detach true_theta.b)) in
  let c = Prms.free (Maths.primal (make_c ())) in
  let log_obs_var =
    Maths.(log (f Float.(square 1.) * ones_1))
    |> Maths.primal
    |> Prms.create ~above:(Tensor.f Float.(log (square 0.001)))
    (* |> Prms.pin *)
  in
  let scaling_factor =
    Prms.create ~above:(Tensor.f 0.1) (Tensor.ones [ 1 ] ~device:base.device)
  in
  PP.{ q; d; b; c; log_obs_var }

let std_of_log_var log_var = Maths.(exp (f 0.5 * log_var))
let precision_of_log_var log_var = Maths.(exp (neg log_var))
let sqrt_precision_of_log_var log_var = Maths.(exp (f (-0.5) * log_var))

let save_summary ~out (theta : P.M.t) =
  let a =
    let a_tmp = a_reparam (theta.q, theta.d) in
    Maths.primal a_tmp |> Tensor.to_bigarray ~kind:base.ba_kind
  in
  let b = Maths.primal theta.b |> Tensor.to_bigarray ~kind:base.ba_kind in
  let c = Maths.primal theta.c |> Tensor.to_bigarray ~kind:base.ba_kind in
  let avg_spatial_cov =
    let q1 = Mat.(transpose b *@ b) in
    let _, q_accu =
      List.fold (List.range 0 tmax) ~init:(q1, q1) ~f:(fun accu _ ->
        let q_prev, q_accu = accu in
        let q_new = Mat.((transpose a *@ q_prev *@ a) + q1) in
        q_new, Mat.(q_new + q_accu))
    in
    Mat.(q_accu /$ Float.of_int tmax)
  in
  let q = Mat.(transpose c *@ avg_spatial_cov *@ c) in
  let noise_term =
    let obs_var = Maths.(exp theta.log_obs_var) |> Maths.primal |> Tensor.to_float0_exn in
    Mat.(obs_var $* eye o)
  in
  let q = Mat.(q + noise_term) in
  q |> (fun x -> Owl.Mat.reshape x [| -1; 1 |]) |> Owl.Mat.save_txt ~out

(* impulse response *)
(* let u_input = List.init tmax ~f:(fun i ->
    match i with 
    | 0 -> Maths.const (Tensor.ones ~device:base.device ~kind:base.kind [bs; m])
    | _ -> Maths.const (Tensor.zeros ~device:base.device ~kind:base.kind [bs; m])
    ) in 
 rollout ~a:theta.a ~b:theta.b ~c:theta.c u_input *)

let _ = save_summary ~out:(in_dir "true_summary") true_theta

let solver a y =
  let ell = Maths.cholesky a in
  let ell_t = Maths.transpose ~dim0:0 ~dim1:1 ell in
  let _x = Maths.linsolve_triangular ~left:false ~upper:true ell_t y in
  Maths.linsolve_triangular ~left:false ~upper:false ell _x

(* solves for xA = y, A = ell (ell)^T *)
let solver_chol ell y =
  let ell_t = Maths.transpose ~dim0:0 ~dim1:1 ell in
  let _x = Maths.linsolve_triangular ~left:false ~upper:true ell_t y in
  Maths.linsolve_triangular ~left:false ~upper:false ell _x

(* This has been TESTED and it works.
   Gradients also successfully tested against reverse-mode *)
let lqr ~a ~b ~c ~obs_precision y =
  let open Maths in
  (* augment observations with dummy y0 *)
  let _T = List.length y in
  let y = f 0. :: y in
  let _Czz = einsum [ c, "ij"; obs_precision, "j"; c, "kj" ] "ik" in
  let _cz_fun y = neg (einsum [ c, "ij"; obs_precision, "j"; y, "mj" ] "mi") in
  let yT = List.last_exn y in
  let backward_info, _, _, _ =
    List.fold_right
      (List.sub y ~pos:0 ~len:_T)
      ~init:([], _cz_fun yT, _Czz, Int.(_T - 1))
      ~f:(fun y (accu, _v, _V, t) ->
        Stdlib.Gc.major ();
        (* we only have state costs for t>=1 *)
        let _cz = if t > 0 then _cz_fun y else f 0. in
        let _Czz = if t > 0 then _Czz else f 0. in
        let _Qzz = _Czz + einsum [ a, "ij"; _V, "jl"; a, "kl" ] "ik" in
        let _Quz = einsum [ b, "ij"; _V, "jl"; a, "kl" ] "ki" in
        let _Quu = eye_m + einsum [ b, "ij"; _V, "jl"; b, "kl" ] "ik" in
        let _qz = _cz + einsum [ a, "ij"; _v, "mj" ] "mi" in
        let _qu = einsum [ b, "ij"; _v, "mj" ] "mi" in
        let _Quu_chol = cholesky _Quu in
        let _K = neg (solver_chol _Quu_chol _Quz) in
        let _k = neg (solver_chol _Quu_chol _qu) in
        let _V = _Qzz + einsum [ _Quz, "ki"; _K, "ji" ] "kj" in
        let _v = _qz + einsum [ _qu, "mi"; _K, "ji" ] "mj" in
        (* important to symmetrize the value function otherwise
             it can drift and Cholesky will fail *)
        let _V = Maths.(f 0.5 * (_V + transpose ~dim0:0 ~dim1:1 _V)) in
        (_k, _K, _Quu_chol) :: accu, _v, _V, Int.(t - 1))
  in
  assert (List.length backward_info = _T);
  let _k0, _, _ = List.hd_exn backward_info in
  let u0 = _k0 in
  let z1 = einsum [ b, "ij"; u0, "mi" ] "mj" in
  let us, _ =
    List.fold
      (List.sub backward_info ~pos:1 ~len:Int.(_T - 1))
      ~init:([ u0 ], z1)
      ~f:(fun (accu, z) (_k, _K, _) ->
        let u = _k + einsum [ _K, "ji"; z, "mj" ] "mi" in
        let z = einsum [ a, "ij"; z, "mi" ] "mj" + einsum [ b, "ij"; u, "mi" ] "mj" in
        u :: accu, z)
  in
  List.rev us, backward_info

let rollout ~a ~b ~c u =
  let open Maths in
  let u0 = List.hd_exn u in
  let y_of z = einsum [ c, "ij"; z, "mi" ] "mj" in
  let z1 = einsum [ b, "ij"; u0, "mi" ] "mj" in
  let y, _ =
    List.fold
      (List.tl_exn u)
      ~init:([ y_of z1 ], z1)
      ~f:(fun (accu, z) u ->
        let z' = einsum [ a, "ij"; z, "mi" ] "mj" + einsum [ b, "ij"; u, "mi" ] "mj" in
        y_of z' :: accu, z')
  in
  List.rev y

let sample_data (theta : P.M.t) =
  let sigma = std_of_log_var theta.log_obs_var in
  let u =
    List.init tmax ~f:(fun _ ->
      Tensor.(randn ~device:base.device ~kind:base.kind [ bs; m ]) |> Maths.const)
  in
  let a = a_reparam (theta.q, theta.d) in
  let y =
    rollout ~a ~b:theta.b ~c:theta.c u
    |> List.map ~f:(fun y ->
      Maths.(y + (sigma * const (Tensor.randn_like (Maths.primal y)))))
  in
  u, y

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

(* ell ell^T = precision *)
let gaussian_llh_chol ?mu ~precision_chol:ell x =
  let d = x |> Maths.primal |> Tensor.shape |> List.last_exn in
  let error =
    match mu with
    | None -> x
    | Some mu -> Maths.(x - mu)
  in
  let error_term = Maths.einsum [ error, "ma"; ell, "ai"; ell, "bi"; error, "mb" ] "m" in
  let cov_term =
    Maths.(neg (sum (log (sqr (diagonal ~offset:0 ell)))) |> reshape ~shape:[ 1 ])
  in
  let const_term = Float.(log (2. * pi) * of_int d) in
  Maths.(-0.5 $* (const_term $+ error_term + cov_term))

(* save the first element of the batch as a time series *)
let save_time_series ~out x =
  List.map x ~f:(fun x ->
    Maths.primal x
    |> Tensor.to_bigarray ~kind:base.ba_kind
    |> Mat.get_slice [ [ 5 ] ]
    |> fun x -> Mat.reshape x [| 1; -1 |])
  |> List.to_array
  |> Mat.concatenate ~axis:0
  |> Mat.save_txt ~out

let save_evals ~out a =
  let _, s, _ =
    Owl.Linalg.D.svd (Tensor.to_bigarray ~kind:base.ba_kind (Maths.primal a))
  in
  Mat.(save_txt ~out (transpose s))

(* -----------------------------------------------
     --  SOME TESTS
     ----------------------------------------------- *)

let u, y = sample_data true_theta
let _ = save_time_series ~out:(in_dir "u") u
let _ = save_time_series ~out:(in_dir "y") y

let u_recov, _ =
  let a = a_reparam (true_theta.q, true_theta.d) in
  let obs_precision = Maths.(precision_of_log_var true_theta.log_obs_var * ones_o) in
  lqr ~a ~b:true_theta.b ~c:true_theta.c ~obs_precision y

let y_recov =
  let a = a_reparam (true_theta.q, true_theta.d) in
  rollout ~a ~b:true_theta.b ~c:true_theta.c u_recov

let _ = save_time_series ~out:(in_dir "urecov") u_recov
let _ = save_time_series ~out:(in_dir "yrecov") y_recov

(* let sample_and_kl ~a ~b ~c ~obs_precision ~scaling_factor ustars ys =
  let open Maths in
  let scaling_factor = reshape scaling_factor ~shape:[ 1; -1 ] in
  let z0 = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n ] |> Maths.const in
  let btrinv = einsum [ b, "ij"; c, "jo"; obs_precision, "o" ] "io" in
  (* posterior precision of filtered covariance of u *)
  let precision_chol =
    (eye_m + einsum [ btrinv, "io"; c, "jo"; b, "kj" ] "ik" |> cholesky) * scaling_factor
  in
  let _, kl, us =
    List.fold2_exn
      ustars
      ys
      ~init:(z0, f 0., [])
      ~f:(fun (z, kl, us) ustar y ->
        Stdlib.Gc.major ();
        let zpred = (z *@ a) + (ustar *@ b) in
        let ypred = zpred *@ c in
        let delta = y - ypred in
        (* posterior mean of filtered u *)
        let mu =
          let tmp = einsum [ btrinv, "io"; delta, "mo" ] "mi" in
          solver_chol precision_chol tmp
        in
        (* sample from posterior filtered covariance of u. *)
        let u_diff_elbo =
          (* if precision = L L^T
             then cov = L^(-T) L^(-1)
             so we get a sample as epsilon L^(-1), i.e solving epsilon = X L *)
          Maths.linsolve_triangular
            ~left:false
            ~upper:false
            precision_chol
            (const (Tensor.randn ~device:base.device ~kind:base.kind [ bs; m ]))
        in
        let u_sample = mu + u_diff_elbo in
        (* propagate that sample to update z *)
        let z = zpred + (u_sample *@ b) in
        (* update the KL divergence *)
        let kl =
          let prior_term =
            let u_tmp = ustar + u_sample in
            gaussian_llh ~inv_std:ones_u u_tmp
          in
          let q_term = gaussian_llh_chol ~precision_chol:precision_chol u_diff_elbo in
          kl + q_term - prior_term
        in
        z, kl, (ustar + u_sample) :: us)
  in
  kl, List.rev us   *)

(* approximate kalman smoothed distribution of u *)
let sample_and_kl ~a ~b backward_info =
  let open Maths in
  let z0 = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n ] |> const in
  let dummy_u = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; m ] in
  let us, kl, _ =
    List.fold
      backward_info
      ~init:([], f 0., z0)
      ~f:(fun (accu, kl, z) (_k, _K, _Quu_chol) ->
        let du =
          linsolve_triangular
            ~left:false
            ~upper:false
            _Quu_chol
            (const (Tensor.randn_like dummy_u))
        in
        let u = du + _k + einsum [ _K, "ji"; z, "mj" ] "mi" in
        let dkl =
          let prior_term = gaussian_llh ~inv_std:ones_u u in
          let q_term = gaussian_llh_chol ~precision_chol:_Quu_chol du in
          q_term - prior_term
        in
        let z = einsum [ a, "ij"; z, "mi" ] "mj" + einsum [ b, "ij"; u, "mi" ] "mj" in
        u :: accu, kl + dkl, z)
  in
  kl, List.rev us

let elbo ~data:(y : Maths.t list) (theta : P.M.t) =
  let a = a_reparam (theta.q, theta.d) in
  let obs_precision = Maths.(precision_of_log_var theta.log_obs_var * ones_o) in
  let ustars, backward_info = lqr ~a ~b:theta.b ~c:theta.c ~obs_precision y in
  let kl, u_sampled =
    (* let scaling_factor = Maths.(theta.scaling_factor * ones_u) in
    sample_and_kl ~a ~b:theta.b ~c:theta.c ~obs_precision ~scaling_factor ustars y   *)
    sample_and_kl ~a ~b:theta.b backward_info
  in
  let y_pred = rollout ~a ~b:theta.b ~c:theta.c u_sampled in
  let lik_term =
    let inv_sigma_o_expanded =
      Maths.(sqrt_precision_of_log_var theta.log_obs_var * ones_o)
    in
    List.fold2_exn
      y
      y_pred
      ~init:Maths.(f 0.)
      ~f:(fun accu y y_pred ->
        Stdlib.Gc.major ();
        Maths.(accu + gaussian_llh ~mu:y_pred ~inv_std:inv_sigma_o_expanded y))
  in
  Maths.(neg (lik_term - kl) / f Float.(of_int tmax * of_int o)), y_pred

module M = struct
  module P = P

  type data = Maths.t list
  type args = unit

  let ggn ~y_pred (theta : P.M.t) =
    let obs_precision = precision_of_log_var theta.log_obs_var in
    let obs_precision_p = Maths.(const (primal obs_precision)) in
    let sigma2_t =
      Maths.(tangent (exp theta.log_obs_var)) |> Option.value_exn |> Maths.const
    in
    List.fold y_pred ~init:(Maths.f 0.) ~f:(fun accu y_pred ->
      let mu_t = Maths.tangent y_pred |> Option.value_exn |> Maths.const in
      let ggn_part1 =
        Maths.(einsum [ mu_t, "kmo"; mu_t, "lmo" ] "kl" * obs_precision_p)
      in
      (* CHECKED this agrees with mine *)
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

  let f_elbo ~update ~data:y ~init ~args:() (theta : P.M.t) =
    let neg_elbo, y_pred_ggn = elbo ~data:y theta in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let ggn = ggn ~y_pred:y_pred_ggn theta in
      let _ =
        let _, s, _ = Owl.Linalg.D.svd (Tensor.to_bigarray ~kind:base.ba_kind ggn) in
        Mat.(save_txt ~out:(in_dir "svals") (transpose s))
      in
      u init (Some (neg_elbo, Some ggn))

  let f = f_elbo
end

(* --------------------------------
   -- Generic type of optimiser
   -------------------------------- *)

module type Do_with_T = sig
  module O :
    Optimizer.T
    with type 'a W.P.p = 'a M.P.p
     and type W.data = Maths.t list
     and type W.args = unit

  val name : string
  val config : iter:int -> (float, Bigarray.float64_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let extract a = a |> Prms.value |> Maths.const

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state running_avg =
      Stdlib.Gc.major ();
      let _, y = sample_data true_theta in
      let loss, new_state = O.step ~config:(config ~iter) ~state ~data:y ~args:() in
      let running_avg =
        let loss_avg =
          match running_avg with
          | [] -> loss
          | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
        in
        (* save params *)
        if iter % 10 = 0
        then (
          (* current a svals *)
          let theta_curr = O.params state in
          (* avg error *)
          Convenience.print [%message (iter : int) (loss_avg : float)];
          let t = iter in
          let std_o_curr =
            theta_curr.log_obs_var |> Prms.value |> Tensor.exp |> Tensor.to_float0_exn
          in
          let ground_truth_loss =
            let ground_truth_elbo, _ = elbo ~data:y true_theta in
            Maths.primal ground_truth_elbo |> Tensor.mean |> Tensor.to_float0_exn
          in
          let ground_truth_std_o_loss =
            let theta_std_o =
              PP.
                { q = extract theta_curr.q
                ; d = extract theta_curr.q
                ; b = extract theta_curr.b
                ; c = extract theta_curr.c
                ; log_obs_var =
                    true_theta.log_obs_var
                    (* ; scaling_factor = true_theta.scaling_factor *)
                }
            in
            let ground_truth_std_o_elbo, _ = elbo ~data:y theta_std_o in
            Maths.primal ground_truth_std_o_elbo |> Tensor.mean |> Tensor.to_float0_exn
          in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array
                 [| Float.of_int t
                  ; loss_avg
                  ; ground_truth_loss
                  ; ground_truth_std_o_loss
                  ; std_o_curr
                 |]
                 1
                 5));
          save_summary
            ~out:(in_dir "summary")
            (O.params new_state |> O.W.P.value |> O.W.P.map ~f:Maths.const));
        []
      in
      if iter < max_iter then loop ~iter:(iter + 1) ~state:new_state (loss :: running_avg)
    in
    loop ~iter:0 ~state:init []
end

(* --------------------------------
   -- SOFO
   -------------------------------- *)

module Do_with_SOFO : Do_with_T = struct
  module O = Optimizer.SOFO (M)

  let name = "sofo"

  let config ~iter:_ =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 0.1
      ; n_tangents = 128
      ; sqrt = false
      ; rank_one = false
      ; damping = Some 1e-6
      ; momentum = None
      ; lm = false
      ; perturb_thresh = None
      }

  let init = O.init ~config:(config ~iter:0) theta
end

(* --------------------------------
   -- Adam
   --------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (M)

  let config ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.01 }

  let init = O.init theta
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
