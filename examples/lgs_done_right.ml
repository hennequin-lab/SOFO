(* Linear Gaussian Dynamics, with same state/control/cost parameters constant across trials and across time *)

open Base
open Forward_torch
open Torch
open Sofo
module Mat = Owl.Dense.Matrix.S
module Linalg = Owl.Linalg.S
module O = Opt.S

let primal_detach (x, _) = Maths.const Tensor.(detach x)

let _ =
  Random.init 1985;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f32)
    ; ba_kind = Bigarray.float32
    }

let m = 5
let n = 10
let o = 40
let tmax = 40
let bs = 64
let _K = 64
let eye_m = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye m)))
let eye_n = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye n)))
let ones_1 = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ 1 ]))
let ones_n = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ n ]))
let ones_o = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
let ones_u = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ m ]))
let ones_u' = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ n ]))
let b0 = Maths.(f 1. * eye_n)

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
      let sa = Linalg.eigvals w |> Owl.Dense.Matrix.C.re |> Mat.max' in
      Mat.(Float.(target_sa / sa) $* w)
    in
    Linalg.expm Mat.((w - eye n) *$ 0.5)
  in
  let p = Linalg.discrete_lyapunov a Mat.(eye n) in
  let u, s, _ = Linalg.svd p in
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

module Generative_P = struct
  type 'a p =
    { q : 'a
    ; d : 'a
    ; b : 'a
    ; c : 'a
    ; log_obs_var : 'a
    }
  [@@deriving prms]
end

module Generative = Generative_P.Make (Prms.P)

let true_theta =
  let q, d = make_a_prms 0.8 in
  let b = make_b () in
  let c = make_c () in
  let log_obs_var = Maths.(log (f Float.(square 0.1))) in
  Generative_P.{ q; d; b; c; log_obs_var }

let theta =
  let q, d = make_a_prms 0.8 in
  let q = Prms.free (Maths.primal q) in
  let d = Prms.free (Maths.primal d) in
  let b = Prms.const (Maths.primal true_theta.b) in
  let c = Prms.free (Maths.primal (make_c ())) in
  let log_obs_var =
    Maths.(log (f Float.(square 1.) * ones_1))
    |> Maths.primal
    |> Prms.create ~above:(Tensor.f Float.(log (square 0.001)))
  in
  Generative_P.{ q; d; b; c; log_obs_var }

let n_params = Generative.M.numel true_theta
let _ = Convenience.print [%message (n_params : int)]
let std_of_log_var log_var = Maths.(exp (f 0.5 * log_var))
let precision_of_log_var log_var = Maths.(exp (neg log_var))
let sqrt_precision_of_log_var log_var = Maths.(exp (f (-0.5) * log_var))

let solver_chol ell y =
  let ell_t = Maths.transpose ~dim0:0 ~dim1:1 ell in
  let _x = Maths.linsolve_triangular ~left:false ~upper:true ell_t y in
  Maths.linsolve_triangular ~left:false ~upper:false ell _x

let solver a = solver_chol (Maths.cholesky a)

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
        let b = if t = 0 then b0 else b in
        Stdlib.Gc.major ();
        (* we only have state costs for t>=1 *)
        let _cz = if t > 0 then _cz_fun y else f 0. in
        let _Cuu = if t = 0 then eye_n else eye_m in
        let _Qzz = _Czz + einsum [ a, "ij"; _V, "jl"; a, "kl" ] "ik" in
        let _Quz = einsum [ b, "ij"; _V, "jl"; a, "kl" ] "ki" in
        let _Quu = _Cuu + einsum [ b, "ij"; _V, "jl"; b, "kl" ] "ik" in
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
  backward_info

let rollout ~a ~b ~c u =
  let open Maths in
  let bs = List.hd_exn (shape (List.hd_exn u)) in
  let z0 = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n ] |> const in
  let y_of z = einsum [ c, "ij"; z, "mi" ] "mj" in
  let y, _ =
    List.foldi u ~init:([], z0) ~f:(fun t (accu, z) u ->
      let b = if t = 0 then b0 else b in
      let z' = einsum [ a, "ij"; z, "mi" ] "mj" + einsum [ b, "ij"; u, "mi" ] "mj" in
      y_of z' :: accu, z')
  in
  List.rev y

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

let ustars ~a ~b backward_info =
  let open Maths in
  let z0 = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n ] |> const in
  let us, _ =
    List.foldi backward_info ~init:([], z0) ~f:(fun t (accu, z) (_k, _K, _Quu_chol) ->
      let b = if t = 0 then b0 else b in
      let u = _k + einsum [ _K, "ji"; z, "mj" ] "mi" in
      let z = einsum [ a, "ij"; z, "mi" ] "mj" + einsum [ b, "ij"; u, "mi" ] "mj" in
      u :: accu, z)
  in
  List.rev us

let sample_and_kl ~a ~b backward_info =
  let open Maths in
  let z0 = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n ] |> const in
  let us, kl, ggn, _ =
    List.foldi
      backward_info
      ~init:([], f 0., f 0., z0)
      ~f:(fun t (accu, kl, ggn, z) (_k, _K, _Quu_chol) ->
        let b, u_dim = if t = 0 then b0, n else b, m in
        let du =
          let eps = Tensor.randn ~device:base.device ~kind:base.kind [ bs; u_dim ] in
          linsolve_triangular ~left:false ~upper:false _Quu_chol (const eps)
        in
        let u = du + _k + einsum [ _K, "ji"; z, "mj" ] "mi" in
        let dkl, dggn =
          let prior_term = gaussian_llh ~inv_std:(if t = 0 then ones_u' else ones_u) u in
          let q_term = gaussian_llh_chol ~precision_chol:_Quu_chol du in
          let q_term_t = Maths.tangent q_term in
          let dggn =
            Option.map q_term_t ~f:(fun qt ->
              Maths.einsum [ const qt, "ki"; const qt, "li" ] "kl")
          in
          q_term - prior_term, dggn
        in
        let z = einsum [ a, "ij"; z, "mi" ] "mj" + einsum [ b, "ij"; u, "mi" ] "mj" in
        let ggn =
          match dggn with
          | Some dggn -> ggn + dggn
          | None -> ggn
        in
        u :: accu, kl + dkl, ggn, z)
  in
  let ggn = ggn / f Float.(of_int tmax) in
  kl, List.rev us, ggn

let filtering_kl ~a ~b ~c ~obs_precision ~(ustar : Maths.t list) y =
  let open Maths in
  let z0 = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n ] |> Maths.const in
  let btrinv = einsum [ b, "ij"; c, "jo"; obs_precision, "o" ] "io" in
  let btrinv' = einsum [ b0, "ij"; c, "jo"; obs_precision, "o" ] "io" in
  let precision_chol =
    eye_m + einsum [ btrinv, "io"; c, "jo"; b, "kj" ] "ik" |> cholesky
  in
  let precision_chol' =
    eye_n + einsum [ btrinv', "io"; c, "jo"; b0, "kj" ] "ik" |> cholesky
  in
  let _, _, kl, ggn, us =
    List.fold2_exn
      ustar
      y
      ~init:(0, z0, f 0., f 0., [])
      ~f:(fun (t, z, kl, ggn, us) ustar y ->
        Stdlib.Gc.major ();
        let b = if t = 0 then b0 else b in
        let precision_chol = if t = 0 then precision_chol' else precision_chol in
        let btrinv = if t = 0 then btrinv' else btrinv in
        let zpred = (z *@ a) + (ustar *@ b) in
        let ypred = zpred *@ c in
        let delta = y - ypred in
        let mu = solver_chol precision_chol (einsum [ btrinv, "io"; delta, "mo" ] "mi") in
        let u_diff =
          (* if precision = L L^T
             then cov = L^(-T) L^(-1)
             so we get a sample as epsilon L^(-1), i.e solving epsilon = X L *)
          Maths.linsolve_triangular
            ~left:false
            ~upper:false
            precision_chol
            (const (Tensor.randn_like (primal ustar)))
        in
        let u_sample = mu + u_diff in
        (* propagate that sample to update z *)
        let z = zpred + (u_sample *@ b) in
        (* update the KL divergence *)
        let kl, dggn =
          let prior_term =
            gaussian_llh ~inv_std:(if t = 0 then ones_u' else ones_u) (ustar + u_sample)
          in
          let q_term = gaussian_llh_chol ~precision_chol u_diff in
          let q_term_t = Maths.tangent q_term |> Option.value_exn |> const in
          let dggn = Maths.einsum [ q_term_t, "ki"; q_term_t, "li" ] "kl" in
          kl + q_term - prior_term, dggn
        in
        Int.(t + 1), z, kl, ggn + dggn, (ustar + u_sample) :: us)
  in
  let ggn = ggn / f Float.(of_int tmax) in
  kl, List.rev us, ggn

let sample_data (theta : Generative.M.t) =
  let open Maths in
  let sigma = std_of_log_var theta.log_obs_var * ones_o |> reshape ~shape:[ 1; -1 ] in
  let u =
    List.init tmax ~f:(fun t ->
      let u_dim = if t = 0 then n else m in
      const Tensor.(randn ~device:base.device ~kind:base.kind [ bs; u_dim ]))
  in
  let a = a_reparam (theta.q, theta.d) in
  let y =
    rollout ~a ~b:theta.b ~c:theta.c u
    |> List.map ~f:(fun y -> y + (const (Tensor.randn_like (primal y)) * sigma))
  in
  u, y

(* save the first element of the batch as a time series *)
let save_time_series ~out x =
  List.map (List.tl_exn x) ~f:(fun x ->
    Maths.primal x
    |> Tensor.to_bigarray ~kind:base.ba_kind
    |> Mat.get_slice [ [ 0 ] ]
    |> fun x -> Mat.reshape x [| 1; -1 |])
  |> List.to_array
  |> Mat.concatenate ~axis:0
  |> Mat.save_txt ~out

let save_summary_youjing ~out (theta : Generative.M.t) =
  let a = a_reparam (theta.q, theta.d) in
  let a = Maths.primal a |> Tensor.to_bigarray ~kind:base.ba_kind in
  let b = Maths.primal theta.b |> Tensor.to_bigarray ~kind:base.ba_kind in
  let c = Maths.primal theta.c |> Tensor.to_bigarray ~kind:base.ba_kind in
  let bbt = Mat.(transpose b *@ b) in
  let avg_spatial_cov =
    let _, q_accu =
      List.foldi
        (List.range 0 tmax)
        ~init:(Mat.(zeros n n), Mat.(zeros n n))
        ~f:(fun t (q_prev, q_accu) _ ->
          let delta = if t = 0 then Mat.eye n else bbt in
          let q_new = Mat.((transpose a *@ q_prev *@ a) + delta) in
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
  q |> (fun x -> Mat.reshape x [| -1; 1 |]) |> Mat.save_txt ~out

let _ = save_summary_youjing ~out:(in_dir "true_summary_y") true_theta

(* -----------------------------------------------
   --  SOME TESTS
   ----------------------------------------------- *)

let u, y = sample_data true_theta
let _ = save_time_series ~out:(in_dir "u") u
let _ = save_time_series ~out:(in_dir "y") y

let u_recov =
  let a = a_reparam (true_theta.q, true_theta.d) in
  let obs_precision = Maths.(precision_of_log_var true_theta.log_obs_var * ones_o) in
  lqr ~a ~b:true_theta.b ~c:true_theta.c ~obs_precision y |> ustars ~a ~b:true_theta.b

let y_recov =
  let a = a_reparam (true_theta.q, true_theta.d) in
  rollout ~a ~b:true_theta.b ~c:true_theta.c u_recov

let _ = save_time_series ~out:(in_dir "urecov") u_recov
let _ = save_time_series ~out:(in_dir "yrecov") y_recov

(*
   let _ =
   let u_sampled () =
   let p = true_theta in
   let obs_precision = Maths.(precision_of_log_var p.log_obs_var * ones_o) in
   let utilde, ytilde = sample_data p in
   let delta_y = List.map2_exn y ytilde ~f:Maths.( - ) in
   let delta_u = lqr ~a:p.a ~b:p.b ~c:p.c ~obs_precision delta_y in
   List.map2_exn utilde delta_u ~f:(fun u du -> Maths.(u + du))
   in
   let n_samples = 1000 in
   let mu = List.hd_exn u_recov in
   Array.init n_samples ~f:(fun _ ->
   let u = u_sampled () |> List.hd_exn in
   let u = Maths.(u - mu) |> Maths.primal |> Tensor.to_bigarray ~kind:base.ba_kind in
   Mat.get_slice [ [ 0 ] ] u)
   |> Mat.concatenate ~axis:0
   |> (fun x -> Mat.(transpose x *@ x /$ Float.(of_int Int.(pred n_samples))))
   |> Mat.save_txt ~out:(in_dir "post_cov")

   let _ =
   let bc = Maths.(true_theta.b *@ true_theta.c) in
   let obs_precision = Maths.(precision_of_log_var true_theta.log_obs_var * ones_o) in
   let posterior_precision =
   Maths.(eye_m + einsum [ bc, "ij"; obs_precision, "j"; bc, "kj" ] "ik")
   in
   posterior_precision
   |> Maths.primal
   |> Tensor.to_bigarray ~kind:base.ba_kind
   |> Owl.Linalg.D.inv
   |> Owl.Mat.save_txt ~out:(in_dir "post_cov_true")

   let _ = assert false
*)

let elbo ~data:(y : Maths.t list) (theta : Generative.M.t) =
  let inv_sigma_o_expanded =
    Maths.(sqrt_precision_of_log_var theta.log_obs_var * ones_o)
  in
  let a = a_reparam (theta.q, theta.d) in
  let kl, u_sampled, ggn =
    let obs_precision = Maths.(precision_of_log_var theta.log_obs_var * ones_o) in
    let backward_info = lqr ~a ~b:theta.b ~c:theta.c ~obs_precision y in
    sample_and_kl ~a ~b:theta.b backward_info
  in
  let y_pred = rollout ~a ~b:theta.b ~c:theta.c u_sampled in
  let lik_term =
    List.fold2_exn
      y
      y_pred
      ~init:Maths.(f 0.)
      ~f:(fun accu y y_pred ->
        Stdlib.Gc.major ();
        Maths.(accu + gaussian_llh ~mu:y_pred ~inv_std:inv_sigma_o_expanded y))
  in
  let neg_elbo =
    Maths.(lik_term - kl)
    |> Maths.neg
    |> fun x -> Maths.(x / f Float.(of_int o * of_int tmax))
  in
  neg_elbo, Maths.primal ggn

let elbo_hack ~data:(y : Maths.t list) (theta : Generative.M.t) =
  let inv_sigma_o_expanded =
    Maths.(sqrt_precision_of_log_var theta.log_obs_var * ones_o)
  in
  let a = a_reparam (theta.q, theta.d) in
  let obs_precision = Maths.(precision_of_log_var theta.log_obs_var * ones_o) in
  let ustar = lqr ~a ~b:theta.b ~c:theta.c ~obs_precision y |> ustars ~a ~b:theta.b in
  let kl, u_sampled, ggn =
    filtering_kl ~a ~b:theta.b ~c:theta.c ~obs_precision ~ustar y
  in
  let y_pred = rollout ~a ~b:theta.b ~c:theta.c u_sampled in
  let lik_term =
    List.fold2_exn
      y
      y_pred
      ~init:Maths.(f 0.)
      ~f:(fun accu y y_pred ->
        Stdlib.Gc.major ();
        Maths.(accu + gaussian_llh ~mu:y_pred ~inv_std:inv_sigma_o_expanded y))
  in
  let neg_elbo =
    Maths.(lik_term - kl)
    |> Maths.neg
    |> fun x -> Maths.(x / f Float.(of_int o * of_int tmax))
  in
  neg_elbo, Maths.primal ggn

module M = struct
  module P = Generative

  type data = Maths.t list
  type args = unit

  let f ~update ~data:y ~init ~args:() (theta : P.M.t) =
    let neg_elbo, ggn = elbo ~data:y theta in
    let s = Linalg.svdvals (Tensor.to_bigarray ~kind:base.ba_kind ggn) in
    Mat.(save_txt ~out:(in_dir "svals") (transpose s));
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u -> u init (Some (neg_elbo, Some ggn))
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
  val config : iter:int -> (float, Bigarray.float32_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state running_avg =
      Stdlib.Gc.major ();
      let _, y = sample_data true_theta in
      let loss, new_state = O.step ~config:(config ~iter) ~state ~data:y () in
      let running_avg =
        let loss_avg =
          match running_avg with
          | [] -> loss
          | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
        in
        (* save params *)
        if iter % 10 = 0
        then (
          (* avg error *)
          Convenience.print [%message (iter : int) (loss_avg : float)];
          let t = iter in
          let theta = O.params new_state |> O.W.P.value |> O.W.P.map ~f:Maths.const in
          let ground_truth_elbo =
            elbo ~data:y true_theta
            |> fst
            |> Maths.primal
            |> Tensor.mean
            |> Tensor.to_float0_exn
          in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array [| Float.of_int t; loss_avg; ground_truth_elbo |] 1 3));
          save_summary_youjing ~out:(in_dir "summary_y") theta);
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

  let config ~iter:k =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate =
          (let lr = Cmdargs.(get_float "-lr" |> default 0.1) in
           Some Float.(lr / sqrt (1. + (1. * (of_int k / 10.)))))
      ; n_tangents = _K
      ; sqrt = false
      ; rank_one = false
      ; damping = Some 1e-4
      ; momentum = None
      }

  let init = O.init theta
end

(* --------------------------------
   -- Adam
   -------------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (M)

  let config ~iter:k =
    Optimizer.Config.Adam.
      { default with
        base
      ; learning_rate =
          (let lr = Cmdargs.(get_float "-lr" |> default 0.02) in
           Some Float.(lr / sqrt (1. + (0. * (of_int k / 100.)))))
      }

  let init = O.init theta
end

let _ =
  let max_iter = 100000 in
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
