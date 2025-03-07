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
let o = 40
let tmax = 10
let bs = 32
let eye_m = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye m)))
let eye_o = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye o)))
let eye_n = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye n)))
let eye_t = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye tmax)))
let ones_1 = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ 1 ]))
let ones_o = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
let ones_u = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ m ]))

(* let make_a () =
  let a =
    let w =
      let w = Mat.(gaussian n n) in
      let sa = Owl.Linalg.D.eigvals w |> Owl.Dense.Matrix.Z.re |> Mat.max' in
      Mat.(Float.(0.9 / sa) $* w)
    in
    Mat.(add_diag (0.9 $* w) 0.1)
  in
  Tensor.of_bigarray ~device:base.device a |> Maths.const  *)

let make_a () =
  let w =
    let tmp = Mat.gaussian n n in
    let r = tmp |> Owl.Linalg.D.eigvals |> Owl.Dense.Matrix.Z.re |> Mat.max' in
    Mat.(Float.(0.8 / r) $* tmp)
  in
  let w_i = Mat.((w - eye n) *$ 0.1) in
  Owl.Linalg.Generic.expm w_i |> Tensor.of_bigarray ~device:base.device |> Maths.const

(* let make_a () = 
  let q, r, _ = Owl.Linalg.D.qr Mat.(gaussian n n) in
  let q = Mat.(q * signum (diag r)) in
  let d = Mat.gaussian 1 n |> Mat.abs in
  let w = Mat.(transpose (sqrt d) * q * sqrt (reci (d +$ 1.))) in
  Tensor.of_bigarray ~device:base.device w |> Maths.const *)

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
    { a : 'a
    ; b : 'a
    ; c : 'a
    ; log_obs_var : 'a
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

let true_theta =
  let a = make_a () in
  let b = make_b () in
  let c = make_c () in
  let log_obs_var = Maths.(log (f Float.(square 0.1))) in
  PP.{ a; b; c; log_obs_var }

let theta =
  let a = Prms.free (Maths.primal (primal_detach true_theta.a)) in
  let b = Prms.const (Maths.primal (primal_detach true_theta.b)) in
  let c = Prms.free (Maths.primal (make_c ())) in
  let log_obs_var =
    Maths.(log (f Float.(square 0.1) * ones_1))
    |> Maths.primal
    |> Prms.create
         ~above:(Tensor.f Float.(log (square 0.001)))
         ~below:(Tensor.f Float.(log (square 10.)))
    (* |> Prms.pin *)
  in
  PP.{ a; b; c; log_obs_var }

let std_of_log_var log_var = Maths.(exp (f 0.5 * log_var))
let precision_of_log_var log_var = Maths.(exp (neg log_var))
let sqrt_precision_of_log_var log_var = Maths.(exp (f (-0.5) * log_var))

let save_summary ~out (theta : P.M.t) =
  let a = Maths.primal theta.a |> Tensor.to_bigarray ~kind:base.ba_kind in
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
  let gains, _, _, _ =
    List.fold_right
      (List.sub y ~pos:0 ~len:_T)
      ~init:([], _cz_fun yT, _Czz, Int.(_T - 1))
      ~f:(fun y (accu, _v, _V, t) ->
        (* we only have state costs for t>=1 *)
        let _cz = if t > 0 then _cz_fun y else f 0. in
        let _Czz = if t > 0 then _Czz else f 0. in
        let _Qzz = _Czz + einsum [ a, "ij"; _V, "jl"; a, "kl" ] "ik" in
        let _Quz = einsum [ b, "ij"; _V, "jl"; a, "kl" ] "ki" in
        (* or maybe ik *)
        let _Quu = einsum [ b, "ij"; _V, "jl"; b, "kl" ] "ik" + eye_m in
        let _qz = _cz + einsum [ a, "ij"; _v, "mj" ] "mi" in
        let _qu = einsum [ b, "ij"; _v, "mj" ] "mi" in
        let _K = neg (solver _Quu _Quz) in
        let _k = neg (solver _Quu _qu) in
        let _V = _Qzz + einsum [ _Quz, "ki"; _K, "ji" ] "kj" in
        let _v = _qz + einsum [ _qu, "mi"; _K, "ji" ] "mj" in
        (* important to symmetrize the value function otherwise
           it can drift and Cholesky will fail *)
        let _V = Maths.(f 0.5 * (_V + transpose ~dim0:0 ~dim1:1 _V)) in
        (_k, _K) :: accu, _v, _V, Int.(t - 1))
  in
  assert (List.length gains = _T);
  let _k0, _ = List.hd_exn gains in
  let u0 = _k0 in
  let z1 = einsum [ b, "ij"; u0, "mi" ] "mj" in
  let us, _ =
    List.fold
      (List.sub gains ~pos:1 ~len:Int.(_T - 1))
      ~init:([ u0 ], z1)
      ~f:(fun (accu, z) (_k, _K) ->
        let u = _k + einsum [ _K, "ji"; z, "mj" ] "mi" in
        let z = einsum [ a, "ij"; z, "mi" ] "mj" + einsum [ b, "ij"; u, "mi" ] "mj" in
        u :: accu, z)
  in
  List.rev us

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
  let y =
    rollout ~a:theta.a ~b:theta.b ~c:theta.c u
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

let gaussian_llh_chol ?mu ~chol x =
  let ell_t = Maths.transpose ~dim0:0 ~dim1:1 chol in
  let error_term =
    let error =
      match mu with
      | None -> x
      | Some mu -> Maths.(x - mu)
    in
    let error = Maths.linsolve_triangular ~left:false ~upper:true ell_t error in
    Maths.einsum [ error, "ma"; error, "ma" ] "m"
  in
  let cov_term =
    Maths.(sum (log (sqr (diagonal ~offset:0 chol))) |> reshape ~shape:[ 1 ])
  in
  let const_term =
    let o = x |> Maths.primal |> Tensor.shape |> List.last_exn in
    Float.(log (2. * pi) * of_int o)
  in
  Maths.(0.5 $* (const_term $+ error_term + cov_term)) |> Maths.neg

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
let _ = save_evals ~out:(in_dir "true_a_svals") true_theta.a

let u_recov =
  let obs_precision = Maths.(precision_of_log_var true_theta.log_obs_var * ones_o) in
  lqr ~a:true_theta.a ~b:true_theta.b ~c:true_theta.c ~obs_precision y

let y_recov = rollout ~a:true_theta.a ~b:true_theta.b ~c:true_theta.c u_recov
let _ = save_time_series ~out:(in_dir "urecov") u_recov
let _ = save_time_series ~out:(in_dir "yrecov") y_recov

let _ =
  let u_sampled () =
    let p = true_theta in
    let obs_precision = Maths.(precision_of_log_var p.log_obs_var * ones_o) in
    let utilde, ytilde = sample_data p in
    let delta_y = List.map2_exn y ytilde ~f:Maths.( - ) in
    let delta_u = lqr ~a:p.a ~b:p.b ~c:p.c ~obs_precision delta_y in
    List.map2_exn utilde delta_u ~f:(fun u du -> Maths.(u + du))
  in
  let n_samples = 100 in
  Array.init n_samples ~f:(fun _ ->
    let u =
      u_sampled () |> List.hd_exn |> Maths.primal |> Tensor.to_bigarray ~kind:base.ba_kind
    in
    Mat.get_slice [ [ 0 ] ] u)
  |> Mat.concatenate ~axis:0
  |> (fun x -> Mat.(transpose x *@ x /$ Float.(of_int Int.(pred n_samples))))
  |> Mat.save_txt ~out:(in_dir "post_cov")

let inv_symm a =
  let eye_a =
    Tensor.(eye ~n:(List.last_exn (Maths.shape a)) ~options:(base.kind, base.device))
    |> Maths.const
  in
  let _a_chol = Maths.cholesky a in
  let tmp = Maths.linsolve_triangular _a_chol eye_a ~left:true ~upper:false in
  Maths.(transpose ~dim0:0 ~dim1:1 tmp *@ tmp)

type km_fwd =
  { filtered_mean : Maths.t
  ; filtered_cov : Maths.t
  ; predictive_cov : Maths.t
  }

(* p(u_t | o_1,...,o_T) *)
let kl_u_post ~y ~u_sampled (theta : P.M.t) =
  let noise =
    Maths.(
      exp theta.log_obs_var
      * const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
    |> Maths.diag_embed ~offset:0 ~dim1:0 ~dim2:1
  in
  let _b_inv = theta.b |> Maths.inv_rectangle in
  (* kalman forward filter *)
  let _mu_0, _cov_0 =
    ( Maths.const (Tensor.zeros [ bs; n ] ~device:base.device ~kind:base.kind)
    , Maths.const (Tensor.zeros [ n; n ] ~device:base.device ~kind:base.kind) )
  in
  let _, km_fwd_list_rev =
    List.fold_left
      y
      ~init:((_mu_0, _cov_0), [])
      ~f:(fun accu y_true ->
        let (_mu_prev, _cov_prev), km_fwd_accu = accu in
        let _mu_p_curr = Maths.(_mu_prev *@ theta.a) in
        let _cov_p_curr =
          Maths.(
            einsum [ theta.a, "ab"; _cov_prev, "ac"; theta.a, "cd" ] "bd"
            + einsum [ theta.b, "ab"; theta.b, "ac" ] "bc")
        in
        let _v_curr = Maths.(y_true - (_mu_p_curr *@ theta.c)) in
        let _s_curr =
          Maths.(einsum [ theta.c, "ab"; _cov_p_curr, "ac"; theta.c, "cd" ] "bd" + noise)
        in
        (* CHECKED *)
        let _s_inv = inv_symm _s_curr in
        let _k_curr =
          Maths.(_s_inv *@ transpose theta.c ~dim0:1 ~dim1:0 *@ _cov_p_curr)
        in
        let _mu_curr = Maths.(_mu_p_curr + (_v_curr *@ _k_curr)) in
        let _cov_curr =
          let tmp = Maths.(eye_n - transpose ~dim0:0 ~dim1:1 (theta.c *@ _k_curr)) in
          Maths.(tmp *@ _cov_p_curr)
        in
        ( (_mu_curr, _cov_curr)
        , { filtered_mean = _mu_curr
          ; filtered_cov = _cov_curr
          ; predictive_cov = _cov_p_curr
          }
          :: km_fwd_accu ))
  in
  (* this list goes from 1 to T *)
  let km_fwd_list = List.rev km_fwd_list_rev in
  let _mu_smoothed_last, _cov_smoothed_last =
    let km_fwd_last = List.last_exn km_fwd_list in
    km_fwd_last.filtered_mean, km_fwd_last.filtered_cov
  in
  let (_mu_smoothed_one, _cov_smoothed_one), kl =
    List.fold
      List.(rev (range 0 (tmax - 1)))
      ~init:((_mu_smoothed_last, _cov_smoothed_last), Maths.f 0.)
      ~f:(fun ((_mu_smoothed_next, _cov_smoothed_next), accu) i ->
        let km_fwd_curr = List.nth_exn km_fwd_list i in
        let km_fwd_next = List.nth_exn km_fwd_list (i + 1) in
        let _h_curr =
          Maths.(
            km_fwd_curr.filtered_cov *@ theta.a *@ inv_symm km_fwd_next.predictive_cov)
        in
        let _mu_smoothed_curr =
          Maths.(
            km_fwd_curr.filtered_mean
            + ((_mu_smoothed_next - (km_fwd_curr.filtered_mean *@ theta.a))
               *@ transpose _h_curr ~dim0:1 ~dim1:0))
        in
        let _cov_smoothed_curr =
          Maths.(
            km_fwd_curr.filtered_cov
            + (_h_curr
               *@ (_cov_smoothed_next - km_fwd_next.predictive_cov)
               *@ transpose _h_curr ~dim0:1 ~dim1:0))
        in
        let _u_mu_smoothed_curr =
          Maths.((_mu_smoothed_next - (_mu_smoothed_curr *@ theta.a)) *@ _b_inv)
        in
        (* TODO: is it fine to ignore corss correlations?*)
        let _u_cov_smoothed_curr =
          Maths.(
            transpose ~dim0:1 ~dim1:0 _b_inv
            *@ (_cov_smoothed_next
                + (transpose ~dim0:1 ~dim1:0 theta.a *@ _cov_smoothed_curr *@ theta.a))
            *@ _b_inv)
        in
        (* shift by 1 since u_sampled goes from 0 to T-1 *)
        let u_sampled_curr = List.nth_exn u_sampled Int.(i + 1) in
        let prior_term =
          gaussian_llh
            ~inv_std:(Tensor.ones [ m ] ~device:base.device ~kind:base.kind |> Maths.const)
            u_sampled_curr
        in
        let neg_entropy_term =
          gaussian_llh_chol
            ~mu:_u_mu_smoothed_curr
            ~chol:(Maths.cholesky _u_cov_smoothed_curr)
            u_sampled_curr
        in
        ( (_mu_smoothed_curr, _cov_smoothed_curr)
        , Maths.(accu + neg_entropy_term - prior_term) ))
  in
  (* kl on u_0 *)
  let kl_first =
    let u_sampled_zero = List.hd_exn u_sampled in
    let _u_mu_smoothed_zero = Maths.(_mu_smoothed_one *@ _b_inv) in
    let _u_cov_smoothed_zero =
      Maths.(transpose ~dim0:1 ~dim1:0 _b_inv *@ _cov_smoothed_one *@ _b_inv)
    in
    let prior_term =
      gaussian_llh
        ~inv_std:(Tensor.ones [ m ] ~device:base.device ~kind:base.kind |> Maths.const)
        u_sampled_zero
    in
    let neg_entropy_term =
      gaussian_llh_chol
        ~mu:_u_mu_smoothed_zero
        ~chol:(Maths.cholesky _u_cov_smoothed_zero)
        u_sampled_zero
    in
    Maths.(neg_entropy_term - prior_term)
  in
  Maths.(kl + kl_first)

let neg_elbo_matheron ~data:(y : Maths.t list) (theta : P.M.t) =
  (* Matheron sampling *)
  let u_sampled =
    let p = P.map theta ~f:primal_detach in
    let obs_precision = Maths.(precision_of_log_var p.log_obs_var * ones_o) in
    let utilde, ytilde = sample_data p in
    let delta_y = List.map2_exn y ytilde ~f:Maths.( - ) in
    let delta_u = lqr ~a:p.a ~b:p.b ~c:p.c ~obs_precision delta_y in
    List.map2_exn utilde delta_u ~f:(fun u du -> Maths.(primal_detach (u + du)))
  in
  List.iter u_sampled ~f:(fun u -> assert (Poly.(Maths.tangent u = None)));
  let y_pred = rollout ~a:theta.a ~b:theta.b ~c:theta.c u_sampled in
  let lik_term =
    let inv_sigma_o_expanded =
      Maths.(sqrt_precision_of_log_var theta.log_obs_var * ones_o)
    in
    List.fold2_exn
      y
      y_pred
      ~init:Maths.(f 0.)
      ~f:(fun accu y y_pred ->
        Maths.(accu + gaussian_llh ~mu:y_pred ~inv_std:inv_sigma_o_expanded y))
  in
  let kl_term = kl_u_post ~u_sampled ~y (theta : P.M.t) in
  let neg_elbo =
    Maths.(lik_term - kl_term)
    |> Maths.neg
    |> fun x -> Maths.(x / f Float.(of_int o * of_int tmax))
  in
  neg_elbo, y_pred

let neg_mll ~data:(y : Maths.t list) (theta : P.M.t) =
  let noise =
    Maths.(
      exp theta.log_obs_var
      * const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
    |> Maths.diag_embed ~offset:0 ~dim1:0 ~dim2:1
  in
  let _mu_0, _cov_0 =
    ( Maths.const (Tensor.zeros [ bs; n ] ~device:base.device ~kind:base.kind)
    , Maths.const (Tensor.zeros [ n; n ] ~device:base.device ~kind:base.kind) )
  in
  let _, mll =
    List.fold_left
      y
      ~init:((_mu_0, _cov_0), Maths.f 0.)
      ~f:(fun accu y_true ->
        let (_mu_prev, _cov_prev), mll_accu = accu in
        let _mu_p_curr = Maths.(_mu_prev *@ theta.a) in
        let _cov_p_curr =
          Maths.(
            einsum [ theta.a, "ab"; _cov_prev, "ac"; theta.a, "cd" ] "bd"
            + einsum [ theta.b, "ab"; theta.b, "ac" ] "bc")
        in
        let _v_curr = Maths.(y_true - (_mu_p_curr *@ theta.c)) in
        let _s_curr =
          Maths.(einsum [ theta.c, "ab"; _cov_p_curr, "ac"; theta.c, "cd" ] "bd" + noise)
        in
        (* CHECKED *)
        let _s_inv =
          let _s_chol = Maths.cholesky _s_curr in
          let tmp = Maths.linsolve_triangular _s_chol eye_o ~left:true ~upper:false in
          Maths.(transpose ~dim0:0 ~dim1:1 tmp *@ tmp)
        in
        let _k_curr =
          Maths.(_s_inv *@ transpose theta.c ~dim0:1 ~dim1:0 *@ _cov_p_curr)
        in
        let _mu_curr = Maths.(_mu_p_curr + (_v_curr *@ _k_curr)) in
        let _cov_curr =
          let tmp = Maths.(eye_n - transpose ~dim0:0 ~dim1:1 (theta.c *@ _k_curr)) in
          Maths.(tmp *@ _cov_p_curr)
        in
        let marginal_mu = Maths.(_mu_p_curr *@ theta.c) in
        let mll_curr =
          gaussian_llh_chol ~mu:marginal_mu ~chol:(Maths.cholesky _s_curr) y_true
        in
        (_mu_curr, _cov_curr), Maths.(mll_accu + mll_curr))
  in
  mll |> Maths.neg |> fun x -> Maths.(x / f Float.(of_int o * of_int tmax))

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

  (* ggn matrix when loss is the negative marginal log likelihood *)
  let ggn_marginal_single ~marginal_cov_chol =
    let marginal_cov_chol_inv =
      Maths.linsolve_triangular marginal_cov_chol eye_o ~left:true ~upper:false
    in
    let marginal_cov_chol_inv_p = Maths.primal marginal_cov_chol_inv in
    let marginal_cov_inv_p =
      Tensor.einsum
        [ marginal_cov_chol_inv_p; marginal_cov_chol_inv_p ]
        ~path:None
        ~equation:"ij,ik->jk"
    in
    let marginal_cov_t =
      let marginal_cov =
        Maths.(marginal_cov_chol *@ transpose marginal_cov_chol ~dim0:1 ~dim1:0)
      in
      Maths.tangent marginal_cov |> Option.value_exn
    in
    let ggn_half =
      Tensor.einsum
        [ marginal_cov_t; marginal_cov_inv_p ]
        ~path:None
        ~equation:"koq,qp->kop"
    in
    Tensor.(einsum [ ggn_half; ggn_half ] ~path:None ~equation:"kop,jpo->kj")

  (* ggn matrix when loss is the negative marginal log likelihood *)
  let ggn_marginal (theta : P.M.t) =
    let noise =
      Maths.(
        exp theta.log_obs_var
        * const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
      |> Maths.diag_embed ~offset:0 ~dim1:0 ~dim2:1
    in
    let _mu_0, _cov_0 =
      ( Maths.const (Tensor.zeros [ bs; n ] ~device:base.device ~kind:base.kind)
      , Maths.const (Tensor.zeros [ n; n ] ~device:base.device ~kind:base.kind) )
    in
    let _, ggn_summed =
      List.fold_left
        y
        ~init:((_mu_0, _cov_0), Tensor.f 0.)
        ~f:(fun accu y_true ->
          let (_mu_prev, _cov_prev), ggn_accu = accu in
          let _mu_p_curr = Maths.(_mu_prev *@ theta.a) in
          let _cov_p_curr =
            Maths.(
              einsum [ theta.a, "ab"; _cov_prev, "ac"; theta.a, "cd" ] "bd"
              + einsum [ theta.b, "ab"; theta.b, "ac" ] "bc")
          in
          let _v_curr = Maths.(y_true - (_mu_p_curr *@ theta.c)) in
          let _s_curr =
            Maths.(
              einsum [ theta.c, "ab"; _cov_p_curr, "ac"; theta.c, "cd" ] "bd" + noise)
          in
          let _s_inv =
            let _s_chol = Maths.cholesky _s_curr in
            let tmp = Maths.linsolve_triangular _s_chol eye_o ~left:true ~upper:false in
            Maths.(transpose ~dim0:0 ~dim1:1 tmp *@ tmp)
          in
          let _k_curr =
            Maths.(_s_inv *@ transpose theta.c ~dim0:1 ~dim1:0 *@ _cov_p_curr)
          in
          let _mu_curr = Maths.(_mu_p_curr + (_v_curr *@ _k_curr)) in
          let _cov_curr =
            let tmp = Maths.(eye_n - transpose ~dim0:0 ~dim1:1 (theta.c *@ _k_curr)) in
            Maths.(tmp *@ _cov_p_curr)
          in
          let cov_chol = Maths.cholesky _s_curr in
          ( (_mu_curr, _cov_curr)
          , Tensor.(ggn_accu + ggn_marginal_single ~marginal_cov_chol:cov_chol) ))
    in
    Tensor.(f Float.(of_int bs / (2. * of_int o * of_int tmax)) * ggn_summed)

  let f_elbo ~update ~data:y ~init ~args:() (theta : P.M.t) =
    let neg_elbo, y_pred = neg_elbo_matheron ~data:y theta in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let ggn = ggn ~y_pred theta in
      let _ =
        let _, s, _ = Owl.Linalg.D.svd (Tensor.to_bigarray ~kind:base.ba_kind ggn) in
        Mat.(save_txt ~out:(in_dir "svals") (transpose s))
      in
      u init (Some (neg_elbo, Some ggn))

  let f_marginal ~update ~data:y ~init ~args:() (theta : P.M.t) =
    let neg_mll = neg_mll ~data:y theta in
    match update with
    | `loss_only u -> u init (Some neg_mll)
    | `loss_and_ggn u ->
      let ggn = ggn_marginal theta in
      let _ =
        let _, s, _ = Owl.Linalg.D.svd (Tensor.to_bigarray ~kind:base.ba_kind ggn) in
        Mat.(save_txt ~out:(in_dir "svals") (transpose s))
      in
      u init (Some (neg_mll, Some ggn))

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
          save_evals ~out:(in_dir "a_svals") (extract theta_curr.a);
          (* avg error *)
          Convenience.print [%message (iter : int) (loss_avg : float)];
          let t = iter in
          let std_o_curr =
            theta_curr.log_obs_var |> Prms.value |> Tensor.exp |> Tensor.to_float0_exn
          in
          let ground_truth_loss =
            let ground_truth_elbo, _ = neg_elbo_matheron ~data:y true_theta in
            Maths.primal ground_truth_elbo |> Tensor.mean |> Tensor.to_float0_exn
            (* let ground_truth_mll = neg_mll ~data:y true_theta in
            Maths.primal ground_truth_mll |> Tensor.mean |> Tensor.to_float0_exn *)
          in
          let ground_truth_std_o_loss =
            let theta_std_o =
              PP.
                { a = extract theta_curr.a
                ; b = extract theta_curr.b
                ; c = extract theta_curr.c
                ; log_obs_var = true_theta.log_obs_var
                }
            in
            let ground_truth_std_o_elbo, _ = neg_elbo_matheron ~data:y theta_std_o in
            Maths.primal ground_truth_std_o_elbo |> Tensor.mean |> Tensor.to_float0_exn
            (* let ground_truth_std_o_mll= neg_mll ~data:y theta_std_o in
            Maths.primal ground_truth_std_o_mll |> Tensor.mean |> Tensor.to_float0_exn *)
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
      ; learning_rate = Some 0.001
      ; n_tangents = 128
      ; sqrt = false
      ; rank_one = false
      ; damping = None
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
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.001 }

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
