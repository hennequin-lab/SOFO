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
let true_noise_std = 0.1
let tmax = 50
let bs = 32
let _K = 120
let eye_m = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye m)))
let eye_o = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye o)))
let eye_n = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye n)))
let ones_1 = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ 1 ]))
let ones_o = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
let ones_u = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ m ]))
let ones_x = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ n ]))

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
    (* sometimes we set to 0.1. 1 is coarser *)
    Owl.Linalg.D.expm Mat.((w - eye n) *$ 0.5)
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
    ; b_0 : 'a (* b at time step 0 has same dimension as state *)
    ; c : 'a
    ; log_obs_var : 'a
    ; scaling_factor : 'a
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

let true_theta =
  let q, d = make_a_prms 0.8 in
  let b = make_b () in
  let c = make_c () in
  let log_obs_var =
    Tensor.(of_float0 ~device:base.device Float.(square true_noise_std))
    |> Maths.const
    |> Maths.log
  in
  let scaling_factor = Maths.const (Tensor.of_float0 ~device:base.device 1.) in
  PP.{ q; d; b; b_0 = eye_n; c; log_obs_var; scaling_factor }

let theta =
  let q, d =
    let tmp_q, tmp_d = make_a_prms 0.8 in
    Prms.free (Maths.primal tmp_q), Prms.free (Maths.primal tmp_d)
  in
  let b = Prms.const (Maths.primal (primal_detach true_theta.b)) in
  let b_0 = Prms.const (Maths.primal (primal_detach true_theta.b_0)) in
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
  PP.{ q; d; b; b_0; c; log_obs_var; scaling_factor }

let std_of_log_var log_var = Maths.(exp (f 0.5 * log_var))
let precision_of_log_var log_var = Maths.(exp (neg log_var))
let sqrt_precision_of_log_var log_var = Maths.(exp (f (-0.5) * log_var))
let detach x = x |> Maths.primal_tensor_detach |> Maths.const

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
let lqr ~a ~b_0 ~b ~c ~obs_precision o_list =
  let open Maths in
  (* augment observations with dummy y0 *)
  let _T = List.length o_list in
  let o_list = f 0. :: o_list in
  let _Czz = einsum [ c, "ij"; obs_precision, "j"; c, "kj" ] "ik" in
  let _cz_fun o = neg (einsum [ c, "ij"; obs_precision, "j"; o, "mj" ] "mi") in
  let o_T = List.last_exn o_list in
  let backward_info, _, _, _ =
    List.fold_right2_exn
      (List.range 0 _T)
      (List.sub o_list ~pos:0 ~len:_T)
      ~init:([], _cz_fun o_T, _Czz, Int.(_T - 1))
      ~f:(fun i o (accu, _v, _V, t) ->
        Stdlib.Gc.major ();
        let b = if i = 0 then b_0 else b in
        (* we only have state costs for t>=1 *)
        let _cz = if t > 0 then _cz_fun o else f 0. in
        let _Czz = if t > 0 then _Czz else f 0. in
        let _Qzz = _Czz + einsum [ a, "ij"; _V, "jl"; a, "kl" ] "ik" in
        let _Quz = einsum [ b, "ij"; _V, "jl"; a, "kl" ] "ki" in
        let _Quu =
          (if i = 0 then eye_n else eye_m) + einsum [ b, "ij"; _V, "jl"; b, "kl" ] "ik"
        in
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
  let z1 = einsum [ b_0, "ij"; u0, "mi" ] "mj" in
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

let rollout ~a ~b ~b_0 ~c u =
  let open Maths in
  let u0 = List.hd_exn u in
  let y_of z = einsum [ c, "ij"; z, "mi" ] "mj" in
  let z1 = einsum [ b_0, "ij"; u0, "mi" ] "mj" in
  let y, _ =
    List.fold
      (List.tl_exn u)
      ~init:([ y_of z1 ], z1)
      ~f:(fun (accu, z) u ->
        let z' = einsum [ a, "ij"; z, "mi" ] "mj" + einsum [ b, "ij"; u, "mi" ] "mj" in
        y_of z' :: accu, z')
  in
  List.rev y

(* u goes from 1 to T-1, o goes from 1 to T *)
let sample_data (theta : P.M.t) =
  let x1 = Tensor.randn ~device:base.device ~kind:base.kind [ bs; n ] in
  let sigma = std_of_log_var theta.log_obs_var in
  let u =
    List.init (tmax - 1) ~f:(fun _ ->
      Tensor.(randn ~device:base.device ~kind:base.kind [ bs; m ]) |> Maths.const)
  in
  let a = a_reparam (theta.q, theta.d) in
  let tmp_einsum a b = Maths.einsum [ a, "ma"; b, "ab" ] "mb" in
  let y_list =
    let _, y_list_rev =
      List.fold
        u
        ~init:(Maths.const x1, [ tmp_einsum (Maths.const x1) theta.c ])
        ~f:(fun (x, y_list) u ->
          let new_x = Maths.(tmp_einsum x a + tmp_einsum u theta.b) in
          let new_y = tmp_einsum new_x theta.c in
          new_x, new_y :: y_list)
    in
    List.rev y_list_rev
  in
  let o =
    List.map y_list ~f:(fun y ->
      Maths.(y + (sigma * const (Tensor.randn_like (Maths.primal y)))))
  in
  u, o

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

let u, o_list = sample_data true_theta
let _ = save_time_series ~out:(in_dir "o") o_list

let u_recov, _ =
  let a = a_reparam (true_theta.q, true_theta.d) in
  let obs_precision = Maths.(precision_of_log_var true_theta.log_obs_var * ones_o) in
  lqr ~a ~b:true_theta.b ~b_0:true_theta.b_0 ~c:true_theta.c ~obs_precision o_list

let o_recov =
  let a = a_reparam (true_theta.q, true_theta.d) in
  rollout ~a ~b:true_theta.b ~b_0:true_theta.b_0 ~c:true_theta.c u_recov

let _ = save_time_series ~out:(in_dir "orecov") o_recov

(* approximate filtering distribution of u *)
let sample_and_kl ~a ~b ~b_0 ~c ~obs_precision ~scaling_factor ustars o_list =
  let open Maths in
  let scaling_factor = reshape scaling_factor ~shape:[ 1; -1 ] in
  let z0 = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n ] |> Maths.const in
  let btrinv = einsum [ b, "ij"; c, "jo"; obs_precision, "o" ] "io" in
  let btrinv_0 = einsum [ b_0, "ij"; c, "jo"; obs_precision, "o" ] "io" in
  (* posterior precision of filtered covariance of u *)
  let precision_chol_0 =
    (eye_n + einsum [ btrinv_0, "io"; c, "jo"; b_0, "kj" ] "ik" |> cholesky)
    * scaling_factor
  in
  let precision_chol =
    (eye_m + einsum [ btrinv, "io"; c, "jo"; b, "kj" ] "ik" |> cholesky) * scaling_factor
  in
  let _, kl_list, us =
    let u_star_y_list = List.map2_exn ustars o_list ~f:(fun u o -> u, o) in
    List.fold2_exn
      (List.range 0 tmax)
      u_star_y_list
      ~init:(z0, [], [])
      ~f:(fun (z, kl_list, us) i (ustar, ostar) ->
        Stdlib.Gc.major ();
        let b = if i = 0 then b_0 else b in
        let precision_chol = if i = 0 then precision_chol_0 else precision_chol in
        let btrinv = if i = 0 then btrinv_0 else btrinv in
        let zpred = (z *@ a) + (ustar *@ b) in
        let ypred = zpred *@ c in
        let delta = ostar - ypred in
        (* posterior mean of filtered u *)
        let mu =
          let tmp = einsum [ btrinv, "io"; delta, "mo" ] "mi" in
          solver_chol precision_chol tmp
        in
        (* sample from posterior filtered covariance of u. *)
        let u_diff_elbo =
          Maths.linsolve_triangular
            ~left:false
            ~upper:false
            precision_chol
            (const
               (Tensor.randn
                  ~device:base.device
                  ~kind:base.kind
                  [ bs; (if i = 0 then n else m) ]))
        in
        let u_sample = mu + u_diff_elbo in
        (* propagate that sample to update z *)
        let z = zpred + (u_sample *@ b) in
        (* update the KL divergence *)
        let kl =
          let prior_term =
            let u_tmp = ustar + u_sample in
            gaussian_llh ~inv_std:(if i = 0 then ones_x else ones_u) u_tmp
          in
          (* sticking the landing idea where gradients w.r.t variational parameters removed. *)
          let q_term = gaussian_llh_chol ~mu:(detach mu) ~precision_chol:(detach precision_chol) u_sample in
          q_term - prior_term
        in
        z, kl :: kl_list, (ustar + u_sample) :: us)
  in
  List.rev kl_list, List.rev us

let elbo ~data:(o_list : Maths.t list) (theta : P.M.t) =
  let a = a_reparam (theta.q, theta.d) in
  let obs_precision = Maths.(precision_of_log_var theta.log_obs_var * ones_o) in
  let ustars, _ = lqr ~a ~b:theta.b ~b_0:theta.b_0 ~c:theta.c ~obs_precision o_list in
  let kl_list, u_sampled =
    let scaling_factor = theta.scaling_factor in
    sample_and_kl
      ~a
      ~b:theta.b
      ~b_0:theta.b_0
      ~c:theta.c
      ~obs_precision
      ~scaling_factor
      ustars
      o_list
  in
  let y_pred = rollout ~a ~b:theta.b ~b_0:theta.b_0 ~c:theta.c u_sampled in
  let lik_list =
    let inv_sigma_o_expanded =
      Maths.(sqrt_precision_of_log_var theta.log_obs_var * ones_o)
    in
    List.map2_exn o_list y_pred ~f:(fun o y_pred ->
      Maths.(gaussian_llh ~mu:y_pred ~inv_std:inv_sigma_o_expanded o))
  in
  let elbo_list =
    List.map2_exn lik_list kl_list ~f:(fun lik kl ->
      Maths.(neg (lik - kl) / f Float.(of_int tmax * of_int o)))
  in
  elbo_list, y_pred

module M = struct
  module P = P

  type data = Maths.t list
  type args = unit

  let empirical_ggn ~loss =
    List.fold loss ~init:(Maths.f 0.) ~f:(fun accu loss ->
      let loss_t = Maths.(tangent loss) |> Option.value_exn |> Maths.const in
      Maths.(accu + einsum [ loss_t, "km"; loss_t, "lm" ] "kl"))
    |> Maths.primal

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
    let neg_elbo_list, y_pred_ggn = elbo ~data:y theta in
    let neg_elbo =
      List.fold neg_elbo_list ~init:(Maths.f 0.) ~f:(fun accu elbo -> Maths.(accu + elbo))
    in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      (* let ggn = ggn ~y_pred:y_pred_ggn theta in *)
      let ggn = empirical_ggn ~loss:neg_elbo_list in
      let _ =
        let _, s, _ = Owl.Linalg.D.svd (Tensor.to_bigarray ~kind:base.ba_kind ggn) in
        Mat.(save_txt ~out:(in_dir "svals") (transpose s))
      in
      u init (Some (neg_elbo, Some ggn))

  let f = f_elbo
end

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
type param_name =
  | Q
  | D
  | C
  | Log_obs_var
  | Scaling_factor
[@@deriving compare]

let param_names_list = [ Q; D; C; Log_obs_var; Scaling_factor ]
let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
let n_params_q = 50
let n_params_d = 10
let n_params_c = Int.(_K - 2 - n_params_d - n_params_q)
let n_params_log_obs_var = 1
let n_params_scaling_factor = 1
let cycle = true

let n_params_list =
  [ n_params_q; n_params_d; n_params_c; n_params_log_obs_var; n_params_scaling_factor ]

module GGN : Wrapper.Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { q_left : 'a
      ; q_right : 'a
      ; d_left : 'a
      ; d_right : 'a
      ; c_left : 'a
      ; c_right : 'a
      ; log_obs_var_left : 'a
      ; log_obs_var_right : 'a
      ; scaling_factor_left : 'a
      ; scaling_factor_right : 'a
      }
    [@@deriving prms]
  end

  module P = P
  module A = Make (Prms.P)

  type sampling_state = int

  let init_sampling_state () = 0

  let zero_params ~shape _K =
    Tensor.zeros ~device:base.device ~kind:base.kind (_K :: shape)

  let random_params ~shape _K =
    Tensor.randn ~device:base.device ~kind:base.kind (_K :: shape)

  let get_shapes (param_name : param_name) =
    match param_name with
    | Q -> [ n; n ]
    | D -> [ 1; n ]
    | C -> [ n; o ]
    | Log_obs_var -> [ 1 ]
    | Scaling_factor -> [ 1 ]

  let get_n_params (param_name : param_name) =
    match param_name with
    | Q -> n_params_q
    | D -> n_params_d
    | C -> n_params_c
    | Log_obs_var -> n_params_log_obs_var
    | Scaling_factor -> n_params_scaling_factor

  let get_total_n_params (param_name : param_name) =
    let list_prod l = List.fold l ~init:1 ~f:(fun accu i -> accu * i) in
    list_prod (get_shapes param_name)

  let get_n_params_before_after (param_name : param_name) =
    let n_params_prefix_suffix_sums = Convenience.prefix_suffix_sums n_params_list in
    let param_idx =
      match param_name with
      | Q -> 0
      | D -> 1
      | C -> 2
      | Log_obs_var -> 3
      | Scaling_factor -> 4
    in
    List.nth_exn n_params_prefix_suffix_sums param_idx

  (* approximation defined implicitly via Gv products *)
  let g12v ~(lambda : A.M.t) (v : P.M.t) : P.M.t =
    let open Maths in
    let q = einsum [ lambda.q_left, "in"; v.q, "aij"; lambda.q_right, "jm" ] "anm" in
    let d = einsum [ lambda.d_left, "in"; v.d, "aij"; lambda.d_right, "jm" ] "anm" in
    (* TODO: is there a more ergonomic way to deal with constant parameters? *)
    let b =
      Tensor.zeros [ _K; m; n ] ~device:base.device ~kind:base.kind |> Maths.const
    in
    let b_0 =
      Tensor.zeros [ _K; n; n ] ~device:base.device ~kind:base.kind |> Maths.const
    in
    let c = einsum [ lambda.c_left, "in"; v.c, "aij"; lambda.c_right, "jm" ] "anm" in
    let log_obs_var =
      einsum
        [ lambda.log_obs_var_left, "in"
        ; reshape v.log_obs_var ~shape:[ -1; 1; 1 ], "aij"
        ; lambda.log_obs_var_right, "jm"
        ]
        "anm"
      |> reshape ~shape:[ -1; 1 ]
    in
    let scaling_factor =
      einsum
        [ lambda.scaling_factor_left, "in"
        ; reshape v.scaling_factor ~shape:[ -1; 1; 1 ], "aij"
        ; lambda.scaling_factor_right, "jm"
        ]
        "anm"
      |> reshape ~shape:[ -1; 1 ]
    in
    { q; d; b; b_0; c; log_obs_var; scaling_factor }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~param_name ~n_per_param v =
    let q = zero_params ~shape:(get_shapes Q) n_per_param in
    let d = zero_params ~shape:(get_shapes D) n_per_param in
    let c = zero_params ~shape:(get_shapes C) n_per_param in
    let log_obs_var = zero_params ~shape:(get_shapes Log_obs_var) n_per_param in
    let scaling_factor = zero_params ~shape:(get_shapes Scaling_factor) n_per_param in
    let b = zero_params ~shape:[ m; n ] n_per_param in
    let b_0 = zero_params ~shape:[ n; n ] n_per_param in
    let params_tmp = PP.{ q; d; b; b_0; c; log_obs_var; scaling_factor } in
    match param_name with
    | Q -> { params_tmp with q = v }
    | D -> { params_tmp with d = v }
    | C -> { params_tmp with c = v }
    | Log_obs_var -> { params_tmp with log_obs_var = v }
    | Scaling_factor -> { params_tmp with scaling_factor = v }

  let random_localised_vs _K : P.T.t =
    let random_localised_param_name param_name =
      let w_shape = get_shapes param_name in
      let before, after = get_n_params_before_after param_name in
      let w = random_params ~shape:w_shape (get_n_params param_name) in
      let zeros_before = zero_params ~shape:w_shape before in
      let zeros_after = zero_params ~shape:w_shape after in
      let final = Tensor.concat [ zeros_before; w; zeros_after ] ~dim:0 in
      final
    in
    { q = random_localised_param_name Q
    ; d = random_localised_param_name D
    ; c = random_localised_param_name C
    ; log_obs_var = random_localised_param_name Log_obs_var
    ; scaling_factor = random_localised_param_name Scaling_factor
    ; b = zero_params ~shape:[ m; n ] _K
    ; b_0 = zero_params ~shape:[ n; n ] _K
    }

  (* compute sorted eigenvalues, u_left and u_right. *)
  let eigenvectors_for_params ~lambda ~param_name =
    let left, right =
      match param_name with
      | Q -> lambda.q_left, lambda.q_right
      | D -> lambda.d_left, lambda.d_right
      | C -> lambda.c_left, lambda.c_right
      | Log_obs_var -> lambda.log_obs_var_left, lambda.log_obs_var_right
      | Scaling_factor -> lambda.scaling_factor_left, lambda.scaling_factor_right
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
    s_all, u_left, u_right

  (* cache storage with a ref to memoize computed results *)
  let s_u_cache = ref []

  (* given param name, get eigenvalues and eigenvectors. *)
  let get_s_u ~lambda ~param_name =
    match List.Assoc.find !s_u_cache param_name ~equal:equal_param_name with
    | Some s -> s
    | None ->
      let s = eigenvectors_for_params ~lambda ~param_name in
      s_u_cache := (param_name, s) :: !s_u_cache;
      s

  let extract_local_vs ~s_all ~param_name ~u_left ~u_right ~selection =
    let n_per_param = get_n_params param_name in
    let local_vs =
      List.map selection ~f:(fun idx ->
        let il, ir, _ = s_all.(idx) in
        let slice_and_squeeze t dim idx =
          Tensor.squeeze_dim
            ~dim
            (Tensor.slice t ~dim ~start:(Some idx) ~end_:(Some (idx + 1)) ~step:1)
        in
        let u_l = slice_and_squeeze u_left 1 il in
        let u_r = slice_and_squeeze u_right 1 ir in
        let tmp =
          match param_name with
          | Log_obs_var | Scaling_factor -> Tensor.(u_l * u_r)
          | _ -> Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_l; u_r ]
        in
        Tensor.unsqueeze tmp ~dim:0)
      |> Tensor.concatenate ~dim:0
    in
    localise ~param_name ~n_per_param local_vs

  let eigenvectors_for_each_param ~lambda ~param_name ~sampling_state =
    let n_per_param = get_n_params param_name in
    let n_params = get_total_n_params param_name in
    let s_all, u_left, u_right = get_s_u ~lambda ~param_name in
    let selection =
      if cycle
      then
        List.init n_per_param ~f:(fun i ->
          ((sampling_state * n_per_param) + i) % n_params)
      else List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    extract_local_vs ~s_all ~param_name ~u_left ~u_right ~selection

  let eigenvectors ~(lambda : A.M.t) ~switch_to_learn t (_K : int) =
    let eigenvectors_each =
      List.map param_names_list ~f:(fun param_name ->
        eigenvectors_for_each_param ~lambda ~param_name ~sampling_state:t)
    in
    let vs =
      List.fold eigenvectors_each ~init:None ~f:(fun accu local_vs ->
        match accu with
        | None -> Some local_vs
        | Some a -> Some (P.map2 a local_vs ~f:(fun x y -> Tensor.concat ~dim:0 [ x; y ])))
    in
    (* reset s_u_cache and set sampling state to 0 if learn again *)
    if switch_to_learn then s_u_cache := [];
    let new_sampling_state = if switch_to_learn then 0 else t + 1 in
    Option.value_exn vs, new_sampling_state

  let init () =
    let init_eye size =
      Mat.(0.1 $* eye size) |> Tensor.of_bigarray ~device:base.device |> Prms.free
    in
    { q_left = init_eye n
    ; q_right = init_eye n
    ; d_left = init_eye 1
    ; d_right = init_eye n
    ; c_left = init_eye n
    ; c_right = init_eye o
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
    Bos.Cmd.(v "rm" % "-f" % in_dir "aux") |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state running_avg =
      Stdlib.Gc.major ();
      let _, o_list = sample_data true_theta in
      let loss, new_state = O.step ~config:(config ~iter) ~state ~data:o_list () in
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
            let ground_truth_elbo_list, _ = elbo ~data:o_list true_theta in
            let ground_truth_elbo =
              List.fold ground_truth_elbo_list ~init:(Maths.f 0.) ~f:(fun accu elbo ->
                Maths.(accu + elbo))
            in
            Maths.primal ground_truth_elbo |> Tensor.mean |> Tensor.to_float0_exn
          in
          let ground_truth_std_o_loss =
            let theta_std_o =
              PP.
                { q = extract theta_curr.q
                ; d = extract theta_curr.q
                ; b = extract theta_curr.b
                ; b_0 = extract theta_curr.b_0
                ; c = extract theta_curr.c
                ; log_obs_var = true_theta.log_obs_var
                ; scaling_factor = true_theta.scaling_factor
                }
            in
            let ground_truth_std_o_elbo_list, _ = elbo ~data:o_list theta_std_o in
            let ground_truth_std_o_elbo =
              List.fold
                ground_truth_std_o_elbo_list
                ~init:(Maths.f 0.)
                ~f:(fun accu elbo -> Maths.(accu + elbo))
            in
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
  module O = Optimizer.SOFO (M) (GGN)

  let name = "sofo"

  let config ~iter:_ =
    let aux =
      Optimizer.Config.SOFO.
        { (default_aux (in_dir "aux")) with
          config =
            Optimizer.Config.Adam.
              { default with base; learning_rate = Some 1e-3; eps = 1e-4 }
        ; steps = 50
        ; learn_steps = 100
        ; exploit_steps = 100
        }
    in
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 0.0005
      ; n_tangents = _K
      ; rank_one = false
      ; damping = Some 1e-5
      ; aux = Some aux
      }

  let init = O.init theta
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
