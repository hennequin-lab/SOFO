(* Linear Gaussian Dynamics, with same state/control/cost parameters constant across trials and across time *)
open Base
open Forward_torch
open Torch
open Sofo
module Mat = Owl.Dense.Matrix.D

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f64)
    ; ba_kind = Bigarray.float64
    }

(* -----------------------------------------
   --- Utility Functions ---
   ----------------------------------------- *)
let primal_detach (x, _) = Maths.const Tensor.(detach x)

(* solves for xA = y, A = ell (ell)^T. NOTE: since linsolve_triangular only deals with rectangular matrix B 
but not vector, this function does not apply to ell with a batch dimension in front! *)
let solver_chol ell y =
  let ell_t = Maths.transpose ~dim0:0 ~dim1:1 ell in
  let _x = Maths.linsolve_triangular ~left:false ~upper:true ell_t y in
  Maths.linsolve_triangular ~left:false ~upper:false ell _x

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

let precision_of_log_var log_var = Maths.(exp (neg log_var))
let sqrt_precision_of_log_var log_var = Maths.(exp (f (-0.5) * log_var))

(* -----------------------------------------
   -- Control Problem / Data Generation ---
   ----------------------------------------- *)
module Dims = struct
  let n = 10
  let m = 5
  let o = 40
  let tmax = 10
  let bs = 32
  let batch_const = true
  let kind = base.kind
  let device = base.device
end

let _K = 120
let true_noise_std = 0.1
let x0 = Tensor.zeros ~device:base.device ~kind:base.kind [ Dims.bs; Dims.n ]
let eye_m = Maths.(const (Tensor.eye ~n:Dims.m ~options:(base.kind, base.device)))
let eye_n = Tensor.eye ~n:Dims.n ~options:(base.kind, base.device) |> Maths.const
let ones_u = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ Dims.m ]))
let ones_x = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ Dims.n ]))
let ones_o = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ Dims.o ]))

let make_a_prms ~n target_sa =
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

let make_fu () =
  Tensor.(
    f Float.(1. /. sqrt (of_int Dims.m))
    * randn ~device:base.device ~kind:base.kind [ Dims.m; Dims.n ])
  |> Maths.const

let make_c () =
  Tensor.(
    f Float.(1. /. sqrt (of_int Dims.n))
    * randn ~device:base.device ~kind:base.kind [ Dims.n; Dims.o ])
  |> Maths.const

let _q_true, _d_true = make_a_prms ~n:Dims.n 0.8
let _Fx_true = _Fx_reparam (_q_true, _d_true)
let eye_n = Tensor.eye ~n:Dims.n ~options:(base.kind, base.device) |> Maths.const
let _Fu_true = make_fu ()

(* _Fu_0 is identity *)
let _Fu_0_true = eye_n
let _c_true = make_c ()
let _obs_var_true = Maths.const Tensor.(square (f true_noise_std))

(* u goes from 1 to T-1, o goes from 1 to T *)
let sample_data () =
  let x1 = Tensor.randn ~device:base.device ~kind:base.kind [ Dims.bs; Dims.n ] in
  let sigma = Maths.sqrt _obs_var_true in
  let u =
    List.init (Dims.tmax - 1) ~f:(fun _ ->
      Tensor.(randn ~device:base.device ~kind:base.kind [ Dims.bs; Dims.m ])
      |> Maths.const)
  in
  let rollout u_list =
    let tmp_einsum a b =
      if Dims.batch_const
      then Maths.einsum [ a, "ma"; b, "ab" ] "mb"
      else Maths.einsum [ a, "ma"; b, "mab" ] "mb"
    in
    let _, y_list_rev =
      List.fold
        u_list
        ~init:(Maths.const x1, [ tmp_einsum (Maths.const x1) _c_true ])
        ~f:(fun (x, y_list) u ->
          let new_x = Maths.(tmp_einsum x _Fx_true + tmp_einsum u _Fu_true) in
          let new_y = tmp_einsum new_x _c_true in
          new_x, new_y :: y_list)
    in
    List.rev y_list_rev
  in
  let o =
    rollout u
    |> List.map ~f:(fun y ->
      Maths.(primal (y + (sigma * const (Tensor.randn_like (Maths.primal y))))))
  in
  u, o

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)
let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run

module PP = struct
  type 'a p =
    { _q : 'a
    ; _d : 'a
    ; _c : 'a
    ; _log_obs_var : 'a (* log of covariance of emission noise *)
    ; _scaling_factor : 'a
    }
  [@@deriving prms]
end

let true_theta =
  PP.
    { _q = _q_true
    ; _d = _d_true
    ; _c = _c_true
    ; _log_obs_var = Maths.log _obs_var_true
    ; _scaling_factor = Maths.f 1.
    }

module P = PP.Make (Prms.P)

module LGS = struct
  module P = P

  type args = unit
  type data = Tensor.t list

  (* average spatial covariance *)
  let save_summary ~out (theta : P.M.t) =
    let a =
      let a_tmp = _Fx_reparam (theta._q, theta._d) in
      Maths.primal a_tmp |> Tensor.to_bigarray ~kind:base.ba_kind
    in
    let b = Maths.primal _Fu_true |> Tensor.to_bigarray ~kind:base.ba_kind in
    let c = Maths.primal theta._c |> Tensor.to_bigarray ~kind:base.ba_kind in
    let avg_spatial_cov =
      let q1 = Owl.Mat.(transpose b *@ b) in
      let _, q_accu =
        List.fold (List.range 0 Dims.tmax) ~init:(q1, q1) ~f:(fun accu _ ->
          let q_prev, q_accu = accu in
          let q_new = Owl.Mat.((transpose a *@ q_prev *@ a) + q1) in
          q_new, Owl.Mat.(q_new + q_accu))
      in
      Owl.Mat.(q_accu /$ Float.of_int Dims.tmax)
    in
    let q = Owl.Mat.(transpose c *@ avg_spatial_cov *@ c) in
    let noise_term =
      let obs_var =
        Maths.(exp theta._log_obs_var) |> Maths.primal |> Tensor.to_float0_exn
      in
      Owl.Mat.(obs_var $* eye Dims.o)
    in
    let q = Owl.Mat.(q + noise_term) in
    q |> (fun x -> Owl.Mat.reshape x [| -1; 1 |]) |> Owl.Mat.save_txt ~out

  (* create params for lds from f *)
  let params_from_f ~(theta : P.M.t) ~x0 ~o_list
    : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
    =
    (* set o at time 0 as 0 *)
    let o_list_tmp = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
    let _obs_var_inv = Maths.(exp (neg theta._log_obs_var) * ones_o) in
    let _Cxx =
      Maths.(einsum [ theta._c, "ab"; _obs_var_inv, "b"; theta._c, "cb" ] "ac")
    in
    let _Fx_prod = _Fx_reparam (theta._q, theta._d) in
    Lqr.Params.
      { x0 = Some x0
      ; params =
          List.mapi o_list_tmp ~f:(fun i o ->
            let _cx =
              Maths.(
                neg (einsum [ const o, "ab"; _obs_var_inv, "b"; theta._c, "cb" ] "ac"))
            in
            Lds_data.Temp.
              { _f = None
              ; _Fx_prod
              ; _Fu_prod = (if i = 0 then _Fu_0_true else _Fu_true)
              ; _cx = Some _cx
              ; _cu = None
              ; _Cxx
              ; _Cxu = None
              ; _Cuu = (if i = 0 then eye_n else eye_m)
              })
      }

  (* rollout y list under sampled u (u_0, ..., u_{T-1})*)
  let rollout ~u_list (theta : P.M.t) =
    let tmp_einsum a b =
      if Dims.batch_const
      then Maths.einsum [ a, "ma"; b, "ab" ] "mb"
      else Maths.einsum [ a, "ma"; b, "mab" ] "mb"
    in
    let _Fx_prod = _Fx_reparam (theta._q, theta._d) in
    let _, y_list_rev =
      List.foldi
        u_list
        ~init:(Maths.const x0, [])
        ~f:(fun i (x, y_list) u ->
          let new_x =
            Maths.(
              tmp_einsum x _Fx_prod
              + tmp_einsum u (if i = 0 then _Fu_0_true else _Fu_true))
          in
          let new_y = tmp_einsum new_x theta._c in
          new_x, new_y :: y_list)
    in
    List.rev y_list_rev

  (* approximate kalman filtered distribution of u *)
  let sample_and_kl ~_Fx ~_Fu ~_Fu_0 ~_c ~obs_precision ~scaling_factor ustars o_list =
    let open Maths in
    let o_list = List.map ~f:Maths.const o_list in
    let z0 =
      Tensor.zeros ~device:base.device ~kind:base.kind [ Dims.bs; Dims.n ] |> Maths.const
    in
    let btrinv_0 = einsum [ _Fu_0, "ij"; _c, "jo"; obs_precision, "o" ] "io" in
    let btrinv = einsum [ _Fu, "ij"; _c, "jo"; obs_precision, "o" ] "io" in
    (* posterior precision of filtered covariance of u *)
    let precision_chol_0 =
      let tmp = scaling_factor * ones_x |> reshape ~shape:[ 1; -1 ] in
      (eye_n + einsum [ btrinv_0, "io"; _c, "jo"; _Fu_0, "kj" ] "ik" |> cholesky) * tmp
    in
    let precision_chol =
      let tmp = scaling_factor * ones_u |> reshape ~shape:[ 1; -1 ] in
      (eye_m + einsum [ btrinv, "io"; _c, "jo"; _Fu, "kj" ] "ik" |> cholesky) * tmp
    in
    let ustars_o_list = List.map2_exn ustars o_list ~f:(fun u o -> u, o) in
    let _, kl, us =
      List.foldi
        ustars_o_list
        ~init:(z0, f 0., [])
        ~f:(fun i (z, kl, us) (ustar, o) ->
          Stdlib.Gc.major ();
          let _Fu = if i = 0 then _Fu_0 else _Fu in
          let precision_chol = if i = 0 then precision_chol_0 else precision_chol in
          let btrinv = if i = 0 then btrinv_0 else btrinv in
          let zpred = (z *@ _Fx) + (ustar *@ _Fu) in
          let ypred = zpred *@ _c in
          let delta = o - ypred in
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
                    [ Dims.bs; (if i = 0 then Dims.n else Dims.m) ]))
          in
          let u_sample = mu + u_diff_elbo in
          (* u_final used to propagate dynamics *)
          let u_final = ustar + u_sample in
          (* propagate that sample to update z *)
          let z = zpred + (u_sample *@ _Fu) in
          (* update the KL divergence *)
          let kl =
            let prior_term =
              gaussian_llh ~inv_std:(if i = 0 then ones_x else ones_u) u_final
            in
            let q_term = gaussian_llh_chol ~precision_chol u_diff_elbo in
            kl + q_term - prior_term
          in
          z, kl, u_final :: us)
    in
    kl, List.rev us

  let elbo ~data:(o_list : Tensor.t list) (theta : P.M.t) =
    let _Fx = _Fx_reparam (theta._q, theta._d) in
    let obs_precision = Maths.(precision_of_log_var theta._log_obs_var * ones_o) in
    (* use lqr to obtain the optimal u *)
    let p =
      params_from_f ~x0:(Maths.const x0) ~theta ~o_list
      |> Lds_data.map_naive ~batch_const:Dims.batch_const
    in
    let sol, _ = Lqr._solve ~batch_const:Dims.batch_const p in
    let ustars = List.map sol ~f:(fun s -> s.u) in
    let kl, u_sampled =
      sample_and_kl
        ~_Fx
        ~_Fu:_Fu_true
        ~_Fu_0:_Fu_0_true
        ~_c:theta._c
        ~obs_precision
        ~scaling_factor:theta._scaling_factor
        ustars
        o_list
    in
    let y_pred = rollout ~u_list:u_sampled theta in
    let lik_term =
      let inv_sigma_o_expanded =
        Maths.(sqrt_precision_of_log_var theta._log_obs_var * ones_o)
      in
      List.fold2_exn
        o_list
        y_pred
        ~init:Maths.(f 0.)
        ~f:(fun accu o y_pred ->
          Stdlib.Gc.major ();
          Maths.(
            accu + gaussian_llh ~mu:y_pred ~inv_std:inv_sigma_o_expanded (Maths.const o)))
    in
    Maths.(neg (lik_term - kl) / f Float.(of_int Dims.tmax * of_int Dims.o)), y_pred

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
      (* CHECKED this agrees with mine *)
      let ggn_part2 =
        Maths.(
          einsum
            [ ( f Float.(0.5 * of_int Dims.o * of_int Dims.bs)
                * sigma2_t
                * sqr obs_precision_p
              , "ky" )
            ; sigma2_t, "ly"
            ]
            "kl")
      in
      Maths.(
        accu
        + const
            (primal
               ((ggn_part1 + ggn_part2) / f Float.(of_int Dims.o * of_int Dims.tmax)))))
    |> Maths.primal

  let f ~update ~data:y ~init ~args:() (theta : P.M.t) =
    let neg_elbo, y_pred_ggn = elbo ~data:y theta in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let ggn = ggn ~y_pred:y_pred_ggn theta in
      let _ =
        let _, s, _ = Owl.Linalg.D.svd (Tensor.to_bigarray ~kind:base.ba_kind ggn) in
        Owl.Mat.(save_txt ~out:(in_dir "svals") (transpose s))
      in
      u init (Some (neg_elbo, Some ggn))

  let init : P.tagged =
    let _q, _d =
      let tmp_q, tmp_d = make_a_prms ~n:Dims.n 0.8 in
      Prms.free (Maths.primal tmp_q), Prms.free (Maths.primal tmp_d)
    in
    let _c = Prms.free (Maths.primal (make_c ())) in
    let _log_obs_var =
      Tensor.(log (f Float.(square 1.) * ones ~device:base.device ~kind:base.kind [ 1 ]))
      |> Prms.free
    in
    let _scaling_factor =
      Prms.create ~above:(Tensor.f 0.1) (Tensor.ones [ 1 ] ~device:base.device)
    in
    { _q; _d; _c; _log_obs_var; _scaling_factor }
end

let _ = LGS.save_summary ~out:(in_dir "true_summary") true_theta

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

let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
let param_names_list = [ Q; D; C; Log_obs_var; Scaling_factor ]
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
    | Q -> [ Dims.n; Dims.n ]
    | D -> [ 1; Dims.n ]
    | C -> [ Dims.n; Dims.o ]
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
    let _q = einsum [ lambda.q_left, "in"; v._q, "aij"; lambda.q_right, "jm" ] "anm" in
    let _d = einsum [ lambda.d_left, "in"; v._d, "aij"; lambda.d_right, "jm" ] "anm" in
    let _c = einsum [ lambda.c_left, "in"; v._c, "aij"; lambda.c_right, "jm" ] "anm" in
    let _log_obs_var =
      einsum
        [ lambda.log_obs_var_left, "in"
        ; reshape v._log_obs_var ~shape:[ -1; 1; 1 ], "aij"
        ; lambda.log_obs_var_right, "jm"
        ]
        "anm"
      |> reshape ~shape:[ -1; 1 ]
    in
    let _scaling_factor =
      einsum
        [ lambda.scaling_factor_left, "in"
        ; reshape v._scaling_factor ~shape:[ -1; 1; 1 ], "aij"
        ; lambda.scaling_factor_right, "jm"
        ]
        "anm"
      |> reshape ~shape:[ -1; 1 ]
    in
    { _q; _d; _c; _log_obs_var; _scaling_factor }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~param_name ~n_per_param v =
    let _q = zero_params ~shape:(get_shapes Q) n_per_param in
    let _d = zero_params ~shape:(get_shapes D) n_per_param in
    let _c = zero_params ~shape:(get_shapes C) n_per_param in
    let _log_obs_var = zero_params ~shape:(get_shapes Log_obs_var) n_per_param in
    let _scaling_factor = zero_params ~shape:(get_shapes Scaling_factor) n_per_param in
    let params_tmp = PP.{ _q; _d; _c; _log_obs_var; _scaling_factor } in
    match param_name with
    | Q -> { params_tmp with _q = v }
    | D -> { params_tmp with _d = v }
    | C -> { params_tmp with _c = v }
    | Log_obs_var -> { params_tmp with _log_obs_var = v }
    | Scaling_factor -> { params_tmp with _scaling_factor = v }

  let random_localised_vs () : P.T.t =
    let random_localised_param_name param_name =
      let w_shape = get_shapes param_name in
      let before, after = get_n_params_before_after param_name in
      let w = random_params ~shape:w_shape (get_n_params param_name) in
      let zeros_before = zero_params ~shape:w_shape before in
      let zeros_after = zero_params ~shape:w_shape after in
      let final = Tensor.concat [ zeros_before; w; zeros_after ] ~dim:0 in
      final
    in
    { _q = random_localised_param_name Q
    ; _d = random_localised_param_name D
    ; _c = random_localised_param_name C
    ; _log_obs_var = random_localised_param_name Log_obs_var
    ; _scaling_factor = random_localised_param_name Scaling_factor
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
        (* let tmp = Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_l; u_r ] in *)
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
    { q_left = init_eye Dims.n
    ; q_right = init_eye Dims.n
    ; d_left = init_eye 1
    ; d_right = init_eye Dims.n
    ; c_left = init_eye Dims.n
    ; c_right = init_eye Dims.o
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
    with type 'a W.P.p = 'a LGS.P.p
     and type W.data = Tensor.t list
     and type W.args = unit

  val name : string
  val config : iter:int -> (float, Bigarray.float64_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    Bos.Cmd.(v "rm" % "-f" % in_dir "aux") |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state ~time_elapsed running_avg =
      Stdlib.Gc.major ();
      let data =
        let _, o_list = sample_data () in
        o_list
      in
      let t0 = Unix.gettimeofday () in
      let loss, new_state = O.step ~config:(config ~iter) ~state ~data () in
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
          (* avg error *)
          Convenience.print [%message (iter : int) (loss_avg : float)];
          let t = iter in
          let ground_truth_loss =
            let ground_truth_elbo, _ = LGS.elbo ~data true_theta in
            Maths.primal ground_truth_elbo |> Tensor.mean |> Tensor.to_float0_exn
          in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array
                 [| Float.of_int t; time_elapsed; loss_avg; ground_truth_loss |]
                 1
                 4));
          LGS.save_summary
            ~out:(in_dir "summary")
            (O.params new_state |> O.W.P.value |> O.W.P.map ~f:Maths.const));
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
  module O = Optimizer.SOFO (LGS) (GGN)

  let name = "sofo"

  let config ~iter:_ =
    let aux =
      Optimizer.Config.SOFO.
        { (default_aux (in_dir "aux")) with
          config =
            Optimizer.Config.Adam.
              { default with base; learning_rate = Some 1e-3; eps = 1e-8 }
        ; steps = 5
        ; learn_steps = 50
        ; exploit_steps = 10
        ; local = true
        }
    in
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 1.
      ; n_tangents = _K
      ; rank_one = false
      ; damping = Some 1e-3
      ; aux = Some aux
      ; orthogonalize = false
      }

  let init = O.init LGS.init
end

(* --------------------------------
     -- Adam
     --------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (LGS)

  let config ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.01 }

  let init = O.init LGS.init
end

let _ =
  let max_iter = 2000 in
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
