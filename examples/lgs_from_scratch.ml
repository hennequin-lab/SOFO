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
let tmax = 100
let bs = 64
let eye_m = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye m)))
let eye_t = Maths.(const (Tensor.of_bigarray ~device:base.device Mat.(eye tmax)))
let ones_1 = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ 1 ]))
let ones_o = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
let ones_u = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ m ]))

let make_a target_sa =
  let a =
    let w =
      let w = Mat.(gaussian n n) in
      let sa = Owl.Linalg.D.eigvals w |> Owl.Dense.Matrix.Z.re |> Mat.max' in
      Mat.(Float.(target_sa / sa) $* w)
    in
    Owl.Linalg.D.expm Mat.((w - eye n) *$ 0.1)
  in
  Tensor.of_bigarray ~device:base.device a |> Maths.const

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
  let a = make_a 0.9 in
  let b = make_b () in
  let c = make_c () in
  let log_obs_var = Maths.(log (f Float.(square 0.1))) in
  PP.{ a; b; c; log_obs_var }

let theta =
  let a = Prms.free (Maths.primal (make_a 0.1)) in
  let b = Prms.const (Maths.primal true_theta.b) in
  let c = Prms.free (Maths.primal (make_c ())) in
  let log_obs_var =
    Maths.(log (f Float.(square 1.) * ones_1))
    |> Maths.primal
    |> Prms.create ~above:(Tensor.f Float.(log (square 0.001)))
  in
  PP.{ a; b; c; log_obs_var }

let std_of_log_var log_var = Maths.(exp (f 0.5 * log_var))
let precision_of_log_var log_var = Maths.(exp (neg log_var))
let sqrt_precision_of_log_var log_var = Maths.(exp (f (-0.5) * log_var))

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
        let _Quu = eye_m + einsum [ b, "ij"; _V, "jl"; b, "kl" ] "ik" in
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

let solver chol y =
  let ell = chol in
  let ell_t = Maths.transpose ~dim0:0 ~dim1:1 ell in
  let _x = Maths.linsolve_triangular ~left:false ~upper:true ell_t y in
  Maths.linsolve_triangular ~left:false ~upper:false ell _x

let sample_and_kl ~a ~b ~c ~obs_precision ustars ys =
  let open Maths in
  let z0 = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n ] |> Maths.const in
  let _, kl =
    List.fold2_exn
      ustars
      ys
      ~init:(z0, f 0.)
      ~f:(fun (z, kl) ustar y ->
        let zpred = (z *@ a) + (ustar *@ b) in
        let ypred = zpred *@ c in
        let delta = y - ypred in
        let precision_chol =
          let tmp = b *@ c in
          eye_m + (einsum [ tmp, "ij"; tmp, "kj" ] "ik" * obs_precision) |> cholesky
        in
        let mu =
          obs_precision
          * solver precision_chol (einsum [ b, "ij"; c, "jo"; delta, "mo" ] "mi")
        in
        let u_diff =
          let ell_t = Maths.transpose ~dim0:0 ~dim1:1 precision_chol in
          Maths.linsolve_triangular
            ~left:false
            ~upper:true
            ell_t
            (const (Tensor.randn_like (primal ustar)))
        in
        let u_sample = mu + u_diff in
        (* propagate that sample to update z *)
        let z = zpred + (u_sample *@ b) in
        (* update the KL divergence *)
        let kl =
          let prior_term = gaussian_llh ~inv_std:ones_u (ustar + u_sample) in
          let q_term = gaussian_llh_chol ~mu:ustar ~precision_chol u_sample in
          kl + q_term - prior_term
        in
        z, kl)
  in
  kl

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

(* save the first element of the batch as a time series *)
let save_time_series ~out x =
  List.map x ~f:(fun x ->
    Maths.primal x
    |> Tensor.to_bigarray ~kind:base.ba_kind
    |> Mat.get_slice [ [ 0 ] ]
    |> fun x -> Mat.reshape x [| 1; -1 |])
  |> List.to_array
  |> Mat.concatenate ~axis:0
  |> Mat.save_txt ~out

let save_summary ~out (theta : P.M.t) =
  let us =
    List.init tmax ~f:(function
      | 0 -> eye_m
      | _ -> Maths.(const (Tensor.zeros_like (primal eye_m))))
  in
  let ys = rollout ~a:theta.a ~b:theta.b ~c:theta.c us in
  save_time_series ~out ys

let _ = save_summary ~out:(in_dir "true_summary") true_theta

let save_summary_youjing ~out (theta : P.M.t) =
  let a = Maths.primal theta.a |> Tensor.to_bigarray ~kind:base.ba_kind in
  let b = Maths.primal theta.b |> Tensor.to_bigarray ~kind:base.ba_kind in
  let c = Maths.primal theta.c |> Tensor.to_bigarray ~kind:base.ba_kind in
  let avg_spatial_cov =
    let q1 = Mat.(transpose b *@ b) in
    let _, q_accu =
      List.fold (List.range 0 tmax) ~init:(q1, q1) ~f:(fun (q_prev, q_accu) _ ->
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

let _ = save_summary_youjing ~out:(in_dir "true_summary_y") true_theta

(* -----------------------------------------------
     --  SOME TESTS
     ----------------------------------------------- *)

let u, y = sample_data true_theta
let _ = save_time_series ~out:(in_dir "u") u
let _ = save_time_series ~out:(in_dir "y") y

let u_recov =
  let obs_precision = Maths.(precision_of_log_var true_theta.log_obs_var * ones_o) in
  lqr ~a:true_theta.a ~b:true_theta.b ~c:true_theta.c ~obs_precision y

let y_recov = rollout ~a:true_theta.a ~b:true_theta.b ~c:true_theta.c u_recov
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

let elbo ~data:(y : Maths.t list) (theta : P.M.t) =
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
  let kl_term =
    Maths.f 0.
    (*
       let prior_term = gaussian_llh ~inv_std:ones_u u_sampled in
    let q_term = gaussian_llh_chol ~precision_chol:post_prec_chol u_diff in
    Maths.(const (primal (q_term - prior_term))) *)
  in
  let neg_elbo =
    Maths.(lik_term - kl_term)
    |> Maths.neg
    |> fun x -> Maths.(x / f Float.(of_int o * of_int tmax))
  in
  neg_elbo, y_pred

module M = struct
  module P = P

  type data = Maths.t list
  type args = unit

  let f ~update ~data:y ~init ~args:() (theta : P.M.t) =
    let neg_elbo, y_pred = elbo ~data:y theta in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let obs_precision = precision_of_log_var theta.log_obs_var in
      let obs_precision_p = Maths.(const (primal obs_precision)) in
      let sigma2_t =
        Maths.(tangent (exp theta.log_obs_var)) |> Option.value_exn |> Maths.const
      in
      let preconditioner =
        List.fold y_pred ~init:(Maths.f 0.) ~f:(fun accu y_pred ->
          let mu_t = Maths.tangent y_pred |> Option.value_exn |> Maths.const in
          let ggn_part1 =
            Maths.(einsum [ mu_t, "kmo"; mu_t, "lmo" ] "kl" * obs_precision_p)
          in
          let ggn_part2 =
            Maths.(
              einsum
                [ ( f Float.(0.5 * of_int o * of_int bs) * sigma2_t * sqr obs_precision_p
                  , "ky" )
                ; sigma2_t, "ly"
                ]
                "kl")
          in
          Maths.(
            accu
            + const (primal ((ggn_part1 + ggn_part2) / f Float.(of_int o * of_int tmax)))))
        |> Maths.primal
      in
      let _ =
        let _, s, _ =
          Owl.Linalg.D.svd (Tensor.to_bigarray ~kind:base.ba_kind preconditioner)
        in
        Mat.(save_txt ~out:(in_dir "svals") (transpose s))
      in
      u init (Some (neg_elbo, Some preconditioner))
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
          let ground_truth_elbo, _ = elbo ~data:y true_theta in
          let ground_truth_elbo =
            Maths.primal ground_truth_elbo |> Tensor.mean |> Tensor.to_float0_exn
          in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array [| Float.of_int t; loss_avg; ground_truth_elbo |] 1 3));
          save_summary
            ~out:(in_dir "summary")
            (O.params new_state |> O.W.P.value |> O.W.P.map ~f:Maths.const);
          save_summary_youjing
            ~out:(in_dir "summary_y")
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

  let config ~iter:k =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some Float.(0.1 / sqrt (1. + (of_int k / 200.)))
      ; n_tangents = 64
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
   -------------------------------- *)

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
