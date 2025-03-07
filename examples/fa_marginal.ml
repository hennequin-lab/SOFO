(* Linear Gaussian Dynamics, with same state/control/cost parameters constant across trials and across time *)
open Printf
open Base
open Forward_torch
open Torch
open Sofo

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

let m = 10
let o = 40
let bs = 16
let _K = 32
let n_elbo_samples = 1
let id_m = Maths.(const (Tensor.of_bigarray ~device:base.device (Owl.Mat.eye m)))
let id_o = Maths.(const (Tensor.of_bigarray ~device:base.device (Owl.Mat.eye o)))
let ones_1 = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ 1 ]))
let ones_o = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
let ones_u = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ m ]))

let make_c ~sigma (m, o) =
  Tensor.(
    f Float.(sigma /. sqrt (of_int m))
    * randn ~device:base.device ~kind:base.kind [ m; o ])
  |> Maths.const

module PP = struct
  type 'a p =
    { c : 'a
    ; log_obs_var : 'a
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

let true_theta =
  let log_obs_var = Maths.(log (f Float.(square 0.01))) in
  let c = make_c ~sigma:1. (m, o) in
  PP.{ log_obs_var; c }

let _ =
  true_theta.c
  |> Maths.primal
  |> Tensor.to_bigarray ~kind:base.ba_kind
  |> (fun x -> Owl.Mat.(reshape (transpose x *@ x) [| -1; 1 |]))
  |> Owl.Mat.save_txt ~out:(in_dir "true_c")

let theta =
  let c = Prms.free (Maths.primal (make_c ~sigma:1. (m, o))) in
  let log_obs_var =
    Maths.(log (f Float.(square 1.) * ones_1))
    |> Maths.primal
    |> Prms.create ~above:(Tensor.f Float.(log (square 0.01)))
  in
  PP.{ c; log_obs_var }

let std_of_log_var log_var = Maths.(exp (f 0.5 * log_var))
let precision_of_log_var log_var = Maths.(exp (neg log_var))
let sqrt_precision_of_log_var log_var = Maths.(exp (f (-0.5) * log_var))

let sample_data =
  let sigma = std_of_log_var true_theta.log_obs_var in
  fun () ->
    let us =
      Tensor.(randn ~device:base.device ~kind:base.kind [ bs; m ]) |> Maths.const
    in
    let xs = Maths.(us *@ true_theta.c) in
    let ys = Maths.(xs + (const (Tensor.randn_like (primal xs)) * sigma)) in
    us, ys

(* tested *)
let solver a y =
  let ell = Maths.cholesky a in
  let ell_t = Maths.transpose ~dim0:0 ~dim1:1 ell in
  let _x = Maths.linsolve_triangular ~left:false ~upper:true ell_t y in
  Maths.linsolve_triangular ~left:false ~upper:false ell _x

let posterior_precision ~c ~obs_precision =
  Maths.(id_m + einsum [ c, "ij"; obs_precision, "j"; c, "kj" ] "ik")

let u_opt ~c ~obs_precision y =
  let a = posterior_precision ~c ~obs_precision in
  let b = Maths.einsum [ c, "ij"; obs_precision, "j"; y, "mj" ] "mi" in
  solver a b

let _ =
  let u, y = sample_data () in
  let obs_precision = Maths.(precision_of_log_var true_theta.log_obs_var * ones_o) in
  let u_recov = u_opt ~c:true_theta.c ~obs_precision y in
  let u =
    u
    |> Maths.primal
    |> Tensor.to_bigarray ~kind:base.ba_kind
    |> fun x -> Owl.Mat.reshape x [| -1; 1 |]
  in
  let u_recov =
    u_recov
    |> Maths.primal
    |> Tensor.to_bigarray ~kind:base.ba_kind
    |> fun x -> Owl.Mat.reshape x [| -1; 1 |]
  in
  Owl.Mat.(save_txt ~out:(in_dir "u") (u @|| u_recov))

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
  Maths.(-0.5 $* (const_term $+ error_term + cov_term))

let elbo ~screwup ~data:y (theta : P.M.t) =
  let obs_precision = Maths.(precision_of_log_var theta.log_obs_var * ones_o) in
  let obs_precision_std = Maths.(sqrt_precision_of_log_var theta.log_obs_var * ones_o) in
  let u_opt = u_opt ~c:theta.c ~obs_precision y in
  let post_prec_chol =
    posterior_precision ~c:theta.c ~obs_precision
    |> Maths.cholesky
    |> fun x -> if screwup then Maths.(id_m * x) else x
  in
  let sample () =
    let u_diff =
      let e = Maths.(const (Tensor.randn_like (primal u_opt))) in
      let post_prec_chol_t = Maths.transpose ~dim0:0 ~dim1:1 post_prec_chol in
      Maths.linsolve_triangular ~left:false ~upper:true post_prec_chol_t e
    in
    let u_sampled = Maths.(u_opt + u_diff) in
    let y_pred = Maths.(const (primal u_sampled) *@ theta.c) in
    u_diff, u_sampled, y_pred
  in
  let u_diff, u_sampled, y_pred = sample () in
  let lik_term = gaussian_llh ~mu:y_pred ~inv_std:obs_precision_std y in
  let kl_term =
    let prior_term = gaussian_llh ~inv_std:ones_u u_sampled in
    let q_term = gaussian_llh_chol ~precision_chol:post_prec_chol u_diff in
    Maths.(q_term - prior_term)
  in
  let neg_elbo =
    Maths.(lik_term - kl_term) |> Maths.neg |> fun x -> Maths.(x / f Float.(of_int o))
  in
  neg_elbo, y_pred

let marginal_covariance ~c ~log_obs_var =
  let obs_var = Maths.(ones_o * exp log_obs_var) |> Maths.reshape ~shape:[ 1; -1 ] in
  Maths.((id_o * obs_var) + einsum [ c, "ji"; c, "jk" ] "ik")

(* tested *)
let marginal_precision_chol cov =
  let ell = cov |> Maths.cholesky in
  Maths.linsolve_triangular ~left:true ~upper:false ell id_o
  |> Maths.transpose ~dim0:0 ~dim1:1

module M = struct
  module P = P

  type data = Maths.t
  type args = unit

  let f_marginal ~update ~data:y ~init ~args:() (theta : P.M.t) =
    let mc = marginal_covariance ~c:theta.c ~log_obs_var:theta.log_obs_var in
    let mpc = marginal_precision_chol mc in
    let nmll =
      Maths.(neg (gaussian_llh_chol ~precision_chol:mpc y) / f Float.(of_int o))
    in
    match update with
    | `loss_only u -> u init (Some nmll)
    | `loss_and_ggn u ->
      let mpc_p = Maths.primal mpc in
      let mc_t = Maths.tangent mc |> Option.value_exn in
      let preconditioner =
        let z =
          Tensor.(
            f Float.(0.5 * of_int bs / of_int o)
            * einsum
                ~path:None
                ~equation:"ik,mk,tmn,nl,jl-> tij"
                [ mpc_p; mpc_p; mc_t; mpc_p; mpc_p ])
        in
        Tensor.einsum ~path:None ~equation:"kij,lij->kl" [ z; mc_t ]
      in
      let _, s, _ = Tensor.svd ~some:true ~compute_uv:true preconditioner in
      let s =
        Tensor.to_bigarray ~kind:base.ba_kind s |> fun x -> Owl.Mat.reshape x [| -1; 1 |]
      in
      Owl.Mat.save_txt ~out:(in_dir "svals") s;
      u init (Some (nmll, Some preconditioner))

  let f_elbo ~update ~data:y ~init ~args:() (theta : P.M.t) =
    let neg_elbo, y_pred = elbo ~screwup:false ~data:y theta in
    (* obtain y_pred from another independent sample *)
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let obs_precision = precision_of_log_var theta.log_obs_var in
      let obs_precision_p = Maths.(const (primal obs_precision)) in
      let sigma2_t =
        Maths.(tangent (exp theta.log_obs_var)) |> Option.value_exn |> Maths.const
      in
      let preconditioner =
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
        Maths.(primal ((ggn_part1 + ggn_part2) / f Float.(of_int o)))
      in
      let _, s, _ = Tensor.svd ~some:true ~compute_uv:true preconditioner in
      let s =
        Tensor.to_bigarray ~kind:base.ba_kind s |> fun x -> Owl.Mat.reshape x [| -1; 1 |]
      in
      Owl.Mat.save_txt ~out:(in_dir "svals") s;
      u init (Some (neg_elbo, Some preconditioner))

  let f = f_marginal
end

(* --------------------------------
   -- Generic type of optimiser
   -------------------------------- *)

module type Do_with_T = sig
  module O :
    Optimizer.T
    with type 'a W.P.p = 'a M.P.p
     and type W.data = Maths.t
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
      let _, y = sample_data () in
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
          let ground_truth_elbo, _ = elbo ~screwup:false ~data:y true_theta in
          let ground_truth_elbo =
            Maths.primal ground_truth_elbo |> Tensor.mean |> Tensor.to_float0_exn
          in
          (* avg error *)
          Convenience.print [%message (iter : int) (loss_avg : float)];
          let t = iter in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array [| Float.of_int t; loss_avg; ground_truth_elbo |] 1 3));
          O.params new_state
          |> P.value
          |> (fun x -> x.PP.c)
          |> Tensor.to_bigarray ~kind:base.ba_kind
          |> (fun x -> Owl.Mat.(reshape (transpose x *@ x) [| -1; 1 |]))
          |> Owl.Mat.save_txt ~out:(in_dir "c"));
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
      ; learning_rate = Some Float.(0.2)
      ; n_tangents = _K
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
