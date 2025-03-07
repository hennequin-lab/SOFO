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
let bs = 128
let _K = 128
let n_samples = 1
let id_m = Maths.(const (Tensor.of_bigarray ~device:base.device (Owl.Mat.eye m)))
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
    ; sigma_o_prms : 'a
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

let true_theta =
  let sigma_o_prms = Maths.(log (f 0.1) * ones_o) in
  let c = make_c ~sigma:1. (m, o) in
  PP.{ sigma_o_prms; c }

let _ =
  true_theta.c
  |> Maths.primal
  |> Tensor.to_bigarray ~kind:base.ba_kind
  |> (fun x -> Owl.Mat.(reshape (transpose x *@ x) [| -1; 1 |]))
  (*   |> (fun x -> Owl.(Mat.reshape (Linalg.D.svdvals x) [| -1; 1 |])) *)
  |> Owl.Mat.save_txt ~out:(in_dir "true_c")

let theta =
  let c = Prms.free (Maths.primal (make_c ~sigma:1. (m, o))) in
  let sigma_o_prms =
    Prms.create
      ~below:(Tensor.f 5.)
      Tensor.(f Float.(log 0.1) + zeros ~device:base.device ~kind:base.kind [ 1 ])
    |> Prms.pin
  in
  PP.{ c; sigma_o_prms }

let sample_data =
  let sigma_o = Maths.(exp true_theta.sigma_o_prms) in
  let c = true_theta.c in
  fun () ->
    let us =
      Tensor.(randn ~device:base.device ~kind:base.kind [ bs; m ]) |> Maths.const
    in
    let xs = Maths.(us *@ c) in
    let ys = Maths.(xs + (sigma_o * const (Tensor.randn_like (primal xs)))) in
    us, ys

let solver a y =
  let ell = Maths.cholesky a in
  let ell_t = Maths.transpose ~dim0:0 ~dim1:1 ell in
  let _x = Maths.linsolve_triangular ~left:false ~upper:true ell_t y in
  Maths.linsolve_triangular ~left:false ~upper:false ell _x

let posterior_precision_chol ~c ~inv_sigma_o =
  let r = Maths.(sqr inv_sigma_o) in
  Maths.(id_m + einsum [ c, "ij"; r, "j"; c, "kj" ] "ik") |> Maths.cholesky

let u_opt ~c ~inv_sigma_o y =
  let r = Maths.(sqr inv_sigma_o) in
  let a = Maths.(id_m + einsum [ c, "ij"; r, "j"; c, "kj" ] "ik") in
  let b = Maths.einsum [ c, "ij"; r, "j"; y, "mj" ] "mi" in
  solver a b

let _ =
  let u, y = sample_data () in
  let inv_sigma_o = Maths.(exp (neg true_theta.sigma_o_prms) * ones_o) in
  let u_recov = u_opt ~c:true_theta.c ~inv_sigma_o y in
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

(* log p = -1/2 * (error_term + d * log 2pi + log | Sigma|) *)

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

let elbo ~data:y (theta : P.M.t) =
  let sigma_o = Maths.(exp theta.sigma_o_prms) in
  let inv_sigma_o = Maths.(exp (neg theta.sigma_o_prms)) in
  let inv_sigma_o_expanded = Maths.(inv_sigma_o * ones_o) in
  let u_opt = u_opt ~c:theta.c ~inv_sigma_o:inv_sigma_o_expanded y in
  let post_prec_chol =
    posterior_precision_chol ~c:theta.c ~inv_sigma_o:inv_sigma_o_expanded
  in
  let results =
    List.init n_samples ~f:(fun _ ->
      let u_diff =
        let e = Maths.(const (Tensor.randn_like (primal u_opt))) in
        Maths.linsolve_triangular ~left:false ~upper:false post_prec_chol e
      in
      let u_sampled = Maths.(u_opt + u_diff) in
      (* m x o *)
      let y_pred = Maths.(u_sampled *@ theta.c) in
      let lik_term = gaussian_llh ~mu:y_pred ~inv_std:inv_sigma_o_expanded y in
      let kl_term =
        let prior_term = gaussian_llh ~inv_std:ones_u u_sampled in
        let q_term = gaussian_llh_chol ~precision_chol:post_prec_chol u_diff in
        Maths.(const (primal (q_term - prior_term)))
      in
      let neg_elbo =
        Maths.(lik_term - kl_term) |> Maths.neg |> fun x -> Maths.(x / f Float.(of_int o))
      in
      neg_elbo, y_pred)
  in
  let y_preds = List.map ~f:snd results in
  let neg_elbo =
    List.fold results ~init:(Maths.f 0.) ~f:(fun accu (x, _) -> Maths.(accu + x))
    |> fun x -> Maths.(x / f Float.(of_int n_samples))
  in
  neg_elbo, y_preds, sigma_o

module M = struct
  module P = P

  type data = Maths.t
  type args = unit

  let f ~update ~data:y ~init ~args:() (theta : P.M.t) =
    let neg_elbo, y_pred, sigma_o = elbo ~data:y theta in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let sigma2 = Maths.sqr sigma_o in
      let sigma2_p = Maths.(const (primal sigma2)) in
      let preconditioner =
        List.fold y_pred ~init:(Tensor.f 0.) ~f:(fun accu y_pred ->
          (*         let sigma2_t = Maths.tangent sigma2 |> Option.value_exn |> Maths.const in *)
          let mu_t = Maths.tangent y_pred |> Option.value_exn |> Maths.const in
          let ggn_part1 = Maths.(einsum [ mu_t, "kmo"; mu_t, "lmo" ] "kl" / sigma2_p) in
          let ggn_part2 = Maths.f 0. in
          (*let ggn_part2 =
          Maths.(
            einsum
              [ f Float.(0.5 * of_int o * of_int bs) * sigma2_t / sqr sigma2_p, "ky"
              ; sigma2_t, "ly"
              ]
              "kl")
        in*)
          Maths.(primal ((ggn_part1 + ggn_part2) / f Float.(of_int o)))
          |> Tensor.( + ) accu)
        |> fun x -> Tensor.(x / f Float.(of_int n_samples))
      in
      let _, s, _ = Tensor.svd ~some:true ~compute_uv:true preconditioner in
      let s =
        Tensor.to_bigarray ~kind:base.ba_kind s |> fun x -> Owl.Mat.reshape x [| -1; 1 |]
      in
      Owl.Mat.save_txt ~out:(in_dir "svals") s;
      u init (Some (neg_elbo, Some preconditioner))

  let update_good_tangents ~data:y ~tangents (theta : P.tagged) =
    let theta = P.make_dual theta ~t:tangents in
    let neg_elbo, _, _ = elbo ~data:y theta in
    let neg_elbo_t = Maths.tangent neg_elbo |> Option.value_exn in
    let u, _, _ = Tensor.svd ~some:true ~compute_uv:true neg_elbo_t in
    (*     let u = Tensor.transpose ~dim0:0 ~dim1:1 u in *)
    let u = Tensor.slice ~dim:1 ~start:(Some 0) ~end_:(Some Int.(_K / 2)) ~step:1 u in
    P.map tangents ~f:(fun v ->
      let v = Maths.tangent' v in
      let s = Tensor.shape v |> List.tl_exn in
      let v = Tensor.reshape v ~shape:[ _K; -1 ] in
      let v = Tensor.einsum ~path:None ~equation:"ki,kj->ji" [ v; u ] in
      let v = Tensor.reshape v ~shape:((_K / 2) :: s) in
      Maths.Direct v)
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

  let init_tangents ~n_tangents theta =
    O.W.P.map theta ~f:(function
      | Prms.Const x ->
        Tensor.zeros
          ~device:(Tensor.device x)
          ~kind:(Tensor.kind x)
          (n_tangents :: Tensor.shape x)
      | Prms.Free x ->
        Tensor.randn
          ~device:(Tensor.device x)
          ~kind:(Tensor.kind x)
          (n_tangents :: Tensor.shape x)
      | Prms.Bounded (x, _, _) ->
        Tensor.randn
          ~device:(Tensor.device x)
          ~kind:(Tensor.kind x)
          (n_tangents :: Tensor.shape x))
    |> O.W.P.map ~f:(fun x -> Maths.Direct x)

  (* orthogonalise tangents *)
  let orthonormalise tangents =
    let vtv =
      O.W.P.fold tangents ~init:(Tensor.f 0.) ~f:(fun accu (v, _) ->
        let v = Maths.tangent' v in
        let n_tangents = List.hd_exn Tensor.(shape v) in
        let v = Tensor.reshape v ~shape:[ n_tangents; -1 ] in
        Tensor.(accu + einsum ~equation:"ij,kj->ik" ~path:None [ v; v ]))
    in
    let u, s, _ = Tensor.svd ~some:true ~compute_uv:true vtv in
    let normalizer = Tensor.(u / sqrt s |> transpose ~dim0:0 ~dim1:1) in
    O.W.P.map tangents ~f:(fun v ->
      let v = Maths.tangent' v in
      let n_tangents = List.hd_exn Tensor.(shape v) in
      let s = Tensor.shape v in
      let v = Tensor.reshape v ~shape:[ n_tangents; -1 ] in
      let v = Tensor.(matmul normalizer v) in
      Maths.Direct (Tensor.reshape v ~shape:s))

  let complete_tangents good_tangents =
    O.W.P.map good_tangents ~f:(fun v ->
      let v = Maths.tangent' v in
      let v = Tensor.(concat [ v; randn_like v ] ~dim:0) in
      Maths.Direct v)
    |> orthonormalise

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state ~good_tangents running_avg =
      Stdlib.Gc.major ();
      let _, y = sample_data () in
      let tangents = complete_tangents good_tangents in
      let loss, new_state = O.step ~tangents ~config:(config ~iter) ~state ~data:y () in
      let running_avg =
        let loss_avg =
          match running_avg with
          | [] -> loss
          | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
        in
        (* save params *)
        if iter % 10 = 0
        then (
          let ground_truth_elbo, _, _ = elbo ~data:y true_theta in
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
          (*           |> (fun x -> Owl.(Mat.reshape (Linalg.D.svdvals x) [| -1; 1 |])) *)
          |> Owl.Mat.save_txt ~out:(in_dir "c"));
        []
      in
      if iter < max_iter
      then (
        let good_tangents = M.update_good_tangents ~data:y ~tangents (O.params state) in
        loop ~iter:(iter + 1) ~state:new_state ~good_tangents (loss :: running_avg))
    in
    let good_tangents = init_tangents ~n_tangents:(_K / 2) theta in
    loop ~iter:0 ~state:init ~good_tangents []
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
      ; learning_rate = Some 0.2
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
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.006 }

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
