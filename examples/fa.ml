(* Linear Gaussian Dynamics, with same state/control/cost parameters constant across trials and across time *)
open Printf
open Base
open Forward_torch
open Torch
open Sofo

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f64)
    ; ba_kind = Bigarray.float64
    }

let marginal_llh = true
let marginal_ggn = true
let sampling = false
let n_fisher = 30
let m = 10
let o = 40
let bs = 16
let ones_o = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
let id_o = Maths.(const (Tensor.eye ~n:o ~options:(base.kind, base.device)))
let ones_u = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ m ]))

let make_c (m, o) =
  Tensor.(
    f Float.(1. /. sqrt (of_int m)) * randn ~device:base.device ~kind:base.kind [ m; o ])
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
  let sigma_o_prms = Maths.f 0.1 |> Maths.log in
  let c = make_c (m, o) in
  PP.{ sigma_o_prms; c }

let _ =
  Maths.(transpose true_theta.c ~dim0:1 ~dim1:0 *@ true_theta.c)
  |> Maths.primal
  |> Tensor.reshape ~shape:[ -1; 1 ]
  |> Tensor.to_bigarray ~kind:base.ba_kind
  |> Owl.Mat.save_txt ~out:(in_dir (sprintf "true_cct"))

let theta =
  let c = Prms.free (Maths.primal (make_c (m, o))) in
  let sigma_o_prms =
    Prms.create
      ~above:(Tensor.f (-7.))
      Tensor.(zeros ~device:base.device ~kind:base.kind [ 1 ])
  in
  PP.{ c; sigma_o_prms }

let sample_data () =
  let us = Tensor.(randn ~device:base.device ~kind:base.kind [ bs; m ]) |> Maths.const in
  let xs = Maths.(us *@ true_theta.c) in
  let ys =
    Maths.(xs + (exp true_theta.sigma_o_prms * const (Tensor.randn_like (primal xs))))
  in
  us, ys

let solver a y =
  let ell = Maths.cholesky a in
  let ell_t = Maths.transpose ~dim0:0 ~dim1:1 ell in
  let _x = Maths.linsolve_triangular ~left:false ~upper:true ell_t y in
  Maths.linsolve_triangular ~left:false ~upper:false ell _x

let u_opt ~c ~sigma_o y =
  let a = Maths.(einsum [ c, "ji"; c, "jk" ] "ik") in
  let a = Maths.(a + diag_embed ~offset:0 ~dim1:0 ~dim2:1 (sqr sigma_o)) in
  let solution = solver a y in
  Maths.(einsum [ c, "ij"; solution, "mj" ] "mi")

let d_opt_inv ~c ~sigma_o =
  let sigma_o_inv_vec =
    Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ m ]) / sqr sigma_o)
  in
  let cct = Maths.(einsum [ c, "ij"; c, "kj" ] "ik") in
  let a = Maths.(einsum [ cct, "ik"; sigma_o_inv_vec, "k" ] "ik") in
  let d_opt_inv = Maths.(const (Tensor.eye ~n:m ~options:(base.kind, base.device)) + a) in
  d_opt_inv

let gaussian_llh ?mu ?(fisher_batched = false) ~std x =
  let inv_std = Maths.(f 1. / std) in
  let error_term =
    if fisher_batched
    then (
      (* batch dimension l is number of fisher samples *)
      let error =
        match mu with
        | None -> Maths.(einsum [ x, "lma"; inv_std, "a" ] "lma")
        | Some mu -> Maths.(einsum [ x - mu, "lma"; inv_std, "a" ] "lma")
      in
      Maths.einsum [ error, "lma"; error, "lma" ] "lm")
    else (
      let error =
        match mu with
        | None -> Maths.(einsum [ x, "ma"; inv_std, "a" ] "ma")
        | Some mu -> Maths.(einsum [ x - mu, "ma"; inv_std, "a" ] "ma")
      in
      Maths.einsum [ error, "ma"; error, "ma" ] "m")
  in
  let cov_term =
    let cov_term_shape = if fisher_batched then [ 1; 1 ] else [ 1 ] in
    Maths.(sum (log (sqr std))) |> Maths.reshape ~shape:cov_term_shape
  in
  let const_term =
    let o = x |> Maths.primal |> Tensor.shape |> List.last_exn in
    Float.(log (2. * pi) * of_int o)
  in
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

let fisher ?(fisher_batched = false) ~n:_ lik_term =
  let neg_lik_t = Maths.(tangent lik_term) |> Option.value_exn in
  let n_tangents = List.hd_exn (Tensor.shape neg_lik_t) in
  let fisher =
    if fisher_batched
    then (
      let fisher_half = Tensor.reshape neg_lik_t ~shape:[ n_tangents; n_fisher; -1 ] in
      Tensor.einsum ~equation:"kla,jla->lkj" [ fisher_half; fisher_half ] ~path:None)
    else (
      let fisher_half = Tensor.reshape neg_lik_t ~shape:[ n_tangents; -1 ] in
      Tensor.(matmul fisher_half (transpose fisher_half ~dim0:0 ~dim1:1)))
  in
  fisher

let ggn ~y_pred ~std_o =
  let precision = Maths.(f 1. / sqr std_o) in
  let ggn_y =
    let vtgt = Maths.tangent y_pred |> Option.value_exn in
    let vtgt_h = Tensor.(vtgt * Maths.(primal precision)) in
    Tensor.(
      einsum ~equation:"kma,jma->kj" [ vtgt_h; vtgt ] ~path:None / f (Float.of_int o))
  in
  let ggn_sigma_o =
    let vtgt = Maths.tangent precision |> Option.value_exn in
    let vtgt_h =
      Tensor.(f Float.(of_int bs / 2.) * vtgt / square (Maths.primal precision))
    in
    Tensor.einsum ~equation:"ka,ja->kj" [ vtgt_h; vtgt ] ~path:None
  in
  Tensor.(ggn_y + ggn_sigma_o)

(* ggn matrix when loss is the negative marginal log likelihood *)
let ggn_marginal ~marginal_cov_chol =
  let marginal_cov_chol_inv =
    Maths.linsolve_triangular marginal_cov_chol id_o ~left:true ~upper:false
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
  Tensor.(
    f Float.(of_int bs / (2. * of_int o))
    * einsum [ ggn_half; ggn_half ] ~path:None ~equation:"kop,jpo->kj")

module M = struct
  module P = P

  type data = Maths.t
  type args = unit

  let kl ~u_sampled ~u_opt ~u_diff ~d_opt_chol =
    if sampling
    then (
      let prior_term = gaussian_llh ~std:ones_u u_sampled in
      let q_term = gaussian_llh_chol ~chol:d_opt_chol u_diff in
      Maths.(q_term - prior_term))
    else (
      let d_opt_chol_diag = Maths.diagonal d_opt_chol ~offset:0 in
      let det1 = Maths.(2. $* sum (log d_opt_chol_diag)) in
      let _const = Maths.const (Tensor.f Float.(of_int m)) in
      let tr = d_opt_chol_diag |> Maths.sqr |> Maths.sum in
      let quad =
        Maths.(einsum [ u_opt, "mb"; u_opt, "mb" ] "m") |> Maths.reshape ~shape:[ -1 ]
      in
      let tmp = Maths.(tr - _const - det1) |> Maths.reshape ~shape:[ 1 ] in
      Maths.(0.5 $* tmp + quad))

  let neg_elbo ~y_pred ~y ~u_sampled ~u_opt ~u_diff ~d_opt_chol (theta : P.M.t) =
    let sigma_o = Maths.exp theta.sigma_o_prms in
    let sigma_o_extended = Maths.(sigma_o * ones_o) in
    let llh = gaussian_llh ~mu:y_pred ~std:sigma_o_extended y in
    let kl = kl ~u_sampled ~u_opt ~u_diff ~d_opt_chol in
    Maths.(llh - kl) |> Maths.neg

  let marginal_cov_chol (theta : P.M.t) =
    let sigma_o = Maths.exp theta.sigma_o_prms in
    let sigma_o_extended = Maths.(sigma_o * ones_o) in
    Maths.(
      (transpose theta.c ~dim0:1 ~dim1:0 *@ theta.c)
      + diag_embed ~offset:0 ~dim1:0 ~dim2:1 (sqr sigma_o_extended))
    |> Maths.cholesky

  let neg_marginal_llh ~marginal_cov_chol ~y =
    gaussian_llh_chol ~chol:marginal_cov_chol y |> Maths.neg

  let d_opt_chol (theta : P.M.t) =
    let sigma_o = Maths.exp theta.sigma_o_prms in
    let d_opt_inv = d_opt_inv ~c:theta.c ~sigma_o in
    let d_opt_inv_chol = Maths.cholesky d_opt_inv in
    Maths.linsolve
      d_opt_inv_chol
      (Maths.const (Tensor.eye ~n:m ~options:(base.kind, base.device)))
      ~left:true

  let f ~update ~data:y ~init ~args:() (theta : P.M.t) =
    (* sample u *)
    let d_opt_chol = d_opt_chol theta in
    let u_opt = u_opt ~c:theta.c ~sigma_o:Maths.(exp theta.sigma_o_prms * ones_o) y in
    let u_diff =
      let e = Maths.(const (Tensor.randn_like (primal u_opt))) in
      Maths.einsum [ e, "mj"; d_opt_chol, "ij" ] "mi"
    in
    let u_sampled =
      (* Maths.(const (primal_tensor_detach (u_opt + u_diff))) *)
      Maths.(u_opt + u_diff)
    in
    let y_pred = Maths.(u_sampled *@ theta.c) in
    let marginal_cov_chol = marginal_cov_chol theta in
    let loss =
      let tmp =
        if marginal_llh
        then neg_marginal_llh ~marginal_cov_chol ~y
        else neg_elbo ~y_pred ~y ~u_sampled ~u_opt ~u_diff ~d_opt_chol (theta : P.M.t)
      in
      Maths.(tmp / f Float.(of_int o))
    in
    match update with
    | `loss_only u -> u init (Some loss)
    | `loss_and_ggn u ->
      let preconditioner =
        if marginal_ggn
        then ggn_marginal ~marginal_cov_chol
        else ggn ~y_pred ~std_o:(Maths.exp theta.sigma_o_prms)
      in
      let _, final_s, _ = Tensor.svd ~some:true ~compute_uv:false preconditioner in
      final_s
      |> Tensor.reshape ~shape:[ -1; 1 ]
      |> Tensor.to_bigarray ~kind:base.ba_kind
      |> Owl.Mat.save_txt ~out:(in_dir (sprintf "svals"));
      u init (Some (loss, Some preconditioner))
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
  val config_f : iter:int -> (float, Bigarray.float64_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state ~time_elapsed running_avg =
      Stdlib.Gc.major ();
      let _, y = sample_data () in
      let config = config_f ~iter in
      let t0 = Unix.gettimeofday () in
      let loss, new_state = O.step ~config ~state ~data:y () in
      let std_o_mean =
        let a = M.P.value (O.params state) in
        a.sigma_o_prms |> Tensor.mean |> Tensor.to_float0_exn
      in
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
          (* learned cct *)
          let _ =
            let c =
              let a = M.P.value (O.params state) in
              a.c
            in
            Tensor.(matmul (transpose c ~dim0:0 ~dim1:1) c)
            |> Tensor.reshape ~shape:[ -1; 1 ]
            |> Tensor.to_bigarray ~kind:base.ba_kind
            |> Owl.Mat.save_txt ~out:(in_dir (sprintf "learned_cct"))
          in
          (* avg error *)
          Convenience.print [%message (iter : int) (loss_avg : float)];
          let t = iter in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array [| Float.of_int t; time_elapsed; loss_avg; std_o_mean |] 1 4));
          O.W.P.T.save
            (M.P.value (O.params new_state))
            ~kind:base.ba_kind
            ~out:(in_dir name ^ "_params"));
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
  module O = Optimizer.SOFO (M) (GGN)

  let config_f ~iter =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some Float.(0.3 / (1. +. (0. * sqrt (of_int iter))))
      ; n_tangents = 16
      ; rank_one = false
      ; damping = None
      ; aux= None
      }

  let init = O.init theta

  let name =
    let init_config = config_f ~iter:0 in
    let gamma_name =
      Option.value_map init_config.damping ~default:"none" ~f:Float.to_string
    in
    sprintf
      "ggn_lr_%s_damp_%s"
      (Float.to_string (Option.value_exn init_config.learning_rate))
      gamma_name
end

(* --------------------------------
   -- Adam
   -------------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (M)

  let config_f ~iter:_ =
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
