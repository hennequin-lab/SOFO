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

type pred_cond =
  | GGN
  | EmpFisher
  | TrueFisher

let pred_cond = GGN
let sampling = false
let n_fisher = 30
let m = 10
let o = 40
let bs = 128
let ones_o = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
let ones_u = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ m ]))

let make_c (m, o) =
  Tensor.(
    f Float.(1. /. sqrt (of_int m)) * randn ~device:base.device ~kind:base.kind [ m; o ])
  |> Maths.const

module PP = struct
  type 'a p =
    { c : 'a
    ; sigma_o_prms : 'a
    ; d_prms : 'a
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

let theta =
  let c = Prms.free (Maths.primal (make_c (m, o))) in
  let sigma_o_prms =
    Prms.create
      ~above:(Tensor.f (-5.))
      Tensor.(zeros ~device:base.device ~kind:base.kind [ 1 ])
  in
  let d_prms = Prms.create ~above:(Tensor.f (-5.)) Maths.(primal (f (-2.) * ones_u)) in
  PP.{ c; sigma_o_prms; d_prms }

let sample_data =
  let sigma_o = Maths.(f 0.1) in
  let c = make_c (m, o) in
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

let u_opt ~c ~sigma_o y =
  let a = Maths.(einsum [ c, "ji"; c, "jk" ] "ik") in
  let a = Maths.(a + diag_embed ~offset:0 ~dim1:0 ~dim2:1 (sqr sigma_o)) in
  let solution = solver a y in
  Maths.(einsum [ c, "ij"; solution, "mj" ] "mi")

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

let ggn ~like_hess ~vtgt =
  let vtgt_hess = Tensor.einsum ~equation:"kma,a->kma" [ vtgt; like_hess ] ~path:None in
  Tensor.einsum ~equation:"kma,jma->kj" [ vtgt_hess; vtgt ] ~path:None

module M = struct
  module P = P

  type data = Maths.t
  type args = unit

  let f ~update ~data:y ~init ~args:() (theta : P.M.t) =
    let sigma_o = Maths.exp theta.sigma_o_prms in
    let precision = Maths.(f 1. / sqr sigma_o) in
    let sigma_o_extended = Maths.(sigma_o * ones_o) in
    let d = Maths.(exp theta.d_prms) in
    let u_opt = u_opt ~c:theta.c ~sigma_o:sigma_o_extended y in
    let u_diff = Maths.(const (Tensor.randn_like (primal u_opt)) * unsqueeze ~dim:0 d) in
    let u_sampled = Maths.(u_opt + u_diff) in
    let y_pred = Maths.(u_sampled *@ theta.c) in
    let lik_term = gaussian_llh ~mu:y_pred ~std:sigma_o_extended y in
    let kl_term =
      if sampling
      then (
        let prior_term = gaussian_llh ~std:ones_u u_sampled in
        let q_term = gaussian_llh ~std:d u_diff in
        Maths.(q_term - prior_term))
      else (
        let det1 = Maths.(2. $* sum theta.d_prms) in
        let _const = Maths.const (Tensor.f Float.(of_int m)) in
        let tr = d |> Maths.sqr |> Maths.sum in
        let quad =
          Maths.(einsum [ u_opt, "mb"; u_opt, "mb" ] "m") |> Maths.reshape ~shape:[ -1 ]
        in
        let tmp = Maths.(tr - _const - det1) |> Maths.reshape ~shape:[ 1 ] in
        Maths.(0.5 $* tmp + quad))
    in
    let neg_elbo =
      Maths.(lik_term - kl_term) |> Maths.neg |> fun x -> Maths.(x / f Float.(of_int o))
    in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let preconditioner =
        match pred_cond with
        | TrueFisher ->
          let true_fisher =
            let y_pred_unsqueezed =
              let tmp = Maths.unsqueeze ~dim:0 y_pred in
              List.init n_fisher ~f:(fun _ -> tmp) |> Maths.concat_list ~dim:0
            in
            let y_pred_primal = Maths.primal y_pred |> Tensor.unsqueeze ~dim:0 in
            let sigma_extended =
              sigma_o_extended
              |> Maths.primal
              |> Tensor.unsqueeze ~dim:0
              |> Tensor.unsqueeze ~dim:0
            in
            let y_samples_batched =
              let noise =
                Tensor.(
                  sigma_extended
                  * Tensor.(
                      randn
                        (n_fisher :: Maths.shape y)
                        ~device:base.device
                        ~kind:base.kind))
              in
              Maths.(const Tensor.(y_pred_primal + noise))
            in
            let lik_term_sampled_batched =
              gaussian_llh
                ~mu:y_pred_unsqueezed
                ~std:sigma_o_extended
                ~fisher_batched:true
                y_samples_batched
            in
            let fisher = fisher ~n:o lik_term_sampled_batched ~fisher_batched:true in
            Tensor.mean_dim fisher ~dim:(Some [ 0 ]) ~keepdim:false ~dtype:base.kind
          in
          true_fisher
        | EmpFisher -> fisher ~n:o lik_term
        | GGN ->
          let ggn_y =
            let vtgt = Maths.tangent y_pred |> Option.value_exn in
            let vtgt_h = Tensor.(vtgt * Maths.(primal precision)) in
            Tensor.einsum ~equation:"kma,jma->kj" [ vtgt_h; vtgt ] ~path:None
          in
          let ggn_sigma_o =
            let vtgt = Maths.tangent precision |> Option.value_exn in
            let vtgt_h =
              Tensor.(
                f Float.(of_int o * of_int bs / 2.)
                * vtgt
                / square (Maths.primal precision))
            in
            Tensor.einsum ~equation:"ka,ja->kj" [ vtgt_h; vtgt ] ~path:None
          in
          Tensor.(ggn_y + ggn_sigma_o)
      in
      let _, final_s, _ = Tensor.svd ~some:true ~compute_uv:false preconditioner in
      final_s
      |> Tensor.reshape ~shape:[ -1; 1 ]
      |> Tensor.to_bigarray ~kind:base.ba_kind
      |> Owl.Mat.save_txt ~out:(in_dir (sprintf "svals"));
      u init (Some (neg_elbo, Some preconditioner))
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
  val config : (float, Bigarray.float64_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state ~time_elapsed running_avg =
      Stdlib.Gc.major ();
      let _, y = sample_data () in
      let t0 = Unix.gettimeofday () in
      let loss, new_state = O.step ~config ~state ~data:y ~args:() in
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
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array [| Float.of_int t; time_elapsed; loss_avg |] 1 3));
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
  module O = Optimizer.SOFO (M)

  let name =
    match pred_cond with
    | TrueFisher -> "true_fisher"
    | EmpFisher -> "emp_fisher"
    | GGN -> "ggn"

  let config =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 30.
      ; n_tangents = 64
      ; sqrt = false
      ; rank_one = false
      ; damping = Some 1e-3
      ; momentum = None
      ; lm = false
      ; perturb_thresh = None
      }

  let init = O.init ~config theta
end

(* --------------------------------
   -- FGD
   -------------------------------- *)

module Do_with_FGD : Do_with_T = struct
  module O = Optimizer.FGD (M)

  let name = "fgd"

  let config =
    Optimizer.Config.FGD.
      { default with base; n_tangents = 256; learning_rate = Some 0.03 }

  let init = O.init ~config theta
end

(* --------------------------------
   -- Adam
   -------------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (M)

  let config = Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.01 }
  let init = O.init theta
end

let _ =
  let max_iter = 2000 in
  let optimise =
    match Cmdargs.get_string "-m" with
    | Some "sofo" ->
      let module X = Make (Do_with_SOFO) in
      X.optimise
    | Some "fgd" ->
      let module X = Make (Do_with_FGD) in
      X.optimise
    | Some "adam" ->
      let module X = Make (Do_with_Adam) in
      X.optimise
    | _ -> failwith "-m [sofo | fgd | adam]"
  in
  optimise max_iter
