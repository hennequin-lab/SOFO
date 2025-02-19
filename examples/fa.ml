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

<<<<<<< Updated upstream
type pred_cond =
  | GGN
  | EmpFisher
  | TrueFisher

let pred_cond = GGN
let sampling = false
let matheron = true
let n_fisher = 30
let m = 10
let o = 40
let bs = 64
=======
let m = 10
let o = 40
let bs = 32
let id_m = Maths.(const (Tensor.of_bigarray ~device:base.device (Owl.Mat.eye m)))
>>>>>>> Stashed changes
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
  let sigma_o_prms = Maths.(log (f 0.01)) in
  let c = make_c ~sigma:1. (m, o) in
  PP.{ sigma_o_prms; c }

let theta =
  let c = Prms.free (Maths.primal (make_c ~sigma:1. (m, o))) in
  let sigma_o_prms =
    Prms.create
      ~below:(Tensor.f 5.)
      Tensor.(f 0. + zeros ~device:base.device ~kind:base.kind [ 1 ])
  in
  PP.{ c; sigma_o_prms }

let sample_data =
<<<<<<< Updated upstream
  let sigma_o = Maths.(f 0.001) in
  let c = make_c (m, o) in
=======
  let sigma_o = Maths.(exp true_theta.sigma_o_prms) in
  let c = true_theta.c in
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
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
=======
let u_opt ~c ~inv_sigma_o y =
  let r = Maths.(sqr inv_sigma_o) in
  let a = Maths.(id_m + einsum [ c, "ij"; r, "j"; c, "kj" ] "ik") in
  let b = Maths.einsum [ c, "ij"; r, "j"; y, "mj" ] "mi" in
  solver a b

let gaussian_llh_chol ?mu ~precision_chol:ell x =
  let error =
    match mu with
    | None -> x
    | Some mu -> Maths.(x - mu)
>>>>>>> Stashed changes
  in
  let error_term = Maths.einsum [ error, "ma"; ell, "ai"; ell, "bi"; error, "mb" ] "m" in
  let cov_term =
    Maths.(sum (log (sqr (diagonal ~offset:0 ell))) |> reshape ~shape:[ 1 ])
  in
  let const_term =
    let o = x |> Maths.primal |> Tensor.shape |> List.last_exn in
    Float.(log (2. * pi) * of_int o)
  in
  Maths.(0.5 $* (const_term $+ error_term - cov_term)) |> Maths.neg

let gaussian_llh ?mu ~inv_std x =
  let error_term =
    let error =
      match mu with
      | None -> Maths.(einsum [ x, "ma"; inv_std, "a" ] "ma")
      | Some mu -> Maths.(einsum [ x - mu, "ma"; inv_std, "a" ] "ma")
    in
    Maths.einsum [ error, "ma"; error, "ma" ] "m"
  in
  let cov_term = Maths.(neg (sum (log (sqr inv_std))) |> reshape ~shape:[ 1 ]) in
  let const_term =
    let o = x |> Maths.primal |> Tensor.shape |> List.last_exn in
    Float.(log (2. * pi) * of_int o)
  in
  Maths.(0.5 $* (const_term $+ error_term + cov_term)) |> Maths.neg

<<<<<<< Updated upstream
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
    Tensor.einsum ~equation:"kma,jma->kj" [ vtgt_h; vtgt ] ~path:None
  in
  let ggn_sigma_o =
    let vtgt = Maths.tangent precision |> Option.value_exn in
    let vtgt_h =
      Tensor.(
        f Float.(of_int o * of_int bs / 2.) * vtgt / square (Maths.primal precision))
    in
    Tensor.einsum ~equation:"ka,ja->kj" [ vtgt_h; vtgt ] ~path:None
  in
  Tensor.(ggn_y + ggn_sigma_o)

let ggn_natural ~y_pred ~std_o =
  let _mean = y_pred in
  let beta = Maths.(f 1. / sqr std_o) in
  let alpha = Maths.einsum [ Maths.unsqueeze ~dim:2 _mean, "mdc"; beta, "c" ] "md" in
  let alpha_pri = Maths.primal alpha in
  let beta_pri = Maths.primal beta in
  let alpha_sum =
    Tensor.einsum ~equation:"mo,mo->m" [ alpha_pri; alpha_pri ] ~path:None |> Tensor.sum
  in
  let h_11 = Tensor.(f 1. / beta_pri) in
  let h_12 = Tensor.(neg alpha_pri / square beta_pri) in
  let h_22 =
    Tensor.(
      ((f (Float.of_int Int.(o * bs)) / (f 2. * beta_pri)) + (alpha_sum / square beta_pri))
      / beta_pri)
  in
  let alpha_t = Maths.tangent alpha |> Option.value_exn in
  let beta_t = Maths.tangent beta |> Option.value_exn |> Tensor.squeeze in
  (* H JV *)
  let tmp1 =
    Tensor.((alpha_t * h_11) + einsum ~equation:"md,k->kmd" [ h_12; beta_t ] ~path:None)
  in
  let tmp2 =
    Tensor.(einsum ~equation:"md,kmd->k" [ h_12; alpha_t ] ~path:None + (h_22 * beta_t))
  in
  (* J^T V^T H JV *)
  let tmp3 = Tensor.einsum ~equation:"kmd,jmd->kj" [ alpha_t; tmp1 ] ~path:None in
  let tmp4 = Tensor.einsum ~equation:"k,j->kj" [ beta_t; tmp2 ] ~path:None in
  Tensor.(tmp3 + tmp4)
=======
let elbo ~data:y (theta : P.M.t) =
  let sigma_o = Maths.(exp theta.sigma_o_prms) in
  let inv_sigma_o = Maths.(exp (neg theta.sigma_o_prms)) in
  let inv_sigma_o_expanded = Maths.(inv_sigma_o * ones_o) in
  let u_opt = u_opt ~c:theta.c ~inv_sigma_o:inv_sigma_o_expanded y in
  let post_prec_chol =
    posterior_precision_chol ~c:theta.c ~inv_sigma_o:inv_sigma_o_expanded
  in
  let u_diff =
    let e = Maths.(const (Tensor.randn_like (primal u_opt))) in
    let ell = post_prec_chol in
    let ell_t = Maths.transpose ~dim0:0 ~dim1:1 ell in
    let _x = Maths.linsolve_triangular ~left:false ~upper:true ell_t e in
    Maths.linsolve_triangular ~left:false ~upper:false ell _x
  in
  let u_sampled = Maths.(const (primal (u_opt + u_diff))) in
  (* m x o *)
  let y_pred = Maths.(u_sampled *@ theta.c) in
  let lik_term = gaussian_llh ~mu:y_pred ~inv_std:inv_sigma_o_expanded y in
  let kl_term =
    Maths.f 0.
    (*
       let prior_term = gaussian_llh ~inv_std:ones_u u_sampled in
    let q_term = gaussian_llh_chol ~precision_chol:post_prec_chol u_diff in
    Maths.(const (primal (q_term - prior_term))) *)
  in
  let neg_elbo =
    Maths.(lik_term - kl_term) |> Maths.neg |> fun x -> Maths.(x / f Float.(of_int o))
  in
  neg_elbo, y_pred, sigma_o
>>>>>>> Stashed changes

module M = struct
  module P = P

  type data = Maths.t
  type args = unit

  let f ~update ~data:y ~init ~args:() (theta : P.M.t) =
<<<<<<< Updated upstream
    let sigma_o = Maths.exp theta.sigma_o_prms in
    let sigma_o_extended = Maths.(sigma_o * ones_o) in
    let d_opt_chol =
      let d_opt_inv = d_opt_inv ~c:theta.c ~sigma_o in
      let d_opt_inv_chol = Maths.cholesky d_opt_inv in
      Maths.linsolve
        d_opt_inv_chol
        (Maths.const (Tensor.eye ~n:m ~options:(base.kind, base.device)))
        ~left:true
    in
    let u_opt = u_opt ~c:theta.c ~sigma_o:sigma_o_extended y in
    let u_diff =
      let e = Maths.(const (Tensor.randn_like (primal u_opt))) in
      Maths.einsum [ e, "mj"; d_opt_chol, "ij" ] "mi"
    in
    let u_sampled =
      if matheron
      then Maths.(const (primal_tensor_detach (u_opt + u_diff)))
      else Maths.(u_opt + u_diff)
    in
    let y_pred = Maths.(u_sampled *@ theta.c) in
    let lik_term = gaussian_llh ~mu:y_pred ~std:sigma_o_extended y in
    let kl_term =
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
    in
    let neg_elbo =
      (if matheron
       then
         Maths.(lik_term )
         (* then Maths.(lik_term - const (primal_tensor_detach kl_term)) *)
       else Maths.(lik_term))
      |> Maths.neg
      |> fun x -> Maths.(x / f Float.(of_int o))
    in
=======
    let neg_elbo, y_pred, sigma_o = elbo ~data:y theta in
>>>>>>> Stashed changes
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let preconditioner =
<<<<<<< Updated upstream
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
        | GGN -> ggn ~y_pred ~std_o:sigma_o
        (* ggn_natural ~_mean:y_pred ~std:sigma_o *)
=======
        let sigma2 = Maths.sqr sigma_o in
        let sigma2_p = Maths.(const (primal sigma2)) in
        let sigma2_t = Maths.tangent sigma2 |> Option.value_exn |> Maths.const in
        let mu_t = Maths.tangent y_pred |> Option.value_exn |> Maths.const in
        let ggn_part1 = Maths.(einsum [ mu_t, "kmo"; mu_t, "lmo" ] "kl" / sigma2_p) in
        let ggn_part2 =
          Maths.(
            einsum
              [ f Float.(0.5 * of_int o * of_int bs) * sigma2_t / sqr sigma2_p, "ky"
              ; sigma2_t, "ly"
              ]
              "kl")
        in
        Maths.(primal ((ggn_part1 + ggn_part2) / f Float.(of_int o)))
>>>>>>> Stashed changes
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
    let rec loop ~iter ~state ~time_elapsed running_avg =
      Stdlib.Gc.major ();
      let _, y = sample_data () in
      let t0 = Unix.gettimeofday () in
<<<<<<< Updated upstream
      let loss, new_state = O.step ~config ~state ~data:y ~args:() in
      let std_o_mean =
        let a = M.P.value (O.params state) in
        a.sigma_o_prms |> Tensor.mean |> Tensor.to_float0_exn
      in
=======
      let loss, new_state = O.step ~config:(config ~iter) ~state ~data:y ~args:() in
>>>>>>> Stashed changes
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
          let theta = O.params new_state |> O.W.P.value |> O.W.P.const in
          let ground_truth_elbo, _, _ = elbo ~data:y true_theta in
          let ground_truth_elbo =
            Maths.primal ground_truth_elbo |> Tensor.mean |> Tensor.to_float0_exn
          in
          let sigma2_o =
            Maths.exp theta.sigma_o_prms
            |> Maths.sum
            |> Maths.primal
            |> Tensor.to_float0_exn
          in
          (* avg error *)
          Convenience.print [%message (iter : int) (loss_avg : float)];
          let t = iter in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
<<<<<<< Updated upstream
              (of_array [| Float.of_int t; time_elapsed; loss_avg; std_o_mean |] 1 4));
          O.W.P.T.save
            (M.P.value (O.params new_state))
            ~kind:base.ba_kind
            ~out:(in_dir name ^ "_params"));
=======
              (of_array
                 [| Float.of_int t; time_elapsed; loss_avg; ground_truth_elbo; sigma2_o |]
                 1
                 5)));
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
  let config =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 50.
      ; n_tangents = 64
      ; sqrt = false
      ; rank_one = false
      ; damping = None
=======
  let name = "sofo"

  let config ~iter:_ =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some Float.(0.2)
      ; n_tangents = 256
      ; sqrt = false
      ; rank_one = true
      ; damping = Some 0.0001
>>>>>>> Stashed changes
      ; momentum = None
      ; lm = false
      ; perturb_thresh = None
      }

<<<<<<< Updated upstream
  let init = O.init ~config theta

  let name =
    match pred_cond with
    | TrueFisher -> "true_fisher"
    | EmpFisher -> "emp_fisher"
    | GGN ->
      let gamma_name =
        Option.value_map config.damping ~default:"none" ~f:Float.to_string
      in
      sprintf
        "ggn_lr_%s_damp_%s_k_%s_matheron_no_kl"
        (Float.to_string (Option.value_exn config.learning_rate))
        gamma_name
        (Int.to_string config.n_tangents)
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
=======
  let init = O.init ~config:(config ~iter:0) theta
>>>>>>> Stashed changes
end

(* --------------------------------
   -- Adam
   -------------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam_matheron_no_kl"

  module O = Optimizer.Adam (M)

  let config ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.01 }

  let init = O.init theta
end

let _ =
<<<<<<< Updated upstream
  let max_iter = 2000 in
=======
  let max_iter = 10000 in
>>>>>>> Stashed changes
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
