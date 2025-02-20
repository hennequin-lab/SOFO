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
let n = 20
let o = 40
let tmax = 130
let bs = 32
let id_m = Maths.(const (Tensor.of_bigarray ~device:base.device (Owl.Mat.eye m)))
let ones_o = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
let ones_u = Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ m ]))

let make_a () =
  let a =
    let open Owl in
    let w =
      let w = Mat.(gaussian n n) in
      let sa = Linalg.D.eigvals w |> Dense.Matrix.Z.re |> Mat.max' in
      Mat.(Float.(0.8 / sa) $* w)
    in
    (*     let w = Mat.(0.5 $* w + transpose w) in *)
    Mat.(add_diag (0.1 $* w) 0.9)
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
    ; sigma_o_prms : 'a
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

let true_theta =
  let a = make_a () in
  let b = make_b () in
  let c = make_c () in
  let sigma_o_prms = Maths.(log (f 0.001)) in
  PP.{ a; b; c; sigma_o_prms }

let theta =
  let a = Prms.free (Maths.primal (make_a ())) in
  let b = Prms.free (Maths.primal (make_b ())) in
  let c = Prms.free (Maths.primal (make_c ())) in
  let sigma_o_prms =
    Prms.create
      ~below:(Tensor.f 5.)
      Tensor.(f 0. + zeros ~device:base.device ~kind:base.kind [ 1 ])
  in
  PP.{ a; b; c; sigma_o_prms }

let solver a y =
  let ell = Maths.cholesky a in
  let ell_t = Maths.transpose ~dim0:0 ~dim1:1 ell in
  let _x = Maths.linsolve_triangular ~left:false ~upper:true ell_t y in
  Maths.linsolve_triangular ~left:false ~upper:false ell _x

let lqr ~a ~b ~c ~inv_sigma_o y =
  let open Maths in
  (* augment observations with dummy y0 *)
  let _T = List.length y in
  let y = f 0. :: y in
  let _Czz = einsum [ c, "ij"; c, "kj" ] "ik" * sqr inv_sigma_o in
  let _cz_fun y = neg (einsum [ c, "ji"; y, "mi" ] "mj") * sqr inv_sigma_o in
  let yT = List.last_exn y in
  let gains, _, _, _ =
    List.fold_right
      (List.sub y ~pos:0 ~len:_T)
      ~init:([], _cz_fun yT, _Czz, Int.(_T - 1))
      ~f:(fun y (accu, _v, _V, t) ->
        (* we only have state costs for t>=1 *)
        let _cz = if t > 0 then _cz_fun y else f 0. in
        let _Czz = if t > 0 then _Czz else f 0. in
        let _Qzz = _Czz + einsum [ a, "ji"; _V, "jl"; a, "lk" ] "ik" in
        let _Quz = einsum [ b, "ij"; _V, "jl"; a, "lk" ] "ki" in
        (* or maybe ik *)
        let _Quu = einsum [ b, "ij"; _V, "jl"; b, "kl" ] "ik" in
        let _qz = _cz + einsum [ a, "ji"; _v, "mj" ] "mi" in
        let _qu = einsum [ b, "ij"; _v, "mj" ] "mi" in
        let _K = neg (solver _Quu _Quz) in
        let _k = neg (solver _Quu _qu) in
        let _V = _Qzz + einsum [ _Quz, "ki"; _K, "ji" ] "kj" in
        let _v = _qz + einsum [ _qu, "mi"; _K, "ji" ] "mj" in
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
        let z = einsum [ a, "ji"; z, "mi" ] "mj" + einsum [ b, "ij"; u, "mi" ] "mj" in
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
        let z' = einsum [ a, "ji"; z, "mi" ] "mj" + einsum [ b, "ij"; u, "mi" ] "mj" in
        y_of z' :: accu, z')
  in
  List.rev y

let sample_data =
  let sigma_o = Maths.(exp true_theta.sigma_o_prms) in
  fun () ->
    let u =
      List.init tmax ~f:(fun _ ->
        Tensor.(randn ~device:base.device ~kind:base.kind [ bs; m ]) |> Maths.const)
    in
    let y =
      rollout ~a:true_theta.a ~b:true_theta.b ~c:true_theta.c u
      |> List.map ~f:(fun y ->
        Maths.(y + (sigma_o * const (Tensor.randn_like (Maths.primal y)))))
    in
    Convenience.print [%message (List.length u : int) (List.length y : int)];
    u, y

let gaussian_llh_chol ?mu ~precision_chol:ell x =
  let d = x |> Maths.primal |> Tensor.shape |> List.last_exn in
  let error =
    match mu with
    | None -> x
    | Some mu -> Maths.(x - mu)
  in
  let error_term = Maths.einsum [ error, "ma"; ell, "ai"; ell, "bi"; error, "mb" ] "m" in
  let cov_term =
    Maths.(sum (log (sqr (diagonal ~offset:0 ell))) |> reshape ~shape:[ 1 ])
  in
  let const_term = Float.(log (2. * pi) * of_int d) in
  Maths.(0.5 $* (const_term $+ error_term - cov_term)) |> Maths.neg

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

(* small test *)
let save_time_series ~out x =
  List.map x ~f:(fun x ->
    Maths.primal x
    |> Tensor.to_bigarray ~kind:base.ba_kind
    |> Owl.Mat.get_slice [ [ 0 ] ]
    |> fun x -> Owl.Mat.reshape x [| 1; -1 |])
  |> List.to_array
  |> Owl.Mat.concatenate ~axis:0
  |> Owl.Mat.save_txt ~out

let u, y = sample_data ()
let _ = save_time_series ~out:(in_dir "u") u
let _ = save_time_series ~out:(in_dir "y") y

let u_recov =
  let inv_sigma_o = Maths.(exp (neg true_theta.sigma_o_prms)) in
  lqr ~a:true_theta.a ~b:true_theta.b ~c:true_theta.c ~inv_sigma_o y

let _ = save_time_series ~out:(in_dir "urecov") u_recov

let err =
  List.fold2_exn
    u
    u_recov
    ~init:Maths.(f 0.)
    ~f:(fun accu u u_recov -> Maths.(accu + sum (sqr (u - u_recov))))
  |> Maths.primal
  |> Tensor.to_float0_exn

let _ = Convenience.print [%message (err : float)]
(*
   let elbo ~data:(y : Maths.t list) (theta : P.M.t) =
  let sigma_o = Maths.(exp theta.sigma_o_prms) in
  let inv_sigma_o = Maths.(exp (neg theta.sigma_o_prms)) in
  let inv_sigma_o_expanded = Maths.(inv_sigma_o * ones_o) in
  let u_opt = lqr ~a:theta.a ~b:theta.b ~c:theta.c ~inv_sigma_o:inv_sigma_o_expanded y in
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

module M = struct
  module P = P

  type data = Maths.t
  type args = unit

  let f ~update ~data:y ~init ~args:() (theta : P.M.t) =
    let neg_elbo, y_pred, sigma_o = elbo ~data:y theta in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let preconditioner =
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
    let rec loop ~iter ~state running_avg =
      Stdlib.Gc.major ();
      let _, y = sample_data () in
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
              (of_array [| Float.of_int t; loss_avg; ground_truth_elbo; sigma2_o |] 1 4)));
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
      ; learning_rate = Some Float.(0.2)
      ; n_tangents = 256
      ; sqrt = false
      ; rank_one = true
      ; damping = Some 0.0001
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
*)
