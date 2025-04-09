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

(* list of length T of [m x b] to matrix of [m x b x T]*)
let concat_time u_list =
  List.map u_list ~f:(fun u -> Maths.unsqueeze ~dim:(-1) u) |> Maths.concat_list ~dim:2

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
let std_of_log_var log_var = Maths.(exp (f 0.5 * log_var))
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

let x0 = Tensor.zeros ~device:Dims.device ~kind:Dims.kind [ Dims.bs; Dims.n ]
let eye_m = Maths.(const (Tensor.eye ~n:Dims.m ~options:(Dims.kind, Dims.device)))
let ones_u = Maths.(const (Tensor.ones ~device:Dims.device ~kind:Dims.kind [ Dims.m ]))

let ones_tmax =
  Maths.(const (Tensor.ones ~device:Dims.device ~kind:Dims.kind [ Dims.tmax ]))

let ones_o = Maths.(const (Tensor.ones ~device:Dims.device ~kind:Dims.kind [ Dims.o ]))
let sample = false

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
let _Fu_true = make_fu ()
let _c_true = make_c ()
let _obs_var_true = Maths.const Tensor.(square (f 0.1))

let sample_data () =
  let sigma = Maths.sqrt _obs_var_true in
  let u =
    List.init Dims.tmax ~f:(fun _ ->
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
        ~init:(Maths.const x0, [])
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
    ; _log_space_var : 'a (* log of covariance of space factor *)
    ; _log_time_var : 'a (* log of covariance of time factor *)
    }
  [@@deriving prms]
end

let true_theta =
  PP.
    { _q = _q_true
    ; _d = _d_true
    ; _c = _c_true
    ; _log_obs_var = Maths.log _obs_var_true
    ; _log_space_var = Maths.log ones_u
    ; _log_time_var = Maths.log ones_tmax
    }

module LGS = struct
  module P = PP.Make (Prms.P)

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
          List.map o_list_tmp ~f:(fun o ->
            let _cx =
              Maths.(
                neg (einsum [ const o, "ab"; _obs_var_inv, "b"; theta._c, "cb" ] "ac"))
            in
            Lds_data.Temp.
              { _f = None
              ; _Fx_prod
              ; _Fu_prod = _Fu_true
              ; _cx = Some _cx
              ; _cu = None
              ; _Cxx
              ; _Cxu = None
              ; _Cuu = eye_m
              })
      }

  (* rollout y list under sampled u *)
  let rollout ~u_list (theta : P.M.t) =
    let tmp_einsum a b =
      if Dims.batch_const
      then Maths.einsum [ a, "ma"; b, "ab" ] "mb"
      else Maths.einsum [ a, "ma"; b, "mab" ] "mb"
    in
    let _Fx_prod = _Fx_reparam (theta._q, theta._d) in
    let _, y_list_rev =
      List.fold
        u_list
        ~init:(Maths.const x0, [])
        ~f:(fun (x, y_list) u ->
          let new_x = Maths.(tmp_einsum x _Fx_prod + tmp_einsum u _Fu_true) in
          let new_y = tmp_einsum new_x theta._c in
          new_x, new_y :: y_list)
    in
    List.rev y_list_rev

  (* optimal u determined from lqr *)
  let pred_u ~data:o_list (theta : P.M.t) =
    (* use lqr to obtain the optimal u *)
    let p =
      params_from_f ~x0:(Maths.const x0) ~theta ~o_list
      |> Lds_data.map_naive ~batch_const:Dims.batch_const
    in
    let sol, _ = Lqr._solve  ~batch_const:Dims.batch_const p in
    let optimal_u_list = List.map sol ~f:(fun s -> s.u) in
    (* sample u from the kronecker formation *)
    let u_list =
      let optimal_u = concat_time optimal_u_list in
      let xi =
        Tensor.randn ~device:Dims.device ~kind:Dims.kind [ Dims.bs; Dims.m; Dims.tmax ]
        |> Maths.const
      in
      let xi_space =
        Maths.einsum [ xi, "mbt"; std_of_log_var theta._log_space_var, "b" ] "mbt"
      in
      let xi_time =
        Maths.einsum [ xi_space, "mat"; std_of_log_var theta._log_time_var, "t" ] "mat"
      in
      let meaned = Maths.(xi_time + optimal_u) in
      List.init Dims.tmax ~f:(fun i ->
        Maths.slice ~dim:2 ~start:(Some i) ~end_:(Some (i + 1)) ~step:1 meaned
        |> Maths.reshape ~shape:[ Dims.bs; Dims.m ])
    in
    optimal_u_list, u_list

  let elbo ~data ~sample (theta : P.M.t) =
    (* obtain u from lqr *)
    let optimal_u_list, u_sampled = pred_u ~data theta in
    (* calculate the likelihood term *)
    let _Fx_prod = _Fx_reparam (theta._q, theta._d) in
    let y_pred = rollout ~u_list:u_sampled theta in
    let lik_term =
      let inv_sigma_o_expanded =
        Maths.(sqrt_precision_of_log_var theta._log_obs_var * ones_o)
      in
      List.fold2_exn
        data
        y_pred
        ~init:Maths.(f 0.)
        ~f:(fun accu o y_pred ->
          Stdlib.Gc.major ();
          Maths.(
            accu + gaussian_llh ~mu:y_pred ~inv_std:inv_sigma_o_expanded (Maths.const o)))
    in
    (* M1: calculate the kl term using samples *)
    let optimal_u = concat_time optimal_u_list in
    let kl =
      if sample
      then (
        let prior =
          List.foldi u_sampled ~init:None ~f:(fun t accu u ->
            if t % 1 = 0 then Stdlib.Gc.major ();
            let increment =
              gaussian_llh
                ~inv_std:
                  (Tensor.ones [ Dims.m ] ~device:base.device ~kind:base.kind
                   |> Maths.const)
                u
            in
            match accu with
            | None -> Some increment
            | Some accu -> Some Maths.(accu + increment))
          |> Option.value_exn
        in
        let neg_entropy =
          let u = concat_time u_sampled |> Maths.reshape ~shape:[ Dims.bs; -1 ] in
          let optimal_u = Maths.reshape optimal_u ~shape:[ Dims.bs; -1 ] in
          let inv_std =
            Maths.(
              f 1.
              / kron
                  (std_of_log_var theta._log_space_var)
                  (std_of_log_var theta._log_time_var))
          in
          gaussian_llh ~mu:optimal_u ~inv_std u
        in
        Maths.(neg_entropy - prior))
      else (
        (* M2: calculate the kl term analytically *)
        let std2 =
          Maths.(
            kron
              (std_of_log_var theta._log_space_var)
              (std_of_log_var theta._log_time_var))
        in
        let det1 = Maths.(2. $* sum (log std2)) in
        let _const = Float.of_int (Dims.m * Dims.tmax) in
        let tr =
          let tmp2 = Maths.(exp theta._log_space_var) in
          let tmp3 = Maths.(kron tmp2 (exp theta._log_time_var)) in
          Maths.sum tmp3
        in
        let quad =
          Maths.einsum [ optimal_u, "mbt"; optimal_u, "mbt" ] "m"
          |> Maths.unsqueeze ~dim:1
        in
        let tmp = Maths.(tr - det1 -$ _const) |> Maths.reshape ~shape:[ 1; 1 ] in
        Maths.(0.5 $* tmp + quad) |> Maths.squeeze ~dim:1)
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

  let f ~update ~data ~init ~args:() (theta : P.M.t) =
    let neg_elbo, y_pred = elbo ~data theta ~sample in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let ggn = ggn ~y_pred theta in
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
      Tensor.(log (f Float.(square 1.) * ones ~device:Dims.device ~kind:Dims.kind [ 1 ]))
      |> Prms.free
    in
    let _log_space_var =
      Tensor.(
        log (f Float.(square 1.) * ones ~device:Dims.device ~kind:Dims.kind [ Dims.m ]))
      |> Prms.free
    in
    let _log_time_var =
      Tensor.(
        log (f Float.(square 1.) * ones ~device:Dims.device ~kind:Dims.kind [ Dims.tmax ]))
      |> Prms.free
    in
    { _q; _d; _c; _log_obs_var; _log_space_var; _log_time_var }
end

let _ = LGS.save_summary ~out:(in_dir "true_summary") true_theta

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
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array [| Float.of_int t; time_elapsed; loss_avg |] 1 3));
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
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 0.1
      ; n_tangents = 128
      ; rank_one = false
      ; damping = None  
      ; aux=None
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
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.001 }

  let init = O.init LGS.init
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
