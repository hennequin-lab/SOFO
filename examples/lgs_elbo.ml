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

let n_fisher = 100

(* -----------------------------------------
   -- Control Problem / Data Generation ---
   ----------------------------------------- *)
module Dims = struct
<<<<<<< Updated upstream
  let a = 24
  let b = 10
  let o = 40
=======
  let b = 5
  let a = 5
  let o = 20
>>>>>>> Stashed changes
  let tmax = 10
  let m = 32
  let batch_const = true
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.cuda_if_available ()
end

module Data = Lds_data.Make_LDS_Tensor (Dims)

let x0 = Tensor.zeros ~device:Dims.device ~kind:Dims.kind [ Dims.m; Dims.a ]

(* in the linear gaussian case, _Fx, _Fu, c, b and cov invariant across time *)
(* TODO: noise cov as identity as the simplest case *)
let _cov_o =
  Tensor.(
    diag_embed
      ~offset:0
      ~dim1:0
      ~dim2:1
      (f 0.01 * ones ~device:Dims.device ~kind:Dims.kind [ Dims.o ]))

let _std_u = Tensor.ones ~device:Dims.device ~kind:Dims.kind [ Dims.b ]
let eye_a = Owl.Mat.(0.9 $* eye Dims.a) |> Tensor.of_bigarray ~device:Dims.device

let _Fx =
  let open Owl in
  let w = Mat.gaussian Dims.a Dims.a in
  let sa = Linalg.D.eigvals w |> Dense.Matrix.Z.re |> Mat.max' in
  let w = Mat.(Float.(0.8 / sa) $* w) in
  let a = Mat.((0.9 $* eye Dims.a) + (0.1 $* w)) in
  Tensor.of_bigarray ~device:Dims.device a

let _Fu = Data.sample_fu ()
let c = Data.sample_c ()

let f_list : Tensor.t Lds_data.f_params list =
  List.init (Dims.tmax + 1) ~f:(fun _ ->
    Lds_data.
      { _Fx_prod = _Fx
      ; _Fu_prod = _Fu
      ; _f = None
      ; _c = Some c
      ; _b = None
      ; _cov = Some _cov_o
      })

let sample_data () =
  (* generate ground truth params and data *)
  let u_list = Data.sample_u_list ~std_u:_std_u in
  let x_list, o_list = Data.traj_rollout ~x0 ~f_list ~u_list in
  let o_list = List.map o_list ~f:(fun o -> Option.value_exn o) in
  u_list, x_list, o_list

(* -----------------------------------------
   ----- Utilitiy Functions ------
   ----------------------------------------- *)

let tmp_einsum a b =
  if Dims.batch_const
  then Maths.einsum [ a, "ma"; b, "ab" ] "mb"
  else Maths.einsum [ a, "ma"; b, "mab" ] "mb"

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)
let in_dir = Cmdargs.in_dir "-d"

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f64)
    ; ba_kind = Bigarray.float64
    }

let laplace = false
let sample = true
let std_transform x = Maths.(0.01 $+ exp x)

module PP = struct
  (* note that all std live in log space *)
  type 'a p =
    { _Fx_prod : 'a (* generative model *)
    ; _Fu_prod : 'a
    ; _c : 'a
<<<<<<< Updated upstream
    ; _b : 'a
    ; _std_o : 'a (* sqrt of the diagonal of covariance of emission noise *)
    ; _std_space : 'a
      (* recognition model; sqrt of the diagonal of covariance of space factor *)
    ; _std_time : 'a (* sqrt of the diagonal of covariance of the time factor *)
=======
    ; _std_o_prms : 'a (* parameterises of the sqrt of the diagonal *)
    ; _std_space_prms : 'a
    ; _std_time_prms : 'a
>>>>>>> Stashed changes
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

module LGS = struct
  module P = P

  type args = unit
  type data = Tensor.t list

  (* 1/ (x^2) *)
  let sqr_inv x = Maths.(1. $/ sqr x)

  (* list of length T of [m x b] to matrix of [m x b x T]*)
  let concat_time u_list =
    List.map u_list ~f:(fun u -> Maths.unsqueeze ~dim:(-1) u) |> Maths.concat_list ~dim:2

  let gaussian_llh ?mu ?(fisher_batched = false) ~std x =
    let inv_std = Maths.(f 1. / std) in
    let error_term =
      if fisher_batched
      then (
        (* dimension l is number of fisher samples *)
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
      Maths.(2. $* sum (log std)) |> Maths.reshape ~shape:cov_term_shape
    in
    let const_term =
      let o = x |> Maths.primal |> Tensor.shape |> List.last_exn in
      Float.(log (2. * pi) * of_int o)
    in
    Maths.(0.5 $* (const_term $+ error_term + cov_term)) |> Maths.neg

  (* special care to be taken when dealing with elbo loss *)
  module Elbo_loss = struct
    let fisher lik_term =
      let neg_lik_t =
        Maths.(tangent lik_term)
        |> Option.value_exn
        |> fun x -> Tensor.(x / f Float.(sqrt (of_int Int.(Dims.m * n_fisher))))
      in
      let n_tangents = List.hd_exn (Tensor.shape neg_lik_t) in
      let fisher =
        let fisher_half = Tensor.reshape neg_lik_t ~shape:[ n_tangents; -1 ] in
        Tensor.einsum ~equation:"kl,jl->kj" [ fisher_half; fisher_half ] ~path:None
      in
      fisher

    (* this is u sampled from posterior *)
    let true_fisher ~u_list (theta : P.M.t) =
      let _std_o = theta._std_o_prms |> std_transform in
      let _std_o_extended =
        _std_o |> Maths.primal |> Tensor.unsqueeze ~dim:0 |> Tensor.unsqueeze ~dim:0
      in
      let _, fisher_rollout =
        List.fold
          u_list
          ~init:(Maths.const x0, Tensor.f 0.)
          ~f:(fun accu u ->
            let prev_x, fisher_accu = accu in
            let new_x =
              Maths.(
                (f 0.9 * prev_x)
                + tmp_einsum prev_x theta._Fx_prod
                + tmp_einsum u theta._Fu_prod)
            in
            let new_o = Maths.(tmp_einsum new_x theta._c) in
            let new_o_primal = Maths.primal new_o in
            let new_o_unsqueezed =
              List.init n_fisher ~f:(fun _ -> Maths.unsqueeze new_o ~dim:0)
              |> Maths.concat_list ~dim:0
            in
            let noise =
              Tensor.(
                _std_o_extended
                * randn
                    (n_fisher :: Maths.shape new_o)
                    ~device:base.device
                    ~kind:base.kind)
            in
            let o_samples_batched = Maths.(const Tensor.(new_o_primal + noise)) in
            let lik_term_sampled_batched =
              gaussian_llh
                ~mu:new_o_unsqueezed
                ~std:_std_o
                ~fisher_batched:true
                o_samples_batched
            in
            let fisher = fisher lik_term_sampled_batched in
            Stdlib.Gc.major ();
            new_x, Tensor.(fisher + fisher_accu))
      in
      let fisher = Tensor.(fisher_rollout / f (Float.of_int Dims.tmax)) in
      let _, final_s, _ = Tensor.svd ~some:true ~compute_uv:false fisher in
      final_s
      |> Tensor.reshape ~shape:[ -1; 1 ]
      |> Tensor.to_bigarray ~kind:base.ba_kind
      |> Owl.Mat.save_txt ~out:(in_dir (sprintf "svals"));
      fisher
  end

  (* create params for lds from f *)
  let params_from_f ~(theta : P.M.t) ~x0 ~o_list
    : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
    =
    (* set o at time 0 as 0 *)
    let o_list_tmp = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
    let _cov_u_inv =
<<<<<<< Updated upstream
      Tensor.eye ~n:Dims.b ~options:(base.kind, base.device) |> Maths.const
    in
    let c_trans = Maths.transpose theta._c ~dim0:1 ~dim1:0 in
    let _cov_o_inv = theta._std_o |> Maths.exp |> sqr_inv in
    let _Cxx =
      let tmp = Maths.(einsum [ theta._c, "ab"; _cov_o_inv, "b" ] "ab") in
      Maths.(tmp *@ c_trans)
    in
    let _cx_common =
      let tmp = Maths.(einsum [ theta._b, "ab"; _cov_o_inv, "b" ] "ab") in
      Maths.(tmp *@ c_trans)
=======
      Maths.const _std_u |> sqr_inv |> Maths.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
>>>>>>> Stashed changes
    in
    let _cov_o_inv = theta._std_o_prms |> std_transform |> sqr_inv in
    let _Cxx = Maths.(einsum [ theta._c, "ab"; _cov_o_inv, "b"; theta._c, "db" ] "ad") in
    let _Fx_prod = Maths.(const eye_a + theta._Fx_prod) in
    Lqr.Params.
      { x0 = Some x0
      ; params =
          List.map o_list_tmp ~f:(fun o ->
            let _cx =
              Maths.(neg (einsum [ const o, "ab"; _cov_o_inv, "b"; theta._c, "db" ] "ad"))
            in
            Lds_data.Temp.
              { _f = None
              ; _Fx_prod
              ; _Fu_prod = theta._Fu_prod
              ; _cx = Some _cx
              ; _cu = None
              ; _Cxx
              ; _Cxu = None
              ; _Cuu = _cov_u_inv
              })
      }

  (* rollout x list under sampled u *)
  let rollout_x ~u_list ~x0 (theta : P.M.t) =
    let x0_tan = Maths.const x0 in
    let _, x_list =
      List.fold u_list ~init:(x0_tan, [ x0_tan ]) ~f:(fun (x, x_list) u ->
        let new_x =
          Maths.((f 0.9 * x) + tmp_einsum x theta._Fx_prod + tmp_einsum u theta._Fu_prod)
        in
        new_x, new_x :: x_list)
    in
    List.rev x_list

  (* optimal u determined from lqr *)
  let pred_u ~data:o_list (theta : P.M.t) =
    (* use lqr to obtain the optimal u *)
    let p =
      params_from_f ~x0:(Maths.const x0) ~theta ~o_list
      |> Lds_data.map_naive ~batch_const:Dims.batch_const
    in
    let sol, _ = Lqr._solve ~laplace ~batch_const:Dims.batch_const p in
    let optimal_u_list = List.map sol ~f:(fun s -> s.u) in
    (* sample u from the kronecker formation *)
    let u_list =
      let optimal_u = concat_time optimal_u_list in
      let xi =
        Tensor.randn ~device:Dims.device ~kind:Dims.kind [ Dims.m; Dims.b; Dims.tmax ]
        |> Maths.const
      in
      let std_space = theta._std_space_prms |> std_transform in
      let std_time = theta._std_time_prms |> std_transform in
      let xi_space = Maths.einsum [ xi, "mbt"; std_space, "b" ] "mbt" in
      let xi_time = Maths.einsum [ xi_space, "mat"; std_time, "t" ] "mat" in
      let meaned = Maths.(xi_time + optimal_u) in
      List.init Dims.tmax ~f:(fun i ->
        Maths.slice ~dim:2 ~start:(Some i) ~end_:(Some (i + 1)) ~step:1 meaned
        |> Maths.reshape ~shape:[ Dims.m; Dims.b ])
    in
    optimal_u_list, u_list

  let elbo ~o_list ~u_list ~optimal_u_list ~sample (theta : P.M.t) =
    (* calculate the likelihood term *)
    let u_o_list = List.map2_exn u_list o_list ~f:(fun u o -> u, o) in
    let llh =
      let std_o = theta._std_o_prms |> std_transform in
      let _, llh =
        List.foldi
          u_o_list
          ~init:(Maths.const x0, None)
          ~f:(fun t accu (u, o) ->
            if t % 1 = 0 then Stdlib.Gc.major ();
            let x_prev, llh_summed = accu in
            let new_x =
              Maths.(
                (f 0.9 * x_prev)
                + tmp_einsum x_prev theta._Fx_prod
                + tmp_einsum u theta._Fu_prod)
            in
            let increment =
              gaussian_llh ~mu:o ~std:std_o Maths.(tmp_einsum new_x theta._c)
            in
            let new_llh_summed =
              match llh_summed with
              | None -> Some increment
              | Some accu -> Some Maths.(accu + increment)
            in
            Stdlib.Gc.major ();
            new_x, new_llh_summed)
      in
      Option.value_exn llh
    in
    (* M1: calculate the kl term using samples *)
    let optimal_u = concat_time optimal_u_list in
    let kl =
      let std_u = Maths.const _std_u in
      let std_space = std_transform theta._std_space_prms in
      let std_time = std_transform theta._std_time_prms in
      let std_spacetime = Maths.kron std_space std_time in
      if sample
      then (
        let prior =
          List.foldi u_list ~init:None ~f:(fun t accu u ->
            if t % 1 = 0 then Stdlib.Gc.major ();
<<<<<<< Updated upstream
            let increment =
              gaussian_llh
                ~std:
                  (Tensor.ones [ Dims.b ] ~device:base.device ~kind:base.kind
                   |> Maths.const)
                u
            in
=======
            let increment = gaussian_llh ~std:std_u u in
>>>>>>> Stashed changes
            match accu with
            | None -> Some increment
            | Some accu -> Some Maths.(accu + increment))
          |> Option.value_exn
        in
        let neg_entropy =
          let u = concat_time u_list |> Maths.reshape ~shape:[ Dims.m; -1 ] in
          let optimal_u = Maths.reshape optimal_u ~shape:[ Dims.m; -1 ] in
          gaussian_llh ~mu:optimal_u ~std:std_spacetime u
        in
        Maths.(neg_entropy - prior))
      else (
<<<<<<< Updated upstream
        (* M2: calculate the kl term analytically *)
        let std2 = Maths.(kron (exp theta._std_space) (exp theta._std_time)) in
        let det1 = Maths.(2. $* sum (log std2)) in
        let _const = Float.of_int (Dims.b * Dims.tmax) in
        let tr =
          let tmp2 = Maths.(sqr (exp theta._std_space)) in
          let tmp3 = Maths.(kron tmp2 (sqr (exp theta._std_time))) in
=======
        let det1 = Maths.(2. $* sum (log std_spacetime)) in
        let det2 = Maths.(Float.(2. * of_int Dims.tmax) $* sum (log std_u)) in
        let _const = Float.of_int (Dims.b * Dims.tmax) in
        let _cov_u_inv = std_u |> Maths.exp |> sqr_inv in
        let tr =
          let tmp2 = Maths.(_cov_u_inv * sqr std_space) in
          let tmp3 = Maths.(kron tmp2 (sqr std_time)) in
>>>>>>> Stashed changes
          Maths.sum tmp3
        in
        let quad =
          Maths.einsum [ optimal_u, "mbt"; optimal_u, "mbt" ] "m"
          |> Maths.unsqueeze ~dim:1
        in
        let tmp = Maths.(tr - det1 -$ _const) |> Maths.reshape ~shape:[ 1; 1 ] in
        Maths.(0.5 $* tmp + quad) |> Maths.squeeze ~dim:1)
    in
    Maths.((llh - kl) /$ Float.of_int Int.(Dims.tmax * Dims.o))

  let f ~update ~data ~init ~args:() (theta : P.M.t) =
    let module L = Elbo_loss in
    let optimal_u_list, u_list = pred_u ~data theta in
    let neg_elbo =
      Maths.(
        neg
          (elbo
             ~o_list:(List.map data ~f:Maths.const)
             ~u_list
             ~optimal_u_list
             theta
             ~sample))
    in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let ggn = L.true_fisher ~u_list theta in
      u init (Some (neg_elbo, Some ggn))

  let init : P.tagged =
    let _Fx_prod =
      let open Owl in
      Tensor.zeros ~device:Dims.device ~kind:Dims.kind Dims.[ a; a ] |> Prms.free
    in
    let _Fu_prod =
      Convenience.gaussian_tensor_2d_normed
        ~device:Dims.device
        ~kind:Dims.kind
        ~a:Dims.b
        ~b:Dims.a
        ~sigma:0.1
      |> Prms.free
    in
    let _c =
      Convenience.gaussian_tensor_2d_normed
        ~device:Dims.device
        ~kind:Dims.kind
        ~a:Dims.a
        ~b:Dims.o
        ~sigma:0.1
      |> Prms.free
    in
    let _b =
      Tensor.zeros ~device:base.device ~kind:base.kind [ 1; Dims.o ] |> Prms.const
    in
<<<<<<< Updated upstream
    let _std_o =
      (* Tensor.diag ~diagonal:0 _std_o |> Prms.const *)
      (* Tensor.(f 1. * ones ~device:Dims.device ~kind:Dims.kind [ Dims.o ])
      |> Prms.create ~above:(Tensor.f 0.1)  *)
=======
    let _std_o_prms =
>>>>>>> Stashed changes
      Prms.create
        ~above:(Tensor.f (-5.))
        Tensor.(zeros ~device:base.device ~kind:base.kind [ Dims.o ])
    in
    let _std_space_prms =
      Prms.free Tensor.(zeros ~device:base.device ~kind:base.kind [ Dims.b ])
    in
    let _std_time_prms =
      Prms.free Tensor.(zeros ~device:base.device ~kind:base.kind [ Dims.tmax ])
    in
<<<<<<< Updated upstream
    { _Fx_prod; _Fu_prod; _c; _b; _std_o; _std_space; _std_time }
=======
    { _Fx_prod; _Fu_prod; _c; _std_o_prms; _std_space_prms; _std_time_prms }
>>>>>>> Stashed changes

  (* calculate the error between latents *)
  let simulate ~data ~(theta : P.M.t) =
    (* rollout under the given u *)
    let x_list, u_list = data in
    (* rollout to obtain x *)
    let rolled_out_x_list =
      rollout_x ~u_list:(List.map u_list ~f:Maths.const) ~x0 theta
    in
    let error =
      List.fold2_exn rolled_out_x_list x_list ~init:0. ~f:(fun accu x1 x2 ->
        let error = Tensor.(norm (Maths.primal x1 - x2)) |> Tensor.to_float0_exn in
        accu +. error)
    in
    Float.(error / of_int Dims.tmax)
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
  val config_f : iter:int -> (float, Bigarray.float64_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state ~time_elapsed running_avg =
      Stdlib.Gc.major ();
      let u_list, _, o_list = sample_data () in
      let t0 = Unix.gettimeofday () in
      let config = config_f ~iter in
      let loss, new_state = O.step ~config ~state ~data:o_list ~args:() in
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
          (* ground truth elbo *)
          let elbo_true =
            let theta_true =
              let theta_curr = O.params new_state in
              let _Fx_prod = Maths.const _Fx in
              let _Fu_prod = Maths.const _Fu in
              let _c = Maths.const c in
<<<<<<< Updated upstream
              let _b = Maths.const b in
              let _std_o = Tensor.diag ~diagonal:0 (Tensor.log _std_o) |> Maths.const in
              let _std_space = theta_curr._std_space |> Prms.value |> Maths.const in
              let _std_time = theta_curr._std_time |> Prms.value |> Maths.const in
              PP.{ _Fx_prod; _Fu_prod; _c; _b; _std_o; _std_space; _std_time }
=======
              let _std_o_prms =
                Tensor.diag ~diagonal:0 (Tensor.log _cov_o) |> Maths.const
              in
              let _std_space_prms =
                theta_curr._std_space_prms |> Prms.value |> Maths.const
              in
              let _std_time_prms =
                theta_curr._std_time_prms |> Prms.value |> Maths.const
              in
              PP.{ _Fx_prod; _Fu_prod; _c; _std_o_prms; _std_space_prms; _std_time_prms }
>>>>>>> Stashed changes
            in
            let u_list = List.map u_list ~f:Maths.const in
            let elbo_tmp =
              LGS.elbo
                ~o_list:(List.map o_list ~f:Maths.const)
                ~u_list
                ~optimal_u_list:u_list
                ~sample
                theta_true
              |> Maths.primal
              |> Tensor.neg
              |> Tensor.mean
              |> Tensor.to_float0_exn
            in
            Float.(elbo_tmp / of_int Dims.tmax)
          in
          (* simulation error *)
          let o_error =
            let u_list, x_list, _ = sample_data () in
            let data = x_list, u_list in
            LGS.simulate ~theta:(LGS.P.const (LGS.P.value (O.params new_state))) ~data
          in
          (* avg error *)
          Convenience.print [%message (iter : int) (loss_avg : float)];
          let t = iter in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array
                 [| Float.of_int t; time_elapsed; loss_avg; o_error; elbo_true |]
                 1
                 5));
          O.W.P.T.save
            (LGS.P.value (O.params new_state))
            ~kind:base.ba_kind
            ~out:(in_dir name ^ "_params"));
        []
      in
      if iter < max_iter
      then loop ~iter:(iter + 1) ~state:new_state ~time_elapsed (loss :: running_avg)
    in
    (* ~config:(config_f ~iter:0) *)
    loop ~iter:0 ~state:init ~time_elapsed:0. []
end

(* --------------------------------
   -- SOFO
   -------------------------------- *)

module Do_with_SOFO : Do_with_T = struct
  module O = Optimizer.SOFO (LGS)

  let config_f ~iter:_ =
    Optimizer.Config.SOFO.
      { base
<<<<<<< Updated upstream
      ; learning_rate = Some Float.(1e-3 / (1. +. (0.0 * sqrt (of_int iter))))
      ; n_tangents = 128
=======
      ; learning_rate = Some 0.03
      ; n_tangents = 32
>>>>>>> Stashed changes
      ; sqrt = true
      ; rank_one = false
      ; damping = Some 1e-3
      ; momentum = None
      ; lm = false
      ; perturb_thresh = None
      }

  let name = "sofo"
  let init = O.init ~config:(config_f ~iter:0) LGS.init
end

(* --------------------------------
     -- Adam
     -------------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (LGS)

  let config_f ~iter:_ =
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
