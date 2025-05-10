open Base
open Forward_torch
open Torch
include Optimizer_typ

(* given info, update loss and ggn at each step *)
let update_each_step ((loss, ggn) as accu) info =
  match info with
  | None -> accu
  | Some (ell, z) ->
    let loss = Maths.(loss + ell) in
    let ggn =
      match z with
      | None -> ggn
      | Some delta_ggn -> Tensor.(ggn + delta_ggn)
    in
    loss, ggn

(* given infor, update loss at each step *)
let update_loss_each_step accu info =
  match info with
  | None -> accu
  | Some ell ->
    let loss = Maths.(accu + ell) in
    loss

(* given [base] configuration, draw elements of tensor of [n_tangents;shape] 
  from standard Gaussian such that the tangent is rank-one *)
let rank_one_tensor ~(base : ('a, 'b) Config.Base.t) ~n_tangents shape =
  let components =
    List.map shape ~f:(fun dim ->
      Tensor.randn ~device:base.device ~kind:base.kind [ n_tangents; dim ])
  in
  let equation =
    (* "ab,ac,ad,ae->abcde" *)
    let idx =
      List.mapi shape ~f:(fun i _ -> Char.(to_string (of_int_exn (to_int 'a' + i + 1))))
    in
    let part1 = String.concat ~sep:"," (List.map idx ~f:(fun x -> "a" ^ x)) in
    let part2 = "->" in
    let part3 = String.concat ("a" :: idx) in
    String.concat [ part1; part2; part3 ]
  in
  Tensor.einsum ~equation ~path:None components

(* given [base] configuration, draw elements of tensor of [shape] 
from standard Gaussian; optionally [rank_one] *)
let sample_rand_tensor ~base ~rank_one ~shape =
  let[@warning "-8"] (n_tangents :: param_shape) = shape in
  if rank_one
  then rank_one_tensor ~base ~n_tangents param_shape
  else Tensor.randn ~device:base.device ~kind:base.kind shape

(* apply momentum to gradient *)
module Momentum (P : Prms.T) = struct
  let apply ?momentum ~avg g =
    match momentum with
    | None -> g
    | Some beta ->
      (match avg with
       | None -> P.T.(Float.(1. - beta) $* g)
       | Some g_avg -> P.T.((beta $* g_avg) + (Float.(1. - beta) $* g)))
end

(* Adam needs to be defined first, because SOFO makes use of Adam for optimising
   the auxiliary loss for learning a GGN approximation *)
module Adam (W : Wrapper.T) = struct
  module W = W

  type ('a, 'b) config = ('a, 'b) Config.Adam.t
  type ('c, _, _) init_opts = W.P.tagged -> 'c

  type state =
    { theta : W.P.tagged
    ; m : Tensor.t W.P.p option
    ; v : Tensor.t W.P.p option
    ; beta1_t : float
    ; beta2_t : float
    }

  let params state = state.theta
  let init theta = { theta; m = None; v = None; beta1_t = 1.; beta2_t = 1. }

  (* given adam [config], current [state] and gradient g, gradient descent on theta *)
  let update_theta_m_v ~(config : ('a, 'b) config) ~state g =
    let c = config in
    let beta1_t = Float.(state.beta1_t * c.beta_1) in
    let beta2_t = Float.(state.beta2_t * c.beta_2) in
    let m, v =
      let g_squared = W.P.T.sqr g in
      match state.m, state.v with
      | None, None ->
        W.P.T.(Float.(1. - c.beta_1) $* g, Float.(1. - c.beta_2) $* g_squared)
      | Some m, Some v ->
        let m = W.P.T.((c.beta_1 $* m) + (Float.(1. - c.beta_1) $* g)) in
        let v = W.P.T.((c.beta_2 $* v) + (Float.(1. - c.beta_2) $* g_squared)) in
        m, v
      | _ -> assert false
    in
    let theta =
      match c.learning_rate with
      | None -> state.theta
      | Some eta ->
        let m_hat = W.P.T.(Float.(1. / (1. - beta1_t)) $* m) in
        let v_hat = W.P.T.(Float.(1. / (1. - beta2_t)) $* v) in
        let v_hat_sqrt = W.P.T.sqrt v_hat in
        let dtheta = W.P.T.(m_hat / (c.eps $+ v_hat_sqrt)) in
        let open W.P.Let_syntax in
        let+ theta = state.theta
        and++ dtheta = dtheta in
        let dtheta_decayed =
          match c.weight_decay with
          | None -> dtheta
          | Some lambda -> Tensor.(dtheta + mul_scalar theta (Scalar.f lambda))
        in
        Tensor.(theta - mul_scalar dtheta_decayed (Scalar.f eta))
    in
    { theta; m = Some m; v = Some v; beta1_t; beta2_t }

  (* given adam [config], current [state], [data] and args, take an adam step
    and returns loss and new state *)
  let step ~(config : ('a, 'b) config) ~state ~data args =
    Stdlib.Gc.major ();
    (* initialise tangents *)
    let theta = params state in
    let theta_ = W.P.value theta in
    let theta_dual =
      W.P.map theta_ ~f:(fun x ->
        let x = Tensor.copy x |> Tensor.to_device ~device:config.base.device in
        let x = Tensor.set_requires_grad x ~r:true in
        Tensor.zero_grad x;
        Maths.const x)
    in
    (* define update function *)
    let update = `loss_only update_loss_each_step in
    (* initialise losses and ggn *)
    let init = Maths.const (Tensor.f 0.) in
    (* compute the losses and tangents with sgd *)
    let final_losses = W.f ~update ~data ~init ~args theta_dual in
    let loss, true_g =
      let loss = Tensor.mean (Maths.primal final_losses) in
      Tensor.backward loss;
      ( Tensor.to_float0_exn loss
      , W.P.map2 theta (W.P.primal theta_dual) ~f:(fun tagged p ->
          match tagged with
          | Prms.Const _ -> Tensor.(f 0.)
          | _ -> Tensor.grad p) )
    in
    let new_state : state = update_theta_m_v ~config ~state true_g in
    loss, new_state
end

(* SOFO optimizer *)
module SOFO (W : Wrapper.T) (A : Wrapper.Auxiliary with module P = W.P) = struct
  module W = W
  module Mom = Momentum (W.P)

  type ('a, 'b) config = ('a, 'b) Config.SOFO.t
  type ('c, _, _) init_opts = W.P.tagged -> 'c

  (* adam optimizer for training aux parameters *)
  module O = Adam (struct
      module P = A.A

      type data = W.P.T.t * Tensor.t
      type args = unit

      (* given update method [update], tangent [vs] and v^TGv [true_sketch], [init] state, [args]
        and aux parameters lambda, compute mse between v^TGv and v^T\hat{G}v  *)
      let f ~update ~data:(vs, true_sketch) ~init ~args:() lambda =
        let open Maths in
        let g12v = A.g12v ~lambda (W.P.map vs ~f:const) in
        let sketch =
          W.P.fold
            g12v
            ~init:Maths.(f 0.)
            ~f:(fun accu (gv, _) ->
              let n_tangents = Convenience.first_dim (primal gv) in
              let gv = reshape gv ~shape:[ n_tangents; -1 ] in
              accu + einsum [ gv, "ki"; gv, "li" ] "kl")
        in
        let normaliser = mean (sqr (const true_sketch)) in
        let loss = mean (sqr (sketch - const true_sketch)) / normaliser in
        match update with
        | `loss_only u -> u init (Some loss)
        | `loss_and_ggn _ -> assert false
    end)

  type state =
    { t : int
    ; theta : W.P.tagged
    ; aux : O.state
    ; sampling_state : A.sampling_state
    }

  (* obtain parameters from state *)
  let params state = state.theta

  (* given parameters, initialise state *)
  let init theta =
    let lambda = A.init () in
    let aux = O.init lambda in
    let sampling_state = A.init_sampling_state () in
    { t = 0; theta; aux; sampling_state }

  (* orthogonalise vs *)
  let orthonormalise vs =
    let vtv =
      W.P.fold vs ~init:(Tensor.f 0.) ~f:(fun accu (v, _) ->
        let n_tangents = Convenience.first_dim v in
        let v = Tensor.reshape v ~shape:[ n_tangents; -1 ] in
        Tensor.(accu + einsum ~equation:"ij,kj->ik" ~path:None [ v; v ]))
    in
    let u, s, _ = Tensor.svd ~some:true ~compute_uv:true vtv in
    let normalizer = Tensor.(u / sqrt s |> transpose ~dim0:0 ~dim1:1) in
    W.P.map vs ~f:(fun v ->
      let n_tangents = Convenience.first_dim v in
      let s = Tensor.shape v in
      let v = Tensor.reshape v ~shape:[ n_tangents; -1 ] in
      let v = Tensor.(matmul normalizer v) in
      Tensor.reshape v ~shape:s)

  (* given [base] configuration, [rank_one] condition, [n_tangents],
    initialise tangents where each tangent is normalised *)
  let init_tangents ~base ~rank_one ~n_tangents theta =
    W.P.map theta ~f:(function
      | Prms.Const x ->
        Tensor.zeros
          ~device:(Tensor.device x)
          ~kind:(Tensor.kind x)
          (n_tangents :: Tensor.shape x)
      | Prms.Free x ->
        sample_rand_tensor ~base ~rank_one ~shape:(n_tangents :: Tensor.shape x)
      | Prms.Bounded (x, _, _) ->
        sample_rand_tensor ~base ~rank_one ~shape:(n_tangents :: Tensor.shape x))

  (* fold vs over sets of v_i s, multiply with associated [weights]. *)
  let weighted_vs_sum vs ~weights =
    W.P.map vs ~f:(fun v ->
      let v_i = Maths.tangent' v in
      let[@warning "-8"] (n_samples :: s) = Tensor.shape v_i in
      let n_ws = Convenience.first_dim weights in
      let v_i = Tensor.view v_i ~size:[ n_samples; -1 ] in
      let s = if n_ws = 1 then s else n_ws :: s in
      Tensor.(view (matmul weights v_i) ~size:s))

  (* calculate natural gradient = [vs] ([ggn] + damping)^-1 [vtg] with [damping] coefficient *)
  let natural_g ?damping ~vs ~ggn vtg =
    let u, s, _ = Tensor.svd ~some:true ~compute_uv:true ggn in
    (* how each V should be weighted, as a row array *)
    let weights =
      let tmp = Convenience.a_trans_b u vtg in
      let ds =
        match damping with
        | None -> s
        | Some gamma ->
          let offset = Float.(gamma * Tensor.(maximum s |> to_float0_exn)) in
          Tensor.(s + f offset)
      in
      Tensor.(matmul (u / ds) tmp) |> Convenience.trans_2d
    in
    weighted_vs_sum vs ~weights

  (* given [learning_rate], [theta] and gradient [dtheta], gradient descent on [theta]
    to return new theta *)
  let update_theta ?learning_rate ~theta dtheta =
    match learning_rate with
    | Some eta ->
      let open W.P.Let_syntax in
      let+ theta = theta
      and++ g = dtheta in
      Tensor.(theta - mul_scalar g (Scalar.f eta))
    | None -> theta

  (* given sofo [config], current [state], [data] and args, take sofo step on params
  and returns loss and new state *)
  let step ~(config : ('a, 'b) config) ~state ~data args =
    Stdlib.Gc.major ();
    (* initialise tangents *)
    let theta = params state in
    (* even trials, use random vs; odd trials, use vs sampled from eigenvectors *)
    let _vs, new_sampling_state =
      match config.aux with
      | Some _ ->
        if state.t % 2 = 0
        then A.random_localised_vs config.n_tangents, state.sampling_state
        else (
          let lambda = O.params state.aux in
          let lambda = A.A.map ~f:(fun x -> Maths.const (Prms.value x)) lambda in
          A.eigenvectors ~lambda state.sampling_state config.n_tangents)
      | None ->
        ( init_tangents
            ~base:config.base
            ~rank_one:config.rank_one
            ~n_tangents:config.n_tangents
            theta
        , state.sampling_state )
    in
    let _vs = orthonormalise _vs in
    let vs = W.P.map _vs ~f:(fun x -> Maths.Direct x) in
    let theta_dual = W.P.make_dual theta ~t:vs in
    (* define update function *)
    let update = `loss_and_ggn update_each_step in
    (* initialise losses and ggn *)
    let init = Maths.const (Tensor.f 0.), Tensor.f 0. in
    (* compute the losses and tangents *)
    let final_losses, final_ggn = W.f ~update ~data ~init ~args theta_dual in
    (* compute loss and vtg *)
    let batch_size = final_losses |> Maths.primal |> Convenience.first_dim in
    let loss = final_losses |> Maths.primal |> Tensor.mean |> Tensor.to_float0_exn in
    (* normalise ggn by batch size *)
    let final_ggn = Tensor.div_scalar_ final_ggn (Scalar.f Float.(of_int batch_size)) in
    let loss_tangents = final_losses |> Maths.tangent |> Option.value_exn in
    let vtg =
      Tensor.(
        mean_dim
          loss_tangents
          ~dtype:(type_ loss_tangents)
          ~dim:(Some [ 1 ])
          ~keepdim:true)
    in
    (* compute natural gradient and update theta *)
    let natural_g = natural_g ?damping:config.damping ~vs ~ggn:final_ggn vtg in
    let new_theta = update_theta ?learning_rate:config.learning_rate ~theta natural_g in
    (* update the auxiliary state by running a few iterations of Adam if using random vs (odd trial) *)
    let new_aux =
      match config.aux with
      | Some aux_config when state.t % 2 = 0 ->
        let rec aux_loop ~iter ~state =
          let data = _vs, final_ggn in
          let loss, new_state = O.step ~config:aux_config.config ~state ~data () in
          Owl.Mat.(save_txt ~append:true ~out:aux_config.file (of_array [| loss |] 1 1));
          if iter < aux_config.steps
          then aux_loop ~iter:(iter + 1) ~state:new_state
          else new_state
        in
        aux_loop ~iter:1 ~state:state.aux
      | _ -> state.aux
    in
    ( loss
    , { t = state.t + 1
      ; theta = new_theta
      ; aux = new_aux
      ; sampling_state = new_sampling_state
      } )
end

(* forward gradient descent; g = V V^T g where V^Tg is obtained with forward AD (Kozak 2021) *)
module FGD (W : Wrapper.T) = struct
  module W = W

  type ('a, 'b) config = ('a, 'b) Config.FGD.t
  type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.FGD.t -> W.P.tagged -> 'c

  type state =
    { theta : W.P.tagged
    ; g_avg : Tensor.t W.P.p option
    ; beta_t : float option
    }

  (* obtain parameters from state *)
  let params state = state.theta

  (* given parameters, initialise state *)
  let init ~(config : ('a, 'b) config) theta =
    { theta; g_avg = None; beta_t = Option.map config.momentum ~f:(fun _ -> 1.) }

  (* given [base] configuration, [rank_one] condition, [n_tangents],
    initialise tangents where each tangent is normalised *)
  let init_tangents ~base ~rank_one ~n_tangents theta =
    let vs =
      W.P.map theta ~f:(function
        | Prms.Const x ->
          Tensor.zeros
            ~device:(Tensor.device x)
            ~kind:(Tensor.kind x)
            (n_tangents :: Tensor.shape x)
        | Prms.Free x ->
          sample_rand_tensor ~base ~rank_one ~shape:(n_tangents :: Tensor.shape x)
        | Prms.Bounded (x, _, _) ->
          sample_rand_tensor ~base ~rank_one ~shape:(n_tangents :: Tensor.shape x))
    in
    (* normalise each tangent *)
    let normalize vs =
      let normalizer =
        W.P.fold vs ~init:(Tensor.f 0.) ~f:(fun accu (v, _) ->
          Tensor.(accu + Convenience.sum_except_dim0 (square v)))
        |> Tensor.sqrt_
        |> Tensor.reciprocal_
      in
      let normed_vs =
        W.P.map vs ~f:(fun v ->
          (* reshape normalizer from [n_tangents] to [n_tangents, 1, ...] with same shape as v. *)
          let normalizer =
            Tensor.view
              normalizer
              ~size:(List.mapi (Tensor.size v) ~f:(fun i si -> if i = 0 then si else 1))
          in
          Maths.Direct Tensor.(v * normalizer))
      in
      normed_vs
    in
    normalize vs

  (* fold vs over sets of v_i s, multiply with associated [weights]. *)
  let weighted_vs_sum vs ~weights =
    W.P.map vs ~f:(fun v ->
      let v_i = Maths.tangent' v in
      let[@warning "-8"] (n_samples :: s) = Tensor.shape v_i in
      let n_ws = Convenience.first_dim weights in
      let v_i = Tensor.view v_i ~size:[ n_samples; -1 ] in
      let s = if n_ws = 1 then s else n_ws :: s in
      Tensor.(view (matmul weights v_i) ~size:s))

  (* given [learning_rate], [theta] and gradient [dtheta], gradient descent on [theta]
    to return new theta *)
  let update_theta ?learning_rate ~theta dtheta =
    match learning_rate with
    | Some eta ->
      let open W.P.Let_syntax in
      let+ theta = theta
      and++ g = dtheta in
      Tensor.(theta - mul_scalar g (Scalar.f eta))
    | None -> theta

  (* given fgd [config], current [state], [data] and args, take fgd step on params
  and returns loss and new state *)
  let step ~(config : ('a, 'b) config) ~state ~data args =
    Stdlib.Gc.major ();
    let beta_t =
      Option.map2 state.beta_t config.momentum ~f:(fun b beta -> Float.(b * beta))
    in
    (* initialise tangents *)
    let theta = params state in
    let vs =
      init_tangents
        ~base:config.base
        ~rank_one:config.rank_one
        ~n_tangents:config.n_tangents
        theta
    in
    let theta_dual = W.P.make_dual theta ~t:vs in
    (* define update function *)
    let update = `loss_and_ggn update_each_step in
    (* initialise losses and ggn *)
    let init = Maths.const (Tensor.f 0.), Tensor.f 0. in
    (* compute the losses and tangents *)
    let final_losses, _ = W.f ~update ~data ~init ~args theta_dual in
    (* compute loss and vtg *)
    let loss = final_losses |> Maths.primal |> Tensor.mean |> Tensor.to_float0_exn in
    let loss_tangents = final_losses |> Maths.tangent |> Option.value_exn in
    let vtg =
      Tensor.(
        mean_dim
          loss_tangents
          ~dtype:(type_ loss_tangents)
          ~dim:(Some [ 1 ])
          ~keepdim:true)
      (* calculate vanilla g *)
    in
    let num_params = Float.(of_int (W.P.T.numel (W.P.value theta))) in
    let vanilla_g =
      let weights =
        Tensor.(
          div_scalar
            (transpose ~dim0:1 ~dim1:0 vtg)
            Scalar.(f Float.(of_int config.n_tangents / num_params)))
      in
      weighted_vs_sum vs ~weights
    in
    let vanilla_g_avg =
      let module M = Momentum (W.P) in
      M.apply ?momentum:config.momentum ~avg:state.g_avg vanilla_g
    in
    let learning_rate =
      Option.map config.learning_rate ~f:(fun eta ->
        Option.value_map beta_t ~default:eta ~f:(fun b -> Float.(eta / (1. - b))))
    in
    let new_theta = update_theta ?learning_rate ~theta vanilla_g in
    loss, { theta = new_theta; g_avg = Some vanilla_g_avg; beta_t }
end

(* stochastic gradient descent; theta = theta - learning_rate * g *)
module SGD (W : Wrapper.T) = struct
  module W = W

  type ('a, 'b) config = ('a, 'b) Config.SGD.t
  type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.SGD.t -> W.P.tagged -> 'c

  type state =
    { theta : W.P.tagged
    ; g_avg : Tensor.t W.P.p option
    ; beta_t : float option
    }

  (* obtain parameters from state *)
  let params state = state.theta

  (* given parameters, initialise state *)
  let init ~(config : ('a, 'b) config) theta =
    { theta; g_avg = None; beta_t = Option.map config.momentum ~f:(fun _ -> 1.) }

  (* given [learning_rate], [theta] and gradient [dtheta], gradient descent on [theta]
    to return new theta *)
  let update_theta ?learning_rate ~theta dtheta =
    match learning_rate with
    | Some eta ->
      let open W.P.Let_syntax in
      let+ theta = theta
      and++ g = dtheta in
      Tensor.(theta - mul_scalar g (Scalar.f eta))
    | None -> theta

  (* given sgd [config], current [state], [data] and args, take sgd step on params
  and returns loss and new state *)
  let step ~(config : ('a, 'b) config) ~state ~data args =
    Stdlib.Gc.major ();
    let beta_t =
      Option.map2 state.beta_t config.momentum ~f:(fun b beta -> Float.(b * beta))
    in
    (* initialise tangents *)
    let theta = params state in
    let theta_ = W.P.value theta in
    let theta_dual =
      W.P.map theta_ ~f:(fun x ->
        let x = Tensor.copy x |> Tensor.to_device ~device:config.base.device in
        let x = Tensor.set_requires_grad x ~r:true in
        Tensor.zero_grad x;
        Maths.const x)
    in
    (* define update function *)
    let update = `loss_only update_loss_each_step in
    (* initialise losses and ggn *)
    let init = Maths.const (Tensor.f 0.) in
    (* compute the losses and tangents with sgd *)
    let final_losses = W.f ~update ~data ~init ~args theta_dual in
    let loss, true_g =
      let loss = Tensor.mean (Maths.primal final_losses) in
      Tensor.backward loss;
      Tensor.to_float0_exn loss, W.P.map (W.P.primal theta_dual) ~f:Tensor.grad
    in
    let g_avg =
      let module M = Momentum (W.P) in
      M.apply ?momentum:config.momentum ~avg:state.g_avg true_g
    in
    (* momentum correction in the learning rate *)
    let learning_rate =
      Option.map config.learning_rate ~f:(fun eta ->
        Option.value_map beta_t ~default:eta ~f:(fun b -> Float.(eta / (1. - b))))
    in
    let new_theta = update_theta ?learning_rate ~theta g_avg in
    loss, { theta = new_theta; g_avg = Some g_avg; beta_t }
end
