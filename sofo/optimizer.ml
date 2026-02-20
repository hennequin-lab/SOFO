open Base
open Forward_torch
open Maths
open Torch
include Optimizer_typ

(* update parameters, respecting any specified bounds *)
module Update_params (P : Prms.T) = struct
  let update ~learning_rate:eta ~theta delta =
    let open Prms in
    let update x delta = Tensor.(sub_ x (mul_scalar_ (primal delta) (Scalar.f eta))) in
    P.map2 theta delta ~f:(fun theta delta ->
      match theta with
      | Pinned x -> Pinned x
      | Free x -> Free (update x delta)
      | Bounded { v = x; lb; ub } ->
        Bounded { v = Prms.enforce_bounds ?lb ?ub (update x delta); lb; ub })

  let manual_replace ~theta (theta' : P.t) =
    let open Prms in
    P.map2 theta theta' ~f:(fun theta theta' ->
      match theta with
      | Pinned x -> Pinned x
      | Free _ -> Free (primal theta')
      | Bounded { v = _; lb; ub } ->
        Bounded { v = Prms.enforce_bounds ?lb ?ub (primal theta'); lb; ub })
end

(* apply momentum to anything *)
module Momentum (P : Prms.T) = struct
  let apply ?momentum ~avg g =
    match momentum with
    | None -> g
    | Some beta ->
      (match avg with
       | None -> P.(g *$ Float.(1. - beta))
       | Some g_avg -> P.((g_avg *$ beta) + (g *$ Float.(1. - beta))))
end

module SOFO (P : Prms.T) = struct
  module P = P

  type ('a, 'b) config = ('a, 'b) Config.SOFO.t
  type ('a, 'b, 'c) init_opts = P.param -> 'c
  type state = { theta : P.param }
  type info = P.t sofo_info

  let params state = state.theta
  let init theta = { theta }

  let clone_state (state : state) : state =
    let open Prms in
    let theta_cloned =
      P.map state.theta ~f:(fun theta ->
        match theta with
        | Pinned x -> Pinned Tensor.(x + f 0.)
        | Free x -> Free Tensor.(x + f 0.)
        | Bounded { v = x; lb; ub } ->
          Bounded { v = Prms.enforce_bounds ?lb ?ub Tensor.(x + f 0.); lb; ub })
    in
    { theta = theta_cloned }

  (* orthonormalize tangents *)
  let orthonormalise vs =
    let vtv =
      P.fold vs ~init:(f 0.) ~f:(fun accu (v, _) ->
        let n_tangents = first_dim v in
        let v = view v ~size:[ n_tangents; -1 ] in
        accu + einsum [ v, "ij"; v, "kj" ] "ik")
    in
    let u, s, _ = Const.svd vtv in
    let normalizer = u / sqrt s |> transpose in
    P.map vs ~f:(fun v ->
      let n_tangents = first_dim v in
      let s = shape v in
      let v = reshape v ~shape:[ n_tangents; -1 ] in
      let v = normalizer *@ v in
      reshape v ~shape:s)

  (* initialise tangents, where each tangent is normalised. *)
  let random_tangents ~n_tangents:k (theta : P.param) =
    (* note that even [Pinned] parameters have non-zero tangents;
       that's important, because we orthonormalize everything;
       however, these tangents don't matter anyway as those parameters
       won't be udpated *)
    P.map theta ~f:(fun x -> randn_like_k ~k (Prms.value x))
    |> orthonormalise
    |> P.map ~f:primal

  let prepare ~(config : (_, _) config) state =
    let theta = params state in
    let vs = random_tangents ~n_tangents:config.n_tangents theta in
    P.dual ~tangent:vs (P.value theta), P.map ~f:const vs

  (* fold vs over sets of v_i s, multiply with associated weights. *)
  let weighted_vs_sum ~vs weights =
    P.map vs ~f:(fun v_i ->
      let[@warning "-8"] (n_samples :: s) = shape v_i in
      let v_i = view v_i ~size:[ n_samples; -1 ] in
      view (reshape weights ~shape:[ 1; -1 ] *@ v_i) ~size:s)

  (* calculate natural gradient = V(VtGtGV)^-1 V^t g *)
  let sofo_update ~damping ~tangents:vs ~ggn vtg =
    let u, s, _ = Const.svd ggn in
    (* how each V should be weighted, as a row array *)
    let weights =
      let tmp = transpose u *@ const vtg in
      let s =
        match damping with
        | `none -> s
        | `relative_from_top gamma ->
          let offset = Float.(gamma * Tensor.(maximum (primal s) |> to_float0_exn)) in
          s +$ offset
        | `relative_from_bottom gamma ->
          let offset = Float.(gamma * Tensor.(minimum (primal s) |> to_float0_exn)) in
          s +$ offset
      in
      u / s *@ tmp
    in
    weighted_vs_sum ~vs weights

  module U = Update_params (P)

  let step ~(config : ('a, 'b) config) ~info state =
    match config.learning_rate with
    | None -> state
    | Some eta ->
      Stdlib.Gc.major ();
      let loss_t = tangent_exn info.loss in
      let delta =
        sofo_update ~damping:config.damping ~tangents:info.tangents ~ggn:info.ggn loss_t
      in
      let new_theta = U.update ~learning_rate:eta ~theta:(params state) delta in
      { theta = new_theta }

  let manual_state_update state f =
    let p = P.value state.theta in
    { theta = U.manual_replace ~theta:state.theta (f p) }
end

module SGDm (P : Prms.T) = struct
  module P = P

  type ('a, 'b) config = ('a, 'b) Config.SGDm.t
  type ('a, 'b, 'c) init_opts = P.param -> 'c
  type info = P.t

  type state =
    { theta : P.param
    ; g_avg : P.t option
    ; beta_t : float
    }

  let params state = state.theta
  let init theta = { theta; g_avg = None; beta_t = 1. }

  let clone_state (state : state) : state =
    let open Prms in
    let theta_cloned =
      P.map state.theta ~f:(fun theta ->
        match theta with
        | Pinned x -> Pinned Tensor.(x + f 0.)
        | Free x -> Free Tensor.(x + f 0.)
        | Bounded { v = x; lb; ub } ->
          Bounded { v = Prms.enforce_bounds ?lb ?ub Tensor.(x + f 0.); lb; ub })
    in
    { theta = theta_cloned; g_avg = state.g_avg; beta_t = state.beta_t }

  module M = Momentum (P)
  module U = Update_params (P)

  let step ~(config : ('a, 'b) config) ~info:g state =
    match config.learning_rate with
    | None -> state
    | Some eta ->
      Stdlib.Gc.major ();
      let beta_t = Float.(config.momentum * state.beta_t) in
      let g_avg = M.apply ~momentum:config.momentum ~avg:state.g_avg g in
      (* momentum correction of learning rate *)
      let eta = Float.(eta / (1. - beta_t)) in
      let new_theta = U.update ~learning_rate:eta ~theta:(params state) g_avg in
      { theta = new_theta; g_avg = Some g_avg; beta_t }

  let manual_state_update state f =
    let p = P.value state.theta in
    { state with theta = U.manual_replace ~theta:state.theta (f p) }
end

module Adam (P : Prms.T) = struct
  module P = P

  type ('a, 'b) config = ('a, 'b) Config.Adam.t
  type (_, _, 'c) init_opts = P.param -> 'c
  type info = P.t

  type state =
    { theta : P.param
    ; m : P.t option
    ; v : P.t option
    ; beta1_t : float
    ; beta2_t : float
    }

  let params state = state.theta
  let init theta = { theta; m = None; v = None; beta1_t = 1.; beta2_t = 1. }

  let clone_state (state : state) : state =
    let open Prms in
    let theta_cloned =
      P.map state.theta ~f:(fun theta ->
        match theta with
        | Pinned x -> Pinned Tensor.(x + f 0.)
        | Free x -> Free Tensor.(x + f 0.)
        | Bounded { v = x; lb; ub } ->
          Bounded { v = Prms.enforce_bounds ?lb ?ub Tensor.(x + f 0.); lb; ub })
    in
    { theta = theta_cloned
    ; m = state.m
    ; v = state.v
    ; beta1_t = state.beta1_t
    ; beta2_t = state.beta2_t
    }

  module M = Momentum (P)
  module U = Update_params (P)

  let step ~(config : ('a, 'b) config) ~info:g state =
    match config.learning_rate with
    | None -> state
    | Some eta ->
      Stdlib.Gc.major ();
      let c = config in
      let beta1_t = Float.(state.beta1_t * c.beta_1) in
      let beta2_t = Float.(state.beta2_t * c.beta_2) in
      let g_squared = P.map ~f:sqr g in
      let m = M.apply ~momentum:config.beta_1 ~avg:state.m g in
      let v = M.apply ~momentum:config.beta_2 ~avg:state.v g_squared in
      let m_hat = if config.debias then P.(m *$ Float.(1. / (1. - beta1_t))) else m in
      let v_hat = P.(v *$ Float.(1. / (1. - beta2_t))) in
      let v_hat_sqrt = P.map ~f:sqrt v_hat in
      (* momentum correction of learning rate *)
      let eta = Float.(eta / (1. - beta1_t)) in
      let dtheta = P.(m_hat / (v_hat_sqrt +$ c.eps)) in
      let new_theta = U.update ~learning_rate:eta ~theta:(params state) dtheta in
      { theta = new_theta; beta1_t; beta2_t; m = Some m; v = Some v }

  let manual_state_update state f =
    let p = P.value state.theta in
    { state with theta = U.manual_replace ~theta:state.theta (f p) }
end
