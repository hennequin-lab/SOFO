open Base
open Forward_torch
open Torch
include Optimizer_typ

(* update loss and ggn at each step *)
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

(* update loss at each step *)
let update_loss_each_step accu info =
  match info with
  | None -> accu
  | Some ell ->
    let loss = Maths.(accu + ell) in
    loss

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

(* sample tensors from Gaussian *)
let sample_rand_tensor ~base ~rank_one ~shape =
  let[@warning "-8"] (n_tangents :: param_shape) = shape in
  let vs =
    if rank_one
    then rank_one_tensor ~base ~n_tangents param_shape
    else Tensor.randn ~device:base.device ~kind:base.kind shape
  in
  vs

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

module SOFO (W : Wrapper.T) = struct
  module W = W

  type ('a, 'b) config = ('a, 'b) Config.SOFO.t
  type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.SOFO.t -> W.P.tagged -> 'c

  type state =
    { theta : W.P.tagged
    ; g_avg : Tensor.t W.P.p option
    ; beta_t : float option
    ; damping : float option
    ; prev_losses : float list
    }

  let params state = state.theta

  let init ~(config : ('a, 'b) config) theta =
    { theta
    ; g_avg = None
    ; beta_t = Option.map config.momentum ~f:(fun _ -> 1.)
    ; damping = config.damping
    ; prev_losses = []
    }

  (* initialise tangents, where each tangent is normalised. *)
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

  (* fold vs over sets of v_i s, multiply with associated weights. *)
  let weighted_vs_sum vs ~weights =
    W.P.map vs ~f:(fun v ->
      let v_i = Maths.tangent' v in
      let[@warning "-8"] (n_samples :: s) = Tensor.shape v_i in
      let n_ws = Convenience.first_dim weights in
      let v_i = Tensor.view v_i ~size:[ n_samples; -1 ] in
      let s = if n_ws = 1 then s else n_ws :: s in
      Tensor.(view (matmul weights v_i) ~size:s))

  (* calculate natural gradient = V(VtGtGV)^-1 V^t g *)
  let natural_g ?damping ~vs ~ggn vtg =
    let u, s, _ = Tensor.svd ~some:true ~compute_uv:true ggn in
    (* how each V should be weighted, as a row array *)
    let weights =
      let tmp = Convenience.a_trans_b u vtg in
      let s =
        match damping with
        | None -> s
        | Some gamma ->
          let offset = Float.(gamma * Tensor.(maximum s |> to_float0_exn)) in
          Tensor.(s + f offset)
      in
      Tensor.(matmul (u / s) tmp) |> Convenience.trans_2d
    in
    weighted_vs_sum vs ~weights

  (* gradient descent on theta *)
  let update_theta ?learning_rate ~theta dtheta =
    match learning_rate with
    | Some eta ->
      let open W.P.Let_syntax in
      let+ theta = theta
      and++ g = dtheta in
      Tensor.(theta - mul_scalar g (Scalar.f eta))
    | None -> theta

  let step ~(config : ('a, 'b) config) ~state ~data ~args =
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
    let learning_rate =
      Option.map config.learning_rate ~f:(fun eta ->
        Option.value_map beta_t ~default:eta ~f:(fun b -> Float.(eta / (1. - b))))
    in
    let updated_damping =
      if config.lm
      then (
        let natural_g_tmp = natural_g ?damping:state.damping ~vs ~ggn:final_ggn vtg in
        (* levenberg-marquardt *)
        let lm =
          (* use natural_g_tmp for a forward pass *)
          let theta_tmp = update_theta ?learning_rate ~theta natural_g_tmp in
          (* loss after updating *)
          let vs_tmp =
            init_tangents
              ~base:config.base
              ~rank_one:config.rank_one
              ~n_tangents:config.n_tangents
              theta_tmp
          in
          let theta_dual_tmp = W.P.make_dual theta_tmp ~t:vs_tmp in
          (* compute the losses *)
          let final_losses_tmp, loss_tangents_tmp =
            W.f ~update ~data ~init ~args theta_dual_tmp
          in
          let vtg_tmp =
            Tensor.(
              mean_dim
                loss_tangents_tmp
                ~dtype:(type_ loss_tangents_tmp)
                ~dim:(Some [ 1 ])
                ~keepdim:true)
          in
          let num =
            let loss_tmp =
              final_losses_tmp |> Maths.primal |> Tensor.mean |> Tensor.to_float0_exn
            in
            Float.(loss_tmp - loss)
          in
          let denom =
            let tmp1 =
              let lr = Option.value_map learning_rate ~default:0. ~f:(fun x -> x) in
              Float.(lr * (-1. + (lr / 2.)))
            in
            let tmp2 =
              let vanilla_g_tmp =
                let num_params = Float.(of_int (W.P.T.numel (W.P.value theta_tmp))) in
                let weights =
                  Tensor.(
                    div_scalar
                      (transpose ~dim0:1 ~dim1:0 vtg_tmp)
                      Scalar.(f Float.(of_int config.n_tangents / num_params)))
                in
                weighted_vs_sum vs_tmp ~weights
              in
              W.P.fold2 natural_g_tmp vanilla_g_tmp ~init:None ~f:(fun accu (x, y, _) ->
                let (z : Tensor.t) = Tensor.(sum (x * y)) in
                match accu with
                | None -> Some z
                | Some a -> Some Tensor.(a + z))
              |> Option.value_exn
              |> Tensor.to_float0_exn
            in
            Float.(tmp1 * tmp2)
          in
          Float.(num / denom)
        in
        let lm_damping =
          if Float.(lm > 1.)
          then Option.map state.damping ~f:(fun x -> Float.(x * 2. / 3.))
          else if Float.(lm < 0.)
          then Option.map state.damping ~f:(fun x -> Float.(x * 1.5))
          else state.damping
        in
        Convenience.print [%message (lm : float) (Option.value_exn lm_damping : float)];
        lm_damping)
      else state.damping
    in
    (* compute natural gradient and update theta *)
    let natural_g = natural_g ?damping:updated_damping ~vs ~ggn:final_ggn vtg in
    let natural_g_avg =
      let module M = Momentum (W.P) in
      M.apply ?momentum:config.momentum ~avg:state.g_avg natural_g
    in
    let new_theta = update_theta ?learning_rate ~theta natural_g_avg in
    let prev_losses =
      if List.length state.prev_losses < 100
      then loss :: state.prev_losses
      else loss :: List.drop_last_exn state.prev_losses
    in
    (* perturb with noise if stuck in local minima *)
    let perturbed =
      match config.perturb_thresh with
      | None -> false
      | Some thresh ->
        if List.length prev_losses = 100
        then (
          let recent =
            List.fold ~init:0. (List.take prev_losses 50) ~f:(fun acc x -> x +. acc)
          in
          let old =
            List.fold
              ~init:0.
              (List.take (List.rev prev_losses) 50)
              ~f:(fun acc x -> x +. acc)
          in
          let diff = Float.(abs ((old - recent) / old)) in
          Float.(diff < thresh))
        else false
    in
    let new_theta =
      if perturbed
      then (
        let _ = Convenience.print [%message "perturbed"] in
        let open W.P.Let_syntax in
        let+ new_theta = new_theta in
        Tensor.(new_theta + mul_scalar (rand_like new_theta) (Scalar.f 0.01)))
      else new_theta
    in
    let prev_losses = if perturbed then [] else prev_losses in
    ( loss
    , { theta = new_theta
      ; g_avg = Some natural_g_avg
      ; beta_t
      ; damping = updated_damping
      ; prev_losses
      } )
end

(* forward gradient descent;: g = V V^T g where V^Tg is obtained with forward AD (Kozak 2021) *)
module FGD (W : Wrapper.T) = struct
  module W = W

  type ('a, 'b) config = ('a, 'b) Config.FGD.t
  type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.FGD.t -> W.P.tagged -> 'c

  type state =
    { theta : W.P.tagged
    ; g_avg : Tensor.t W.P.p option
    ; beta_t : float option
    }

  let params state = state.theta

  let init ~(config : ('a, 'b) config) theta =
    { theta; g_avg = None; beta_t = Option.map config.momentum ~f:(fun _ -> 1.) }

  (* initialise tangents, where each tangent is normalised. *)
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

  (* fold vs over sets of v_i s, multiply with associated weights. *)
  let weighted_vs_sum vs ~weights =
    W.P.map vs ~f:(fun v ->
      let v_i = Maths.tangent' v in
      let[@warning "-8"] (n_samples :: s) = Tensor.shape v_i in
      let n_ws = Convenience.first_dim weights in
      let v_i = Tensor.view v_i ~size:[ n_samples; -1 ] in
      let s = if n_ws = 1 then s else n_ws :: s in
      Tensor.(view (matmul weights v_i) ~size:s))

  (* gradient descent on theta *)
  let update_theta ?learning_rate ~theta dtheta =
    match learning_rate with
    | Some eta ->
      let open W.P.Let_syntax in
      let+ theta = theta
      and++ g = dtheta in
      Tensor.(theta - mul_scalar g (Scalar.f eta))
    | None -> theta

  let step ~(config : ('a, 'b) config) ~state ~data ~args =
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

module SGD (W : Wrapper.T) = struct
  module W = W

  type ('a, 'b) config = ('a, 'b) Config.SGD.t
  type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.SGD.t -> W.P.tagged -> 'c

  type state =
    { theta : W.P.tagged
    ; g_avg : Tensor.t W.P.p option
    ; beta_t : float option
    }

  let params state = state.theta

  let init ~(config : ('a, 'b) config) theta =
    { theta; g_avg = None; beta_t = Option.map config.momentum ~f:(fun _ -> 1.) }

  (* gradient descent on theta *)
  let update_theta ?learning_rate ~theta dtheta =
    match learning_rate with
    | Some eta ->
      let open W.P.Let_syntax in
      let+ theta = theta
      and++ g = dtheta in
      Tensor.(theta - mul_scalar g (Scalar.f eta))
    | None -> theta

  let step ~(config : ('a, 'b) config) ~state ~data ~args =
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

  (* gradient descent on theta *)
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

  let step ~(config : ('a, 'b) config) ~state ~data ~args =
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
      Tensor.to_float0_exn loss, W.P.map (W.P.primal theta_dual) ~f:Tensor.grad
    in
    let new_state : state = update_theta_m_v ~config ~state true_g in
    loss, new_state
end
