open Base
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S
module C = Owl.Dense.Matrix.C

type lds =
  { tau : float
  ; b : float
  ; beta : float
  ; sigma_eps : float
  }

type ('a, 'b) state =
  { x : 'a (* Markov state of the LDS *)
  ; y : 'a (* observations *)
  ; lds : 'b
  }

module Default = struct
  let tau_bounds = 1., 10.
  let b_bounds = -3., 3.
  let beta_bounds = 0.1, 3.
  let sigma = 1.
  let do_not_refresh_before = 100
  let refresh_every = 100
end

module Make (X : module type of Default) = struct
  (* generate a random LDS *)
  let fresh_lds () =
    let uniform (a, b) = Float.(a + Random.float (b - a)) in
    let tau = uniform X.tau_bounds in
    let sigma_eps = Float.(X.sigma *. sqrt ((2. / tau) - (1. /. square tau))) in
    { tau; b = uniform X.b_bounds; beta = uniform X.beta_bounds; sigma_eps }

  let sample t_max =
    (* sample initial state of lds *)
    let fresh_lds () =
      let lds = fresh_lds () in
      let x = Float.(lds.b + (X.sigma * Owl_stats.gaussian_rvs ~mu:0. ~sigma:1.)) in
      (lds, 0), x
    in
    (* populate state with y and lds parameters *)
    let state_of lds x =
      let noise = Owl_stats.gaussian_rvs ~mu:0. ~sigma:1. in
      let y = Float.(x + (lds.beta * noise)) in
      { x; y; lds }
    in
    let (lds, k), x = fresh_lds () in
    let state = state_of lds x in
    let states = Array.init t_max ~f:(fun _ -> state) in
    let rec iter t state (active_lds, k) =
      if t < t_max
      then (
        (* first, decide whether we want to keep the current LDS or refresh *)
        let (lds, k), x =
          if
            k < X.do_not_refresh_before
            || (k >= X.do_not_refresh_before && Random.int X.refresh_every > 0)
          then (
            let lds = active_lds in
            let eps = Owl_stats.gaussian_rvs ~mu:0. ~sigma:1. in
            (* one step forward in state evolution *)
            let x =
              Float.(state.x + ((lds.b - state.x) / lds.tau) + (lds.sigma_eps * eps))
            in
            (lds, k + 1), x)
          else fresh_lds ()
        in
        (* update the state *)
        let state = state_of lds x in
        states.(t) <- state;
        iter (t + 1) state (lds, k))
    in
    iter 0 (state_of lds x) (lds, k);
    states

  let minibatch ~t_max bs = Array.init bs ~f:(fun _ -> sample t_max)

  (* refactor from batch array of time array of states to a time list of (batch by x, batch by y and batch by lds parameters )*)
  let minibatch_as_data minibatch =
    let bs = Array.length minibatch in
    let t_max = Array.length minibatch.(0) in
    List.init t_max ~f:(fun t ->
      let x = Mat.init bs 1 (fun i -> minibatch.(i).(t).x) in
      let y = Mat.init bs 1 (fun i -> minibatch.(i).(t).y) in
      let lds = Array.init bs ~f:(fun i -> minibatch.(i).(t).lds) in
      { x; y; lds })

  (* recover stationary sigma^2 given sigma_rescaled *)
  let recover_sigma ~tau ~sigma_rescaled =
    let sigma = Float.(tau * sigma_rescaled / sqrt ((2. * tau) - 1.)) in
    Float.(int_pow sigma 2)

  (* kalman params for this particular lds. if random then sample randomly from global distribution. *)
  let kalman_params ~random lds =
    let uniform (a, b) = Float.(a + Random.float (b - a)) in
    let tau, beta =
      if random then uniform X.tau_bounds, uniform X.beta_bounds else lds.tau, lds.beta
    in
    let sigma_eps =
      if random
      then Float.(X.sigma *. sqrt ((2. / tau) - (1. /. square tau)))
      else lds.sigma_eps
    in
    let a = Float.(1. - (1. / lds.tau)) in
    let b = sigma_eps in
    let c = 1. in
    let r = Float.(int_pow beta 2) in
    let drift = Float.(lds.b / tau) in
    a, b, c, r, drift

  (* given data, return kalman predicted mean of x. if random, given data, return kalman predicted mean of x with true b but tau and sigma sampled randomly from the global distribution. *)
  let kalman_filter ~random data =
    Array.fold data ~init:(None, []) ~f:(fun (moments, pred_means) datum ->
      let a, b, c, r, drift = kalman_params ~random datum.lds in
      (* if first data point, or if prev_tau different from tau, reset prev_mean and prev_conv *)
      let prev_mean_recal, prev_cov_recal, curr_tau =
        match moments with
        | Some (prev_mean, prev_cov, prev_tau) when Float.(datum.lds.tau = prev_tau) ->
          prev_mean, prev_cov, prev_tau
        | _ -> datum.lds.b, Float.square X.sigma, datum.lds.tau
      in
      let curr_cov_inv =
        Float.((square c / r) + (1. / ((square a * prev_cov_recal) + square b)))
      in
      let curr_cov = Float.(1. / curr_cov_inv) in
      let curr_mean =
        Float.(
          (a * prev_mean_recal)
          + drift
          + (curr_cov * c * (datum.y - (c * ((a * prev_mean_recal) + drift))) / r))
      in
      (* update moments *)
      let moments = curr_mean, curr_cov, curr_tau in
      Some moments, curr_mean :: pred_means)
    |> snd
    |> List.rev
    |> Array.of_list

  (* baseline where we return the observation as predicted x *)
  let zero_filtering data = Array.map data ~f:(fun datum -> datum.y)

  let mse ~filter_fun minibatch =
    Array.map minibatch ~f:(fun data ->
      let pred = filter_fun data in
      Array.mapi pred ~f:(fun t xpred -> Float.(square (xpred - data.(t).x)))
      |> Owl.Stats.mean)
    |> Owl.Stats.mean

  (* mat of time by [x; y; b; kalman_x] for one session *)
  let to_save ~random data =
    let kf_pred = kalman_filter ~random data in
    Array.mapi data ~f:(fun t datum ->
      Mat.of_array [| datum.x; datum.y; datum.lds.b; kf_pred.(t) |] 1 4)
    |> Mat.concatenate ~axis:0
end

(** 2d; x_{t+1} = A x_{t} + b_tilde + Sigma_eps. b is the stationary mean and b_tilde is the drift; d is diagonal matrix used in sampling of a. sigma^2 (actual stationary covariance ) is (d+I) * 2/alpha. *)
type lds_2d =
  { a : Mat.mat
  ; b : Mat.mat
  ; b_tilde : Mat.mat
  ; beta : Mat.mat
  ; d : Mat.mat
  ; sigma : Mat.mat
  ; sigma_eps : Mat.mat
  }

module Default_2d = struct
  let tau_bounds = 1., 10.
  let b_bounds = -3., 3.
  let beta_bounds = 0.1, 3.

  (* for sampling of a *)
  let mu = 3.
  let a_d_bounds = 0.1, 0.9

  (* for lds transitions *)
  let do_not_refresh_before = 100
  let refresh_every = 100
end

module Make_2d (X : module type of Default_2d) = struct
  module Complex = Stdlib.Complex

  let uniform (a, b) = Float.(a + Random.float (b - a))
  let within x (a, b) = Float.(a < x) && Float.(x < b)

  (* sample a from a truncated exponential distribution *)
  let rec sample_a mu =
    let q, r, _ = Owl.Linalg.S.qr Mat.(gaussian 2 2) in
    let q = Mat.(q * signum (diag r)) in
    let d =
      Mat.init_2d 1 2 (fun _ _ -> 0.1 +. Owl_stats.exponential_rvs ~lambda:(1. /. mu))
    in
    let d = Mat.diagm d in
    let a = Mat.(transpose (sqrt d) * q * sqrt (reci (d +$ 1.))) in
    (* calculate eigvals of a *)
    let a_eigvals = Owl.Linalg.S.eigvals a in
    let a_eigvals_norm_1 = Complex.(norm (C.get a_eigvals 0 0)) in
    let a_eigvals_norm_2 = Complex.(norm (C.get a_eigvals 0 1)) in
    (* reject if eigvals too big or too small *)
    if
      within a_eigvals_norm_1 X.a_d_bounds && within a_eigvals_norm_2 X.a_d_bounds
      (* global covariance set to I + D *)
    then a, d
    else sample_a mu

  let sample_beta () =
    let beta_1, beta_2 = uniform X.beta_bounds, uniform X.beta_bounds in
    Mat.of_array [| beta_1; 0.; 0.; beta_2 |] 2 2

  (* calculate sigma_eps and sigma from d. *)
  let sigma_eps d =
    let alpha = Mat.(trace (eye 2 + d)) in
    let sigma_eps_2 = Mat.(eye 2 *$ Float.(2. / alpha)) in
    (* scale stationary sigma as well *)
    let sigma_2 = Mat.((eye 2 + d) *$ Float.(2. / alpha)) in
    let sigma_eps = Owl.Linalg.S.chol sigma_eps_2 ~upper:false in
    let sigma = Owl.Linalg.S.chol sigma_2 ~upper:false in
    sigma_eps, sigma

  (* given stationary mean, calculate drift *)
  let sample_b ~a =
    let b_1, b_2 = uniform X.b_bounds, uniform X.b_bounds in
    let b = Mat.of_array [| b_1; b_2 |] 2 1 in
    b, Mat.((eye 2 - a) *@ b)

  (* generate a random LDS *)
  let fresh_lds () =
    (* transition matrix and momentary global covariance matrix *)
    let a, d = sample_a X.mu in
    (* mean *)
    let b, b_tilde = sample_b ~a in
    (* transition std and stationary std *)
    let sigma_eps, sigma = sigma_eps d in
    (* observation noise *)
    let beta = sample_beta () in
    { a; b; b_tilde; beta; d; sigma; sigma_eps }

  let sample t_max =
    (* sample initial state of lds *)
    let fresh_lds () =
      let lds = fresh_lds () in
      let eps = Mat.gaussian ~mu:0. ~sigma:1. 2 1 in
      let x = Mat.(lds.b + (lds.sigma *@ eps)) in
      (lds, 0), x
    in
    (* populate state with y and lds parameters *)
    let state_of lds x =
      let noise = Mat.gaussian ~mu:0. ~sigma:1. 2 1 in
      let y = Mat.(x + (lds.beta *@ noise)) in
      { x; y; lds }
    in
    let (lds, k), x = fresh_lds () in
    let state = state_of lds x in
    let states = Array.init t_max ~f:(fun _ -> state) in
    let rec iter t state (active_lds, k) =
      if t < t_max
      then (
        (* first, decide whether we want to keep the current LDS or refresh *)
        let (lds, k), x =
          if
            k < X.do_not_refresh_before
            || (k >= X.do_not_refresh_before && Random.int X.refresh_every > 0)
          then (
            let lds = active_lds in
            let eps = Mat.gaussian ~mu:0. ~sigma:1. 2 1 in
            (* one step forward in state evolution *)
            let x = Mat.((lds.a *@ state.x) + lds.b_tilde + (lds.sigma_eps *@ eps)) in
            (lds, k + 1), x)
          else fresh_lds ()
        in
        (* update the state *)
        let state = state_of lds x in
        states.(t) <- state;
        iter (t + 1) state (lds, k))
    in
    iter 0 (state_of lds x) (lds, k);
    states

  let minibatch ~t_max bs = Array.init bs ~f:(fun _ -> sample t_max)

  (* refactor from batch array of time array of states to a time list of (batch by x, batch by y and batch by lds parameters )*)
  let minibatch_as_data minibatch =
    let bs = Array.length minibatch in
    let t_max = Array.length minibatch.(0) in
    List.init t_max ~f:(fun t ->
      let x =
        let x_array =
          Array.init bs ~f:(fun i -> Mat.(reshape minibatch.(i).(t).x [| 1; 2 |]))
        in
        Mat.concatenate x_array ~axis:0
      in
      let y =
        let y_array =
          Array.init bs ~f:(fun i -> Mat.(reshape minibatch.(i).(t).y [| 1; 2 |]))
        in
        Mat.concatenate y_array ~axis:0
      in
      let lds = Array.init bs ~f:(fun i -> minibatch.(i).(t).lds) in
      { x; y; lds })

  (* kalman params for this particular lds. if random then sample randomly from global distribution. *)
  let kalman_params ~random lds =
    let lds_a, d = if random then sample_a X.mu else lds.a, lds.d in
    let lds_b_tilde = if random then snd (sample_b ~a:lds_a) else lds.b_tilde in
    let beta = if random then sample_beta () else lds.beta in
    let sigma_eps, _ = sigma_eps d in
    let a = lds_a in
    let b = sigma_eps in
    let c = Mat.eye 2 in
    let r = Mat.(beta *@ beta) in
    let drift = lds_b_tilde in
    a, b, c, r, drift

  (* given data, return kalman predicted mean of x. if random, given data, return kalman predicted mean of x with true b but tau and sigma sampled randomly from the global distribution. *)
  let kalman_filter ~random data =
    Array.fold data ~init:(None, []) ~f:(fun (moments, pred_means) datum ->
      let a, b, _, r, drift = kalman_params ~random datum.lds in
      (* if first data point, or if prev_a different from a, reset prev_mean and prev_conv *)
      let prev_mean_recal, prev_cov_recal, curr_tau =
        match moments with
        | Some (prev_mean, prev_cov, prev_a) when Mat.(datum.lds.a = prev_a) ->
          prev_mean, prev_cov, prev_a
        | _ ->
          datum.lds.b, Mat.(datum.lds.sigma *@ transpose datum.lds.sigma), datum.lds.a
      in
      (* c is identity matrix; a is symmetric *)
      let r_inv = Mat.inv r in
      let curr_cov_inv = Mat.(r_inv + inv ((a *@ prev_cov_recal *@ a) + (b *@ b))) in
      let curr_cov = Mat.(inv curr_cov_inv) in
      let curr_mean =
        Mat.(
          (a *@ prev_mean_recal)
          + drift
          + (curr_cov * r_inv *@ (datum.y - ((a *@ prev_mean_recal) + drift))))
      in
      let moments = curr_mean, curr_cov, curr_tau in
      Some moments, curr_mean :: pred_means)
    |> snd
    |> List.rev
    |> Array.of_list

  (* baseline where we return the observation as predicted x *)
  let zero_filtering data = Array.map data ~f:(fun datum -> datum.y)

  let mse ~filter_fun minibatch =
    Array.map minibatch ~f:(fun data ->
      let pred = filter_fun data in
      let b =
        Array.mapi pred ~f:(fun t xpred ->
          Mat.(xpred - data.(t).x) |> Mat.sqr |> Mat.mean')
        |> Owl.Stats.mean
      in
      b)
    |> Owl.Stats.mean

  (* mat of time by [x; y; b; kalman_x] for one session *)
  let to_save ~random data =
    let kf_pred = kalman_filter ~random data in
    Array.mapi data ~f:(fun t datum ->
      let save_array =
        let save_list =
          [ Mat.to_array datum.x
          ; Mat.to_array datum.y
          ; Mat.to_array datum.lds.b
          ; Mat.to_array kf_pred.(t)
          ]
        in
        Array.concat save_list
      in
      Mat.of_array save_array 1 8)
    |> Mat.concatenate ~axis:0
end
