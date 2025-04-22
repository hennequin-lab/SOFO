open Base
open Torch
open Forward_torch
open Sofo
open Motor
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

let print = Convenience.print
let in_dir = Cmdargs.in_dir "-d"
let base = Arm.base
let dt = 2e-3
let bins t = Float.(to_int (t / dt))
let t_tot = bins 2.
let bs = 256 (* batch size *)
let n = 64

module R = Rnn.Make (struct
    let dt = dt
    let tau = 20e-3
    let n = n
    let internal_noise_std = Some Float.(0.2 * sqrt (tau / dt))
  end)

type trial =
  { target : float Arm.state (* 1x2 tensor, desired hand pos *)
  ; is_idle : int -> bool
  ; is_prep : int -> bool
  ; is_late_move : int -> bool
  }

(* generation of a random trial *)
let trial ?idle_time ?prep_time ?angle ?radius () =
  (* random target *)
  let target = Arm.reach_target ?angle ?radius () in
  let prep_time =
    match prep_time with
    | Some t -> bins t
    | None -> bins Float.(0.2 + Random.float 0.6)
  in
  let idle_time =
    match idle_time with
    | Some t -> bins t
    | None -> bins Float.(0.1 + Random.float 0.4)
  in
  let go_cue = idle_time + prep_time in
  let is_idle t = t < idle_time in
  let is_prep t = t >= idle_time && t < go_cue in
  let is_late_move t = t > go_cue + bins 0.6 in
  { target; is_idle; is_prep; is_late_move }

module W = struct
  module P = R.P

  type data = int * trial list
  type args = unit

  let input_of trials =
    let open Arm in
    let n_trials = List.length trials in
    let input = Arr.zeros [| t_tot; n_trials; 3 |] in
    let set ~t ~k (a, b) =
      Arr.set input [| t; k; 0 |] a;
      Arr.set input [| t; k; 1 |] b.pos.x1;
      Arr.set input [| t; k; 2 |] b.pos.x2
    in
    let zero =
      let zero = { x1 = 0.; x2 = 0. } in
      { pos = zero; vel = zero }
    in
    List.iteri trials ~f:(fun k trial ->
      for t = 0 to t_tot - 1 do
        if trial.is_idle t
        then set ~t ~k (1., Arm.central_spot)
        else if trial.is_prep t
        then set ~t ~k (1., trial.target)
        else set ~t ~k (0., zero)
      done);
    let input = Tensor.of_bigarray ~device:base.device input in
    fun t ->
      Tensor.slice input ~dim:0 ~start:(Some t) ~end_:(Some (t + 1)) ~step:1
      |> Tensor.squeeze

  let loss_items_of trials =
    let open Arm in
    let open Loss in
    let n_trials = List.length trials in
    let targets = Arr.zeros [| t_tot; n_trials; 2 |] in
    let weights = Arr.zeros [| t_tot; n_trials |] in
    let set ~t ~k b =
      Arr.set targets [| t; k; 0 |] b.pos.x1;
      Arr.set targets [| t; k; 1 |] b.pos.x2
    in
    let setw ~t ~k = Arr.set weights [| t; k |] in
    List.iteri trials ~f:(fun k trial ->
      for t = 0 to t_tot - 1 do
        if trial.is_idle t || trial.is_prep t
        then (
          set ~t ~k Arm.central_spot;
          setw ~t ~k 1.)
        else (
          set ~t ~k trial.target;
          if trial.is_late_move t then setw ~t ~k 4.)
      done);
    let targets = Tensor.of_bigarray ~device:base.device targets in
    let weights = Tensor.of_bigarray ~device:base.device weights in
    let target_vel =
      { x1 = Tensor.zeros ~device:base.device [ n_trials; 1 ]
      ; x2 = Tensor.zeros ~device:base.device [ n_trials; 1 ]
      }
    in
    fun t hand ->
      let target_pos =
        let targ =
          Tensor.slice targets ~dim:0 ~start:(Some t) ~end_:(Some (t + 1)) ~step:1
          |> Tensor.squeeze
        in
        { x1 = Tensor.slice targ ~dim:1 ~start:(Some 0) ~end_:(Some 1) ~step:1
        ; x2 = Tensor.slice targ ~dim:1 ~start:(Some 1) ~end_:(Some 2) ~step:1
        }
      in
      let weights_pos =
        Tensor.slice weights ~dim:0 ~start:(Some t) ~end_:(Some (t + 1)) ~step:1
        |> Tensor.transpose ~dim0:0 ~dim1:1
      in
      let weights_vel = weights_pos in
      [ { weights = weights_pos; target = target_pos; output = hand.pos }
      ; { weights = weights_vel; target = target_vel; output = hand.vel }
      ]

  let squared_loss = Loss.squared_loss ~scaling_factor:Float.(10. / of_int t_tot)

  let f ~update ~data ~init ~args:_ prms =
    let t_max, trials = data in
    let bs = List.length trials in
    let input = input_of trials in
    let loss_items = loss_items_of trials in
    let z0 =
      Maths.concat_list ~dim:0 (List.init bs ~f:(fun _ -> prms.Rnn.PP.init_cond))
    in
    let a0 = Arm.theta_init bs in
    let noise = R.draw_noise ~device:base.device ~t_max ~bs in
    let result, _, _ =
      List.range 0 t_max
      |> List.fold ~init:(init, z0, a0) ~f:(fun (accu, z, a) t ->
        if t % 1 = 0 then Stdlib.Gc.major ();
        (* update the state of the system *)
        let noise = R.noise_slice noise t in
        let z, a, _ = R.step_forward ?noise ~prms (input t) (z, a) in
        let hand = Arm.hand_of a in
        let accu =
          let ell, dggn = squared_loss (loss_items t hand) in
          match update with
          | `loss_only u -> u accu (Some ell)
          | `loss_and_ggn u -> u accu (Some (ell, Some (dggn ())))
        in
        accu, z, a)
    in
    result
end
