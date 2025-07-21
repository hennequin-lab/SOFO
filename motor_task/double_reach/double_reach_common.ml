open Base
open Torch
open Forward_torch
open Motor
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

let print s = Stdio.printf "%s\n%!" (Base.Sexp.to_string_hum s)
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
  { target1 : float Arm.state
  ; target2 : float Arm.state
  ; is_prep : int -> bool
  ; is_move1 : int -> bool
  ; is_late_move1 : int -> bool
  ; is_late_move2 : int -> bool
  }

(* generation of a random trial *)
let trial ?prep_time () =
  (* random target *)
  let target1 = Arm.reach_target () in
  let target2 = Arm.reach_target () in
  (* exponentially distributed delay period in each trial, mean 300ms *)
  let go_cue1 =
    match prep_time with
    | Some t -> bins t
    | None -> bins Float.(0.2 + Random.float 0.4)
  in
  let go_cue2 = go_cue1 + bins 0.7 in
  let is_prep t = t < go_cue1 in
  let is_move1 t = t >= go_cue1 && t < go_cue2 in
  let is_late_move1 t = (t > Int.(go_cue1 + bins 0.6)) && t <= go_cue2 in
  let is_late_move2 t = t > go_cue2 + bins 0.6 in
  { target1; target2; is_prep; is_move1; is_late_move1; is_late_move2 }

module W = struct
  module P = R.P

  type data = int * trial list
  type args = unit

  let input_of trials =
    let open Arm in
    let n_trials = List.length trials in
    let input = Arr.zeros [| t_tot; n_trials; 6 |] in
    let set ~t ~k (a1, a2, b, c) =
      Arr.set input [| t; k; 0 |] a1;
      Arr.set input [| t; k; 1 |] a2;
      Arr.set input [| t; k; 2 |] b.pos.x1;
      Arr.set input [| t; k; 3 |] b.pos.x2;
      Arr.set input [| t; k; 4 |] c.pos.x1;
      Arr.set input [| t; k; 5 |] c.pos.x2
    in
    let zero =
      let zero = { x1 = 0.; x2 = 0. } in
      { pos = zero; vel = zero }
    in
    List.iteri trials ~f:(fun k trial ->
      for t = 0 to t_tot - 1 do
        if trial.is_prep t
        then set ~t ~k (1., 1., trial.target1, trial.target2)
        else if trial.is_move1 t
        then set ~t ~k (0., 1., zero, zero)
        else set ~t ~k (0., 0., zero, zero)
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
        if trial.is_prep t
        then (
          set ~t ~k Arm.central_spot;
          setw ~t ~k 1.)
        else if trial.is_late_move1 t
        then (
          set ~t ~k trial.target1;
          setw ~t ~k 4.)
        else if trial.is_late_move2 t
        then (
          set ~t ~k trial.target2;
          setw ~t ~k 4.)
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

  let f ~data prms =
    let t_max, trials = data in
    let bs = List.length trials in
    let input = input_of trials in
    let loss_items = loss_items_of trials in
    let z0 = Maths.concat ~dim:0 (List.init bs ~f:(fun _ -> prms.Rnn.PP.init_cond)) in
    let a0 = Arm.map (Arm.theta_init bs) ~f:Maths.any in
    let noise = R.draw_noise ~device:base.device ~t_max ~bs in
    let result, _, _ =
      List.range 0 t_max
      |> List.fold ~init:(None, z0, a0) ~f:(fun (accu, z, a) t ->
        if t % 1 = 0 then Stdlib.Gc.major ();
        (* update the state of the system *)
        let noise = R.noise_slice noise t in
        let z, a, _ = R.step_forward ?noise ~prms (input t) (z, a) in
        let hand = Arm.hand_of a in
        let delta_ell, delta_dggn = squared_loss (loss_items t hand) in
        let accu =
          match accu with
          | None -> Some (delta_ell, delta_dggn ())
          | Some accu ->
            let ell_accu, ggn_accu = accu in
            Some (Maths.(ell_accu + delta_ell), Maths.C.(ggn_accu + delta_dggn ()))
        in
        accu, z, a)
    in
    Option.value_exn result
end
