open Printf
open Base
open Torch
open Forward_torch
open Motor
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S
open Double_reach_common

let prms_file = Cmdargs.(get_string "-load" |> force ~usage:"-load prms.bin")
let prms = R.P.T.load ~device:base.device prms_file |> R.P.map ~f:Maths.const

type 'a result =
  { network : 'a
  ; hand : 'a
  ; torques : 'a
  }

let prep_time = 0.5

let run_trials ~n_trials m =
  let trials =
    Random.init 1985;
    Owl_stats_prng.init 1985;
    List.init m ~f:(fun _ -> trial ~prep_time ())
  in
  let targets =
    trials
    |> List.map ~f:(fun trial ->
      Mat.of_array
        [| trial.target1.pos.x1
         ; trial.target1.pos.x2
         ; trial.target2.pos.x1
         ; trial.target2.pos.x2
        |]
        1
        4)
  in
  let input = W.input_of trials in
  let to_mat x = Tensor.to_bigarray ~kind:base.ba_kind x in
  let to_mat' x = to_mat (Maths.primal x) in
  let network = Arr.zeros [| n_trials; m; t_tot; n |] in
  let hand = Arr.zeros [| n_trials; m; t_tot; 2 |] in
  let torques = Arr.zeros [| n_trials; m; t_tot; 2 |] in
  let shape = [| 1; m; 1; -1 |] in
  for k = 0 to n_trials - 1 do
    print [%message "trial" (k : int)];
    let traj = R.forward ~t_max:t_tot ~prms input in
    List.iteri traj ~f:(fun t r ->
      (* log the network's activity *)
      let x = to_mat' r.network in
      Arr.set_slice [ [ k ]; []; [ t ]; [] ] network (Arr.reshape x shape);
      (* log the hand trajectories *)
      let x1, x2 =
        let h = Arm.hand_of r.arm in
        to_mat' h.pos.x1, to_mat' h.pos.x2
      in
      Arr.set_slice [ [ k ]; []; [ t ]; [ 0 ] ] hand (Arr.reshape x1 shape);
      Arr.set_slice [ [ k ]; []; [ t ]; [ 1 ] ] hand (Arr.reshape x2 shape);
      (* log the torques *)
      let tau1, tau2 = to_mat' (fst r.torques), to_mat' (snd r.torques) in
      Arr.set_slice [ [ k ]; []; [ t ]; [ 0 ] ] torques (Arr.reshape tau1 shape);
      Arr.set_slice [ [ k ]; []; [ t ]; [ 1 ] ] torques (Arr.reshape tau2 shape))
  done;
  let avg = Arr.mean ~keep_dims:false ~axis:0 in
  let first_trial x = Arr.(squeeze (get_slice [ [ 0 ] ] x)) in
  ( targets
  , { network = avg network; hand = first_trial hand; torques = first_trial torques } )

let _ =
  let m = 24 in
  (* number of reaching conditions *)
  let targets, result = run_trials ~n_trials:10 m in
  (* save targets *)
  List.iteri targets ~f:(fun i ti ->
    Mat.save_txt ~out:(in_dir (sprintf "target_%i" i)) ti);
  (* save all 64 neurons for all conditions, average over trials *)
  for i = 0 to 63 do
    result.network
    |> Arr.get_slice [ []; []; [ i ] ]
    |> Arr.squeeze
    |> Arr.transpose
    |> Mat.save_txt ~out:(in_dir (sprintf "neuron_%i" i))
  done;
  (* save all hand trajectories *)
  result.hand
  |> Arr.transpose ~axis:[| 1; 0; 2 |] (* time, cond, xy *)
  |> (fun x -> Arr.reshape x [| -1; m * 2 |])
  |> Mat.save_txt ~out:(in_dir "hand");
  (* save all torques *)
  result.torques
  |> Arr.transpose ~axis:[| 1; 0; 2 |]
  |> (fun x -> Arr.reshape x [| -1; m * 2 |])
  |> Mat.save_txt ~out:(in_dir "torques")
