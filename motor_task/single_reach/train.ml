open Printf
open Base
open Torch
open Forward_torch
open Sofo
open Motor
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S
open Single_reach_common
open Rnn_typ

let n_input_channels = 3

let _ =
  Random.init 1999;
  (* Random.self_init (); *)
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

(* -----------------------------------------
   --- Training for single reaches
   ----------------------------------------- *)

let max_iter = 2000
let name = "sofo"

let save_samples prms =
  let bs = 8 in
  let trials = List.init bs ~f:(fun _ -> trial ()) in
  let input = W.input_of trials in
  let traj = R.forward ~t_max:t_tot ~prms input |> Array.of_list in
  let to_mat x = Tensor.to_bigarray ~kind:base.ba_kind x in
  let to_mat' x = to_mat (Maths.to_tensor x) in
  List.iteri trials ~f:(fun i trial ->
    let hand =
      Array.map traj ~f:(fun r ->
        let h = Arm.hand_of r.arm in
        let x1 = Mat.get (to_mat' h.Arm.pos.x1) i 0 in
        let x2 = Mat.get (to_mat' h.Arm.pos.x2) i 0 in
        Mat.of_array [| x1; x2 |] 1 2)
      |> Mat.concatenate ~axis:0
    in
    let network =
      if i = 0 || i = 1
      then
        Array.map traj ~f:(fun r -> Mat.row (to_mat' r.network) i)
        |> Mat.concatenate ~axis:0
        |> Option.some
      else None
    in
    let torques =
      Array.map traj ~f:(fun r ->
        Mat.(row (to_mat' (fst r.torques)) i @|| row (to_mat' (snd r.torques)) i))
      |> Mat.concatenate ~axis:0
    in
    let target = Mat.(of_array [| trial.target.pos.x1; trial.target.pos.x2 |] 1 2) in
    Mat.save_txt ~out:(in_dir (sprintf "target%i_%s" i name)) target;
    Option.iter network ~f:(Mat.save_txt ~out:(in_dir (sprintf "x%i_%s" i name)));
    Mat.save_txt ~out:(in_dir (sprintf "h%i_%s" i name)) hand;
    Mat.save_txt ~out:(in_dir (sprintf "tau%i_%s" i name)) torques)

(* -----------------------------------------
   -- Optimization with SOFO    ------
   ----------------------------------------- *)

module O = Optimizer.SOFO (R.P)

let config =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some 0.1
    ; n_tangents = 128
    ; damping = `relative_from_top 1e-5
    }

let rec loop k wallclock state =
  if k < max_iter
  then (
    let data = t_tot, List.init bs ~f:(fun _ -> trial ()) in
    let theta, tangents = O.prepare ~config state in
    let tic = Unix.gettimeofday () in
    let loss, ggn = W.f ~data theta in
    let new_state = O.step ~config ~info:{ loss; ggn; tangents } state in
    let toc = Unix.gettimeofday () in
    let it_took = Float.(toc - tic) in
    let loss = Maths.to_float_exn (Maths.const loss) in
    (* guards against spikes in the loss *)
    let wallclock = Float.(wallclock + it_took) in
    print [%message (k : int) (loss : float)];
    Owl.Mat.(
      save_txt
        ~append:true
        ~out:(in_dir name)
        (of_array [| Float.of_int k; wallclock; loss |] 1 3));
    if k % 10 = 0
    then (
      let prms = W.P.value (O.params state) in
      (* if k % 50 = 0 then save_samples (W.P.map ~f:Maths.const prms); *)
      (* save the parameters so they can later be loaded for analysis *)
      prms |> W.P.C.save ~kind:base.ba_kind ~out:(in_dir (sprintf "prms_%s.bin" name)));
    loop (k + 1) wallclock new_state)

let prms = R.init ~base ~n_input_channels

(* Start the sofo loop. *)
let _ =
  Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
  Bos.Cmd.(v "rm" % "-f" % in_dir "aux") |> Bos.OS.Cmd.run |> ignore;
  loop 0 0. (O.init (R.init ~base ~n_input_channels))
