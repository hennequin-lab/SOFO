(* Example on training a vanilla RNN to learn the chaotic Lorenz attractor. *)
open Printf
open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

(* Setting seed. *)
let _ =
  Random.self_init ();
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

(* Command-line instruction on in which directory to save info. *)
let in_dir = Cmdargs.in_dir "-d"

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)
let base = Optimizer.Config.Base.default

(* Neural network. *)
module RNN = struct
  module PP = struct
    type 'a p =
      { w : 'a
      ; c : 'a
      ; b : 'a
      ; a : 'a
      }
    [@@deriving prms]
  end

  module P = PP.Make (Prms.P)

  type input = unit

  (* One-step forward function for the RNN. [theta] is the parameters and y is the network state. *)
  let f ~(theta : P.M.t) ~input:_ y =
    Maths.((y *@ theta.a) + (relu (theta.b + (y *@ theta.c)) *@ theta.w))

  (* Initialise parameters. *)
  let init ~d ~dh : P.tagged =
    let w =
      Convenience.gaussian_tensor_2d_normed
        ~kind:base.kind
        ~device:base.device
        ~a:dh
        ~b:d
        ~sigma:0.1
      |> Prms.free
    and c =
      Convenience.gaussian_tensor_2d_normed
        ~kind:base.kind
        ~device:base.device
        ~a:d
        ~b:dh
        ~sigma:1.
      |> Prms.free
    and b = Tensor.zeros ~device:base.device [ 1; dh ] |> Prms.free
    and a =
      Tensor.(mul_scalar (eye ~n:d ~options:(base.kind, base.device)) (Scalar.f 0.9))
    in
    { w; c; b; a = Prms.free a }
end

(* Feedforward model with MSE loss *)
module FF =
  Wrapper.Recurrent
    (RNN)
    (Loss.MSE (struct
         let scaling_factor = 1.
       end))

(* Instatiate optimiser. *)
(* Sofo configuration with learning rate, number of tangents and damping specified. *)
let config =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some 0.01
    ; n_tangents = 128
    ; rank_one = false
    ; damping = Some 1e-5
    ; momentum = None
    ; tan_from_act = false
    }

module O = Optimizer.SOFO (FF)

(* -----------------------------------------
   -- Generate Lorenz data            ------
   ----------------------------------------- *)

open Lorenz_common

(* Lorenz state dimension, which is the same as the RNN state dimension. *)
let d = 3

(* Inverted bottleneck dimension. *)
let dh = 400
let batch_size = 256
let num_epochs_to_run = 2000
let n_trials_simulation = 10
let train_data = data 32
let full_batch_size = Arr.(shape train_data).(1)
let train_data_batch = get_batch train_data
let test_horizon = 10000

let num_train_loops =
  Convenience.num_train_loops ~full_batch_size ~batch_size num_epochs_to_run

let epoch_of t = Convenience.epoch_of ~full_batch_size ~batch_size t

(* -----------------------------------------
   -- Optimization loop       ------
   ----------------------------------------- *)
let rec iter ~f_name ~config ~t ~state running_avg =
  Stdlib.Gc.full_major ();
  let e = epoch_of t in
  (* Get initial condition and the full trajectory as data. *)
  let init_cond, data =
    let trajectory = train_data_batch batch_size in
    let init_cond = List.hd_exn trajectory in
    ( Tensor.of_bigarray ~device:base.device init_cond
    , List.map trajectory ~f:(fun x ->
        let x = Tensor.of_bigarray ~device:base.device x in
        (), Some x) )
  in
  (* Take one optimization step to update state and returns loss. *)
  let loss, new_state = O.step ~config ~state ~data ~args:init_cond in
  let running_avg =
    if t % 10 = 0
    then (
      let loss_avg =
        match running_avg with
        | [] -> loss
        | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
      in
      Convenience.print [%message (e : float) (loss_avg : float)];
      (* Save loss information in a text file. *)
      Mat.(
        save_txt
          ~append:true
          ~out:(in_dir f_name)
          (of_array [| epoch_of t; loss_avg |] 1 3));
      [])
    else running_avg
  in
  if t < num_train_loops
  then iter ~f_name ~config ~t:(t + 1) ~state:new_state (loss :: running_avg)

(* Start the loop. *)
let _ =
  let f_name = "lorenz" in
  Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
  iter ~f_name ~config ~t:0 ~state:(O.init ~config RNN.(init ~d ~dh)) []
