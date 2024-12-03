open Printf
open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let _ =
  (* Random.init 1999; *)
  Random.self_init ();
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run
let base = Optimizer.Config.Base.default

let config ~base_lr ~gamma =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some base_lr
    ; n_tangents = 128
    ; rank_one = false
    ; damping = gamma
    ; momentum = None
    }

(* neural network *)
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

  let f ~(theta : P.M.t) ~input:_ y =
    Maths.((y *@ theta.a) + (relu (theta.b + (y *@ theta.c)) *@ theta.w))

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

  let simulate ~(theta : P.M.t) ~horizon y0 =
    let rec iter t accu y =
      if t = 0 then List.rev accu else iter (t - 1) (y :: accu) (f ~theta ~input:() y)
    in
    iter horizon [] (Maths.const y0)
end

(* feedforward model with mse loss *)
module FF =
  Wrapper.Recurrent
    (RNN)
    (Loss.MSE (struct
         let scaling_factor = 1.
       end))

(* optimiser *)
module O = Optimizer.SOFO (FF)

(* -----------------------------------------
   -- Generate Lorenz data            ------
   ----------------------------------------- *)

open Lorenz_common

let d = 3
let dh = 400
let batch_size = 256
let num_epochs_to_run = 2000
let n_trials_simulation = 10
let train_data = data 32
let full_batch_size = Arr.(shape train_data).(1)
let train_data_batch = get_batch train_data
let test_horizon = 10000
let full_batch_size = Arr.(shape train_data).(1)
let _ = Convenience.print [%message (full_batch_size : int)]
let train_data_batch = get_batch train_data

let num_train_loops =
  Convenience.num_train_loops ~full_batch_size ~batch_size num_epochs_to_run

let epoch_of t = Convenience.epoch_of ~full_batch_size ~batch_size t

(* simulate n trials from saved parameters; first 3 columns are predictions and last 3 columns are ground truth *)
let simulate ~f_name n_trials =
  let model_params =
    let params_ba = O.W.P.T.load (in_dir f_name ^ "_params") in
    RNN.P.map params_ba ~f:(fun x -> x |> Maths.const)
  in
  let n_list = List.range 0 n_trials in
  List.iter n_list ~f:(fun j ->
    (* ground truth obtained from integration *)
    let y_true = data_test (test_horizon - 1) in
    (* use same initial condition to simulate with model *)
    let init_cond_sim = Mat.get_slice [ [ 0 ]; [] ] y_true in
    let simulated_arr =
      RNN.simulate
        ~theta:model_params
        ~horizon:test_horizon
        Tensor.(of_bigarray ~device:base.device init_cond_sim)
      |> List.map ~f:(fun yt ->
        let yt = Maths.primal yt in
        let yt = Tensor.to_bigarray ~kind:base.ba_kind yt in
        Arr.expand yt 3)
      |> Array.of_list
      |> Arr.concatenate ~axis:0
      |> Arr.transpose ~axis:[| 1; 0; 2 |]
    in
    simulated_arr
    |> Arr.iter_slice ~axis:0 (fun yi ->
      let yi = Arr.squeeze yi in
      let yi_tot = Mat.concat_horizontal yi y_true in
      Mat.save_txt ~out:(in_dir (sprintf "%s_autonomous%i" f_name j)) yi_tot))

let rec iter ~f_name ~config ~t ~state ~time_elapsed running_avg =
  Stdlib.Gc.full_major ();
  let e = epoch_of t in
  let init_cond, data =
    let trajectory = train_data_batch batch_size in
    let init_cond = List.hd_exn trajectory in
    ( Tensor.of_bigarray ~device:base.device init_cond
    , List.mapi trajectory ~f:(fun tt x ->
        (* only label provided is the end point *)
        if tt = 31
        then (
          let x = Tensor.of_bigarray ~device:base.device x in
          (), Some x)
        else (), None) )
  in
  let t0 = Unix.gettimeofday () in
  let loss, new_state = O.step ~config ~state ~data ~args:init_cond in
  let t1 = Unix.gettimeofday () in
  let time_elapsed = Float.(time_elapsed + t1 - t0) in
  let running_avg =
    if t % 10 = 0
    then (
      (* save params *)
      O.W.P.T.save
        (RNN.P.value (O.params new_state))
        ~kind:base.ba_kind
        ~out:(in_dir f_name ^ "_params");
      (* if t % 500 = 0 then simulate ~f_name n_trials_simulation; *)
      let loss_avg =
        match running_avg with
        | [] -> loss
        | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
      in
      Convenience.print [%message (e : float) (loss_avg : float)];
      Mat.(
        save_txt
          ~append:true
          ~out:(in_dir f_name)
          (of_array [| epoch_of t; time_elapsed; loss_avg |] 1 3));
      [])
    else running_avg
  in
  if t < num_train_loops
  then iter ~f_name ~config ~t:(t + 1) ~state:new_state ~time_elapsed (loss :: running_avg)

let lr_rate = 0.01
let damping = Some 1e-5
let meth = "ggn"

let _ =
  let config = config ~base_lr:0.01 ~gamma:(Some 1e-05) in
  let f_name = "lorenz" in
  Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
  iter ~f_name ~config ~t:0 ~state:(O.init ~config RNN.(init ~d ~dh)) ~time_elapsed:0. []
