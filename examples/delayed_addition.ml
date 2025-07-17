(** Learning a delayed addition task to compare SOFO with adam. *)

open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let batch_size = 256
let max_iter = 20000

let _ =
  Random.init 1999;
  (* Random.self_init (); *)
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default

module Settings = struct
  (* length of data *)
  let n_steps = 1000 (* 20 to 600 *)

  (* first signal upper bound *)
  let t1_bound = 10
  let t2_bound = Int.(n_steps / 2)
end

(* net parameters *)
let n = 128 (* number of neurons *)
let alpha = 0.25

module RNN_P = struct
  type 'a p =
    { c : 'a
    ; b : 'a
    ; o : 'a
    }
  [@@deriving prms]
end

module P = RNN_P.Make (Prms.Single)

(* neural network *)
module RNN = struct
  module P = P

  let phi x = Maths.relu x

  (* input is the (number, signal) pair and z is the internal state *)
  let forward ~(theta : _ Maths.some P.t) ~input z =
    let bs = Tensor.shape input |> List.hd_exn in
    let input =
      Maths.C.concat
        [ Maths.of_tensor input
        ; Maths.of_tensor (Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ])
        ]
        ~dim:1
    in
    match z with
    | Some z ->
      let leak = Maths.(Float.(1. - alpha) $* z) in
      Maths.(leak + (alpha $* phi ((z *@ theta.c) + (input *@ theta.b))))
    | None ->
      (* this is the equivalent of initialising at z = 0 *)
      Maths.(phi (input *@ theta.b))

  let prediction ~(theta : _ Maths.some P.t) z = Maths.(z *@ theta.o)

  let init : P.param =
    let c =
      Sofo.gaussian_tensor_normed ~kind:base.kind ~device:base.device ~sigma:1.0 [ n; n ]
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    let b =
      Sofo.gaussian_tensor_normed ~kind:base.kind ~device:base.device ~sigma:1.0 [ 3; n ]
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    (* initialise to repeat observation *)
    let o =
      Sofo.gaussian_tensor_normed ~kind:base.kind ~device:base.device ~sigma:1.0 [ n; 2 ]
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    { c; b; o }

  (* here data is a list of (x_t, optional labels). labels is x_t. *)
  let f ~data theta =
    let result, _ =
      let input_all, labels_all = data in
      let top_2, _ = List.split_n (Tensor.shape input_all) 2 in
      let time_list = List.range 0 Settings.n_steps in
      List.fold time_list ~init:(None, None) ~f:(fun (accu, z) t ->
        if t % 1 = 0 then Stdlib.Gc.major ();
        let input =
          let tmp =
            Tensor.slice ~dim:2 ~start:(Some t) ~end_:(Some (t + 1)) ~step:1 input_all
          in
          Tensor.reshape tmp ~shape:top_2
        in
        (* loss only calculated at the final timestep *)
        let labels = if t = Settings.n_steps - 1 then Some labels_all else None in
        let z = forward ~theta ~input z in
        let accu =
          match labels with
          | None -> accu
          | Some labels ->
            let pred = prediction ~theta z in
            let delta_ell =
              Loss.mse ~average_over:[ 0; 1 ] Maths.(of_tensor labels - pred)
            in
            let delta_ggn =
              Loss.mse_ggn
                ~average_over:[ 0; 1 ]
                (Maths.const pred)
                ~vtgt:(Maths.tangent_exn pred)
            in
            (match accu with
             | None -> Some (delta_ell, delta_ggn)
             | Some accu ->
               let ell_accu, ggn_accu = accu in
               Some (Maths.(ell_accu + delta_ell), Maths.C.(ggn_accu + delta_ggn)))
        in
        accu, Some z)
    in
    Option.value_exn result
end

(* -----------------------------------------
   -- Generate addition data    ------
   ----------------------------------------- *)

let data_shape = [| 1; 2; Settings.n_steps |]

let sample () =
  let number_trace = Mat.uniform 1 Settings.n_steps in
  let signal_trace = Mat.zeros 1 Settings.n_steps in
  (* set indicator *)
  let t1 = Random.int_incl 0 (Settings.t1_bound - 1) in
  let t2 = Random.int_incl (t1 + 1) Settings.t2_bound in
  Mat.set signal_trace 0 t1 1.;
  Mat.set signal_trace 0 t2 1.;
  let target = Mat.(get number_trace 0 t1) +. Mat.(get number_trace 0 t2) in
  let target_mat = Mat.of_array [| target |] 1 1 in
  let input_mat = Mat.concat_horizontal number_trace signal_trace in
  let input_array = Arr.reshape input_mat data_shape in
  input_array, target_mat

let sample_data batch_size =
  let data_minibatch = Array.init batch_size ~f:(fun _ -> sample ()) in
  let input_array = Array.map data_minibatch ~f:fst in
  let target_array = Array.map data_minibatch ~f:snd in
  let input_tensor = Arr.concatenate ~axis:0 input_array in
  let target_mat = Mat.concatenate ~axis:0 target_array in
  let to_device = Tensor.of_bigarray ~device:base.device in
  to_device input_tensor, to_device target_mat

(* -----------------------------------------
   -- Optimization with SOFO    ------
   ----------------------------------------- *)

module O = Optimizer.SOFO (RNN.P)

let config =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some 0.1
    ; n_tangents = 128
    ; damping = `relative_from_top 1e-5
    }

let rec loop ~t ~out ~state =
  Stdlib.Gc.major ();
  let data = sample_data batch_size in
  let theta, tangents = O.prepare ~config state in
  let loss, ggn = RNN.f ~data theta in
  let new_state = O.step ~config ~info:{ loss; ggn; tangents } state in
  if t % 10 = 0
  then (
    (* save params *)
    O.P.C.save
      (RNN.P.value (O.params new_state))
      ~kind:base.ba_kind
      ~out:(in_dir "sofo_params");
    let loss = Maths.to_float_exn (Maths.const loss) in
    print [%message (t : int) (loss : float)];
    Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss |] 1 2)));
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state

(* Start the sofo loop. *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init RNN.init)
