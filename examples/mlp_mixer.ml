open Printf
open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let _ =
  Random.init 1985;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let cast = Dense.Matrix.Generic.cast_d2s
let base = Optimizer.Config.Base.default

let config ~base_lr ~gamma =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some base_lr
    ; n_tangents = 256
    ; rank_one = false
    ; damping = gamma
    ; momentum = None
    ; adaptive_lr = true
    }

(* let config ~base_lr ~gamma:_ =
  Optimizer.Config.Adam.{ default with learning_rate = Some base_lr } *)

(* for MNIST *)
let input_dim = 28
let output_dim = 10
let full_batch_size = 60_000
let batch_size = 64
let num_epochs_to_run = 70
let max_iter = Convenience.num_train_loops ~full_batch_size ~batch_size num_epochs_to_run
let epoch_of t = Convenience.epoch_of ~full_batch_size ~batch_size t

(* network hyperparameters *)
let n_blocks = 2
let blocks_list = List.range 0 n_blocks
let n_layers = (n_blocks * 4) + 2
let in_channels = 1
let patch_size = 7

(* number of patches *)
let s = (input_dim / patch_size) ** 2

(* hidden size *)
let c = 32

(* expanded hidden size *)
let h = c * 2
let groups = 1

(* -----------------------------------------
   ---- Build MLP-mixer       ------
   ----------------------------------------- *)
module MLP_mixer = struct
  module MLP_Layer = struct
    type 'a t =
      { w : 'a
      ; b : 'a
      }
    [@@deriving prms]
  end

  module P = Prms.Array (MLP_Layer.Make (Prms.P))

  type input = Tensor.t

  let phi = Maths.relu

  (* use conv2d for patchification; https://github.com/rishikksh20/MLP-Mixer-pytorch/blob/master/mlp-mixer.py *)
  let patchify ~(theta : P.t') ~input =
    (* w has shape [out_channels x in_channels x kerl_x x kerl_y] *)
    Maths.conv2d
      (Maths.const input)
      theta.(0).w
      ~bias:theta.(0).b
      ~stride:(patch_size, patch_size)

  let token_mixer ~(theta : P.t') ~layer_idx1 ~layer_idx2 x =
    let hidden = Maths.einsum [ x, "msc"; theta.(layer_idx1).w, "sq" ] "mqc" in
    let hidden_bias = Maths.(phi (hidden + theta.(layer_idx1).b)) in
    let output = Maths.einsum [ hidden_bias, "mqc"; theta.(layer_idx2).w, "qs" ] "msc" in
    Maths.(output + theta.(layer_idx2).b)

  let channel_mixer ~(theta : P.t') ~layer_idx1 ~layer_idx2 x =
    let hidden = Maths.einsum [ x, "msc"; theta.(layer_idx1).w, "cq" ] "mqs" in
    let hidden_bias = Maths.(phi (hidden + theta.(layer_idx1).b)) in
    let output = Maths.einsum [ hidden_bias, "mqs"; theta.(layer_idx2).w, "qc" ] "msc" in
    Maths.(output + theta.(layer_idx2).b)

  let f ~(theta : P.t') ~(input : input) =
    (* patchify and map to hidden -> blocks -> classification *)
    let patchified_image = patchify ~theta ~input in
    let patchified_permute = Maths.permute patchified_image ~dims:[ 0; 2; 3; 1 ] in
    let batch_size = List.hd_exn (Tensor.shape input) in
    let patchified_final = Maths.view patchified_permute ~size:[ batch_size; s; c ] in
    let mixer_blocks =
      List.fold blocks_list ~init:patchified_final ~f:(fun acc i ->
        let token_mixed =
          token_mixer
            ~theta
            ~layer_idx1:Int.((i * 4) + 1)
            ~layer_idx2:Int.((i * 4) + 2)
            acc
        in
        let channel_mixed =
          channel_mixer
            ~theta
            ~layer_idx1:Int.((i * 4) + 3)
            ~layer_idx2:Int.((i + 1) * 4)
            token_mixed
        in
        channel_mixed)
    in
    let mixer_blocked_mean =
      Maths.mean_dim mixer_blocks ~dim:(Some [ 1 ]) ~keepdim:false
    in
    let final_layer = theta.(n_layers - 1) in
    let output = Maths.einsum [ mixer_blocked_mean, "mc"; final_layer.w, "cd" ] "md" in
    Maths.(output + final_layer.b)

  let init =
    let open MLP_Layer in
    (* patchify and map to hidden;  w has shape [out_channels x in_channels x kerl_x x kerl_y] *)
    let patchify =
      let w =
        let n_i = in_channels * patch_size * patch_size in
        let normaliser = Float.(1. / sqrt (of_int n_i)) in
        Tensor.mul_scalar_
          (Tensor.randn
             ~kind:base.kind
             ~device:base.device
             [ c; Int.(in_channels / groups); patch_size; patch_size ])
          (Scalar.f normaliser)
      in
      let b = Tensor.zeros ~kind:base.kind ~device:base.device [ c ] in
      { w; b }
    in
    let mixer_layers =
      List.map blocks_list ~f:(fun _ ->
        let token_hidden =
          let w =
            Convenience.gaussian_tensor_2d_normed
              ~kind:base.kind
              ~device:base.device
              ~a:s
              ~b:h
              ~sigma:1.
          in
          let b = Tensor.zeros ~kind:base.kind ~device:base.device [ 1; h; c ] in
          { w; b }
        in
        let token_output =
          let w =
            Convenience.gaussian_tensor_2d_normed
              ~kind:base.kind
              ~device:base.device
              ~a:h
              ~b:s
              ~sigma:1.
          in
          let b = Tensor.zeros ~kind:base.kind ~device:base.device [ 1; s; c ] in
          { w; b }
        in
        let channel_hidden =
          let w =
            Convenience.gaussian_tensor_2d_normed
              ~kind:base.kind
              ~device:base.device
              ~a:c
              ~b:h
              ~sigma:1.
          in
          let b = Tensor.zeros ~kind:base.kind ~device:base.device [ 1; h; s ] in
          { w; b }
        in
        let channel_output =
          let w =
            Convenience.gaussian_tensor_2d_normed
              ~kind:base.kind
              ~device:base.device
              ~a:h
              ~b:c
              ~sigma:1.
          in
          let b = Tensor.zeros ~kind:base.kind ~device:base.device [ 1; s; c ] in
          { w; b }
        in
        [| token_hidden; token_output; channel_hidden; channel_output |])
    in
    let classification_head =
      let w =
        Convenience.gaussian_tensor_2d_normed
          ~kind:base.kind
          ~device:base.device
          ~a:c
          ~b:output_dim
          ~sigma:1.
      in
      let b = Tensor.zeros ~kind:base.kind ~device:base.device [ 1; output_dim ] in
      { w; b }
    in
    Array.concat ([ [| patchify |] ] @ mixer_layers @ [ [| classification_head |] ])
    |> P.map ~f:Prms.free
end

(* feedforward model with mse loss *)
module FF =
  Wrapper.Feedforward
    (MLP_mixer)
    (Loss.CE (struct
         let scaling_factor = 1.
       end))

(* optimiser *)
module O = Optimizer.SOFO (FF)

(* -----------------------------------------
   ---- Read in MNIST data       ------
   ----------------------------------------- *)

let dataset typ =
  let suffix =
    match typ with
    | `train -> "train"
    | `test -> "test"
  in
  let x = Owl.Arr.load_npy ("_data/x_" ^ suffix ^ ".npy") in
  let y = Owl.Arr.load_npy ("_data/t_" ^ suffix ^ ".npy") |> Owl.Arr.one_hot output_dim in
  let mu = 0.13062754273414612
  and sigma = 0.30810779333114624 in
  let x = Owl.Arr.(((x /$ 255.) -$ mu) /$ sigma) in
  cast x, cast y

let train_set = dataset `train
let test_set = dataset `test

let sample_data (set_x, set_y) =
  let a = Mat.row_num set_x in
  fun batch_size ->
    if batch_size < 0
    then (
      (* reshape x to [batch size x 1 x input_dim x input_dim]. *)
      let total_bs = Mat.row_num set_x in
      let xs_tensor = Tensor.of_bigarray ~device:base.device set_x in
      let xs = Tensor.reshape xs_tensor ~shape:[ total_bs; 1; input_dim; input_dim ] in
      xs, Tensor.of_bigarray ~device:base.device set_y)
    else (
      let ids = List.init batch_size ~f:(fun _ -> Random.int a) in
      let x_tensor =
        Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_x)
      in
      let xs = Tensor.reshape x_tensor ~shape:[ batch_size; 1; input_dim; input_dim ] in
      let ys = Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_y) in
      xs, ys)

let test_eval ~train_data theta =
  let x, y =
    match train_data with
    | None -> sample_data test_set (-1)
    | Some train_data -> train_data
  in
  let logits = MLP_mixer.f ~theta ~input:x |> Maths.primal in
  let _, max_y = Tensor.max_dim ~keepdim:false ~dim:1 y in
  let _, max_ypred = Tensor.max_dim ~keepdim:false ~dim:1 logits in
  Tensor.eq_tensor max_y max_ypred
  |> Tensor.to_dtype ~dtype:base.kind ~non_blocking:false ~copy:false
  |> Tensor.mean
  |> Tensor.to_float0_exn

let optimise ~max_iter ~f_name config =
  let rec loop ~iter ~state ~time_elapsed running_avg =
    Stdlib.Gc.major ();
    let e = epoch_of iter in
    let data = sample_data train_set batch_size in
    let t0 = Unix.gettimeofday () in
    let loss, new_state = O.step ~config ~state ~data ~args:() in
    let t1 = Unix.gettimeofday () in
    let time_elapsed = Float.(time_elapsed + t1 - t0) in
    let running_avg =
      if iter % 10 = 0
      then (
        let loss_avg =
          match running_avg with
          | [] -> loss
          | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
        in
        let test_acc =
          test_eval ~train_data:None MLP_mixer.P.(const (value (O.params new_state)))
        in
        let train_acc =
          test_eval
            ~train_data:(Some data)
            MLP_mixer.P.(const (value (O.params new_state)))
        in
        (* save params *)
        if iter % 100 = 0
        then (
          let ba_params =
            MLP_mixer.P.map
              (MLP_mixer.P.value (O.params new_state))
              ~f:(fun x -> Tensor.to_bigarray ~kind:base.ba_kind x)
          in
          O.W.P.save ba_params ~out:(in_dir f_name ^ "_params");
          Convenience.print [%message (e : float) (loss_avg : float) (test_acc : float)]);
        Mat.(
          save_txt
            ~append:true
            ~out:(in_dir f_name)
            (of_array [| e; time_elapsed; loss_avg; test_acc; train_acc |] 1 5));
        [])
      else running_avg
    in
    if iter < max_iter
    then loop ~iter:(iter + 1) ~state:new_state ~time_elapsed (loss :: running_avg)
  in
  loop
    ~iter:0 (* ~config:(config_f ~iter:0) *)
    ~state:(O.init ~config MLP_mixer.(init))
    ~time_elapsed:0.
    []

let damping_rates = [ None ]
let lr_rates = [ 1e-3; 5e-4; 1e-4 ]
let meth = "ggn"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    List.iter damping_rates ~f:(fun gamma ->
      let config_f = config ~base_lr:eta ~gamma in
      let f_name =
        sprintf
          "mlp_mixer_%s_lr_%s_adaptive_lr_%s"
          meth
          (Float.to_string eta)
          (Bool.to_string config_f.adaptive_lr)
      in
      Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
      optimise ~max_iter ~f_name config_f))

(* let damping_rates = [ None ]
   let lr_rates = [ 1e-3 ]
   let meth = "adam"

   let _ =
   List.iter lr_rates ~f:(fun eta ->
   List.iter damping_rates ~f:(fun gamma ->
   let config_f = config ~base_lr:eta ~gamma in
   let gamma_name = Option.value_map gamma ~default:"none" ~f:Float.to_string in
   let f_name =
   sprintf "mlp_mixer_%s_lr_%s_damp_%s" meth (Float.to_string eta) gamma_name
   in
   Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
   optimise ~max_iter ~f_name config_f)) *)
