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
  (* Random.self_init (); *)
  Owl_stats_prng.init 1985;
  Torch_core.Wrapper.manual_seed 1985

let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run
let base = Optimizer.Config.Base.default

module PP = struct
  type 'a p =
    { w : 'a
    ; c : 'a
    ; a : 'a
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.Single)

(* neural network *)
module RNN = struct
  module P = P

  type input = unit

  let forward ~(theta : _ Maths.some P.t) ~input:_ y =
    let bs = Maths.shape y |> List.hd_exn in
    let y_tmp =
      Maths.concat
        [ y; Maths.(any (ones ~device:base.device ~kind:base.kind [ bs; 1 ])) ]
        ~dim:1
    in
    Maths.((y *@ theta.a) + (relu (y_tmp *@ theta.c) *@ theta.w))

  (* here data is a list of (x, optional labels) *)
  let f ~data ~y0 theta =
    let result, _ =
      List.foldi
        data
        ~init:(None, Maths.(any (of_tensor y0)))
        ~f:(fun t (accu, y) (x, labels) ->
          if t % 1 = 0 then Stdlib.Gc.major ();
          let y = forward ~theta ~input:x y in
          let accu =
            match labels with
            | None -> accu
            | Some labels ->
              let delta_ell =
                Loss.mse ~average_over:[ 0; 1 ] Maths.(of_tensor labels - y)
              in
              let delta_ggn =
                Loss.mse_ggn
                  ~average_over:[ 0; 1 ]
                  (Maths.const y)
                  ~vtgt:(Maths.tangent_exn y)
              in
              (match accu with
               | None -> Some (delta_ell, delta_ggn)
               | Some accu ->
                 let ell_accu, ggn_accu = accu in
                 Some (Maths.(ell_accu + delta_ell), Maths.C.(ggn_accu + delta_ggn)))
          in
          accu, y)
    in
    Option.value_exn result

  let init ~d ~dh : P.param =
    let w =
      Sofo.gaussian_tensor_normed ~kind:base.kind ~device:base.device ~sigma:0.1 [ dh; d ]
      |> Maths.of_tensor
      |> Prms.Single.free
    and c =
      Sofo.gaussian_tensor_normed
        ~kind:base.kind
        ~device:base.device
        ~sigma:1.
        [ d + 1; dh ]
      |> Maths.of_tensor
      |> Prms.Single.free
    and a = Maths.(eye ~device:base.device ~kind:base.kind d) |> Prms.Single.free in
    PP.{ w; c; a }

  let simulate ~(theta : _ Maths.some P.t) ~horizon y0 =
    let rec iter t accu y =
      if t = 0
      then List.rev accu
      else iter (t - 1) (y :: accu) (forward ~theta ~input:() y)
    in
    iter horizon [] Maths.(any (of_tensor y0))
end

(* -----------------------------------------
   -- Generate Lorenz data            ------
   ----------------------------------------- *)

let d = 3
let dh = 400
let batch_size = 128
let num_epochs_to_run = 4000
let n_trials_simulation = 10
let train_data = Lorenz_common.data 32
let test_horizon = 10000
let full_batch_size = Arr.(shape train_data).(1)
let _ = Sofo.print [%message (full_batch_size : int)]
let max_iter = Int.(full_batch_size * num_epochs_to_run / batch_size)

(* simulate n trials from saved parameters; first 3 columns are predictions and last 3 columns are ground truth *)
let simulate ~f_name n_trials =
  let model_params =
    let params_ba = P.C.load (in_dir f_name ^ "_params") in
    RNN.P.map params_ba ~f:(fun x -> x |> Maths.const)
  in
  let n_list = List.range 0 n_trials in
  List.iter n_list ~f:(fun j ->
    (* ground truth obtained from integration *)
    let y_true = Lorenz_common.data_test (test_horizon - 1) in
    (* use same initial condition to simulate with model *)
    let init_cond_sim = Mat.get_slice [ [ 0 ]; [] ] y_true in
    let simulated_arr =
      RNN.simulate
        ~theta:model_params
        ~horizon:test_horizon
        Tensor.(of_bigarray ~device:base.device init_cond_sim)
      |> List.map ~f:(fun yt ->
        let yt = Maths.to_tensor yt in
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

module O = Optimizer.SOFO (RNN.P)

let config =
  Optimizer.Config.SOFO.
    { base; learning_rate = Some 1.; n_tangents = 128; damping = `relative_from_top 1e-5 }

let rec loop ~t ~out ~state =
  Stdlib.Gc.major ();
  let init_cond, data =
    let trajectory = Lorenz_common.get_batch train_data batch_size in
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
  let theta, tangents = O.prepare ~config state in
  let loss, ggn = RNN.f ~data ~y0:init_cond theta in
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
  loop ~t:0 ~out ~state:(O.init (RNN.init ~d ~dh))
