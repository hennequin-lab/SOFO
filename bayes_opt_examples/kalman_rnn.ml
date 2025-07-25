(** Meta learning a kalman filter with a vanilla rnn. *)
open Base

open Owl
open Torch
open Forward_torch
open Sofo
open Bayes_opt_common
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

(* state dimension *)
let s = 1

module Data = Kalman_data.Make (Kalman_data.Default)

let tmax = 5000
let batch_size = 256
let max_iter = 4000
(* let n_trials_simulation = 500 *)
(* let n_trials_baseline = 1000 *)

let _ =
  Random.init 1999;
  (* Random.self_init (); *)
  Owl_stats_prng.init 1999;
  Torch_core.Wrapper.manual_seed 1999

let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default
let bayes_opt = Option.value (Cmdargs.get_bool "-bayes_opt") ~default:false

(* generate some random data to get baseline performance of kalman *)
(* let kalman_baseline =
   let open Data in
   Array.init 2 ~f:(fun _ ->
   let data = minibatch ~tmax n_trials_baseline in
   (* save [x; y; b; kalman_x]*)
   let _ =
   Mat.save_txt ~out:(in_dir "example_session") (to_save ~random:false data.(0))
   in
   mse ~filter_fun:(kalman_filter ~random:false) data)
   |> Stats.mean

   let kalman_random_baseline =
   let open Data in
   Array.init 2 ~f:(fun _ ->
   let data = minibatch ~tmax n_trials_baseline in
   (* save [x; y; b; kalman_random_x]*)
   let _ =
   Mat.save_txt ~out:(in_dir "example_session_random") (to_save ~random:true data.(0))
   in
   mse ~filter_fun:(kalman_filter ~random:true) data)
   |> Stats.mean *)

(* let kalman_baseline = 0.4896232
let kalman_random_baseline = 1.3612788 *)

module RNN_P = struct
  type 'a p =
    { c : 'a
    ; b : 'a
    ; o : 'a
    }
  [@@deriving prms]
end

module P = RNN_P.Make (Prms.Single)

let n = 100
let alpha = 0.25

(* neural network *)
module RNN = struct
  module P = P

  let phi x = Maths.relu x

  (* input is the noisy observation and z is the internal state *)
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
      Sofo.gaussian_tensor_normed
        ~kind:base.kind
        ~device:base.device
        ~sigma:1.0
        [ s + 1; n ]
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    (* initialise to repeat observation *)
    let o =
      let b =
        Prms.value b
        |> Maths.to_tensor
        |> Tensor.slice ~dim:0 ~start:(Some 0) ~end_:(Some s) ~step:1
      in
      if s = 1
      then (
        let b2 = Tensor.(square_ (norm b)) in
        Tensor.(div_ (transpose ~dim0:1 ~dim1:0 b) b2)
        |> Maths.of_tensor
        |> Prms.Single.free)
      else Tensor.pinverse b ~rcond:0. |> Maths.of_tensor |> Prms.Single.free
    in
    { c; b; o }

  (* here data is a list of (x_t, optional labels). labels is x_t. *)
  let f ~data theta =
    let scaling = Float.(1. / of_int tmax) in
    let result, _ =
      List.foldi data ~init:(None, None) ~f:(fun t (accu, z) (input, labels) ->
        if t % 1 = 0 then Stdlib.Gc.major ();
        let z = forward ~theta ~input z in
        let pred = prediction ~theta z in
        let accu =
          match labels with
          | None -> accu
          | Some labels ->
            let delta_ell =
              Maths.(
                scaling $* Loss.mse ~output_dims:[ 1 ] Maths.(of_tensor labels - pred))
            in
            let delta_ggn =
              Maths.C.(
                scaling
                $* Loss.mse_ggn
                     ~output_dims:[ 1 ]
                     (Maths.const pred)
                     ~vtgt:(Maths.tangent_exn pred))
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

  (* TODO: make sure the batch size is 1 here, otherwise nonsensical results. Simulate one sample trajectory *)
  let simulate_1d (data : (float, Kalman_data.lds) Kalman_data.state array) theta =
    Array.foldi data ~init:([], None) ~f:(fun t (accu, z) datum ->
      if t % 1 = 0 then Stdlib.Gc.major ();
      let input =
        Tensor.of_bigarray
          ~device:base.device
          Mat.(of_array [| datum.Kalman_data.y |] 1 1)
      in
      let z = forward ~theta ~input z in
      let pred =
        prediction ~theta z |> Maths.to_tensor |> fun x -> Tensor.get_float2 x 0 0
      in
      let accu =
        Mat.of_array
          [| datum.Kalman_data.lds.tau
           ; datum.Kalman_data.lds.b
           ; datum.Kalman_data.lds.beta
           ; datum.Kalman_data.lds.sigma_eps
           ; datum.Kalman_data.y
           ; datum.Kalman_data.x
           ; pred
          |]
          1
          7
        :: accu
      in
      accu, Some z)
    |> fst
    |> List.rev
    |> Array.of_list
    |> Mat.concatenate ~axis:0
end

let sample_data bs =
  Data.minibatch ~tmax bs
  |> Data.minibatch_as_data
  |> List.map ~f:(fun datum ->
    let to_device = Tensor.of_bigarray ~device:base.device in
    to_device datum.Kalman_data.y, Some (to_device datum.Kalman_data.x))

let simulate_1d ~f_name n_trials =
  let module Data = Kalman_data.Make (Kalman_data.Default) in
  let n_list = List.range 0 n_trials in
  List.iter n_list ~f:(fun i ->
    let data = (Data.minibatch ~tmax 1).(0) in
    (* 1d *)
    let kf_prediction = Mat.of_array (Data.kalman_filter ~random:false data) (-1) 1 in
    let kf_random_prediction =
      Mat.of_array (Data.kalman_filter ~random:true data) (-1) 1
    in
    let model_params =
      let params_ba = RNN.P.C.load (in_dir f_name ^ "_params") in
      RNN.P.map params_ba ~f:(fun x -> x |> Maths.const)
    in
    let model_prediction = RNN.simulate_1d data model_params in
    (* the columns are tau, b, beta, sigma_eps, observed_y, x, model prediction, kalman prediction and kalman random prediction *)
    let to_save = Mat.((model_prediction @|| kf_prediction) @|| kf_random_prediction) in
    Mat.(
      save_txt to_save ~out:(in_dir f_name ^ "_" ^ Int.to_string i ^ "_model_prediction")))

(* -----------------------------------------
   -- Optimization with SOFO    ------
   ----------------------------------------- *)
let alpha_low = -2.
let alpha_high = -0.
let n_alpha = 5
let max_iter_alpha_opt = 5

module O = Optimizer.SOFO (RNN.P)

let init_config =
  Optimizer.Config.SOFO.
    { base; learning_rate = None; n_tangents = 256; damping = `relative_from_top 1e-3 }

(* TODO: fix tangents to be used when line searching for the optimal learning rate *)
let model ~data ~tangents ~state alpha =
  let config = { init_config with learning_rate = Some alpha } in
  let rec bayes_opt_loop t state =
    let theta, _ = O.prepare ~config state in
    let loss, ggn = RNN.f ~data theta in
    let state = O.step ~config ~info:{ loss; ggn; tangents } state in
    (* return neg loss for BayesOpt *)
    if t < max_iter_alpha_opt then bayes_opt_loop Int.(t + 1) state else Maths.neg loss
  in
  bayes_opt_loop 0 state

let rec loop ~t ~out ~state ~alpha_opt running_avg =
  Stdlib.Gc.major ();
  let data = sample_data batch_size in
  let config = { init_config with learning_rate = Some alpha_opt } in
  let theta, tangents = O.prepare ~config state in
  let alpha_opt =
    if bayes_opt && t % 100 = 0 && t > 0
    then (
      let model_fn = model ~data ~tangents ~state:(O.clone_state state) in
      alpha_search ~alpha_low ~alpha_high ~n_alpha model_fn)
    else alpha_opt
  in
  let loss, ggn = RNN.f ~data theta in
  let new_state =
    let config = { init_config with learning_rate = Some alpha_opt } in
    O.step ~config ~info:{ loss; ggn; tangents } state
  in
  let loss = Maths.to_float_exn (Maths.const loss) in
  let running_avg =
    let loss_avg =
      match running_avg with
      | [] -> loss
      | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
    in
    if t % 10 = 0
    then (
      (* save params *)
      O.P.C.save
        (RNN.P.value (O.params new_state))
        ~kind:base.ba_kind
        ~out:(in_dir "sofo_params");
      print [%message (t : int) (loss_avg : float) (alpha_opt : float)];
      Owl.Mat.(
        save_txt
          ~append:true
          ~out
          (of_array [| Float.of_int t; loss_avg; alpha_opt |] 1 3)));
    []
  in
  if t < max_iter
  then loop ~t:Int.(t + 1) ~out ~state:new_state ~alpha_opt (loss :: running_avg)

(* Start the sofo loop. *)
let _ =
  let out =
    let loss_name = if bayes_opt then "loss_bayes" else "loss" in
    in_dir loss_name
  in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  Bos.Cmd.(v "rm" % "-f" % in_dir "gp_info") |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init RNN.init) ~alpha_opt:0.05 []

(* let _ =
      let f_name = "sofo" in
      simulate_1d ~f_name n_trials_simulation *)
