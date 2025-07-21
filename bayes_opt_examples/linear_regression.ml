(* example script testing automatic tuning of learning rate of SOFO. 
  following and simplifying https://arxiv.org/abs/1807.02811. *)
open Owl
open Base
open Forward_torch
open Maths
open Sofo
open Bayes_opt_common
module Mat = Dense.Matrix.S

let in_dir = Cmdargs.in_dir "-d"
let bayes_opt = Option.value (Cmdargs.get_bool "-bayes_opt") ~default:false
let base = Optimizer.Config.Base.default

let _ =
  Random.init 1985;
  (* Random.self_init (); *)
  Owl_stats_prng.init 1985;
  Torch_core.Wrapper.manual_seed 1985

(* -----------------------------------------
   -- Generate linear regression data.
   ----------------------------------------- *)

let d_in, d_out = 100, 3

let teacher =
  let sigma = Float.(1. / sqrt (of_int d_in)) in
  sigma $* randn ~kind:base.kind ~device:base.device [ d_in; d_out ]

let data_minibatch =
  let input_cov_sqrt =
    let u, _ = C.qr (randn ~device:base.device [ d_in; d_in ]) in
    let lambda =
      Array.init d_in ~f:(fun i -> Float.(1. / (1. + square (of_int Int.(i + 1)))))
      |> of_array ~device:base.device ~shape:[ d_in; 1 ]
      |> fun x -> C.(x / mean x)
    in
    C.(sqrt lambda * u)
  in
  fun bs ->
    let x = C.(randn ~device:base.device [ bs; d_in ] *@ input_cov_sqrt) in
    let y = Maths.(x *@ teacher) in
    x, y

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)

module Model = struct
  module P = Prms.Single

  let f ~(theta : _ some P.t) input = Maths.(input *@ theta)

  let init ~d_in ~d_out : P.param =
    let sigma = Float.(1. / sqrt (of_int d_in)) in
    let theta = sigma $* randn ~kind:base.kind ~device:base.device [ d_in; d_out ] in
    P.free theta
end

module O = Optimizer.SOFO (Model.P)

let init_config =
  Optimizer.Config.SOFO.
    { base; learning_rate = None; n_tangents = 10; damping = `relative_from_top 1e-5 }

(* -----------------------------------------
   -- Optimization loop       ------
   ----------------------------------------- *)
(* alpha_low and alpha_high in log(10) space *)
let alpha_low = 1.
let alpha_high = 2.3
let n_alpha = 5
let batch_size = 256
let max_iter = 10_000
let max_iter_alpha_opt = 10

(* TODO: fix tangents to be used when line searching for the optimal learning rate *)
let model ~x ~y ~tangents ~state alpha =
  let config = { init_config with learning_rate = Some alpha } in
  let rec bayes_opt_loop t state =
    let theta, _ = O.prepare ~config state in
    let y_pred = Model.f ~theta x in
    let loss = Loss.mse ~output_dims:[ 1 ] (y - y_pred) in
    let ggn = Loss.mse_ggn ~output_dims:[ 1 ] (const y) ~vtgt:(tangent_exn y_pred) in
    let state = O.step ~config ~info:{ loss; ggn; tangents } state in
    (* return neg loss for BayesOpt *)
    if t < max_iter_alpha_opt then bayes_opt_loop Int.(t + 1) state else Maths.neg loss
  in
  bayes_opt_loop 0 state

let rec loop ~t ~out ~state ~alpha_opt =
  Stdlib.Gc.major ();
  let x, y = data_minibatch batch_size in
  let config = { init_config with learning_rate = Some alpha_opt } in
  let theta, tangents = O.prepare ~config state in
  let alpha_opt =
    if bayes_opt && t % 100 = 0 && t > 0
    then (
      let model_fn = model ~x ~y ~tangents ~state:(O.clone_state state) in
      alpha_search ~alpha_low ~alpha_high ~n_alpha model_fn)
    else alpha_opt
  in
  let y_pred = Model.f ~theta x in
  let loss = Loss.mse ~output_dims:[ 1 ] (y - y_pred) in
  let ggn = Loss.mse_ggn ~output_dims:[ 1 ] (const y) ~vtgt:(tangent_exn y_pred) in
  let new_state =
    let config = { init_config with learning_rate = Some alpha_opt } in
    O.step ~config ~info:{ loss; ggn; tangents } state
  in
  if t % 100 = 0
  then (
    let loss_float = to_float_exn (const loss) in
    print [%message (t : int) (loss_float : float) (alpha_opt : float)];
    Owl.Mat.save_txt
      ~append:true
      ~out
      (Owl.Mat.of_array [| Float.of_int t; loss_float; alpha_opt |] 1 3));
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state ~alpha_opt

(* Start the loop. *)
let _ =
  let out =
    let loss_name = if bayes_opt then "loss_bayes" else "loss" in
    in_dir loss_name
  in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  Bos.Cmd.(v "rm" % "-f" % in_dir "gp_info") |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init (Model.init ~d_in ~d_out)) ~alpha_opt:100.
