(* Example a simple linear regression task to learn W where y = W x. *)
open Base
open Forward_torch
open Maths
open Sofo

let print s = Stdio.print_endline (Sexp.to_string_hum s)
let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default

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

let config =
  Optimizer.Config.SOFO.
    { base; learning_rate = Some 0.1; n_tangents = 10; damping = `none }

(* -----------------------------------------
   -- Generate linear regression data.
   ----------------------------------------- *)

let d_in, d_out = 100, 3

let data_minibatch =
  let teacher = Model.init ~d_in ~d_out |> Model.P.value in
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
    let y = Model.f ~theta:teacher x in
    x, y

(* -----------------------------------------
   -- Optimization loop       ------
   ----------------------------------------- *)

let batch_size = 512
let max_iter = 10_000

let rec loop ~t ~out ~state =
  Stdlib.Gc.major ();
  let x, y = data_minibatch batch_size in
  let theta, tangents = O.prepare ~config state in
  let y_pred = Model.f ~theta x in
  let loss = Loss.mse ~output_dims:[ 1 ] (y - y_pred) in
  let ggn = Loss.mse_ggn ~output_dims:[ 1 ] (const y) ~vtgt:(tangent_exn y_pred) in
  let new_state = O.step ~config ~info:{ loss; ggn; tangents } state in
  if t % 100 = 0
  then (
    let loss = to_float_exn (const loss) in
    print [%message (t : int) (loss : float)];
    Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss |] 1 2)));
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state

(* Start the loop. *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init (Model.init ~d_in ~d_out))
