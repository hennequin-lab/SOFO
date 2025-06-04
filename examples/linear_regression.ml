(* Example a simple linear regression task to learn W where y = W x. *)
open Base
open Torch
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
  module P = Prms.Singleton

  let f ~theta input = theta *@ input

  let init ~n ~d : P.param =
    let sigma = Float.(1. / sqrt (of_int n)) in
    let theta = sigma $* randn ~kind:base.kind ~device:base.device [ d; n ] in
    P.free theta
end

module O = Optimizer.SOFO (Model.P)

let config =
  Optimizer.Config.SOFO.
    { base; learning_rate = Some 0.1; n_tangents = 10; damping = None }

(* -----------------------------------------
   -- Generate linear regression data.
   ----------------------------------------- *)

(* Dimension of x. *)
let n = 100

(* Dimension of y. *)
let d = 3

(* Input covariance ( = Fisher information matrix in this case). *)
let input_cov12 =
  let lambda =
    Array.init n ~f:(fun i -> Float.(1. / (1. + square (of_int Int.(i + 1)))))
    |> Tensor.of_float1 ~device:base.device
    |> of_tensor
  in
  let lambda = C.(lambda / mean lambda) in
  let u, _ = C.qr (randn ~device:base.device [ n; n ]) in
  C.(u * sqrt lambda)

(* True weights. *)
let teacher = Model.init ~n ~d |> Model.P.value

(* Generate data for mini batch. *)
let minibatch bs =
  let x = C.(input_cov12 *@ randn ~device:base.device [ n; bs ]) in
  let y = Model.f ~theta:teacher x in
  x, y

(* -----------------------------------------
   -- Optimization loop       ------
   ----------------------------------------- *)

let batch_size = 512
let max_iter = 10_000

let rec loop ~t ~out ~state =
  Stdlib.Gc.major ();
  (* Generate data. *)
  let x, y = minibatch batch_size in
  let theta, vs = O.prepare ~config state in
  (* we need to compute the loss tangents, and the ggn *)
  let y_pred = Model.f ~theta x in
  let loss = Loss.mse ~average_over:[ 0; 1 ] (sqr (y - y_pred)) in
  let ggn =
    let yt = tangent_exn y_pred in
    let hyt = Loss.mse_hv_prod ~average_over:[ 0; 1 ] (const y) ~v:yt in
    C.einsum [ yt, "kab"; hyt, "lab" ] "kl"
  in
  let new_state = O.step ~config ~info:{ loss; ggn; tangents = vs } state in
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
  loop ~t:0 ~out ~state:(O.init (Model.init ~n ~d))
