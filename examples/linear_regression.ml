(* Example a simple linear regression task to learn W where y = W x. *)
open Base
open Torch
open Forward_torch
open Sofo

(* Command-line instruction on in which directory to save info. *)
let in_dir = Cmdargs.in_dir "-d"

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)
let base = Optimizer.Config.Base.default

(* Linear regression = simple one-layer neural network *)
module One_layer = struct
  module P = Prms.P

  type input = Tensor.t

  (* output = theta *@ input. *)
  let f ~(theta : P.M.t) ~(input : input) = Maths.(theta *@ const input)

  (* Initialise parameters. *)
  let init ~n ~d : P.tagged =
    let sigma = Float.(1. / sqrt (of_int n)) in
    Tensor.(f sigma * randn ~kind:base.kind ~device:base.device [ d; n ]) |> Prms.free
end

(* Feedforward model with MSE loss *)
module FF =
  Wrapper.Feedforward
    (One_layer)
    (Loss.MSE (struct
         let scaling_factor = 1.
       end))

(* Instatiate optimiser. *)
(* Sofo configuration with learning rate, number of tangents and damping specified. *)
module O = Optimizer.SOFO (FF)

let config =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some 100.
    ; n_tangents = 10
    ; rank_one = false
    ; damping = None
    ; momentum = None
    }

(* -----------------------------------------
   -- Generate linear regression data.
   ----------------------------------------- *)
(* Dimension of x. *)
let n = 100

(* Dimension of y. *)
let d = 1

(* Input covariance ( = Fisher information matrix in this case). *)
let input_cov12 =
  let lambda =
    Array.init n ~f:(fun i -> Float.(1. / (1. + square (of_int Int.(i + 1)))))
    |> Tensor.of_float1 ~device:base.device
  in
  let lambda = Tensor.(lambda / mean lambda) in
  let u, _ =
    Tensor.linalg_qr ~a:Tensor.(randn ~device:base.device [ n; n ]) ~mode:"complete"
  in
  Tensor.(u * sqrt lambda)

(* True weights. *)
let teacher : One_layer.P.M.t =
  One_layer.init ~n ~d |> One_layer.P.value |> One_layer.P.const

(* Generate data for mini batch. *)
let minibatch bs =
  let x = Tensor.(matmul input_cov12 (randn ~device:base.device [ n; bs ])) in
  let y = One_layer.f ~theta:teacher ~input:x |> Maths.primal in
  x, y

(* -----------------------------------------
   -- Optimization loop       ------
   ----------------------------------------- *)
let batch_size = 512
let max_iter = 10_000

let rec loop ~t ~out ~state =
  Stdlib.Gc.major ();
  (* Generate data. *)
  let data = minibatch batch_size in
  (* Take one optimization step to update state and returns loss. *)
  let loss, new_state = O.step ~config ~state ~data ~args:() in
  if t % 100 = 0
  then (
    Convenience.print [%message (t : int) (loss : float)];
    (* Save loss information in a text file. *)
    Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss |] 1 2)));
  if t < max_iter then loop ~t:(t + 1) ~out ~state:new_state

(* Start the loop. *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init ~config (One_layer.init ~n ~d))
