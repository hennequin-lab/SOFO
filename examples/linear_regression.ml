open Base
open Torch
open Forward_torch
open Sofo

let in_dir = Cmdargs.in_dir "-d"
let batch_size = 512
let max_iter = 10_000
let base = Optimizer.Config.Base.default

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)

(* Linear regression = simple one-layer neural network *)
module One_layer = struct
  module P = Prms.P

  type input = Tensor.t

  let f ~(theta : P.M.t) ~(input : input) = Maths.(theta *@ const input)

  let init ~n ~d : P.tagged =
    let sigma = Float.(1. / sqrt (of_int n)) in
    Tensor.(f sigma * randn ~kind:base.kind ~device:base.device [ d; n ]) |> Prms.free
end

(* Feedforward model wrapper with MSE loss *)
module FF =
  Wrapper.Feedforward
    (One_layer)
    (Loss.MSE (struct
         let scaling_factor = 1.
       end))

(* Optimiser; here you can switch to Adam to compare. *)
module O = Optimizer.SOFO (FF)

let config =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some 100.
    ; n_tangents = 10
    ; rank_one = false
    ; damping = None
    ; momentum = None
    ; lm = false
    ; perturb_thresh = None
    ; sqrt = false
    }

(* -----------------------------------------
   -- Generate linear regression data.
   ----------------------------------------- *)

let n = 100
let d = 1

(* input covariance ( = Fisher information matrix in this case) *)
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

(* true weights *)
let teacher : One_layer.P.M.t =
  One_layer.init ~n ~d |> One_layer.P.value |> One_layer.P.const

(* data for mini batch. *)
let minibatch bs =
  let x = Tensor.(matmul input_cov12 (randn ~device:base.device [ n; bs ])) in
  let y = One_layer.f ~theta:teacher ~input:x |> Maths.primal in
  x, y

(* optimization loop *)
let rec loop ~t ~out ~state =
  Stdlib.Gc.major ();
  let data = minibatch batch_size in
  let loss, new_state = O.step ~config ~state ~data ~args:() in
  if t % 100 = 0
  then (
    Convenience.print [%message (t : int) (loss : float)];
    Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss |] 1 2)));
  if t < max_iter then loop ~t:(t + 1) ~out ~state:new_state

(* start the loop *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init ~config (One_layer.init ~n ~d))
