(* Example a simple linear regression task to learn W where y = W x. *)
open Base
open Forward_torch
open Reverse
open Sofo

let print s = Stdio.print_endline (Sexp.to_string_hum s)
let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default

(* -----------------------------------------
   -- Model
   ----------------------------------------- *)

module Model = struct
  let f ~theta input = input *@ theta

  (* pure tensor version *)
  let f_tensor ~theta input = Maths.(input *@ theta)

  let init ~d_in ~d_out =
    let sigma = Float.(1. / sqrt (of_int d_in)) in
    Maths.randn ~scale:sigma ~device:base.device ~kind:base.kind [ d_in; d_out ]
    |> Maths.primal
    |> Maths.const
end

(* -----------------------------------------
   -- Data generation
   ----------------------------------------- *)

let d_in, d_out = 100, 3

let data_minibatch =
  let teacher = Model.init ~d_in ~d_out in
  let input_cov_sqrt =
    let u, _ =
      Maths.Const.qr (Maths.randn ~device:base.device ~kind:base.kind [ d_in; d_in ])
    in
    let lambda =
      Array.init d_in ~f:(fun i -> Float.(1. / (1. + square (of_int Int.(i + 1)))))
      |> Maths.of_array ~device:base.device ~shape:[ d_in; 1 ]
      |> fun x -> Maths.(x / mean x)
    in
    Maths.(sqrt lambda * u)
  in
  fun bs ->
    let x =
      Maths.(randn ~device:base.device ~kind:base.kind [ bs; d_in ] *@ input_cov_sqrt)
      |> Maths.primal
      |> Maths.const
    in
    let y = Model.f_tensor ~theta:teacher x in
    x, y

(* -----------------------------------------
   -- Loss
   ----------------------------------------- *)

let mse y y_pred =
  let diff = y - y_pred in
  mean (sqr diff)

(* -----------------------------------------
   -- Optimization loop (SGD)
   ----------------------------------------- *)

let learning_rate = 0.001
let batch_size = 512
let max_iter = 10_000

let rec loop ~t ~theta =
  Stdlib.Gc.major ();
  let x, y = data_minibatch batch_size in
  let x = const x
  and y = const y in
  (* compute gradient via reverse mode *)
  let loss_dual, () =
    grad
      (fun theta ->
         let y_pred = Model.f ~theta x in
         let loss = mse y y_pred in
         loss, ())
      theta
  in
  let loss = primal loss_dual in
  let theta_grad = adjoint theta |> Option.value_exn in
  (* SGD step *)
  let theta =
    zero_adj Maths.(Reverse.primal theta - Maths.(theta_grad *$ learning_rate))
  in
  if t % 100 = 0
  then (
    let loss = Maths.to_float_exn loss in
    print [%message (t : int) (loss : float)]);
  if t < max_iter then loop ~t:Int.(t + 1) ~theta

(* -----------------------------------------
   -- run
   ----------------------------------------- *)

let () =
  let theta0 = Model.init ~d_in ~d_out |> zero_adj in
  loop ~t:0 ~theta:theta0
