(* Linear Gaussian Dynamics, with same state/control/cost parameters constant across trials and across time; use a mini-gru (ilqr-vae appendix c) as generative model. *)
open Printf
open Base
open Forward_torch
open Torch
open Sofo
module Mat = Owl.Dense.Matrix.D
module Arr = Owl.Dense.Ndarray.D

let a_mat = Mat.of_array [| 1.; 2.; 3. |] 3 1
let a_ten = Tensor.of_bigarray a_mat
let w_mat = Mat.of_array (Array.init 12 ~f:(fun x -> Float.of_int x +. 4.)) 3 4
let w_ten = Tensor.of_bigarray w_mat
let a_w = Tensor.(a_ten * w_ten)
let w_diag = Tensor.diag_embed w_ten ~offset:0 ~dim1:(-2) ~dim2:(-1)

(* let a_w_ein = Tensor.einsum ~equation:"a,ab->ab" [(Tensor.squeeze a_ten); w_ten] ~path:None
let _ = Tensor.(print (a_w - a_w_ein)) *)
