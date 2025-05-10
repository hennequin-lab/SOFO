open Base
open Torch
open Forward_torch

(* print message s *)
let print s = Stdio.printf "%s\n%!" (Base.Sexp.to_string_hum s)

(* transpose a in dim 0 and 1. *)
let trans_2d a = Tensor.transpose ~dim0:1 ~dim1:0 a

(* get first dimension of a *)
let first_dim a = List.hd_exn (Tensor.shape a)

(* efficient implementation of a *@ (transpose b) *)
let a_b_trans a b = Tensor.einsum ~path:None [ a; b ] ~equation:"ik,jk->ij"

(* efficient implementation of (transpose a) *@ b *)
let a_trans_b a b = Tensor.einsum ~path:None [ a; b ] ~equation:"ki,kj->ij"

(* int list starting from 1 and ending at the last dim of a *)
let all_dims_but_first a = List.range 1 (List.length (Tensor.shape a))

(* sum all elements in tensor except for dimension 0. *)
let sum_except_dim0 x =
  let dim = List.init (List.length (Tensor.shape x) - 1) ~f:Int.succ in
  Tensor.sum_dim_intlist x ~dim:(Some dim) ~keepdim:false ~dtype:(Tensor.type_ x)

(* get top eigenvalue of x *)
let top_eigval x =
  let _, s, _ = Tensor.svd ~some:true ~compute_uv:false x in
  Tensor.get_float1 s 0

(* given [device] and [kind], draw elements of tensor from N(0, sigma^2) of shape 
  [a x b] and normalised on rows *)
let gaussian_tensor_2d_normed ~device ~kind ~a ~b ~sigma =
  let normaliser = Float.(sigma / sqrt (of_int a)) in
  Tensor.mul_scalar_ (Tensor.randn ~kind ~device [ a; b ]) (Scalar.f normaliser)

(* given the [full_batch_size], [batch_size] and num_epochs_to_run,
  calculate num of training loops. *)
let num_train_loops ~full_batch_size ~batch_size num_epochs_to_run =
  Int.(full_batch_size * num_epochs_to_run / batch_size)

(* given the [full_batch_size], [batch_size] and current iter t, calculate current epoch *)
let epoch_of ~full_batch_size ~batch_size t =
  Float.(of_int t * of_int batch_size / of_int full_batch_size)

(* calculate output heigh, width and channel *)
let calc_out_height_width_channel
      ~padding:(pad_x, pad_y)
      ~dilation:(dil_x, dil_y)
      ~in_out_channels_kernel
      ~stride:(stride_x, stride_y)
      ~in_height
      ~in_width
  =
  let _, out_channel, kerl_x, kerl_y = in_out_channels_kernel in
  let out_height =
    Int.(((in_height + (2 * pad_x) - (dil_x * (kerl_x - 1)) - 1) / stride_x) + 1)
  in
  let out_width =
    Int.(((in_width + (2 * pad_y) - (dil_y * (kerl_y - 1)) - 1) / stride_y) + 1)
  in
  out_height, out_width, out_channel

(* append 1s to the last dim of x *)
let expand_dim x =
  let x_shape = Maths.shape x in
  let x_shape_exp_last = List.drop_last_exn x_shape in
  let x_ = Maths.primal x in
  Maths.concat
    x
    (Maths.const
       (Tensor.ones
          ~device:(Tensor.device x_)
          ~kind:(Tensor.kind x_)
          (x_shape_exp_last @ [ 1 ])))
    ~dim:(List.length x_shape - 1)
