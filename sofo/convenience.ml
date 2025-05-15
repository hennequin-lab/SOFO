open Base
open Torch
open Forward_torch

let print s = Stdio.printf "%s\n%!" (Base.Sexp.to_string_hum s)

(** Transpose a along the first two dimensions. *)
let trans_2d a = Tensor.transpose ~dim0:1 ~dim1:0 a

(** Extend the shape of a from [shape] to [shape; 1]. *)
let extend_one a = Maths.view a ~size:(Tensor.shape (Maths.primal a) @ [ 1 ])

(** Get first dimension of a. *)
let first_dim a = List.hd_exn (Tensor.shape a)

(** a *@ (transpose b) *)
let a_b_trans a b = Tensor.einsum ~path:None [ a; b ] ~equation:"ik,jk->ij"

(** (transpose a) *@ b *)
let a_trans_b a b = Tensor.einsum ~path:None [ a; b ] ~equation:"ki,kj->ij"

(** Returns an int list starting from 1 and ending at the last dim of a. *)
let all_dims_but_first a = List.range 1 (List.length (Tensor.shape a))

(** Sum all elements in x except for dimension 0. *)
let sum_except_dim0 x =
  let dim = List.init (List.length (Tensor.shape x) - 1) ~f:Int.succ in
  Tensor.sum_dim_intlist x ~dim:(Some dim) ~keepdim:false ~dtype:(Tensor.type_ x)

(** Get largest eigenvalue of x *)
let top_eigval x =
  let _, s, _ = Tensor.svd ~some:true ~compute_uv:false x in
  Tensor.get_float1 s 0

(** Return a Random gaussian tensor of standard deviation [sigma] of [kind]
 normalised on rows on [device] with shape [a;b]. *)
let gaussian_tensor_2d_normed ~device ~kind ~a ~b ~sigma =
  let normaliser = Float.(sigma / sqrt (of_int a)) in
  Tensor.mul_scalar_ (Tensor.randn ~kind ~device [ a; b ]) (Scalar.f normaliser)

(** Calculate num of training loops given [full_batch_size], [batch_size] and num_epochs_to_run. *)
let num_train_loops ~full_batch_size ~batch_size num_epochs_to_run =
  Int.(full_batch_size * num_epochs_to_run / batch_size)

(** Calculate current epoch number given [full_batch_size], [batch_size] and iteration number. *)
let epoch_of ~full_batch_size ~batch_size t =
  Float.(of_int t * of_int batch_size / of_int full_batch_size)

(** Calculate output height, width and channel after a conv2d operation. *)
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
