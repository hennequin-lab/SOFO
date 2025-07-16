open Base
open Torch

let print s = Stdio.print_endline (Sexp.to_string_hum s)

let gaussian_tensor_normed ~kind ~device ~sigma shape =
  let normaliser = Float.(sigma / sqrt (of_int (List.hd_exn shape))) in
  Tensor.mul_scalar_ (Tensor.randn ~kind ~device shape) (Scalar.f normaliser)
