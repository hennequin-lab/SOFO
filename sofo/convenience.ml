open Base
open Torch

let print s = Stdio.print_endline (Sexp.to_string_hum s)

let gaussian_tensor_normed ~kind ~device ~sigma shape =
  let normaliser = Float.(sigma / sqrt (of_int (List.hd_exn shape))) in
  Tensor.mul_scalar_ (Tensor.randn ~kind ~device shape) (Scalar.f normaliser)

(* for a list of integers, return a list of (prefix sum, suffix sum) *)
let prefix_suffix_sums (params : int list) : (int * int) list =
  let total = List.fold_left ~f:(fun a b -> a + b) ~init:0 params in
  let rec aux acc before = function
    | [] -> List.rev acc
    | x :: xs ->
      let after = total - before - x in
      aux ((before, after) :: acc) (before + x) xs
  in
  aux [] 0 params
