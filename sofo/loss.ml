open Base
open Forward_torch
open Maths

(* computes the total number of elements being reduced *)
(* let n_reduction dim y =
  let s = Array.of_list (shape y) in
  List.fold dim ~init:1 ~f:(fun accu i -> Int.(accu * s.(i))) *)

let mse ~output_dims err = mean ~keepdim:false ~dim:(0 :: output_dims) (sqr err)

(* let mse_vtgt_h_prod ~average_over y ~vtgt =
  let m = n_reduction average_over y in
  C.(Float.(2. / of_int m) $* vtgt) *)

let mse_ggn ~output_dims:_ ~vtgt =
  let vtgt_shape = Maths.shape vtgt in
  let vtgt_reshaped = reshape vtgt ~shape:[ List.hd_exn vtgt_shape; -1 ] in
  einsum [ vtgt_reshaped, "ka"; vtgt_reshaped, "la" ] "kl"
  *$ Float.(2. / of_int (List.hd_exn vtgt_shape))

let cross_entropy ~output_dims ~(labels : t) y =
  let lse = logsumexp ~keepdim:true ~dim:output_dims y in
  let diff = y - lse in
  let tmp = labels * diff in
  neg (sum ~keepdim:false ~dim:output_dims tmp) |> mean ~dim:[ 0 ] ~keepdim:false

let cross_entropy_ggn ~output_dims y ~vtgt =
  let vtgt_shape = shape vtgt in
  let n_samples = List.hd_exn vtgt_shape in
  let vtgt_mat = reshape vtgt ~shape:[ n_samples; -1 ] in
  let softmaxed_probs = exp (y - logsumexp ~keepdim:true ~dim:output_dims y) in
  let y_bar_row = reshape softmaxed_probs ~shape:[ 1; -1 ] in
  let diag_part = vtgt_mat * y_bar_row in
  let rank_1_part =
    let vtgt_ybar = vtgt * softmaxed_probs in
    let z = sum vtgt_ybar ~dim:[ 2 ] ~keepdim:false in
    let vtgt_h = einsum [ z, "ij"; softmaxed_probs, "jk" ] "ijk" in
    reshape vtgt_h ~shape:[ n_samples; -1 ]
  in
  let vtgt_h = diag_part - rank_1_part in
  einsum [ vtgt_h, "ka"; vtgt_mat, "la" ] "kl"
  *$ Float.(2. / of_int (List.nth_exn vtgt_shape 1))
