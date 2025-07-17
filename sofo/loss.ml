open Base
open Forward_torch
open Maths

(* computes the total number of elements being reduced *)
let n_reduction dim y =
  let s = Array.of_list (shape y) in
  List.fold dim ~init:1 ~f:(fun accu i -> Int.(accu * s.(i)))

let mse ~average_over err = mean ~keepdim:false ~dim:average_over (sqr err)

let mse_vtgt_h_prod ~average_over y ~vtgt =
  let m = n_reduction average_over y in
  C.(Float.(2. / of_int m) $* vtgt)

let mse_ggn ~average_over:_ y ~vtgt =
  C.(Float.(2.) $* einsum [ vtgt, "kab"; vtgt, "lab" ] "kl")

let cross_entropy ~average_over ~logit_dim ~(labels : [ `const ] t) y =
  let lse = logsumexp ~keepdim:true ~dim:[ logit_dim ] y in
  let diff = y - lse in
  let tmp = labels * diff in
  neg (sum ~keepdim:false ~dim:average_over tmp)

let cross_entropy_vtgt_h_prod ~average_over y ~vtgt =
  let n_samples = C.shape vtgt |> List.hd_exn in
  let vtgt_mat = C.reshape vtgt ~shape:[ n_samples; -1 ] in
  let softmaxed_probs = C.(exp (y - logsumexp ~keepdim:true ~dim:average_over y)) in
  let y_bar_row = C.reshape softmaxed_probs ~shape:[ 1; -1 ] in
  let diag_part = C.(vtgt_mat * y_bar_row) in
  let rank_1_part =
    let vtgt_ybar = C.(vtgt * softmaxed_probs) in
    let z = C.sum vtgt_ybar ~dim:[ 2 ] ~keepdim:false in
    let vtgt_h = C.einsum [ z, "ij"; softmaxed_probs, "jk" ] "ijk" in
    C.reshape vtgt_h ~shape:[ n_samples; -1 ]
  in
  let vtgt_h = C.(diag_part - rank_1_part) in
  C.(2. $* vtgt_h)

let cross_entropy_ggn ~average_over y ~vtgt =
  let n_samples = C.shape vtgt |> List.hd_exn in
  let vtgt_mat = C.reshape vtgt ~shape:[ n_samples; -1 ] in
  let softmaxed_probs = C.(exp (y - logsumexp ~keepdim:true ~dim:average_over y)) in
  let y_bar_row = C.reshape softmaxed_probs ~shape:[ 1; -1 ] in
  let diag_part = C.(vtgt_mat * y_bar_row) in
  let rank_1_part =
    let vtgt_ybar = C.(vtgt * softmaxed_probs) in
    let z = C.sum vtgt_ybar ~dim:[ 2 ] ~keepdim:false in
    let vtgt_h = C.einsum [ z, "ij"; softmaxed_probs, "jk" ] "ijk" in
    C.reshape vtgt_h ~shape:[ n_samples; -1 ]
  in
  let vtgt_h = C.(diag_part - rank_1_part) in
  C.(2. $* einsum [ vtgt_h, "ka"; vtgt_mat, "la" ] "kl")
