open Base
open Forward_torch
open Maths

let n_reduction dim y =
  let s = Array.of_list (shape y) in
  List.fold dim ~init:1 ~f:(fun accu i -> Int.(accu * s.(i)))

let mse ~average_over err = mean_dim ~keepdim:false ~dim:average_over (sqr err)

let mse_hv_prod ~average_over y ~v =
  let m = n_reduction average_over y in
  C.(Float.(2. / of_int m) $* v)

let cross_entropy ~average_over ~logit_dim ~(labels : [ `const ] t) y =
  let lse = logsumexp ~keepdim:true ~dim:[ logit_dim ] y in
  let diff = y - lse in
  let tmp = labels * diff in
  neg (sum_dim ~keepdim:false ~dim:average_over tmp)

let cross_entropy_hv_prod ~average_over:_ ~logit_dim:_ _y ~v:_ = assert false
