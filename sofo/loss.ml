open Base
open Forward_torch
open Maths
include Loss_typ

let mse ~dim err = mean_dim ~keepdim:false ~dim (sqr err)
let mse_hv_prod ~dim:_ v = C.(2. $* v)

let cross_entropy ~dim ~(labels : [ `const ] t) y =
  let lse = logsumexp ~keepdim:true ~dim y in
  let diff = y - lse in
  let tmp = labels * diff in
  neg (sum_dim ~keepdim:false ~dim tmp)

let cross_entropy_hv_prod ~dim y ~v =
  let open C in
  let s = shape v in
  let n_samples = first_dim v in
  let v_mat = reshape v ~shape:[ n_samples; -1 ] in
  let softmaxed_probs =
    let y' = to_tensor y in
    Torch.Tensor.(exp_ (y' - logsumexp ~keepdim:true ~dim y')) |> of_tensor
  in
  let y_bar_row = reshape softmaxed_probs ~shape:[ 1; -1 ] in
  let diag_part = v_mat * y_bar_row in
  let rank_1_part =
    let vtgt_ybar = v * softmaxed_probs in
    let z = sum_dim vtgt_ybar ~dim:[ 2 ] ~keepdim:false in
    let vtgt_h = einsum [ z, "ij"; softmaxed_probs, "jk" ] "ijk" in
    reshape vtgt_h ~shape:[ n_samples; -1 ]
  in
  let vtgt_h = diag_part - rank_1_part in
  einsum [ vtgt_h, "ik"; v_mat, "jk" ] "ij" |> reshape ~shape:s
