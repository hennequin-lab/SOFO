open Base
open Forward_torch
open Torch
include Loss_typ

(* mean squared loss *)
module MSE (X : sig
    val scaling_factor : float
  end) =
struct
  type 'a with_args = 'a
  type labels = Tensor.t

  let f ~labels ~reduce_dim_list y =
    Maths.(
      X.scaling_factor
      $* mean_dim ~keepdim:false ~dim:(Some reduce_dim_list) (sqr (y - const labels)))

  let vtgt_hessian_gv ~labels:_ ~vtgt ~reduce_dim_list:_ _ =
    let n_samples = Convenience.first_dim vtgt in
    let vtgt_mat = Tensor.reshape vtgt ~shape:[ n_samples; -1 ] in
    Tensor.(f Float.(2. * X.scaling_factor) * Convenience.a_b_trans vtgt_mat vtgt_mat)
end

(* cross entropy loss *)
module CE (X : sig
    val scaling_factor : float
  end) =
struct
  type 'a with_args = 'a
  type labels = Tensor.t

  let f ~labels ~reduce_dim_list y =
    Maths.(
      neg
        (X.scaling_factor
         $* sum_dim
              ~keepdim:false
              ~dim:(Some reduce_dim_list)
              (const labels * (y - logsumexp ~keepdim:true ~dim:reduce_dim_list y))))

  let vtgt_hessian_gv ~labels:_ ~vtgt ~reduce_dim_list y =
    let n_samples = Convenience.first_dim vtgt in
    let vtgt_mat = Tensor.reshape vtgt ~shape:[ n_samples; -1 ] in
    let softmaxed_probs =
      let y = Maths.primal y in
      Tensor.(exp_ (y - logsumexp ~keepdim:true ~dim:reduce_dim_list y))
    in
    let y_bar_row = Tensor.reshape softmaxed_probs ~shape:[ 1; -1 ] in
    let diag_part = Tensor.(vtgt_mat * y_bar_row) in
    let rank_1_part =
      let vtgt_ybar = Tensor.(vtgt * softmaxed_probs) in
      let z =
        Tensor.sum_dim_intlist
          vtgt_ybar
          ~dim:(Some [ 2 ])
          ~keepdim:false
          ~dtype:(Tensor.type_ softmaxed_probs)
      in
      let vtgt_h =
        Tensor.einsum ~equation:"ij,jk->ijk" [ z; softmaxed_probs ] ~path:None
      in
      Tensor.reshape vtgt_h ~shape:[ n_samples; -1 ]
    in
    let vtgt_h = Tensor.(diag_part - rank_1_part) in
    Tensor.(f X.scaling_factor * Convenience.a_b_trans vtgt_h vtgt_mat)
end

(* mean squared loss with weights *)
module Weighted_MSE (X : sig
    val scaling_factor : float
  end) =
struct
  type 'a with_args = weights:Tensor.t -> 'a
  type labels = Tensor.t

  let f ~weights ~labels ~reduce_dim_list y =
    Maths.(
      X.scaling_factor
      $* mean_dim
           ~keepdim:false
           ~dim:(Some reduce_dim_list)
           (Maths.const weights * sqr (y - const labels)))

  let vtgt_hessian_gv ~weights ~labels:_ ~vtgt ~reduce_dim_list:_ _ =
    let weights = Tensor.squeeze weights in
    let n_samples = Convenience.first_dim vtgt in
    let vtgt_mat = Tensor.reshape vtgt ~shape:[ n_samples; -1 ] in
    Tensor.(
      f Float.(2. * X.scaling_factor)
      * einsum [ vtgt_mat; weights; vtgt_mat ] ~path:None ~equation:"ik,k,jk->ij")
end

(* negative utility, which has been defined upstream *)
module RL_loss (X : sig
    val scaling_factor : float
  end) =
struct
  type 'a with_args = 'a

  let vtgt_gv ~vtgt =
    let n_samples = Convenience.first_dim vtgt in
    let vtgt_mat = Tensor.reshape vtgt ~shape:[ n_samples; -1 ] in
    Tensor.(f Float.(2. * X.scaling_factor) * Convenience.a_b_trans vtgt_mat vtgt_mat)
end
