open Forward_torch
open Torch

(** Module for loss function. *)
module type T = sig
  type 'a with_args
  type labels

  (** Computes the losses average over batch and [reduce_dim_list] given [labels] and  model outputs. *)
  val f : (labels:labels -> reduce_dim_list:int list -> Maths.t -> Maths.t) with_args

  (** Compute vtgtHgv, which is the GGN projected onto the subspace, given [labels], [vtgt] and model outputs 
  computed over [reduce_dim_list]. *)
  val vtgt_hessian_gv
    : (labels:labels -> vtgt:Tensor.t -> reduce_dim_list:int list -> Maths.t -> Tensor.t)
        with_args
end

(** Module for reward function, where loss is usually defined as negative reward. *)
module type Reward = sig
  type 'a with_args

  (** Compute vtgtgv, which is the GGN projected onto the subspace given [vtgt]. *)
  val vtgt_gv : (vtgt:Tensor.t -> Tensor.t) with_args
end
