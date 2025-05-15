(** Loss modules. *)
open Torch
include module type of Loss_typ

(** Mean squared loss. *)
module MSE (_ : sig
    val scaling_factor : float
  end) : T with type labels = Tensor.t and type 'a with_args = 'a

(** Cross-entropy loss. *)
module CE (_ : sig
    val scaling_factor : float
  end) : T with type labels = Tensor.t and type 'a with_args = 'a

(** Weighted mean squared loss. *)
module Weighted_MSE (_ : sig
    val scaling_factor : float
  end) : T with type labels = Tensor.t and type 'a with_args = weights:Tensor.t -> 'a

(** Negative of utility/reward. *)
module RL_loss : Reward with type 'a with_args = 'a
