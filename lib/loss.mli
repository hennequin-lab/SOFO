open Torch
include module type of Loss_typ

module MSE (_ : sig
    val scaling_factor : float
  end) : T with type labels = Tensor.t and type 'a with_args = 'a

module CE (_ : sig
    val scaling_factor : float
  end) : T with type labels = Tensor.t and type 'a with_args = 'a

module Weighted_MSE (_ : sig
    val scaling_factor : float
  end) : T with type labels = Tensor.t and type 'a with_args = weights:Tensor.t -> 'a
