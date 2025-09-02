open Base
open Torch
open Forward_torch

module Spec_typ = Spec_typ
(** Functor to generate GGN utilities from a [Spec] module. *)
module GGN_Common (Spec : Spec_typ.SPEC) : sig
  val zero_params
    :  shape:int list
    -> device:Torch_core.Device.t
    -> kind:Torch_core.Kind.packed
    -> int
    -> Tensor.t

  val random_params
    :  shape:int list
    -> device:Torch_core.Device.t
    -> kind:Torch_core.Kind.packed
    -> int
    -> Tensor.t

  val get_total_n_params : Spec.param_name -> int
  val get_n_params_before_after : Spec.param_name -> int * int

  val eigenvectors_for_params
    :  lambda:'a
    -> get_sides:('a -> 'b -> [< `const | `dual ] Maths.t * [< `const | `dual ] Maths.t)
    -> param_name:'b
    -> (int * int * float) array * Tensor.t * Tensor.t

  val get_s_u
    :  lambda:'a
    -> get_sides:
         ('a
          -> Spec.param_name
          -> [< `const | `dual ] Maths.t * [< `const | `dual ] Maths.t)
    -> param_name:Spec.param_name
    -> (int * int * float) array * Tensor.t * Tensor.t

  val eigenvectors
    :  lambda:'a
    -> switch_to_learn:bool
    -> sampling_state:int
    -> get_sides:
         ('a
          -> Spec.param_name
          -> [< `const | `dual ] Maths.t * [< `const | `dual ] Maths.t)
    -> combine:('b -> 'b -> 'b)
    -> wrap:('b -> 'c)
    -> localise:(param_name:Spec.param_name -> n_per_param:int -> Tensor.t -> 'b)
    -> 'c * int
end
