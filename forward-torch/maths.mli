(** operations on the Maths.t object. Maths.t object is a dual number of (primal, tangent)
  where the primal is of type Tensor.t and the tangent is of type tangent. *)

open Base
open Torch

(* abstract types make it easier for the user to understand the structure *)

type _ tangent_kind =
  | Explicit : Tensor.t -> Tensor.t tangent_kind
  | On_demand : (Device.t -> Tensor.t) -> (Device.t -> Tensor.t) tangent_kind

type tangent = Tangent : 'a tangent_kind -> tangent
type _ t

exception Not_dual
exception Wrong_shape of int list * int list
exception Wrong_device of Device.t * Device.t
exception Check_grad_failed

val to_dual_exn : [ `Const | `Dual ] t -> [> `Dual ] t
val const : Tensor.t -> [> `Const ] t
val dual : dx:Tensor.t -> [ `Const ] t -> [> `Dual ] t
val dual_lazy : dx:(Device.t -> Tensor.t) -> [ `Const ] t -> [> `Dual ] t
val tangent_tensor_of : [ `Dual ] t -> Tensor.t
val primal : [ `Const | `Dual ] t -> Tensor.t
val tangent : [ `Dual ] t -> Tensor.t

(** Get shape of the primal tensor. *)
val shape : [ `Const | `Dual ] t -> int list

(** Get the device of the primal tensor. *)
val device : [ `Const | `Dual ] t -> Device.t

(** Get the kind of the primal tensor. *)
val kind : [ `Const | `Dual ] t -> Torch_core.Kind.packed

(** Create a constant scalar tensor by pairing the primal with None as tangent. *)
val f : float -> [ `Const ] t

module Builder : sig
  type a = Tensor.t

  module Const : sig
    type 'a unary_op = [ `Const ] t -> ([> `Const ] as 'a) t
    type 'a binary_op = [ `Const ] t -> [ `Const ] t -> ([> `Const ] as 'a) t
    type unary_builder = a -> a
    type binary_builder = a -> a -> a

    val make_unary : unary_builder -> 'a unary_op
    val make_binary : binary_builder -> 'a binary_op
  end

  module Any : sig
    type 'a unary_op = [ `Const | `Dual ] t -> ([> `Const | `Dual ] as 'a) t

    type 'a binary_op =
      [ `Const | `Dual ] t -> [ `Const | `Dual ] t -> ([> `Const | `Dual ] as 'a) t

    type unary_builder =
      { f : a -> a
      ; df : f:a -> x:a -> a -> a
      }

    type binary_builder =
      { f : a -> a -> a
      ; dfx : f:a -> x:a -> y:a -> a -> a
      ; dfy : f:a -> x:a -> y:a -> a -> a
      ; dfxy : f:a -> x:a -> y:a -> a -> a -> a
      }

    val make_unary : unary_builder -> 'a unary_op
    val make_binary : binary_builder -> 'a binary_op
  end
end

module Primal : sig
  val view : size:int list -> [ `Const ] t -> [> `Const ] t
  val reshape : shape:int list -> [ `Const ] t -> [> `Const ] t
  val permute : dims:int list -> [ `Const ] t -> [> `Const ] t
  val squeeze : dim:int -> [ `Const ] t -> [> `Const ] t
  val unsqueeze : dim:int -> [ `Const ] t -> [> `Const ] t
end

val view : size:int list -> [ `Const | `Dual ] t -> [> `Const | `Dual ] t
val reshape : shape:int list -> [ `Const | `Dual ] t -> [> `Const | `Dual ] t
val permute : dims:int list -> [ `Const | `Dual ] t -> [> `Const | `Dual ] t
val squeeze : dim:int -> [ `Const | `Dual ] t -> [> `Const | `Dual ] t
val unsqueeze : dim:int -> [ `Const | `Dual ] t -> [> `Const | `Dual ] t
