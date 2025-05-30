(** operations on the Maths.t object. Maths.t object is a dual number of (primal, tangent)
  where the primal is of type Tensor.t and the tangent is of type tangent. *)

open Base
open Torch

type tangent =
  | Explicit of const t
  | On_demand of (Device.t -> const t)

and const = [ `Const ]
and dual = [ `Dual ]
and 'a any = [< `Const | `Dual ] as 'a

and _ t =
  | Const of Tensor.t
  | Dual of Tensor.t * tangent

exception Not_dual
exception Wrong_shape of int list * int list
exception Wrong_device of Device.t * Device.t
exception Check_grad_failed

val as_const : _ any t -> const t
val as_dual_exn : _ any t -> dual t
val const : Tensor.t -> const t
val dual : dx:const t -> const t -> dual t
val dual_lazy : dx:(Device.t -> const t) -> const t -> dual t
val primal : _ any t -> Tensor.t
val tangent : dual t -> Tensor.t

(** Get shape of the primal tensor. *)
val shape : _ any t -> int list

(** Get the device of the primal tensor. *)
val device : _ any t -> Device.t

(** Get the kind of the primal tensor. *)
val kind : _ any t -> Torch_core.Kind.packed

(** Create a constant scalar tensor by pairing the primal with None as tangent. *)
val f : float -> const t

type 'a with_tensor_params = ?device:Device.t -> ?kind:Torch_core.Kind.packed -> 'a

val zeros : (int list -> const t) with_tensor_params
val ones : (?scale:float -> int list -> const t) with_tensor_params
val rand : (?scale:float -> int list -> const t) with_tensor_params
val randn : (?scale:float -> int list -> const t) with_tensor_params
val zeros_like : _ any t -> const t
val ones_like : _ any t -> const t
val rand_like : _ any t -> const t
val randn_like : _ any t -> const t

module Builder : sig
  type a = Tensor.t

  module Const : sig
    type unary_op = const t -> const t
    type binary_op = const t -> const t -> const t
    type unary_builder = a -> a
    type binary_builder = a -> a -> a

    val make_unary : unary_builder -> unary_op
    val make_binary : binary_builder -> binary_op
  end

  module Any : sig
    type 'a unary_op = 'a any t -> 'a any t
    type ('a, 'b, 'c) binary_op = 'a any t -> 'b any t -> 'c any t

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

    val make_unary : unary_builder -> _ unary_op
    val make_binary : binary_builder -> (_, _, _) binary_op
  end
end

module Primal : sig
  val view : size:int list -> const t -> const t
  val reshape : shape:int list -> const t -> const t
  val permute : dims:int list -> const t -> const t
  val squeeze : dim:int -> const t -> const t
  val unsqueeze : dim:int -> const t -> const t
  val neg : const t -> const t
  val trace : const t -> const t
  val sin : const t -> const t
  val cos : const t -> const t
  val sqr : const t -> const t
  val sqrt : const t -> const t
  val log : const t -> const t
  val exp : const t -> const t
  val tanh : const t -> const t
  val relu : const t -> const t
  val sigmoid : const t -> const t
  val softplus : const t -> const t
  val slice : ?start:int -> ?end_:int -> ?step:int -> dim:int -> const t -> const t
  val sum : const t -> const t
  val mean : const t -> const t
  val sum_dim : ?keepdim:bool -> dim:int list -> const t -> const t
  val mean_dim : ?keepdim:bool -> dim:int list -> const t -> const t
  val ( + ) : const t -> const t -> const t
  val ( - ) : const t -> const t -> const t
  val ( * ) : const t -> const t -> const t
  val ( / ) : const t -> const t -> const t
  val ( $+ ) : float -> const t -> const t
  val ( $* ) : float -> const t -> const t
  val ( $/ ) : float -> const t -> const t
  val ( *@ ) : const t -> const t -> const t
  val einsum : (const t * string) list -> string -> const t
end

val numel : 'a any t -> int
val view : size:int list -> 'a any t -> 'a any t
val reshape : shape:int list -> 'a any t -> 'a any t
val permute : dims:int list -> 'a any t -> 'a any t
val squeeze : dim:int -> 'a any t -> 'a any t
val unsqueeze : dim:int -> 'a any t -> 'a any t
val neg : 'a any t -> 'a any t
val trace : 'a any t -> 'a any t
val sin : 'a any t -> 'a any t
val cos : 'a any t -> 'a any t
val sqr : 'a any t -> 'a any t
val sqrt : 'a any t -> 'a any t
val log : 'a any t -> 'a any t
val exp : 'a any t -> 'a any t
val tanh : 'a any t -> 'a any t
val relu : 'a any t -> 'a any t
val sigmoid : 'a any t -> 'a any t
val softplus : 'a any t -> 'a any t
val slice : ?start:int -> ?end_:int -> ?step:int -> dim:int -> 'a any t -> 'a any t
val sum : 'a any t -> 'a any t
val mean : 'a any t -> 'a any t
val sum_dim : ?keepdim:bool -> dim:int list -> 'a any t -> 'a any t
val mean_dim : ?keepdim:bool -> dim:int list -> 'a any t -> 'a any t
val ( + ) : 'a any t -> 'b any t -> 'c any t
val ( - ) : 'a any t -> 'b any t -> 'c any t
val ( * ) : 'a any t -> 'b any t -> 'c any t
val ( / ) : 'a any t -> 'b any t -> 'c any t
val ( $+ ) : float -> 'a any t -> 'a any t
val ( $* ) : float -> 'a any t -> 'a any t
val ( $/ ) : float -> 'a any t -> 'a any t
val ( *@ ) : 'a any t -> 'b any t -> 'c any t
val einsum : ('a any t * string) list -> string -> 'a any t
