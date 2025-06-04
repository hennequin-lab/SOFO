(** operations on the Maths.t object. Maths.t object is a dual number of (primal, tangent)
  where the primal is of type Tensor.t and the tangent is of type tangent. *)

open Base
open Torch

type +'a t

exception Wrong_shape of int list * int list
exception Wrong_device of Device.t * Device.t
exception Check_grad_failed

val any : [< `const | `dual ] t -> [ `const | `dual ] t
val of_tensor : Tensor.t -> [ `const ] t
val to_tensor : [< `const | `dual ] t -> Tensor.t
val of_array : ?device:Device.t -> shape:int list -> float array -> [ `const ] t
val to_float_exn : [ `const ] t -> float
val const : [< `const | `dual ] t -> [ `const ] t
val shape : [< `const | `dual ] t -> int list
val device : [< `const | `dual ] t -> Device.t
val kind : [< `const | `dual ] t -> Torch_core.Kind.packed
val numel : [< `const | `dual ] t -> int
val tangent_exn : [< `const | `dual ] t -> [ `const ] t
val dual : tangent:[ `const ] t -> [ `const ] t -> [ `dual ] t
val dual_on_demand : tangent:(Device.t -> [ `const ] t) -> [ `const ] t -> [ `dual ] t
val first_dim : [< `const | `dual ] t -> int

(** Create a constant scalar tensor by pairing the primal with None as tangent. *)
val f : float -> [ `const ] t

type 'a with_tensor_params = ?device:Device.t -> ?kind:Torch_core.Kind.packed -> 'a

val zeros : (int list -> [ `const ] t) with_tensor_params
val ones : (?scale:float -> int list -> [ `const ] t) with_tensor_params
val rand : (?scale:float -> int list -> [ `const ] t) with_tensor_params
val randn : (?scale:float -> int list -> [ `const ] t) with_tensor_params
val zeros_like : [< `const | `dual ] t -> [ `const ] t
val zeros_like_k : k:int -> [< `const | `dual ] t -> [ `const ] t
val ones_like : [< `const | `dual ] t -> [ `const ] t
val rand_like : [< `const | `dual ] t -> [ `const ] t
val randn_like : [< `const | `dual ] t -> [ `const ] t
val randn_like_k : k:int -> [< `const | `dual ] t -> [ `const ] t

type unary_info =
  { f : Tensor.t -> Tensor.t
  ; df : f:Tensor.t -> x:Tensor.t -> dx:Tensor.t -> Tensor.t
  }

type binary_info =
  { f : Tensor.t -> Tensor.t -> Tensor.t
  ; dfx : f:Tensor.t -> x:Tensor.t -> y:Tensor.t -> dx:Tensor.t -> Tensor.t
  ; dfy : f:Tensor.t -> x:Tensor.t -> y:Tensor.t -> dy:Tensor.t -> Tensor.t
  ; dfxy :
      f:Tensor.t -> x:Tensor.t -> y:Tensor.t -> dx:Tensor.t -> dy:Tensor.t -> Tensor.t
  }

val make_unary : unary_info -> ([< `const | `dual ] as 'a) t -> 'a t

val make_binary
  :  binary_info
  -> [< `const | `dual ] t
  -> [< `const | `dual ] t
  -> [ `const | `dual ] t

val view : size:int list -> ([< `const | `dual ] as 'a) t -> 'a t
val reshape : shape:int list -> ([< `const | `dual ] as 'a) t -> 'a t
val permute : dims:int list -> ([< `const | `dual ] as 'a) t -> 'a t
val squeeze : dim:int -> ([< `const | `dual ] as 'a) t -> 'a t
val unsqueeze : dim:int -> ([< `const | `dual ] as 'a) t -> 'a t
val transpose : ?dims:int list -> ([< `const | `dual ] as 'a) t -> 'a t
val neg : ([< `const | `dual ] as 'a) t -> 'a t
val trace : ([< `const | `dual ] as 'a) t -> 'a t
val sin : ([< `const | `dual ] as 'a) t -> 'a t
val cos : ([< `const | `dual ] as 'a) t -> 'a t
val sqr : ([< `const | `dual ] as 'a) t -> 'a t
val sqrt : ([< `const | `dual ] as 'a) t -> 'a t
val log : ([< `const | `dual ] as 'a) t -> 'a t
val exp : ([< `const | `dual ] as 'a) t -> 'a t
val tanh : ([< `const | `dual ] as 'a) t -> 'a t
val relu : ([< `const | `dual ] as 'a) t -> 'a t
val sigmoid : ([< `const | `dual ] as 'a) t -> 'a t
val softplus : ([< `const | `dual ] as 'a) t -> 'a t

val slice
  :  ?start:int
  -> ?end_:int
  -> ?step:int
  -> dim:int
  -> ([< `const | `dual ] as 'a) t
  -> 'a t

val sum : ([< `const | `dual ] as 'a) t -> 'a t
val mean : ([< `const | `dual ] as 'a) t -> 'a t
val sum_dim : ?keepdim:bool -> dim:int list -> ([< `const | `dual ] as 'a) t -> 'a t
val mean_dim : ?keepdim:bool -> dim:int list -> ([< `const | `dual ] as 'a) t -> 'a t
val logsumexp : ?keepdim:bool -> dim:int list -> ([< `const | `dual ] as 'a) t -> 'a t
val ( + ) : [< `const | `dual ] t -> [< `const | `dual ] t -> [ `const | `dual ] t
val ( - ) : [< `const | `dual ] t -> [< `const | `dual ] t -> [ `const | `dual ] t
val ( * ) : [< `const | `dual ] t -> [< `const | `dual ] t -> [ `const | `dual ] t
val ( / ) : [< `const | `dual ] t -> [< `const | `dual ] t -> [ `const | `dual ] t
val ( $+ ) : float -> ([< `const | `dual ] as 'a) t -> 'a t
val ( $* ) : float -> ([< `const | `dual ] as 'a) t -> 'a t
val ( $/ ) : float -> ([< `const | `dual ] as 'a) t -> 'a t
val ( *@ ) : [< `const | `dual ] t -> [< `const | `dual ] t -> [ `const | `dual ] t
val einsum : ([< `const | `dual ] t * string) list -> string -> [ `const | `dual ] t

(* ---------------------------------------------------
   -- Type-preserving ops on constants
   --------------------------------------------------- *)
module C : sig
  val make_unary : unary_info -> [ `const ] t -> [ `const ] t
  val make_binary : binary_info -> [ `const ] t -> [ `const ] t -> [ `const ] t
  val view : size:int list -> [ `const ] t -> [ `const ] t
  val reshape : shape:int list -> [ `const ] t -> [ `const ] t
  val permute : dims:int list -> [ `const ] t -> [ `const ] t
  val squeeze : dim:int -> [ `const ] t -> [ `const ] t
  val unsqueeze : dim:int -> [ `const ] t -> [ `const ] t
  val transpose : ?dims:int list -> [ `const ] t -> [ `const ] t
  val neg : [ `const ] t -> [ `const ] t
  val trace : [ `const ] t -> [ `const ] t
  val sin : [ `const ] t -> [ `const ] t
  val cos : [ `const ] t -> [ `const ] t
  val sqr : [ `const ] t -> [ `const ] t
  val sqrt : [ `const ] t -> [ `const ] t
  val log : [ `const ] t -> [ `const ] t
  val exp : [ `const ] t -> [ `const ] t
  val tanh : [ `const ] t -> [ `const ] t
  val relu : [ `const ] t -> [ `const ] t
  val sigmoid : [ `const ] t -> [ `const ] t
  val softplus : [ `const ] t -> [ `const ] t

  val slice
    :  ?start:int
    -> ?end_:int
    -> ?step:int
    -> dim:int
    -> [ `const ] t
    -> [ `const ] t

  val sum : [ `const ] t -> [ `const ] t
  val mean : [ `const ] t -> [ `const ] t
  val sum_dim : ?keepdim:bool -> dim:int list -> [ `const ] t -> [ `const ] t
  val mean_dim : ?keepdim:bool -> dim:int list -> [ `const ] t -> [ `const ] t
  val logsumexp : ?keepdim:bool -> dim:int list -> [ `const ] t -> [ `const ] t
  val ( + ) : [ `const ] t -> [ `const ] t -> [ `const ] t
  val ( - ) : [ `const ] t -> [ `const ] t -> [ `const ] t
  val ( * ) : [ `const ] t -> [ `const ] t -> [ `const ] t
  val ( / ) : [ `const ] t -> [ `const ] t -> [ `const ] t
  val ( $+ ) : float -> [ `const ] t -> [ `const ] t
  val ( $* ) : float -> [ `const ] t -> [ `const ] t
  val ( $/ ) : float -> [ `const ] t -> [ `const ] t
  val ( *@ ) : [ `const ] t -> [ `const ] t -> [ `const ] t
  val einsum : ([ `const ] t * string) list -> string -> [ `const ] t
  val svd : [ `const ] t -> [ `const ] t * [ `const ] t * [ `const ] t
  val qr : [ `const ] t -> [ `const ] t * [ `const ] t
end
