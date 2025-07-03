(** operations on the Maths.t object. Maths.t object is a dual number of (primal, tangent)
  where the primal is of type Tensor.t and the tangent is of type tangent. *)

open Base
open Torch

type +'a t
type const = [ `const ]
type dual = [ `dual ]
type 'a some = [< const | dual ] as 'a

type any =
  [ const
  | dual
  ]

exception Wrong_shape of int list * int list
exception Wrong_device of Device.t * Device.t
exception Check_grad_failed

val any : _ some t -> any t
val of_tensor : Tensor.t -> const t
val to_tensor : _ some t -> Tensor.t
val of_array : ?device:Device.t -> shape:int list -> float array -> const t

val of_bigarray
  :  ?device:Device.t
  -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
  -> const t

val to_bigarray
  :  kind:('a, 'b) Bigarray.kind
  -> const t
  -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t

val to_float_exn : const t -> float
val const : _ some t -> const t
val shape : _ some t -> int list
val device : _ some t -> Device.t
val kind : _ some t -> Torch_core.Kind.packed
val numel : _ some t -> int
val tangent_exn : _ some t -> const t
val dual : tangent:const t -> const t -> dual t
val dual_on_demand : tangent:(Device.t -> const t) -> const t -> dual t
val first_dim : _ some t -> int

(** Create a constant scalar tensor by pairing the primal with None as tangent. *)
val f : float -> const t

type 'a with_tensor_params = ?device:Device.t -> ?kind:Torch_core.Kind.packed -> 'a

val eye : (int -> const t) with_tensor_params
val zeros : (int list -> const t) with_tensor_params
val ones : (?scale:float -> int list -> const t) with_tensor_params
val rand : (?scale:float -> int list -> const t) with_tensor_params
val randn : (?scale:float -> int list -> const t) with_tensor_params
val zeros_like : _ some t -> const t
val zeros_like_k : k:int -> _ some t -> const t
val ones_like : _ some t -> const t
val rand_like : _ some t -> const t
val randn_like : _ some t -> const t
val randn_like_k : k:int -> _ some t -> const t

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

val make_unary : unary_info -> 'a some t -> 'a t
val make_binary : binary_info -> _ some t -> _ some t -> any t
val view : size:int list -> 'a some t -> 'a t
val broadcast_to : size:int list -> 'a some t -> 'a t
val reshape : shape:int list -> 'a some t -> 'a t
val permute : dims:int list -> 'a some t -> 'a t
val squeeze : dim:int -> 'a some t -> 'a t
val unsqueeze : dim:int -> 'a some t -> 'a t
val transpose : ?dims:int list -> 'a some t -> 'a t
val neg : 'a some t -> 'a t
val abs : 'a some t -> 'a t
val trace : 'a some t -> 'a t
val sin : 'a some t -> 'a t
val cos : 'a some t -> 'a t
val sqr : 'a some t -> 'a t
val sqrt : 'a some t -> 'a t
val log : 'a some t -> 'a t
val exp : 'a some t -> 'a t
val tanh : 'a some t -> 'a t
val inv_sqr : 'a some t -> 'a t
val inv_rectangle : rcond:float -> 'a some t -> 'a t
val relu : 'a some t -> 'a t
val sigmoid : 'a some t -> 'a t
val softplus : 'a some t -> 'a t
val slice : ?start:int -> ?end_:int -> ?step:int -> dim:int -> 'a some t -> 'a t
val sum : ?keepdim:bool -> ?dim:int list -> 'a some t -> 'a t
val mean : ?keepdim:bool -> ?dim:int list -> 'a some t -> 'a t
val logsumexp : ?keepdim:bool -> dim:int list -> 'a some t -> 'a t
val ( + ) : _ some t -> _ some t -> any t
val ( - ) : _ some t -> _ some t -> any t
val ( * ) : _ some t -> _ some t -> any t
val ( / ) : _ some t -> _ some t -> any t
val ( $+ ) : float -> 'a some t -> 'a t
val ( $* ) : float -> 'a some t -> 'a t
val ( $/ ) : float -> 'a some t -> 'a t
val ( *@ ) : _ some t -> _ some t -> any t
val einsum : (_ some t * string) list -> string -> any t
val concat : dim:int -> _ some t list -> any t
val cholesky : 'a some t -> 'a some t
val linsolve_triangular : ?left:bool -> ?upper:bool -> _ some t -> _ some t -> any t

(* ---------------------------------------------------
   -- Type-preserving ops on constants
   --------------------------------------------------- *)
module C : sig
  val make_unary : unary_info -> const t -> const t
  val make_binary : binary_info -> const t -> const t -> const t
  val view : size:int list -> const t -> const t
  val broadcast_to : size:int list -> const t -> const t
  val reshape : shape:int list -> const t -> const t
  val permute : dims:int list -> const t -> const t
  val squeeze : dim:int -> const t -> const t
  val unsqueeze : dim:int -> const t -> const t
  val transpose : ?dims:int list -> const t -> const t
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
  val sign : const t -> const t
  val slice : ?start:int -> ?end_:int -> ?step:int -> dim:int -> const t -> const t
  val sum : ?keepdim:bool -> ?dim:int list -> const t -> const t
  val mean : ?keepdim:bool -> ?dim:int list -> const t -> const t
  val logsumexp : ?keepdim:bool -> dim:int list -> const t -> const t
  val ( + ) : const t -> const t -> const t
  val ( - ) : const t -> const t -> const t
  val ( * ) : const t -> const t -> const t
  val ( / ) : const t -> const t -> const t
  val ( $+ ) : float -> const t -> const t
  val ( $* ) : float -> const t -> const t
  val ( $/ ) : float -> const t -> const t
  val ( *@ ) : const t -> const t -> const t
  val einsum : (const t * string) list -> string -> const t
  val concat : dim:int -> const t list -> const t
  val svd : const t -> const t * const t * const t
  val qr : const t -> const t * const t
  val cholesky : const t -> const t
  val linsolve_triangular : ?left:bool -> ?upper:bool -> const t -> const t -> const t
end
