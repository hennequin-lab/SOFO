(** operations on the Maths.t object. Maths.t object is a dual number of (primal, tangent)
  where the primal is of type Tensor.t and the tangent is of type tangent. *)

open Base
open Torch

exception Wrong_shape of int list * int list
exception Wrong_device of Device.t * Device.t
exception Check_grad_failed
exception No_tangent
exception Not_const

type tangent =
  | Explicit of Tensor.t
  | On_demand of (Device.t -> Tensor.t)

type t

val const : Tensor.t -> t
val primal : t -> Tensor.t
val of_array : ?device:Device.t -> shape:int list -> float array -> t
val of_bigarray : ?device:Device.t -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t -> t

val to_bigarray
  :  kind:('a, 'b) Bigarray.kind
  -> t
  -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t

val to_float_exn : t -> float
val shape : t -> int list
val device : t -> Device.t
val kind : t -> Torch_core.Kind.packed
val numel : t -> int
val tangent : t -> Tensor.t option
val tangent_exn : t -> Tensor.t
val dual : tangent:Tensor.t -> t -> t
val dual_on_demand : tangent:(Device.t -> Tensor.t) -> t -> t
val first_dim : t -> int

(** Create a constant scalar tensor by pairing the primal with None as tangent. *)
val f : float -> t

type 'a with_tensor_params = ?device:Device.t -> ?kind:Torch_core.Kind.packed -> 'a

val primal_tensor_detach : t -> t
val eye : (int -> t) with_tensor_params
val eye_like : t -> t
val zeros : (int list -> t) with_tensor_params
val ones : (?scale:float -> int list -> t) with_tensor_params
val rand : (?scale:float -> int list -> t) with_tensor_params
val randn : (?scale:float -> int list -> t) with_tensor_params
val zeros_like : t -> t
val zeros_like_k : k:int -> t -> t
val ones_like : t -> t
val rand_like : t -> t
val randn_like : t -> t
val randn_like_k : k:int -> t -> t

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

val make_unary : unary_info -> t -> t
val make_binary : binary_info -> t -> t -> t
val view : size:int list -> t -> t
val broadcast_to : size:int list -> t -> t
val contiguous : t -> t
val reshape : shape:int list -> t -> t
val permute : dims:int list -> t -> t
val squeeze : dim:int -> t -> t
val unsqueeze : dim:int -> t -> t
val transpose : ?dims:int list -> t -> t
val btr : t -> t
val diagonal : offset:int -> t -> t
val diag_embed : offset:int -> dim1:int -> dim2:int -> t -> t
val tril : _diagonal:int -> t -> t
val neg : t -> t
val abs : t -> t
val trace : t -> t
val sin : t -> t
val cos : t -> t
val sqr : t -> t
val sqrt : t -> t
val log : t -> t
val exp : t -> t
val tanh : t -> t
val pdf : t -> t
val erf : t -> t
val inv : t -> t
val pinv : rcond:float -> t -> t
val relu : t -> t
val soft_relu : t -> t
val sigmoid : t -> t
val softplus : t -> t
val lgamma : t -> t
val slice : ?start:int -> ?end_:int -> ?step:int -> dim:int -> t -> t
val sum : ?keepdim:bool -> ?dim:int list -> t -> t
val mean : ?keepdim:bool -> ?dim:int list -> t -> t
val max : ?keepdim:bool -> dim:int -> t -> t
val logsumexp : ?keepdim:bool -> dim:int list -> t -> t
val max_2d_dim1 : keepdim:bool -> t -> t

val maxpool2d
  :  ?padding:int * int
  -> ?dilation:int * int
  -> ?ceil_mode:bool
  -> ?stride:int * int
  -> int * int
  -> t
  -> t

val ( + ) : t -> t -> t
val ( - ) : t -> t -> t
val ( * ) : t -> t -> t
val ( / ) : t -> t -> t
val ( +$ ) : t -> float -> t
val ( -$ ) : t -> float -> t
val ( *$ ) : t -> float -> t
val ( /$ ) : t -> float -> t
val ( *@ ) : t -> t -> t
val einsum : (t * string) list -> string -> t
val concat : dim:int -> t list -> t
val gumbel_softmax : tau:float -> hard:bool -> t -> t
val cholesky : t -> t
val linsolve_triangular : ?left:bool -> ?upper:bool -> t -> t -> t
val linsolve : left:bool -> t -> t -> t
val kron : t -> t -> t

module Const : sig
  val svd : t -> t * t * t
  val eigh : ?uplo:string -> t -> t * t
  val qr : t -> t * t
end
