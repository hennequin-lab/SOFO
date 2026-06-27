(** Dual number implementation for automatic differentiation on Libtorch tensors.
    
    This module implements a dual number system where each value consists of a primal
    tensor and an optional tangent component. The tangent represents the derivative
    information and enables automatic differentiation through complex tensor operations.
    
    The tangent dimension is handled as a batch dimension, allowing multiple tangent
    vectors to be tracked simultaneously (“batched forward-mode AD”). Operations on
    dual numbers maintain this structure by propagating directional derivatives in batches.
*)

open Base
open Torch

(** Exception raised when tensor shapes are incompatible. *)
exception Wrong_shape of int list * int list

(** Exception raised when tensor devices are incompatible. *)
exception Wrong_device of Device.t * Device.t

(** Exception raised during gradient checking. *)
exception Check_grad_failed

(** Exception raised when a tangent is expected but not present. *)
exception No_tangent

(** Exception raised when an operation is not supported on constant tensors. *)
exception Not_const

(** Tangent representation for dual numbers.
    
    A tangent can either be explicitly stored as a tensor or computed on-demand
    through a function that generates the tangent for a specific device. *)
type tangent =
  | Explicit of Tensor.t
  | On_demand of (Device.t -> Tensor.t)

type t

(** [const p] is a dual number with primal tensor p and no tangent component. *)
val const : Tensor.t -> t

(** [primal x] is the primal tensor value of x. *)
val primal : t -> Tensor.t

(** [of_array device shape x] creates a dual number from a float array. *)
val of_array : ?device:Device.t -> shape:int list -> float array -> t

(** [of_bigarray device x] creates a dual number from a bigarray. *)
val of_bigarray : ?device:Device.t -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t -> t

(** [to_bigarray kind x] converts a dual number to a bigarray. *)
val to_bigarray
  :  kind:('a, 'b) Bigarray.kind
  -> t
  -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t

(** [to_float_exn x] converts a scalar dual number to a float. *)
val to_float_exn : t -> float

(** [shape x] is the shape of the primal tensor of x. *)
val shape : t -> int list

(** [shape' x] is the shape of the primal tensor of x as an array. *)
val shape' : t -> int array

(** [device x] is the device of the primal tensor of x. *)
val device : t -> Device.t

(** [kind x] is the data type of the primal tensor of x. *)
val kind : t -> Torch_core.Kind.packed

(** [numel x] is the number of elements in the primal tensor of x. *)
val numel : t -> int

(** [tangent x] is the tangent tensor of x, or None if no tangent exists. *)
val tangent : t -> Tensor.t option

(** [tangent_exn x] is the tangent tensor of x (raises No_tangent if none exists). *)
val tangent_exn : t -> Tensor.t

(** [dual ~tangent x] creates a dual number with explicit tangent tensor. *)
val dual : tangent:Tensor.t -> t -> t

(** [dual_on_demand ~tangent x] creates a dual number with on-demand tangent. *)
val dual_on_demand : tangent:(Device.t -> Tensor.t) -> t -> t

(** [first_dim x] is the first dimension of the primal tensor of x. *)
val first_dim : t -> int

(** [f x] is a constant scalar dual number with value x. *)
val f : float -> t

type 'a with_tensor_params = ?device:Device.t -> ?kind:Torch_core.Kind.packed -> 'a

(** [primal_tensor_detach x] is a dual number with detached primal tensor. *)
val primal_tensor_detach : t -> t

(** [eye device kind n] creates an identity matrix. *)
val eye : (int -> t) with_tensor_params

(** [eye_like x] creates an identity matrix with same shape as x. *)
val eye_like : t -> t

(** [zeros device kind shape] creates a tensor of zeros. *)
val zeros : (int list -> t) with_tensor_params

(** [ones device kind scale shape] creates a tensor of ones. *)
val ones : (?scale:float -> int list -> t) with_tensor_params

(** [rand device kind scale shape] creates a tensor of random values. *)
val rand : (?scale:float -> int list -> t) with_tensor_params

(** [randn device kind scale shape] creates a tensor of random normal values. *)
val randn : (?scale:float -> int list -> t) with_tensor_params

(** [zeros_like x] creates a tensor of zeros with same shape as x. *)
val zeros_like : t -> t

(** [zeros_like_k ~k x] creates a tensor of zeros with k as first dimension. *)
val zeros_like_k : k:int -> t -> t

(** [ones_like x] creates a tensor of ones with same shape as x. *)
val ones_like : t -> t

(** [rand_like x] creates a tensor of random values with same shape as x. *)
val rand_like : t -> t

(** [randn_like x] creates a tensor of random normal values with same shape as x. *)
val randn_like : t -> t

(** [randn_like_k ~k x] creates a tensor of random normal values with k as first dimension. *)
val randn_like_k : k:int -> t -> t

(** [unary_info] describes a unary operation with forward and derivative functions. *)
type unary_info =
  { f : Tensor.t -> Tensor.t
  ; df : f:Tensor.t -> x:Tensor.t -> dx:Tensor.t -> Tensor.t
  }

(** [binary_info] describes a binary operation with forward and partial derivative functions. *)
type binary_info =
  { f : Tensor.t -> Tensor.t -> Tensor.t
  ; dfx : f:Tensor.t -> x:Tensor.t -> y:Tensor.t -> dx:Tensor.t -> Tensor.t
  ; dfy : f:Tensor.t -> x:Tensor.t -> y:Tensor.t -> dy:Tensor.t -> Tensor.t
  ; dfxy :
      f:Tensor.t -> x:Tensor.t -> y:Tensor.t -> dx:Tensor.t -> dy:Tensor.t -> Tensor.t
  }

(** [make_unary info x] creates a unary operation from function information. *)
val make_unary : unary_info -> t -> t

(** [make_binary info x y] creates a binary operation from function information. *)
val make_binary : binary_info -> t -> t -> t

(** [view ~size x] reshapes the primal tensor of x to size. *)
val view : size:int list -> t -> t

(** [broadcast_to ~size x] broadcasts the primal tensor of x to size. *)
val broadcast_to : size:int list -> t -> t

(** [contiguous x] makes the primal tensor of x contiguous. *)
val contiguous : t -> t

(** [reshape ~shape x] reshapes the primal tensor of x to shape. *)
val reshape : shape:int list -> t -> t

(** [permute ~dims x] permutes the dimensions of the primal tensor of x. *)
val permute : dims:int list -> t -> t

(** [squeeze ~dim x] squeezes dimension dim of the primal tensor of x. *)
val squeeze : dim:int -> t -> t

(** [unsqueeze ~dim x] unsqueezes dimension dim of the primal tensor of x. *)
val unsqueeze : dim:int -> t -> t

(** [transpose ?dims x] transposes dimensions of the primal tensor of x. *)
val transpose : ?dims:int list -> t -> t

(** [btr x] transposes last two dimensions of the primal tensor of x. *)
val btr : t -> t

(** [diagonal ~offset x] extracts diagonal elements from the primal tensor of x. *)
val diagonal : offset:int -> t -> t

(** [diag_embed ~offset ~dim1 ~dim2 x] creates diagonal embedding from the primal tensor of x. *)
val diag_embed : offset:int -> dim1:int -> dim2:int -> t -> t

(** [tril ~_diagonal x] extracts lower triangular part of the primal tensor of x. *)
val tril : _diagonal:int -> t -> t

(** [neg x] negates the primal tensor of x. *)
val neg : t -> t

(** [abs x] computes absolute value of the primal tensor of x. *)
val abs : t -> t

(** [trace x] computes trace of the primal tensor of x. *)
val trace : t -> t

(** [sin x] computes sine of the primal tensor of x. *)
val sin : t -> t

(** [cos x] computes cosine of the primal tensor of x. *)
val cos : t -> t

(** [sqr x] computes square of the primal tensor of x. *)
val sqr : t -> t

(** [sqrt x] computes square root of the primal tensor of x. *)
val sqrt : t -> t

(** [log x] computes natural logarithm of the primal tensor of x. *)
val log : t -> t

(** [exp x] computes exponential of the primal tensor of x. *)
val exp : t -> t

(** [tanh x] computes hyperbolic tangent of the primal tensor of x. *)
val tanh : t -> t

(** [pdf x] computes PDF for standard normal of the primal tensor of x. *)
val pdf : t -> t

(** [erf x] computes error function of the primal tensor of x. *)
val erf : t -> t

(** [inv x] computes matrix inverse of the primal tensor of x. *)
val inv : t -> t

(** [pinv ~rcond x] computes pseudo-inverse of the primal tensor of x. *)
val pinv : rcond:float -> t -> t

(** [relu x] applies ReLU activation to the primal tensor of x. *)
val relu : t -> t

(** [soft_relu x] applies soft ReLU activation to the primal tensor of x. *)
val soft_relu : t -> t

(** [sigmoid x] applies sigmoid activation to the primal tensor of x. *)
val sigmoid : t -> t

(** [softplus x] applies softplus activation to the primal tensor of x. *)
val softplus : t -> t

(** [lgamma x] computes log gamma of the primal tensor of x. *)
val lgamma : t -> t

(** [get_slice slice x] slices the primal tensor of x. *)
val get_slice : int list list -> t -> t

(** [get_slice_inplace slice x] performs in-place slicing of the primal tensor of x. *)
val get_slice_inplace : int list list -> t -> t

(** [slice ?start ?end_ ?step ~dim x] slices the primal tensor of x. *)
val slice : ?start:int -> ?end_:int -> ?step:int -> dim:int -> t -> t

(** [sum ?keepdim ?dim x] sums elements of the primal tensor of x. *)
val sum : ?keepdim:bool -> ?dim:int list -> t -> t

(** [mean ?keepdim ?dim x] computes mean of the primal tensor of x. *)
val mean : ?keepdim:bool -> ?dim:int list -> t -> t

(** [max ?keepdim ~dim x] computes maximum elements of the primal tensor of x. *)
val max : ?keepdim:bool -> dim:int -> t -> t

(** [logsumexp ?keepdim ~dim x] computes log sum exp of the primal tensor of x. *)
val logsumexp : ?keepdim:bool -> dim:int list -> t -> t

(** [max_2d_dim1 ~keepdim x] computes max along dimension 1 of the primal tensor of x. *)
val max_2d_dim1 : keepdim:bool -> t -> t

(** [maxpool2d ?padding ?dilation ?ceil_mode ?stride ksize x] performs max pooling. *)
val maxpool2d
  :  ?padding:int * int
  -> ?dilation:int * int
  -> ?ceil_mode:bool
  -> ?stride:int * int
  -> int * int
  -> t
  -> t

(** [x + y] adds two dual numbers. *)
val ( + ) : t -> t -> t

(** [x - y] subtracts two dual numbers. *)
val ( - ) : t -> t -> t

(** [x * y] multiplies two dual numbers. *)
val ( * ) : t -> t -> t

(** [x / y] divides two dual numbers. *)
val ( / ) : t -> t -> t

(** [x +$ z] adds scalar z to dual number x. *)
val ( +$ ) : t -> float -> t

(** [x -$ z] subtracts scalar z from dual number x. *)
val ( -$ ) : t -> float -> t

(** [x *$ z] multiplies dual number x by scalar z. *)
val ( *$ ) : t -> float -> t

(** [x /$ z] divides dual number x by scalar z. *)
val ( /$ ) : t -> float -> t

(** [x *@ y] performs matrix multiplication of dual numbers x and y. *)
val ( *@ ) : t -> t -> t

(** [einsum operands return] performs Einstein summation. *)
val einsum : (t * string) list -> string -> t

(** [concat ~dim x_list] concatenates dual numbers along dimension dim. *)
val concat : dim:int -> t list -> t

(** [gumbel_softmax ~tau ~hard logits] performs Gumbel-Softmax sampling. *)
val gumbel_softmax : tau:float -> hard:bool -> t -> t

(** [cholesky x] computes Cholesky decomposition of the primal tensor of x. *)
val cholesky : t -> t

(** [linsolve_triangular ?left ?upper x y] solves triangular linear system. *)
val linsolve_triangular : ?left:bool -> ?upper:bool -> t -> t -> t

(** [linsolve ~left x y] solves linear system. *)
val linsolve : left:bool -> t -> t -> t

(** [kron x y] computes Kronecker product of dual numbers x and y. *)
val kron : t -> t -> t

(** Constant tensor operations.
    
    Operations that can only be performed on constant tensors (with no tangent component). *)
module Const : sig
  (** [svd x] computes singular value decomposition; returns [U, S, V^T] where U and V are orthogonal matrices and S is diagonal. *)
  val svd : t -> t * t * t

  (** [eigh ?uplo x] computes a symmetric eigenvalue decomposition; returns [eigenvalues, eigenvectors]. *)
  val eigh : ?uplo:string -> t -> t * t

  (** [qr x] computes QR decomposition; returns [Q, R] where Q is orthogonal and R is upper triangular. *)
  val qr : t -> t * t

  (** [sign x] computes sign function. *)
  val sign : t -> t
end
