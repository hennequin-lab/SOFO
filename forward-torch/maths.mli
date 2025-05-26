(** operations on the Maths.t object. Maths.t object is a dual number of (primal, tangent)
  where the primal is of type Tensor.t and the tangent is of type tangent. *)
open Base

open Torch

type ('a, 'b) t = ('a, 'b) Maths_typ.t
type const
type 'a dual


(** Create a constant tensor by pairing the primal with None as tangent. *)
val of_tensor : Tensor.t -> const

val to_tensor : const -> Tensor.t
val dual : tangent:Tensor.t -> const -> Tensor.t dual
val lazy_dual : tangent:(unit -> Tensor.t) -> const -> (unit -> Tensor.t) dual
val primal : (_, _) t -> const
val tangent : 'a dual -> const

(** Get shape of the primal tensor. *)
val shape : (_, _) t -> int list

(** Get the device of the primal tensor. *)
val device : (_, _) t -> Device.t
 
(** Get the kind of the primal tensor. *)
val kind : (_, _) t -> Torch_core.Kind.packed
 
(** Create a constant scalar tensor by pairing the primal with None as tangent. *)
val f : float -> const

type ('a,'b) diff1 = ('a, 'b) t -> ('a, Tensor.t) t

   (** Unary operations *)
(** Reshape x to [size]. *)
val view : size:int list -> (_, _) diff1

(** Reshape x to [size]. *)
val reshape : shape:int list -> ('a, 'b) diff1

(** Permute x along dimensions in [dims]. *)
val permute : dims:int list -> ('a, 'b) diff1

(** Removes dimension of size one at [dim] in x. *)
val squeeze : dim:int -> ('a, 'b) diff1
 
(** Insert dimension of size one at [dim] in x. *)
val unsqueeze : dim:int -> ('a, 'b) diff1

(** Element-wise -x. *)
val neg : (_,_) diff1

(** Returns trace(x). *)
val trace : (_,_) diff1

(** Element-wise sin(x). *)
val sin : (_,_) diff1

(** Element-wise cos(x). *)
val cos : (_,_) diff1

(** Element-wise square(x). *)
val sqr : (_,_)  diff1

(** Element-wise square_root(x). *)
val sqrt : (_,_) diff1

(** Element-wise log(x). *)
val log : (_,_) diff1

(** Element-wise exp(x). *)
val exp : (_,_) diff1

(** Element-wise tanh(x). *)
val tanh : (_,_) diff1

(** Returns x^-1 where x is a square matrix. *)
val inv_sqr : (_,_) diff1

    (*
(** Returns pseudo-inverse of x where x is a rectangular matrix, with [rcond] as the 
reciprocal condition number. *)
val inv_rectangle : ?rcond:float -> t -> t

(** Element-wise relu(x). *)
val relu : t -> t

(** Element-wise sigmoid(x). *)
val sigmoid : t -> t

(** Element-wise softplus(x). *)
val softplus : t -> t

(** Slice x along [dim] from [start] to [end_] with a [step]. *)
val slice : dim:int -> start:int option -> end_:int option -> step:int -> t -> t

(** Returns the sum of all elements in x. *)
val sum : t -> t

(** Returns the sum of x along dimensions in [dim] with option to [keepdim]. *)
val sum_dim : t -> dim:int list option -> keepdim:bool -> t

(** Returns the mean of all elements in x. *)
val mean : t -> t

(** Returns the mean of x along dimensions in [dim] with option to [keepdim]. *)
val mean_dim : t -> dim:int list option -> keepdim:bool -> t

(** Returns the max along dim 1 for a 2d x with option to [keepdim]. *)
val max_2d_dim1 : t -> keepdim:bool -> t

(** Returns x transposed along [dim0] and [dim1]. *)
val transpose : t -> dim0:int -> dim1:int -> t

(** Returns x transpose along the last two dims (batched transpose). *)
val btr : t -> t

(** Returns the lower triangular part of 2d x along [diagonal].  *)
val tril : t -> diagonal:int -> t

(** Returns cholesky(x) where x is 2d. *)
val cholesky : t -> t

(** Returns log of sum of exp(x_i) along dimensions in [dim] with option to [keepdim]. *)
val logsumexp : t -> dim:int list -> keepdim:bool -> t

(** Returns gumbel softmax of x with [tau] as the scaling, [with_noise] option on gumbel
noise and if [discrete] the output is one-hot encoded. 
  @see <https://arxiv.org/abs/1611.01144> the original paper. *)
val gumbel_softmax : t -> tau:float -> with_noise:bool -> discrete:bool -> t

(** Maxpooling a 2d x with kernel size specified as [ksize]. 
  @see <https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html> pytorch documentation
  for all other parameters. *)
val maxpool2d
  :  ?padding:int * int
  -> ?dilation:int * int
  -> ?ceil_mode:bool
  -> ?stride:int * int
  -> t
  -> ksize:int * int
  -> t

(** Binary operations *)
(** Element-wise addition. *)
val ( + ) : t -> t -> t

(** Element-wise subtration. *)
val ( - ) : t -> t -> t

(** Element-wise multiplication. *)
val ( * ) : t -> t -> t

(** Element-wise division. *)
val ( / ) : t -> t -> t

(** Adds a scalar to Maths.t . *)
val ( $+ ) : float -> t -> t

(** Multiply a scalar with Maths.t . *)
val ( $* ) : float -> t -> t

(** Divide a scalar by Maths.t . *)
val ( $/ ) : float -> t -> t

(** Divide Maths.t by a scalar . *)
val ( /$ ) : t -> float -> t

(** Returns z where z = x matrix_multiply y *)
val ( *@ ) : t -> t -> t

(** Given a list of operants and the result, perform einsum. 
  example: A*@ B *@ C = einsum \[ a, "ij"; b, "jk"; c, "km" \] "im". *)
val einsum : (t * string) list -> string -> t

(** If [left] solve for ax = b otherwise solve for xa = b. 
  Note that if left is false, then b must be 3D (i.e. m x p x n) whereas if left is true, b can be 2D (i.e. m x n)*)
val linsolve : t -> t -> left:bool -> t

(** Similar to linsolve but a needs to be either upper-triangular ([upper] = true) or
lower-triangular. *)
val linsolve_triangular : t -> t -> left:bool -> upper:bool -> t

(** Apply a 2d convolution over x, with the convolution weight and bias specified by [bias].  
@see <https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html> pytorch documentation
  for all other parameters. *)
val conv2d
  :  ?padding:int * int
  -> ?dilation:int * int
  -> ?groups:int
  -> bias:t
  -> stride:int * int
  -> t
  -> t
  -> t

(** Returns z which is x and y concatenated along [dim]. *)
val concat : t -> t -> dim:int -> t

(** Concat elements in the list along [dim]. *)
val concat_list : t list -> dim:int -> t

(** Check gradient against finite difference for unary operations given a tensor input. *)
val check_grad1 : (t -> t) -> Tensor.t -> float

(** Check gradient against finite difference for binary operations given two tensor inputs. *)
val check_grad2 : (t -> t -> t) -> Tensor.t -> Tensor.t -> float
*)
