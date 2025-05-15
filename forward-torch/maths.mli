(** operations on the Maths.t object. Maths.t object is a dual number of (primal, tangent)
  where the primal is of type Tensor.t and the tangent is of type tangent. *)
open Torch
include module type of Maths_typ

(** get shape of the primal tensor. *)
val shape : t -> int list

(** create a constant tensor by pairing the primal with None as tangent. *)
val const : Tensor.t -> t

(** create a constant scalar tensor by pairing the primal with None as tangent. *)
val f : float -> t

(** get primal tensor , which is the first element. *)
val primal : t -> Tensor.t

(** make dual number of (primal, [t] as tangent). *)
val make_dual : Tensor.t -> t:tangent -> t

(** get tangent value from the tangent type. *)
val tangent' : tangent -> Tensor.t

(* get tangent value option from the t type. *)
val tangent : t -> Tensor.t option

(** Unary operations *)
(** reshape x to [size]. *)
val view : t -> size:int list -> t

(** reshape x to [size]. *)
val reshape : t -> shape:int list -> t

(** permute x along dimensions in [dims]. *)
val permute : t -> dims:int list -> t

(** insert dimension of size one at [dim] in x. *)
val unsqueeze : t -> dim:int -> t

(** returns -x. *)
val neg : t -> t

(** returns trace(x). *)
val trace : t -> t

(** returns sin(x). *)
val sin : t -> t

(** returns cos(x). *)
val cos : t -> t

(** returns square(x). *)
val sqr : t -> t

(** returns square_root(x). *)
val sqrt : t -> t

(** returns log(x). *)
val log : t -> t

(** returns exp(x). *)
val exp : t -> t

(** returns tanh(x). *)
val tanh : t -> t

(** returns x^-1 where x is a square matrix. *)
val inv_sqr : t -> t

(** returns pseudo-inverse of x where x is a rectangular matrix, with [rcond] as the 
reciprocal condition number. *)
val inv_rectangle : ?rcond:float -> t -> t

(** returns relu(x). *)
val relu : t -> t

(** returns sigmoid(x). *)
val sigmoid : t -> t

(** returns softplus(x). *)
val softplus : t -> t

(** slice x along [dim] from [start] to [end_] with a [step]. *)
val slice : dim:int -> start:int option -> end_:int option -> step:int -> t -> t

(** returns the sum of all elements in x. *)
val sum : t -> t

(** returns the sum of x along dimensions in [dim] with option to [keepdim]. *)
val sum_dim : t -> dim:int list option -> keepdim:bool -> t

(** returns the mean of all elements in x. *)
val mean : t -> t

(** returns the mean of x along dimensions in [dim] with option to [keepdim]. *)
val mean_dim : t -> dim:int list option -> keepdim:bool -> t

(** returns the max along dim 1 for a 2d x with option to [keepdim]. *)
val max_2d_dim1 : t -> keepdim:bool -> t

(** returns x transposed along [dim0] and [dim1]. *)
val transpose : t -> dim0:int -> dim1:int -> t

(** returns x transpose along the last two dims (batched transpose). *)
val btr : t -> t

(** returns the lower triangular part of 2d x along [diagonal].  *)
val tril : t -> diagonal:int -> t

(** returns cholesky(x) where x is 2d. *)
val cholesky : t -> t

(** returns log of sum of exp(x_i) along dimensions in [dim] with option to [keepdim]. *)
val logsumexp : t -> dim:int list -> keepdim:bool -> t

(** returns gumbel softmax of x with [tau] as the scaling, [with_noise] option on gumbel
noise and if [discrete] the output is one-hot encoded. 
  @see <https://arxiv.org/abs/1611.01144> the original paper. *)
val gumbel_softmax : t -> tau:float -> with_noise:bool -> discrete:bool -> t

(** maxpooling a 2d x with kernel size specified as [ksize]. 
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
(** returns z where z = x + y. *)
val ( + ) : t -> t -> t

(** returns z where z = x - y. *)
val ( - ) : t -> t -> t

(** returns z where z = x * y. *)
val ( * ) : t -> t -> t

(** returns z where z = x / y. *)
val ( / ) : t -> t -> t

(** returns z where z = x (scalar) + y. *)
val ( $+ ) : float -> t -> t

(** returns z where z = x (scalar) * y. *)
val ( $* ) : float -> t -> t

(** returns z where z = x (scalar) / y. *)
val ( $/ ) : float -> t -> t

(** returns z where z = x / y (scalar). *)
val ( /$ ) : t -> float -> t

(** returns z where z = x matrix_multiply y *)
val ( *@ ) : t -> t -> t

(** given a list of operants and the result, perform einsum. 
  example: A*@ B *@ C = einsum \[ a, "ij"; b, "jk"; c, "km" \] "im". *)
val einsum : (t * string) list -> string -> t

(** if [left] solve for ax = b otherwise solve for xa = b. 
  Note that if left is false, then b must be 3D (i.e. m x p x n) whereas if left is true, b can be 2D (i.e. m x n)*)
val linsolve : t -> t -> left:bool -> t

(** similar to linsolve but a needs to be either upper-triangular ([upper] = true) or
lower-triangular. *)
val linsolve_triangular : t -> t -> left:bool -> upper:bool -> t

(** apply a 2d convolution over x, with the convolution weight and bias specified by [bias].  
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

(** returns z which is x and y concatenated along [dim]. *)
val concat : t -> t -> dim:int -> t

(** concat elements in the list along [dim]. *)
val concat_list : t list -> dim:int -> t

(** check gradient against finite difference for unary operations given a tensor input. *)
val check_grad1 : (t -> t) -> Tensor.t -> float

(** check gradient against finite difference for binary operations given two tensor inputs. *)
val check_grad2 : (t -> t -> t) -> Tensor.t -> Tensor.t -> float
