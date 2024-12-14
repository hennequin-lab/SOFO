open Torch
include module type of Maths_typ

val shape : t -> int list
val const : Tensor.t -> t
val f : float -> t
val primal : t -> Tensor.t
val make_dual : Tensor.t -> t:tangent -> t

(* get tangent value from the tangent type. *)
val tangent' : tangent -> Tensor.t

(* get tangent value option from the t type (primal, tangent pair). *)
val tangent : t -> Tensor.t option

(** Unary operations *)
val view : t -> size:int list -> t

val reshape : t -> shape:int list -> t
val permute : t -> dims:int list -> t
val unsqueeze : t -> dim:int -> t
val squeeze : t -> dim:int -> t
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
val inv_sqr : t -> t
val inv_rectangle : ?rcond:float -> t -> t
val relu : t -> t
val sigmoid : t -> t
val softplus : t -> t
val slice : dim:int -> start:int option -> end_:int option -> step:int -> t -> t
val sum : t -> t

(* sum along a dimension *)
val sum_dim : t -> dim:int list option -> keepdim:bool -> t
val mean : t -> t

(* mean along a dimension *)
val mean_dim : t -> dim:int list option -> keepdim:bool -> t
val max_2d_dim1 : t -> keepdim:bool -> t

(* transpose along two dimensions *)
val transpose : t -> dim0:int -> dim1:int -> t

(* batched transposition -- always swaps the last two dimensions *)
val btr : t -> t

(* take the batch diagonal *)
val diagonal : t -> offset:int -> t

(* diagonal of certain 2d planes are filled by input *)
val diag_embed : t -> offset:int -> dim1:int -> dim2:int -> t
val tril : t -> diagonal:int -> t
val cholesky : t -> t
val logsumexp : t -> dim:int list -> keepdim:bool -> t
val gumbel_softmax : t -> tau:float -> with_noise:bool -> discrete:bool -> t

val maxpool2d
  :  ?padding:int * int
  -> ?dilation:int * int
  -> ?ceil_mode:bool
  -> ?stride:int * int
  -> t
  -> ksize:int * int
  -> t

(** Binary operations *)
val ( + ) : t -> t -> t

val ( - ) : t -> t -> t
val ( * ) : t -> t -> t
val ( / ) : t -> t -> t

(* scalar addition. *)
val ( $+ ) : float -> t -> t

(* scalar multiplication. *)
val ( $* ) : float -> t -> t

(* scalar divisions *)
val ( $/ ) : float -> t -> t
val ( /$ ) : t -> float -> t

(* matrix multiplication. *)
val ( *@ ) : t -> t -> t

(* Einstein summation *)
val einsum : (t * string) list -> string -> t

(* ax = b, find x*)
val linsolve : t -> t -> left:bool -> t
val linsolve_triangular : t -> t -> left:bool -> upper:bool -> t

val kron: t -> t -> t

val conv2d
  :  ?padding:int * int
  -> ?dilation:int * int
  -> ?groups:int
  -> bias:t
  -> stride:int * int
  -> t
  -> t
  -> t

val concat : t -> t -> dim:int -> t
val concat_list : t list -> dim:int -> t

(* check gradient against finite difference for unary operations. *)
val check_grad1 : (t -> t) -> Tensor.t -> float

(* check gradient against finite difference for binary operations. *)
val check_grad2 : (t -> t -> t) -> Tensor.t -> Tensor.t -> float

(*
   (* check gradient against finite difference for lqr operations. *)
   val check_grad_lqr
   :  (state_params:(Tensor.t * tangent option) state_params
   -> cost_params:(Tensor.t * tangent option) cost_params
   -> (Tensor.t * tangent option) list * 'a)
   -> state_params:Tensor.t state_params
   -> cost_params:Tensor.t cost_params
   -> float
*)
