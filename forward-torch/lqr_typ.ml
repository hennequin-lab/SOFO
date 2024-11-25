open Base

type 'a momentary_params =
  { _f : 'a option
  ; _Fx_prod : 'a -> 'a
      (* product with a matrix of size (x,x); e.g. THE A matrix in a linear system *)
  ; _Fx_prod2 : 'a -> 'a (* same thing, but product from the left *)
  ; _Fu_prod : 'a -> 'a
      (* product with a matrix of size (u,x); e.g. THE B matrix in a linear system *)
  ; _Fu_prod2 : 'a -> 'a (* same thing, but product from the left *)
  ; _cx : 'a option
  ; _cu : 'a option
  ; _Cxx : 'a
  ; _Cxu : 'a
  ; _Cuu : 'a
  }

type 'a params =
  { x0 : 'a
  ; params : 'a momentary_params list
  }

type 'a solution =
  { u : 'a
  ; x : 'a
  }

module type Ops = sig
  type t

  val zero : t
  val shape : t -> int list
  val ( + ) : t -> t -> t
  val ( - ) : t -> t -> t
  val ( * ) : t -> t -> t
  val ( / ) : t -> t -> t
  val neg : t -> t
  val einsum : (t * string) list -> string -> t
  val cholesky : t -> t
  val linsolve_triangular : left:bool -> upper:bool -> t -> t -> t
  val reshape : t -> shape:int list -> t
end

module type T = sig
  type t

  val solve : t params -> t solution list
end
