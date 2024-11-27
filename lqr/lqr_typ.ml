open Base
open Forward_torch

(* conventions is, for each batch element:
   x_(t+1) = x_t A + u_t B *)

type 'a momentary_params =
  { _f : 'a option
  ; _Fx_prod : 'a -> 'a (* Av product *)
  ; _Fx_prod2 : 'a -> 'a (* vA product *)
  ; _Fu_prod : 'a -> 'a (* Bv produt *)
  ; _Fu_prod2 : 'a -> 'a (* vB product *)
  ; _cx : 'a option
  ; _cu : 'a option
  ; _Cxx : 'a
  ; _Cxu : 'a option
  ; _Cuu : 'a
  }

module Params = struct
  type ('a, 'p) p =
    { x0 : 'a
    ; params : 'p
    }
  [@@deriving prms]
end

module Solution = struct
  type 'a p =
    { u : 'a
    ; x : 'a
    }
  [@@deriving prms]
end

module type Ops = sig
  type t

  val print_shape : t -> label:string -> unit
  val zeros: like:t -> shape:int list -> t
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

  val solve : (t, t momentary_params list) Params.p -> t Solution.p list
end
