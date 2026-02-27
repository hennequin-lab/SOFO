open! Base
open Maths

type dual

val const : t -> dual
val primal : dual -> t
val adjoint : dual -> t option
val lift1 : (t -> t) -> dual -> dual
val lift2 : (t -> t -> t) -> dual -> dual -> dual
val lift2_float : (t -> float -> t) -> dual -> float -> dual
val ( + ) : dual -> dual -> dual
val ( - ) : dual -> dual -> dual
val ( * ) : dual -> dual -> dual
val ( / ) : dual -> dual -> dual
val ( +$ ) : dual -> float -> dual
val ( -$ ) : dual -> float -> dual
val ( *$ ) : dual -> float -> dual
val ( /$ ) : dual -> float -> dual
val ( *@ ) : dual -> dual -> dual
val neg : dual -> dual
val sigmoid : dual -> dual
val tanh : dual -> dual
val mean : dual -> dual
val sqr : dual -> dual
val log : dual -> dual
val eval : ('a -> 'b) -> 'a -> 'b
val zero_adj : t -> dual
val update_adj : dual -> t -> unit
val grad : ('a -> dual * 'b) -> 'a -> dual * 'b
val concat : dim:int -> dual list -> dual

module Bernoulli : sig
  val sample : ?beta:float -> dual -> dual
end

module Make (P : Prms.T) : sig
  val const : t P.p -> dual P.p
  val zero_adj : t P.p -> dual P.p
end
