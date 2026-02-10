open! Base
open Maths

type dual

val const : any t -> dual
val primal : dual -> any t
val lift1 : (any t -> any t) -> dual -> dual
val lift2 : (any t -> any t -> any t) -> dual -> dual -> dual
val ( + ) : dual -> dual -> dual
val ( - ) : dual -> dual -> dual
val ( * ) : dual -> dual -> dual
val ( / ) : dual -> dual -> dual
val ( $+ ) : float -> dual -> dual
val ( $- ) : float -> dual -> dual
val ( *@ ) : dual -> dual -> dual
val sigmoid : dual -> dual
val tanh : dual -> dual
val mean : dual -> dual
val sqr : dual -> dual
val eval : ('a -> 'b) -> 'a -> 'b

module Make (P : Prms.T) : sig
  val const : any t P.p -> dual P.p
  val grad : (dual P.p -> dual) -> dual P.p -> any t * const t P.p
end
