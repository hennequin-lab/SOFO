open! Base
open Maths

type dual

val of_const : const t -> dual
val get_primal : dual -> const t

val lift1 : (const t -> const t) -> dual -> dual
val lift2 : (const t -> const t -> const t) -> dual -> dual -> dual
val einsum : (dual * string) list -> string -> dual
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

module Make (P : Prms.T) : sig
  val lift : (const t P.p -> const t) -> dual P.p -> dual
  val lift_dual : (dual P.p -> dual) -> dual P.p -> dual
  val zero_adj : const t -> dual
  val zero_adj_prms : const t P.p -> dual P.p
  val eval : (dual P.p -> 'a) -> const t P.p -> 'a
  val grad : (dual P.p -> dual) -> dual P.p -> const t * const t P.p
end
