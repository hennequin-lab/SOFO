open! Base
open Maths

type dual

val lift1 : (const t -> const t) -> dual -> dual
val lift2 : (const t -> const t -> const t) -> dual -> dual -> dual
val einsum : (dual * string) list -> string -> dual

module Make (P : Prms.T) : sig
  val lift : (const t P.p -> const t) -> dual P.p -> dual
  val eval : (dual P.p -> 'a) -> const t P.p -> 'a
  val grad : (dual P.p -> dual) -> const t P.p -> const t * const t P.p
end
