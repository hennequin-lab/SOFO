open Base
open Maths
include module type of Prms_typ

val value : param -> const t
val maybe_apply_bounds : param -> param

module Make (B : Basic) : T with type 'a p = 'a B.p

module Singleton : sig
  include T with type 'a p = 'a

  val pinned : const t -> param
  val free : const t -> param
  val bounded : ?above:const t -> ?below:const t -> const t -> param
end

module None : T with type 'a p = unit
module Pair (P1 : T) (P2 : T) : T with type 'a p = 'a P1.p * 'a P2.p
module Option (P : T) : T with type 'a p = 'a P.p Option.t
module List (P : T) : T with type 'a p = 'a P.p List.t

(* ------------------------------------------------------------
   -- Arrays of parameters
   ------------------------------------------------------------ *)

module Array (P : T) : T with type 'a p = 'a P.p array
