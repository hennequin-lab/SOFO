open Base
open Torch
open Maths
include module type of Prms_typ

val value : param -> t
val enforce_bounds : ?lb:Tensor.t -> ?ub:Tensor.t -> Tensor.t -> Tensor.t
val cat : string -> path option -> path option

module Make (B : Basic) : T with type 'a p = 'a B.p

module Single : sig
  include T with type +'a p = 'a

  val pinned : t -> param
  val free : t -> param
  val bounded : ?above:t -> ?below:t -> t -> param
end

module None : T with type +'a p = unit
module Pair (P1 : T) (P2 : T) : T with type 'a p = 'a P1.p * 'a P2.p
module Option (P : T) : T with type 'a p = 'a P.p Option.t
module List (P : T) : T with type 'a p = 'a P.p List.t
