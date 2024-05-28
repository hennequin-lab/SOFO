(** {1 Main modules} *)
open Torch

include module type of Prms_typ

val value : ('a, 'a, _) tag -> 'a

(** {1 Basic functions to deal with parameter values} *)

val bind
  :  ('a, 'a, 'bound_a) tag
  -> f:('a -> ('b, 'b, 'bound_b) tag)
  -> ('b, 'b, 'bound_b) tag

val map : t -> f:(Tensor.t -> Tensor.t) -> t

(** Constructs a constant parameter. *)
val const : creator

(** Constructs a free parameter. *)
val free : creator

(** Constructs a free or (partially) bounded parameter. *)
val create : ?above:Tensor.t -> ?below:Tensor.t -> creator

(** Pin (i.e. convert to Const) either to its current value, or to some [to_]. *)
val pin : ?to_:Tensor.t -> t -> t

(** Introduce bounds to a Free parameter; fails otherwise *)
val bound : ?above:Tensor.t -> ?below:Tensor.t -> t -> t

(** Number of float elements in this parameter *)
val numel : Tensor.t -> int

module Let_syntax : sig
  val ( let* ) : ('a, 'a, _) tag -> ('a -> ('b, 'b, 'bound) tag) -> ('b, 'b, 'bound) tag

  val ( and* )
    :  ('a, 'a, 'bound_a) tag
    -> ('b, 'b, _) tag
    -> ('a * 'b, 'a * 'b, 'bound_a) tag

  val ( let+ )
    :  ('a, Tensor.t, Tensor.t) tag
    -> ('a -> Tensor.t)
    -> (Tensor.t, Tensor.t, Tensor.t) tag

  val ( and+ )
    :  ('a, Tensor.t, Tensor.t) tag
    -> (Tensor.t, Tensor.t, Tensor.t) tag
    -> ('a * Tensor.t, Tensor.t, Tensor.t) tag

  val ( and++ )
    :  ('a, Tensor.t, Tensor.t) tag
    -> Tensor.t
    -> ('a * Tensor.t, Tensor.t, Tensor.t) tag
end

(** {1 Parameter sets} *)

val cat : string -> path option -> path option

(** This functor takes a Basic module that supplies map, fold, etc,
    and populates it into a full-fledged T module. If you use the
    [[@@deriving prms]] attribute, you won't be needing this. *)
module Make (B : Basic) : T with type 'a p = 'a B.p

(** Empty parameter set; useful if your model must expose
    a parameter module but the model doesn't actually introduce
    any parameter. *)
module None : T with type 'a p = unit

(** Single parameter *)
module P : T with type 'a p = 'a

(** Single optional parameter *)
module Option (P : T) : T with type 'a p = 'a P.p Option.t

(** List of parameters of the same type. *)
module List (P : T) : T with type 'a p = 'a P.p List.t

(** Array of parameters of the same type. *)
module Array (P : T) : T with type 'a p = 'a P.p array
