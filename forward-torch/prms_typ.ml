open Base
open Torch

type ('a, 'const, 'bound) tag =
  | Const of 'const
  | Free of 'a
  | Bounded of 'a * 'bound option * 'bound option

(* parameter type, can be constant, free or bounded. *)
type tagged = (Tensor.t, Tensor.t, Tensor.t) tag

(* creator creates a parameter type from the original tensor. *)
type creator = Tensor.t -> tagged
type path = String.t List.t

module type Basic = sig
  (** If you want to make your own custom parameter structure
      which for a reason or another cannot be automated using [ppx-prms]
      (e.g. if it is not a record type), the you will need to use {!Make}
      and provide a module of this {!Basic} type. *)

  type 'a p

  val map : 'a p -> f:('a -> 'b) -> 'b p
  val map2 : 'a p -> 'b p -> f:('a -> 'b -> 'c) -> 'c p

  (** [?path] optionally contains a record of the path (a list of strings, in reverse order) taken to
      arrive at the current value, if it is nested within a broader structure. Make sure to use this
      if you want to attach string labels to the various components of your custom ['a p] type
      (e.g. use for saving to files, see {!Prms.T.save_txt}). *)
  val fold : ?path:path -> 'a p -> init:'c -> f:('c -> 'a * path option -> 'c) -> 'c

  val fold2
    :  ?path:path
    -> 'a p
    -> 'b p
    -> init:'c
    -> f:('c -> 'a * 'b * path option -> 'c)
    -> 'c
end

module type Ops = sig
  type elt
  type t

  (** [zeros_of x] has the same structure as [x] but each [AD.t] parameter is replaced
      by zeros of the same shape. *)
  val zeros_like : t -> t

  (** [ones_of x] has the same structure as [x] but each [AD.t] parameter is replaced
      by ones of the same shape. *)
  val ones_like : t -> t

  (** [gaussian_of ~mu ~sigma x] has the same structure as [x] but each [AD.t] parameter is replaced
      by independent normal samples of the same shape (mean [?(mu=0.)] and stdev. [?(sigma=1.)]). *)
  val gaussian_like : ?mu:float -> ?sigma:float -> t -> t

  val numel : t -> int
  val dot_prod : t -> t -> elt
  val sqr : t -> t
  val sqrt : t -> t
  val ( * ) : t -> t -> t
  val ( + ) : t -> t -> t
  val ( - ) : t -> t -> t
  val ( / ) : t -> t -> t
  val ( $* ) : float -> t -> t
  val ( $+ ) : float -> t -> t
end

module type T = sig
  (** Main type for parameter structures. *)

  include Basic

  type nonrec tagged = tagged p

  val iter : 'a p -> f:('a -> unit) -> unit
  val iter2 : 'a p -> 'b p -> f:('a -> 'b -> unit) -> unit

  module T : sig
    include Ops with type t = Tensor.t p and type elt = Tensor.t

    val save : t -> kind:('a, 'b) Bigarray.kind -> out:string -> unit
    val load : ?device:Device.t -> string -> t

    val save_npy
      :  ?prefix:string
      -> kind:('a, 'b) Bigarray.kind
      -> out:string
      -> t
      -> unit
  end

  module M : sig
    include Ops with type t = Maths.t p and type elt = Maths.t
  end

  val const : T.t -> M.t
  val value : tagged -> T.t
  val primal : M.t -> T.t
  val tangent : M.t -> T.t
  val make_dual : T.t -> t:Maths.tangent p -> M.t

  module Let_syntax : sig
    (** This module lifts the {!Let_syntax} module for the [t] type to the [t p] type. *)
    val ( let* )
      :  ('a, 'a, _) tag p
      -> ('a p -> ('b, 'b, 'bound) tag p)
      -> ('b, 'b, 'bound) tag p

    val ( and* )
      :  ('a, 'a, 'bound) tag p
      -> ('b, 'b, _) tag p
      -> ('a * 'b, 'a * 'b, 'bound) tag p

    val ( let+ )
      :  ('a, Tensor.t, Tensor.t) tag p
      -> ('a -> Tensor.t)
      -> (Tensor.t, Tensor.t, Tensor.t) tag p

    val ( and+ )
      :  ('a, Tensor.t, Tensor.t) tag p
      -> (Tensor.t, Tensor.t, Tensor.t) tag p
      -> ('a * Tensor.t, Tensor.t, Tensor.t) tag p

    val ( and++ )
      :  ('a, Tensor.t, Tensor.t) tag p
      -> Tensor.t p
      -> ('a * Tensor.t, Tensor.t, Tensor.t) tag p
  end
end
