open Base
open Torch

type ('a, 'const, 'bound) tag =
  | Const of 'const
  | Free of 'a
  | Bounded of 'a * 'bound option * 'bound option

(* parameter type, can be constant, free or bounded. *)
type t = (Tensor.t, Tensor.t, Tensor.t) tag

(* creator creates a parameter type from the original tensor. *)
type creator = Tensor.t -> t
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

module type T = sig
  (** Main type for parameter structures. *)

  include Basic

  (* p of Maths.t; sets of dual numbers. *)
  type t' = Maths.t p

  (* p of t; sets of parameters. *)
  type nonrec t = t p

  val const : Tensor.t p -> Maths.t p
  val value : t -> Tensor.t p
  val primal : t' -> Tensor.t p
  val tangent : t' -> Tensor.t p
  val iter : 'a p -> f:('a -> unit) -> unit
  val iter2 : 'a p -> 'b p -> f:('a -> 'b -> unit) -> unit
  val numel : Tensor.t p -> int

  (** Below are a few commonly used math operations that can be
      defined on arbitrary parameter sets, and which are provided
      as shortcuts to avoid having to use {!map}. *)

  val sqr : t' -> t'
  val sqrt : t' -> t'
  val dot_prod : t' -> t' -> Maths.t
  val ( * ) : t' -> t' -> t'
  val ( + ) : t' -> t' -> t'
  val ( - ) : t' -> t' -> t'
  val ( / ) : t' -> t' -> t'
  val ( $* ) : float -> t' -> t'
  val ( $+ ) : float -> t' -> t'

  module C : sig
    type t' = Tensor.t p

    val sqr : t' -> t'
    val sqrt : t' -> t'
    val dot_prod : t' -> t' -> Tensor.t
    val ( + ) : t' -> t' -> t'
    val ( - ) : t' -> t' -> t'
    val ( * ) : t' -> t' -> t'
    val ( / ) : t' -> t' -> t'
    val ( $+ ) : float -> t' -> t'
    val ( $* ) : float -> t' -> t'
  end

  (** [zeros_of x] has the same structure as [x] but each [AD.t] parameter is replaced
      by zeros of the same shape. *)
  val zeros_like : Tensor.t p -> Tensor.t p

  (** [ones_of x] has the same structure as [x] but each [AD.t] parameter is replaced
      by ones of the same shape. *)
  val ones_like : Tensor.t p -> Tensor.t p

  (** [gaussian_of ~mu ~sigma x] has the same structure as [x] but each [AD.t] parameter is replaced
      by independent normal samples of the same shape (mean [?(mu=0.)] and stdev. [?(sigma=1.)]). *)
  val gaussian_like : ?mu:float -> ?sigma:float -> Tensor.t p -> Tensor.t p

  val make_dual : Tensor.t p -> t:Maths.tangent p -> t'
  val save : ('a, 'b, 'c) Bigarray.Genarray.t p -> out:string -> unit
  val load : string -> ('a, 'b, 'c) Bigarray.Genarray.t p

  val save_npy
    :  ?prefix:string
    -> out:string
    -> ('a, 'b, 'c) Bigarray.Genarray.t p
    -> unit

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
