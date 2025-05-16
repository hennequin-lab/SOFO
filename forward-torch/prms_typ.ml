open Base
open Torch

type ('a, 'const, 'bound) tag =
  | Const of 'const
  | Free of 'a
  | Bounded of 'a * 'bound option * 'bound option

(** Parameter type, can be constant, free or bounded. *)
type tagged = (Tensor.t, Tensor.t, Tensor.t) tag

(** Creator creates a parameter type from the original tensor. *)
type creator = Tensor.t -> tagged

(** Path specified as a list of strings. *)
type path = String.t List.t

module type Basic = sig
  (** If you want to make your own custom parameter structure
      which for a reason or another cannot be automated using [ppx-prms]
      (e.g. if it is not a record type), the you will need to use {!Forward_torch.Prms.Make}
      and provide a module of this type. *)

  type 'a p

  (** Apply function [f] on all elements in x. *)
  val map : 'a p -> f:('a -> 'b) -> 'b p

  (** Apply function [f] on all elements in x and y. *)
  val map2 : 'a p -> 'b p -> f:('a -> 'b -> 'c) -> 'c p

  (** Fold x with [init] and function [f].
    [?path] optionally contains a record of the path (a list of strings, in reverse order) taken to
      arrive at the current value, if it is nested within a broader structure. Make sure to use this
      if you want to attach string labels to the various components of your custom ['a p] type
      (e.g. use for saving to files, see {!Forward_torch.Prms.module-type-T.T.save_npy}). *)
  val fold : ?path:path -> 'a p -> init:'c -> f:('c -> 'a * path option -> 'c) -> 'c

  (** Fold x and y with [init] and function [f]. *)
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

  (** Similar to gaussian_like, except for that the tensors sampled has an extra dimension [k] in front. *)
  val gaussian_like_k : ?mu:float -> ?sigma:float -> k:int -> t -> t

  (** Returns the number of parameters in x. *)
  val numel : t -> int

  (** Returns the dot product between x and y. *)
  val dot_prod : t -> t -> elt

  (** Each element in x is squared. *)
  val sqr : t -> t

  (** Each element in x is square-rooted. *)
  val sqrt : t -> t

  (** Element-wise multiplication of two parameter sets *)
  val ( * ) : t -> t -> t

  (** Element-wise addition of two parameter sets *)
  val ( + ) : t -> t -> t

  (** Element-wise subtraction of two parameter sets *)
  val ( - ) : t -> t -> t

  (** Element-wise division of two parameter sets *)
  val ( / ) : t -> t -> t

  (** Multiplication of a parameter set with a scalar *)
  val ( $* ) : float -> t -> t

  (** Adds a scalar to a parameter set *)
  val ( $+ ) : float -> t -> t
end

(** Main type for parameter structures. *)
module type T = sig
  include Basic

  type nonrec tagged = tagged p

  (** Apply function [f] to each element in x. *)
  val iter : 'a p -> f:('a -> unit) -> unit

  (** Apply function [f] to each element in x and y. *)
  val iter2 : 'a p -> 'b p -> f:('a -> 'b -> unit) -> unit

  (** Module T contains parameters with elements of type Tensor.t. *)
  module T : sig
    include Ops with type t = Tensor.t p and type elt = Tensor.t

    (** Save x as [kind] in [out]. *)
    val save : t -> kind:('a, 'b) Bigarray.kind -> out:string -> unit

    (** Load params from file onto [device]. *)
    val load : ?device:Device.t -> string -> t

    (** Save x as [kind] in [out] with [prefix] as .npy file. *)
    val save_npy
      :  ?prefix:string
      -> kind:('a, 'b) Bigarray.kind
      -> out:string
      -> t
      -> unit
  end

  (** Module T contains parameters with elements of type Maths.t. *)
  module M : sig
    include Ops with type t = Maths.t p and type elt = Maths.t
  end

  (** Apply Maths.t to all elements in x. *)
  val const : T.t -> M.t

  (** Extract value as Tensor.t in all elements in x. *)
  val value : tagged -> T.t

  (** Extract primal of all elements in x. *)
  val primal : M.t -> T.t

  (** Extract tangents (as Tensor.t) of all elemnts in x. *)
  val tangent : M.t -> T.t

  (** Make dual number of (primal, [t] as tangent). *)
  val make_dual : tagged -> t:Maths.tangent p -> M.t

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
