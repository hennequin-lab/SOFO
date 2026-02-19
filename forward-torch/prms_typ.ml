open Base
open Torch
open Maths

(* ------------------------------------------------------------
   -- Basic types for parameters
   ------------------------------------------------------------ *)

type bounded =
  { v : Tensor.t
  ; lb : Tensor.t option
  ; ub : Tensor.t option
  }

type param =
  | Pinned of Tensor.t
  | Free of Tensor.t
  | Bounded of bounded

(** Path specified as a list of strings. *)
type path = String.t List.t

module type Basic = sig
  (** If you want to make your own custom parameter structure
      which for a reason or another cannot be automated using [ppx-prms]
      (e.g. if it is not a record type), then you will need to use
      {!Forward_torch.Prms.Make} and provide a module of this type. *)

  type +'a p

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

(* ------------------------------------------------------------
   -- Main type for parameter structures
   ------------------------------------------------------------ *)

module type T = sig
  include Basic

  (** Apply function [f] to each element in x. *)
  val iter : 'a p -> f:('a -> unit) -> unit

  (** Apply function [f] to each element in x and y. *)
  val iter2 : 'a p -> 'b p -> f:('a -> 'b -> unit) -> unit

  type nonrec t = t p
  type nonrec param = param p

  (** Extract value as Tensor.t in all elements in x. *)
  val value : param -> t

  val const : Tensor.t p -> t
  val primal : t -> Tensor.t p
  val numel : t -> int
  val tangent_exn : t -> Tensor.t p
  val dual : tangent:Tensor.t p -> t -> t
  val dual_on_demand : tangent:(Device.t -> Tensor.t) p -> t -> t
  val zeros_like : t -> t
  val zeros_like_k : k:int -> t -> t
  val ones_like : t -> t
  val rand_like : t -> t
  val randn_like : t -> t
  val randn_like_k : k:int -> t -> t

  (** Returns the dot product between x and y. *)
  val dot_prod : t -> t -> Maths.t

  (** Element-wise multiplication of two parameter sets *)
  val ( * ) : t -> t -> t

  (** Element-wise addition of two parameter sets *)
  val ( + ) : t -> t -> t

  (** Element-wise subtraction of two parameter sets *)
  val ( - ) : t -> t -> t

  (** Element-wise division of two parameter sets *)
  val ( / ) : t -> t -> t

  (** Multiplication of a parameter set with a scalar *)
  val ( *$ ) : t -> float -> t

  (** Adds a scalar to a parameter set *)
  val ( +$ ) : t -> float -> t

  (** Save the primal part of x as [kind] in [out]. *)
  val save : t -> kind:('a, 'b) Bigarray.kind -> out:string -> unit

  (** Load params from file onto [device]. *)
  val load : ?device:Device.t -> string -> t

  (** Save x as [kind] in [out] with [prefix] as .npy file. *)
  val save_npz : ?prefix:string -> kind:('a, 'b) Bigarray.kind -> out:string -> t -> unit
end
