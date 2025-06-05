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

  type 'a elt = 'a t
  type nonrec 'a t = 'a elt p
  type nonrec param = param p

  (** Extract value as Tensor.t in all elements in x. *)
  val value : param -> const t

  val any : _ some t -> any t
  val of_tensor : Tensor.t p -> const t
  val to_tensor : _ some t -> Tensor.t p
  val const : _ some t -> const t
  val numel : _ some t -> int
  val tangent_exn : _ some t -> const t
  val dual : tangent:const t -> const t -> dual t
  val dual_on_demand : tangent:(Device.t -> const elt) p -> const t -> dual t
  val zeros_like : _ some t -> const t
  val zeros_like_k : k:int -> _ some t -> const t
  val ones_like : _ some t -> const t
  val rand_like : _ some t -> const t
  val randn_like : _ some t -> const t
  val randn_like_k : k:int -> _ some t -> const t

  (** Returns the dot product between x and y. *)
  val dot_prod : _ some t -> _ some t -> any elt

  (** Element-wise multiplication of two parameter sets *)
  val ( * ) : _ some t -> _ some t -> any t

  (** Element-wise addition of two parameter sets *)
  val ( + ) : _ some t -> _ some t -> any t

  (** Element-wise subtraction of two parameter sets *)
  val ( - ) : _ some t -> _ some t -> any t

  (** Element-wise division of two parameter sets *)
  val ( / ) : _ some t -> _ some t -> any t

  (** Multiplication of a parameter set with a scalar *)
  val ( $* ) : float -> 'a some t -> 'a t

  (** Adds a scalar to a parameter set *)
  val ( $+ ) : float -> 'a some t -> 'a t

  module C : sig
    val dot_prod : const t -> const t -> const elt
    val ( * ) : const t -> const t -> const t
    val ( + ) : const t -> const t -> const t
    val ( - ) : const t -> const t -> const t
    val ( / ) : const t -> const t -> const t
    val ( $* ) : float -> const t -> const t
    val ( $+ ) : float -> const t -> const t

    (** Save x as [kind] in [out]. *)
    val save : const t -> kind:('a, 'b) Bigarray.kind -> out:string -> unit

    (** Load params from file onto [device]. *)
    val load : ?device:Device.t -> string -> const t

    (** Save x as [kind] in [out] with [prefix] as .npy file. *)
    val save_npz
      :  ?prefix:string
      -> kind:('a, 'b) Bigarray.kind
      -> out:string
      -> const t
      -> unit
  end
end
