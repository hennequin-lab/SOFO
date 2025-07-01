open Torch
open Base

let print s = Stdio.printf "%s\n%!" (Base.Sexp.to_string_hum s)

module type DeferType = sig
  type t

  exception Not_yet_set
  (* exception Already_set *)

  val empty : unit -> t
  val get_exn : t -> Tensor.t
  val set_exn : t -> Tensor.t -> unit
end

module Deferred : DeferType = struct
  type t = Tensor.t option ref

  exception Not_yet_set
  (* exception Already_set *)

  let empty () = ref None

  let get_exn t =
    match !t with
    | Some v -> v
    | None -> raise Not_yet_set

  let set_exn t v =
    match !t with
    | Some _ -> t := Some v
    | None -> t := Some v
end

(* direct means instantiate it now; lazy means instantiate only when called. *)
type tangent =
  | Direct of Tensor.t
  | Lazy of (unit -> Tensor.t)
  | Deferred of Deferred.t

(* dual number type: (primal, optional tangent batch) pair.*)
type t = Tensor.t * tangent Option.t

exception Not_a_dual_number
exception Wrong_shape of string
exception Check_grad_failed
