open Torch

(* direct means instantiate it now; lazy means instantiate only when called. *)
type tangent =
  | Direct of Tensor.t
  | Lazy of (unit -> Tensor.t)

(* dual number type: (primal, optional tangent batch) pair.*)
type t = Tensor.t * tangent Option.t

exception Not_a_dual_number
exception Wrong_shape of string
exception Check_grad_failed
