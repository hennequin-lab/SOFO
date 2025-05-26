open Base
open Torch

type primal_t = Primal
type dual_t = Dual

(* GADT that will enforce type safety of tangent manipulations *)
type (_, _) t =
  | Primal : Tensor.t -> (primal_t, _) t
  | Dual : Tensor.t * Tensor.t -> (dual_t, Tensor.t) t
  | Dual_lazy : Tensor.t * (unit -> Tensor.t) -> (dual_t, unit -> Tensor.t) t

exception Wrong_shape of int list * int list
exception Wrong_device of Device.t * Device.t
exception Check_grad_failed
