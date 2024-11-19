open Torch
include module type of Lqr_typ

val lqr
  :  state_params:Maths.t state_params
  -> cost_params:Maths.t cost_params
  -> Maths.t list * Maths.t list

val lqr_tensor
  :  state_params:Tensor.t state_params
  -> cost_params:Tensor.t cost_params
  -> Tensor.t list * Tensor.t list

val lqr_sep
  :  state_params:Maths.t state_params
  -> cost_params:Maths.t cost_params
  -> Maths.t list * Maths.t list
