open Torch
include module type of Lqr_type

val lqr
  :  state_params:state_params
  -> cost_params:cost_params
  -> Maths.t list * Maths.t list

val lqr_tensor
  :  state_params:state_params_tensor
  -> cost_params:cost_params_tensor
  -> Tensor.t list * Tensor.t list

val lqr_sep
  :  state_params:state_params
  -> cost_params:cost_params
  -> Maths.t list * Maths.t list
