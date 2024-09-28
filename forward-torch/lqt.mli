open Torch
include module type of Lqt_type

val lqt
  :  state_params:state_params
  -> x_u_desired:x_u_desired
  -> cost_params:cost_params
  -> Maths.t list * Maths.t list

val lqt_tensor
  :  state_params:state_params_tensor
  -> x_u_desired:x_u_desired_tensor
  -> cost_params:cost_params_tensor
  -> Tensor.t list * Tensor.t list
