open Torch
include module type of Lqr_type

val lqr
  :  state_params:state_params
  -> cost_params:cost_params
  -> Maths.t list * Maths.t list
