open Forward_torch
include module type of Lqr_typ

val backward_common
  :  laplace:bool
  -> batch_const:bool
  -> (Maths.t, Maths.t -> Maths.t) momentary_params_common list
  -> backward_common_info list

val backward
  :  ?tangent:bool
  -> batch_const:bool
  -> backward_common_info list
  -> (Maths.t option, (Maths.t, Maths.t -> Maths.t) momentary_params list) Params.p
  -> backward_info list


val covariances
  :  ?batch_const:bool
  -> common_info:backward_common_info list
  -> (Maths.t option, (Maths.t, Maths.t -> Maths.t) momentary_params list) Params.p
  -> Maths.t option list

val _solve
  :  ?batch_const:bool
  -> ?laplace:bool
  -> (Maths.t option, (Maths.t, Maths.t -> Maths.t) momentary_params list) Params.p
  -> Maths.t Solution.p list * Maths.t list option

val solve
  :  ?batch_const:bool
  -> ?laplace:bool
  -> (Maths.t option, (Maths.t, Maths.t prod) momentary_params list) Params.p
  -> Maths.t Solution.p list * Maths.t list option
