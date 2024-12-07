open Forward_torch
include module type of Lqr_typ

val _solve
  :  ?batch_const:bool
  -> (Maths.t option, (Maths.t, Maths.t -> Maths.t) momentary_params list) Params.p
  -> Maths.t Solution.p list

val solve
  :  ?batch_const:bool
  -> (Maths.t option, (Maths.t, Maths.t prod) momentary_params list) Params.p
  -> Maths.t Solution.p list
