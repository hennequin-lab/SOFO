open Forward_torch
include module type of Lqr_typ

val _solve
  :  (Maths.t option, (Maths.t, Maths.t -> Maths.t) momentary_params list) Params.p
  -> Maths.t option Solution.p list

val solve
  :  (Maths.t option, (Maths.t, Maths.t prod) momentary_params list) Params.p
  -> Maths.t Solution.p list
