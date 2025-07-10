open Forward_torch
open Lqr

val _isolve
  :  ?batch_const:bool
  -> gamma:float
  -> f_theta:(x:Maths.t -> u:Maths.t -> Maths.t)
  -> cost_func:
       (batch_const:bool
        -> Maths.t option Solution.p list
        -> (Maths.t option, (Maths.t, Maths.t -> Maths.t) momentary_params list) Params.p
        -> float)
  -> params_func:
       (Maths.t option Solution.p list
        -> (Maths.t option, (Maths.t, Maths.t -> Maths.t) momentary_params list) Params.p)
  -> 
     conv_threshold:
       float
  -> tau_init:Maths.t option Solution.p list
  -> int
  -> Maths.t option Solution.p list * backward_info list option
