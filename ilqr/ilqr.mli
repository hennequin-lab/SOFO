open Forward_torch
open Lqr

val _isolve
  :  ?batch_const:bool
  -> ?laplace:bool
  -> f_theta:(x:Maths.t -> u:Maths.t -> Maths.t)
  -> cost_func:
       (batch_const:bool
        -> Maths.t option Solution.p list
        -> (Maths.t option, (Maths.t, Maths.t -> Maths.t) momentary_params list) Params.p
        -> float)
  -> params_func:
       (Maths.t option Solution.p list
        -> (Maths.t option, (Maths.t, Maths.t -> Maths.t) momentary_params list) Params.p)
  -> (* -> u_init:Maths.t option list *)
     conv_threshold:
       float
       (* -> p_init:(Maths.t option, (Maths.t, Maths.t -> Maths.t) momentary_params list) Params.p *)
  -> tau_init:Maths.t option Solution.p list
  -> max_iter:int
  -> Maths.t option Solution.p list * Maths.t list option
