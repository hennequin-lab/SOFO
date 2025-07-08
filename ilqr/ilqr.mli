open Forward_torch
open Lqr

val _isolve
  :  ?batch_const:bool
  -> gamma:float
  -> f_theta:(x:Maths.any Maths.t -> u:Maths.any Maths.t -> Maths.any Maths.t)
  -> cost_func:
       (batch_const:bool
        -> Maths.any Maths.t option Solution.p list
        -> ( Maths.any Maths.t option
             , ( Maths.any Maths.t
                 , Maths.any Maths.t -> Maths.any Maths.t )
                 momentary_params
                 list )
             Params.p
        -> float)
  -> params_func:
       (Maths.any Maths.t option Solution.p list
        -> ( Maths.any Maths.t option
             , ( Maths.any Maths.t
                 , Maths.any Maths.t -> Maths.any Maths.t )
                 momentary_params
                 list )
             Params.p)
  -> (* -> u_init:Maths.any Maths.t option list *)
     conv_threshold:
       float
       (* -> p_init:(Maths.any Maths.t option, (Maths.any Maths.t, Maths.any Maths.t -> Maths.any Maths.t) momentary_params list) Params.p *)
  -> tau_init:Maths.any Maths.t option Solution.p list
  -> max_iter:int
  -> Maths.any Maths.t option Solution.p list * backward_info list option
