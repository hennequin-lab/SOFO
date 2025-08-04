open Forward_torch
open Lqr

(* if [linesearch], backtrack alpha iteratively; if [ex_reduction], use dC 
  in linesearch criterion. *)
val _isolve
  :  ?linesearch:bool
  -> ?ex_reduction:bool
  -> ?batch_const:bool
  -> gamma:float
  -> f_theta:(i:int -> x:Maths.any Maths.t -> u:Maths.any Maths.t -> Maths.any Maths.t)
  -> cost_func:(Maths.any Maths.t option Solution.p list -> float)
  -> params_func:
       (Maths.any Maths.t option Solution.p list
        -> ( Maths.any Maths.t option
             , ( Maths.any Maths.t
                 , Maths.any Maths.t -> Maths.any Maths.t )
                 momentary_params
                 list )
             Params.p)
  -> conv_threshold:float
  -> tau_init:Maths.any Maths.t option Solution.p list
  -> int
  -> Maths.any Maths.t option Solution.p list * backward_info list option
