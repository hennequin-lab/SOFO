open Forward_torch
include module type of Lqr_typ

(* given momentary_params_common, calculate backward_common_info. batch_const refers to when
  all parameters are the same in the batch *)
val backward_common
  :  batch_const:bool
  -> (Maths.any Maths.t, Maths.any Maths.t -> Maths.any Maths.t) momentary_params_common
       list
  -> backward_common_info list

(* given backward_common_info and momentary_params, calculate backward_info *)
val backward
  :  ?tangent:bool
  -> batch_const:bool
  -> backward_common_info list
  -> ( Maths.any Maths.t option
       , (Maths.any Maths.t, Maths.any Maths.t -> Maths.any Maths.t) momentary_params list
       )
       Params.p
  -> backward_info list

(* given momentary_params, run LQR and return solutions and backward_info; naively differentiate *)
val _solve
  :  ?batch_const:bool
  -> ( Maths.any Maths.t option
       , (Maths.any Maths.t, Maths.any Maths.t -> Maths.any Maths.t) momentary_params list
       )
       Params.p
  -> Maths.any Maths.t Solution.p list * backward_info list

(* given momentary_params, run LQR and return solutions and backward_info; implicit differentiate *)
val solve
  :  ?batch_const:bool
  -> ( Maths.any Maths.t option
       , (Maths.any Maths.t, Maths.any Maths.t prod) momentary_params list )
       Params.p
  -> Maths.any Maths.t Solution.p list
