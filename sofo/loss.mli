open Forward_torch
open Maths

val mse : average_over:int list -> 'a some t -> 'a t
val mse_hv_prod : average_over:int list -> const t -> v:const t -> const t

val cross_entropy
  :  average_over:int list
  -> logit_dim:int
  -> labels:const t
  -> _ some t
  -> any t

val
  [@deprecated] cross_entropy_hv_prod
  :  average_over:int list
  -> logit_dim:int
  -> const t
  -> v:const t
  -> const t
