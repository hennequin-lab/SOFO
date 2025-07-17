open Forward_torch
open Maths

val mse : average_over:int list -> 'a some t -> 'a t
val mse_vtgt_h_prod : average_over:int list -> const t -> vtgt:const t -> const t
val mse_ggn : average_over:int list -> const t -> vtgt:const t -> const t

val cross_entropy
  :  average_over:int list
  -> logit_dim:int
  -> labels:const t
  -> _ some t
  -> any t

val cross_entropy_vtgt_h_prod
  :  average_over:int list
  -> const t
  -> vtgt:const t
  -> const t

val cross_entropy_ggn : average_over:int list -> const t -> vtgt:const t -> const t
