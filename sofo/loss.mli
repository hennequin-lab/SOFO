open Forward_torch
open Maths

val mse : output_dims:int list -> 'a some t -> 'a t

(* val mse_vtgt_h_prod : average_over:int list -> const t -> vtgt:const t -> const t *)
val mse_ggn : output_dims:int list -> const t -> vtgt:const t -> const t
val cross_entropy : output_dims:int list -> labels:const t -> _ some t -> any t
val cross_entropy_ggn : output_dims:int list -> const t -> vtgt:const t -> const t
