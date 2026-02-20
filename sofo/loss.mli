open Forward_torch
open Maths

val mse : output_dims:int list -> t -> t

(* val mse_vtgt_h_prod : average_over:int list -> const t -> vtgt:const t -> const t *)
val mse_ggn : output_dims:int list -> t -> vtgt:t -> t
val cross_entropy : output_dims:int list -> labels:t -> t -> t
val cross_entropy_ggn : output_dims:int list -> t -> vtgt:t -> t
