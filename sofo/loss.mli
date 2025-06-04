open Forward_torch
open Maths

val mse : dim:int list -> ([< `const | `dual ] as 'a) t -> 'a t
val mse_hv_prod : dim:int list -> [ `const ] t -> [ `const ] t

val cross_entropy
  :  dim:int list
  -> labels:[ `const ] t
  -> [< `const | `dual ] t
  -> [ `const | `dual ] t

val cross_entropy_hv_prod : dim:int list -> [ `const ] t -> v:[ `const ] t -> [ `const ] t
