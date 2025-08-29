open Forward_torch
open Torch
open Maths

(* For auxiliary loss when learning the ggn *)
module type Auxiliary = sig
  module P : Prms.T
  module A : Prms.T

  (* initialise sampling state of tangents *)
  val init_sampling_state : unit -> int

  (* given aux parameters [lambda] and tangents v, compute \hat{G}^{1/2}v where \hat{G} is the approximated ggn *)
  val g12v : lambda:([< `const | `dual ] as 'a) A.t -> 'a P.t -> any P.t

  (* draw localised tangents randomly. Do not need tangent parameters now since tangents for each layer distributed 
    upstream. used in the learning phase of learning-the-ggn idea. *)
  val random_localised_vs : unit -> const t P.p

  (* given aux parameters [lambda], whether to switch to learning, current sampling state and the number of tangents, 
    return eigenvectors of \hat{G} and new sampling state *)
  val eigenvectors
    :  lambda:_ some A.t
    -> switch_to_learn:bool
    -> int
    -> int
    -> const t P.p * int

  (* initialise aux parameters *)
  val init : unit -> A.param
end
