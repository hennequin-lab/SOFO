open Forward_torch
open Torch

module type Function = sig
  module P : Prms.T

  type input

  (* forward pass where we compute output of network given parameters [theta] and [input]. *)
  val f : theta:P.M.t -> input:input -> Maths.t
end

module type Recurrent_function = sig
  module P : Prms.T

  type input

  (* recurrent step where we compute new state given parameters [theta], [input] and current state*)
  val f : theta:P.M.t -> input:input -> Maths.t -> Maths.t
end

module type T = sig
  module P : Prms.T

  type data
  type args

  (* given an [update] function, the [data], the initial value of the losses and ggn [init] and [args], return the final losses and ggn. *)
  val f
    :  update:
         [ `loss_only of 'a -> Maths.t option -> 'a
         | `loss_and_ggn of 'a -> (Maths.t * Tensor.t option) option -> 'a
         ]
    -> data:data
    -> init:'a
    -> args:args
    -> P.M.t
    -> 'a
end

module type Auxiliary = sig
  module P : Prms.T
  module A : Prms.T

  type sampling_state

  (* initialise sampling state of tangents *)
  val init_sampling_state : unit -> sampling_state

  (* given aux parameters [lambda] and tangents v, compute \hat{G}^{1/2}v where \hat{G} is the approximated ggn *)
  val g12v : lambda:A.M.t -> P.M.t -> P.M.t

  (* given the number of tangents, draw localised tangents randomly *)
  val random_localised_vs : int -> P.T.t

  (* given aux parameters [lambda], sampling state and the number of tangetns, return eigenvectors of \hat{G} and new sampling state *)
  val eigenvectors : lambda:A.M.t -> sampling_state -> int -> P.T.t * sampling_state

  (* initialise aux parameters *)
  val init : unit -> A.tagged
end
