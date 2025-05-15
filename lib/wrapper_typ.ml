open Forward_torch
open Torch

(** Forward function. *)
module type Function = sig
  module P : Prms.T

  type input

  (** Forward pass that computes output of network given parameters [theta] and [input]. *)
  val f : theta:P.M.t -> input:input -> Maths.t
end

(** Recurrent function. *)
module type Recurrent_function = sig
  module P : Prms.T

  type input

  (** One-step forward that computes the next state given parameters [theta] and [input]. *)
  val f : theta:P.M.t -> input:input -> Maths.t -> Maths.t
end

(** Basic wrapper type, which is the forward computational graph of the network. *)
module type T = sig
  module P : Prms.T

  (** Type of data to be passed into the network. *)
  type data

  (** Type of any additional arguments. *)
  type args

  (** Given an update function [update], [data], the initial value of the losses and ggn [init] and [args],
   return the final losses and ggn. *)
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
