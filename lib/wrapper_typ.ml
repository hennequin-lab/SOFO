open Forward_torch
open Torch

module type Function = sig
  module P : Prms.T

  type input

  (* forward pass where we compute output of network given parameters theta and input x. *)
  val f : theta:P.t' -> input:input -> Maths.t
end

module type Recurrent_function = sig
  module P : Prms.T

  type input

  val f : theta:P.t' -> input:input -> Maths.t -> Maths.t
end

module type T = sig
  module P : Prms.T

  type data
  type args

  (* given an update function, the data, the initial value of the losses and ggn, return the final losses and ggn. *)
  val f
    :  update:
         [ `loss_only of 'a -> Maths.t option -> 'a
         | `loss_and_ggn of 'a -> (Maths.t * Tensor.t option) option -> 'a
         ]
    -> data:data
    -> init:'a
    -> args:args
    -> P.t'
    -> 'a
end

module type A2C = sig
  module P : Prms.T

  type data
  type args

  (* given an update function, the data, the initial value of the losses and ggn, return the final losses and ggn. *)
  val f
    :  update:
         [ `loss_and_ggn_a2c of
           'a -> (Maths.t * Tensor.t option * Maths.t * Tensor.t option) option -> 'a
         ]
    -> data:data
    -> init:'a
    -> args:args
    -> P.t'
    -> 'a
end
