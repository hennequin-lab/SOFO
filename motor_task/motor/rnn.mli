open Base
open Torch
open Forward_torch
open Sofo
include module type of Rnn_typ

module Make (_ : sig
    val dt : float
    val tau : float
    val n : int
    val internal_noise_std : float option
  end) : sig
  module P : Prms.T with type 'a p = 'a PP.p

  val init : base:('a, 'b) Optimizer.Config.Base.t -> n_input_channels:int -> P.tagged

  val step_forward
    :  ?noise:Tensor.t
    -> prms:P.M.t
    -> Tensor.t
    -> Maths.t * Maths.t Arm.state
    -> Maths.t * Maths.t Arm.state * (Maths.t * Maths.t)

  val draw_noise : device:Torch.Device.t -> t_max:int -> bs:int -> Tensor.t option
  val noise_slice : Tensor.t option -> int -> Tensor.t option

  val forward
    :  prms:P.M.t
    -> t_max:int
    -> (int -> Tensor.t)
    -> Maths.t result list
end
