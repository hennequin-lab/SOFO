(** Two common wrappers. *)
include module type of Wrapper_typ

(** Feedforward network where F is the forward function and L is the loss module. *)
module Feedforward (F : Function) (L : Loss.T with type 'a with_args = 'a) :
  T with module P = F.P and type data = F.input * L.labels and type args = unit

(** Recurrent network where F is the one-step recurrent function and L is the loss module. *)
module Recurrent (F : Recurrent_function) (L : Loss.T with type 'a with_args = 'a) :
  T
  with module P = F.P
   and type data = (F.input * L.labels option) list
   and type args = Torch.Tensor.t
