include module type of Wrapper_typ

module Feedforward (F : Function) (L : Loss.T with type 'a with_args = 'a) :
  T with module P = F.P and type data = F.input * L.labels and type args = unit

module Recurrent (F : Recurrent_function) (L : Loss.T with type 'a with_args = 'a) :
  T
  with module P = F.P
   and type data = (F.input * L.labels option) list
   and type args = Torch.Tensor.t
