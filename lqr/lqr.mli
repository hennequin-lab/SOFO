open Forward_torch
include module type of Lqr_typ
module Make (O : Ops) : T with type t = O.t
module TensorOps : Ops with type t = Torch.Tensor.t
module MathsOps : Ops with type t = Maths.t
