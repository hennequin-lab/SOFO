include module type of Lqr_typ
module Make (O : Ops) : T with type t = O.t
module TensorOps : Ops
module MathsOps : Ops
