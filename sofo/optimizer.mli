(** Four types of optimizers supported in this library. *)
include module type of Optimizer_typ

(** SOFO optimizer *)
module SOFO (W : Wrapper.T) :
  T
  with module W = W
   and type ('a, 'b) config = ('a, 'b) Config.SOFO.t
   and type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.SOFO.t -> W.P.tagged -> 'c

(** Forward Gradient Descent
  @see <https://arxiv.org/abs/2202.08587> original paper. *)
module FGD (W : Wrapper.T) :
  T
  with module W = W
   and type ('a, 'b) config = ('a, 'b) Config.FGD.t
   and type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.FGD.t -> W.P.tagged -> 'c

(** Stochastic gradient descent *)
module SGD (W : Wrapper.T) :
  T
  with module W = W
   and type ('a, 'b) config = ('a, 'b) Config.SGD.t
   and type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.SGD.t -> W.P.tagged -> 'c

(** Adam optimizer *)
module Adam (W : Wrapper.T) :
  T
  with module W = W
   and type ('a, 'b) config = ('a, 'b) Config.Adam.t
   and type ('c, _, _) init_opts = W.P.tagged -> 'c
