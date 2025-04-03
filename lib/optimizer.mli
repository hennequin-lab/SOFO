include module type of Optimizer_typ

module SOFO (W : Wrapper.T) (A : Wrapper.Auxiliary  with module P = W.P) :
  T
  with module W = W
   and type ('a, 'b) config = ('a, 'b) Config.SOFO.t
   and type ('c, _, _) init_opts = W.P.tagged -> 'c

module FGD (W : Wrapper.T) :
  T
  with module W = W
   and type ('a, 'b) config = ('a, 'b) Config.FGD.t
   and type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.FGD.t -> W.P.tagged -> 'c

module SGD (W : Wrapper.T) :
  T
  with module W = W
   and type ('a, 'b) config = ('a, 'b) Config.SGD.t
   and type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.SGD.t -> W.P.tagged -> 'c

module Adam (W : Wrapper.T) :
  T
  with module W = W
   and type ('a, 'b) config = ('a, 'b) Config.Adam.t
   and type ('c, _, _) init_opts = W.P.tagged -> 'c
