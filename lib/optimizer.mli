include module type of Optimizer_typ

module SOFO (W : Wrapper.T) :
  T
  with module W = W
   and type ('a, 'b) config = ('a, 'b) Config.SOFO.t
   and type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.SOFO.t -> W.P.t -> 'c

module FGD (W : Wrapper.T) :
  T
  with module W = W
   and type ('a, 'b) config = ('a, 'b) Config.FGD.t
   and type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.FGD.t -> W.P.t -> 'c

module SGD (W : Wrapper.T) :
  T
  with module W = W
   and type ('a, 'b) config = ('a, 'b) Config.SGD.t
   and type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.SGD.t -> W.P.t -> 'c

module Adam (W : Wrapper.T) :
  T
  with module W = W
   and type ('a, 'b) config = ('a, 'b) Config.Adam.t
   and type ('c, _, _) init_opts = W.P.t -> 'c

module SOFO_a2c (W : Wrapper.A2C) :
  A2C
  with module W = W
   and type ('a, 'b) config = ('a, 'b) Config.SOFO_a2c.t
   and type ('c, 'a, 'b) init_opts = config:('a, 'b) Config.SOFO_a2c.t -> W.P.t -> 'c
