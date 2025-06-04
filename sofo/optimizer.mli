open Forward_torch

(** Four types of optimizers supported in this library. *)
include module type of Optimizer_typ

(** SOFO optimizer *)
module SOFO (P : Prms.T) : sig
  include
    T
    with module P = P
     and type ('a, 'b) config = ('a, 'b) Config.SOFO.t
     and type ('a, 'b, 'c) init_opts = P.param -> 'c
     and type info = [ `const ] P.t sofo_info

  (* initialise parameters with random tangents, ready to go into forward pass *)
  val prepare : config:(_, _) config -> state -> [ `dual ] P.t
end

(** Stochastic gradient descent *)
module SGDm (P : Prms.T) :
  T
  with module P = P
   and type ('a, 'b) config = ('a, 'b) Config.SGDm.t
   and type ('a, 'b, 'c) init_opts = P.param -> 'c
   and type info = [ `const ] P.t

(** Adam optimizer *)
module Adam (P : Prms.T) :
  T
  with module P = P
   and type ('a, 'b) config = ('a, 'b) Config.Adam.t
   and type (_, _, 'c) init_opts = P.param -> 'c
   and type info = [ `const ] P.t
