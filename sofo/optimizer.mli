open Forward_torch
open Maths

(** Four types of optimizers supported in this library. *)
include module type of Optimizer_typ

(** SOFO optimizer *)
module SOFO (P : Prms.T) : sig
  include
    T
    with module P = P
     and type ('a, 'b) config = ('a, 'b) Config.SOFO.t
     and type ('a, 'b, 'c) init_opts = P.param -> 'c
     and type info = const P.t sofo_info

  (** Initialise parameters with random tangents (also returned), ready to go into forward pass;
      in your optimization loop, just do
      {[
      let theta, tangents = prepare ~config state in
      let loss, ggn = _ (*.... some function of theta using the Maths module for fwd-mode AD *) in
      let state = step ~config ~info:{ loss; ggn; tangents } state in
      (* carry on with the new optimizer state *)
      ]}
     *)
  val prepare : config:(_, _) config -> state -> dual P.t * const P.t
end

(** Stochastic gradient descent *)
module SGDm (P : Prms.T) :
  T
  with module P = P
   and type ('a, 'b) config = ('a, 'b) Config.SGDm.t
   and type ('a, 'b, 'c) init_opts = P.param -> 'c
   and type info = const P.t

(** Adam optimizer *)
module Adam (P : Prms.T) : sig
  include
    T
    with module P = P
     and type ('a, 'b) config = ('a, 'b) Config.Adam.t
     and type (_, _, 'c) init_opts = P.param -> 'c
     and type info = const P.t

  val value_and_grad : f:(const P.t -> any t * 'a) -> state -> float * const P.t * 'a
end
