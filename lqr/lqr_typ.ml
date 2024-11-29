open Base
open Torch
open Forward_torch

(* conventions is, for each batch element:
   x_(t+1) = x_t A + u_t B *)

type 'a prod =
  { primal : 'a -> 'a
  ; tangent : 'a -> 'a
  }

(* everything has to be optional, because
   perhaps none of those input parameters will have tangents *)
type ('a, 'prod) momentary_params_common =
  { _Fx_prod : 'prod option (* Fx v product *)
  ; _Fx_prod2 : 'prod option (* v Fx product *)
  ; _Fu_prod : 'prod option (* Fu v produt *)
  ; _Fu_prod2 : 'prod option (* v Fu product *)
  ; _Cxx : 'a option
  ; _Cxu : 'a option
  ; _Cuu : 'a option
  }

(* everything has to be optional, because
   perhaps none of those input parameters will have tangents. 
   commpn refers to what is commoro both primal and tangent LQR problems. *)
type ('a, 'prod) momentary_params =
  { common : ('a, 'prod) momentary_params_common
  ; _f : 'a option
  ; _cx : 'a option
  ; _cu : 'a option
  }

module Params = struct
  type ('a, 'p) p =
    { x0 : 'a
    ; params : 'p
    }
  [@@deriving prms]
end

module Solution = struct
  type 'a p =
    { u : 'a
    ; x : 'a
    }
  [@@deriving prms]
end

(*
   module type Ops = sig
   type t

   val print_shape : t -> label:string -> unit
   val zeros : like:t -> shape:int list -> t
   val shape : t -> int list
   val ( + ) : t -> t -> t
   val ( - ) : t -> t -> t
   val ( * ) : t -> t -> t
   val ( / ) : t -> t -> t
   val neg : t -> t
   val btr : t -> t
   val einsum : (t * string) list -> string -> t
   val cholesky : t -> t
   val linsolve_triangular : left:bool -> upper:bool -> t -> t -> t
   val reshape : t -> shape:int list -> t
   end

   module type T = sig
   type t

   val solve : (t, t momentary_params list) Params.p -> t Solution.p list
   end
*)
