open Base
open Forward_torch

(* conventions is, for each batch element:
   x_(t+1) = x_t A + u_t B *)

type 'a prod =
  { primal : 'a -> 'a
  ; tangent : ('a -> 'a) option
  }

(* everything has to be optional, because
   perhaps none of those input parameters will have tangents *)
type ('a, 'prod) momentary_params_common =
  { _Fx_prod : 'prod option (* Av product *)
  ; _Fx_prod2 : 'prod option (* vA product *)
  ; _Fu_prod : 'prod option (* Bv produt *)
  ; _Fu_prod2 : 'prod option (* vB product *)
  ; _Fx_prod_tangent :
      'prod option (* Av product, where the leading dim of v is the tangent dim *)
  ; _Fx_prod2_tangent : 'prod option (* vA product *)
  ; _Fu_prod_tangent : 'prod option (* Bv produt *)
  ; _Fu_prod2_tangent : 'prod option (* vB product *)
  ; _Cxx : 'a option
  ; _Cxu : 'a option
  ; _Cuu : 'a option
  }

(* everything has to be optional, because
   perhaps none of those input parameters will have tangents.
   common refers to what is commoro both primal and tangent LQR problems. *)
type ('a, 'prod) momentary_params =
  { common : ('a, 'prod) momentary_params_common
  ; _f : 'a option
  ; _cx : 'a option
  ; _cu : 'a option
  }

(* params starts at time idx 0 and ends at time index T. Note that at time index T only _Cxx and _cx is used *)
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
