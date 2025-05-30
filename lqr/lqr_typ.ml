(** Types used in linear quadratic regulator (LQR) function. conventions is, for each batch element:
   x_(t+1) = x_t A + u_t B + f_t. *)
open Base
open Forward_torch

(** Type of all elements in LQR pass. This can be lazily evaluated. *)
type 'a prod =
  { primal : 'a -> 'a
  ; tangent : ('a -> 'a) option
  }

(** Everything has to be optional, because
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

(** common refers to what is commoro both primal and tangent LQR problems. *)
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

(** In Solution u indexes from 0 to T-1 while x indexes from 1 to T. *)
module Solution = struct
  type 'a p =
    { u : 'a
    ; x : 'a
    }
  [@@deriving prms]
end

open Maths

type backward_common_info =
  { _Quu_chol : t option
  ; _Quu_chol_T : t option
  ; _V : t option
  ; _K : t option
  }

type backward_info =
  { _K : t option
  ; _k : t option
  }

type backward_info_f =
  { _K : t option
  ; _k : t option
  ; _f : t option
  }

type backward_surrogate =
  { x : t option
  ; u : t option
  ; params : (t, t prod) momentary_params
  }
