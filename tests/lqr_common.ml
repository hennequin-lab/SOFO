open Base
open Torch
open Forward_torch
module Arr = Owl.Arr
module Mat = Owl.Mat
module Linalg = Owl.Linalg.D

let print s = Stdio.print_endline (Sexp.to_string_hum s)
let device = Torch.Device.Cpu
let kind = Torch_core.Kind.(T f64)
let to_device = Tensor.of_bigarray ~device

module Temp = struct
  type ('a, 'o) p =
    { _f : 'o
    ; _Fx_prod : 'a
    ; _Fu_prod : 'a
    ; _cx : 'o
    ; _cu : 'o
    ; _Cxx : 'a
    ; _Cxu : 'o
    ; _Cuu : 'a
    }
  [@@deriving prms]
end

module O = Prms.Option (Prms.P)
module Input = Lqr.Params.Make (O) (Prms.List (Temp.Make (Prms.P) (O)))
module Output = Prms.List (Lqr.Solution.Make (Prms.P))

(* -----------------------------------------
   ---- Utility Functions   ------
   ----------------------------------------- *)

(* Fv prod; if batch_const, F does not have a leading batch dimension. *)
let bmm ~batch_const a b =
  let open Maths in
  if batch_const
  then (
    match List.length (shape b) with
    | 3 -> einsum [ a, "ab"; b, "mbc" ] "mac"
    | 2 -> einsum [ a, "ab"; b, "mb" ] "ma"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 3 -> einsum [ a, "mab"; b, "mbc" ] "mac"
    | 2 -> einsum [ a, "mab"; b, "mb" ] "ma"
    | _ -> failwith "not batch multipliable")

(* vF prod *)
let bmm2 ~batch_const b a =
  let open Maths in
  if batch_const
  then (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b, "ab" ] "mb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b, "mab" ] "mb"
    | _ -> failwith "should not happen")

(* F^-1 v prod; if batch_const, F does not have a leading batch dimension. *)
let bmm_inv ~batch_const a b =
  let open Maths in
  let a_inv = inv_rectangle a in
  if batch_const
  then (
    match List.length (shape b) with
    | 3 -> einsum [ a_inv, "ab"; b, "mbc" ] "mac"
    | 2 -> einsum [ a_inv, "ab"; b, "mb" ] "ma"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 3 -> einsum [ a_inv, "mab"; b, "mbc" ] "mac"
    | 2 -> einsum [ a_inv, "mab"; b, "mb" ] "ma"
    | _ -> failwith "not batch multipliable")

(* vF^-1 prod *)
let bmm2_inv ~batch_const b a =
  let open Maths in
  let b_inv = inv_rectangle b in
  if batch_const
  then (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b_inv, "ab" ] "mb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b_inv, "mab" ] "mb"
    | _ -> failwith "should not happen")

(* F^T v prod; if batch_const, F does not have a leading batch dimension. *)
let bmm_trans ~batch_const a b =
  let open Maths in
  if batch_const
  then (
    match List.length (shape b) with
    | 3 -> einsum [ a, "ba"; b, "mbc" ] "mac"
    | 2 -> einsum [ a, "ba"; b, "mb" ] "ma"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 3 -> einsum [ a, "mba"; b, "mbc" ] "mac"
    | 2 -> einsum [ a, "mba"; b, "mb" ] "ma"
    | _ -> failwith "not batch multipliable")

(* vF^T prod *)
let bmm2_trans ~batch_const b a =
  let open Maths in
  if batch_const
  then (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b, "ba" ] "mb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b, "mba" ] "mb"
    | _ -> failwith "should not happen")

(* F^-T v prod; if batch_const, F does not have a leading batch dimension. *)
let bmm_inv_trans ~batch_const a b =
  let open Maths in
  let a_inv = inv_rectangle a in
  if batch_const
  then (
    match List.length (shape b) with
    | 3 -> einsum [ a_inv, "ba"; b, "mbc" ] "mac"
    | 2 -> einsum [ a_inv, "ba"; b, "mb" ] "ma"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 3 -> einsum [ a_inv, "mba"; b, "mbc" ] "mac"
    | 2 -> einsum [ a_inv, "mba"; b, "mb" ] "ma"
    | _ -> failwith "not batch multipliable")

(* vF^-T prod *)
let bmm2_inv_trans ~batch_const b a =
  let open Maths in
  let b_inv = inv_rectangle b in
  if batch_const
  then (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b_inv, "ba" ] "mb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b_inv, "mba" ] "mb"
    | _ -> failwith "should not happen")

(* Fv prod, tangent on v *)
let bmm_tangent_v ~batch_const a b =
  let open Maths in
  if batch_const
  then (
    match List.length (shape b) with
    | 4 -> einsum [ a, "ab"; b, "kmbc" ] "kmac"
    | 3 -> einsum [ a, "ab"; b, "kmb" ] "kma"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 4 -> einsum [ a, "mab"; b, "kmbc" ] "kmac"
    | 3 -> einsum [ a, "mab"; b, "kmb" ] "kma"
    | _ -> failwith "not batch multipliable")

(* Fv prod, tangent on F *)
let bmm_tangent_F ~batch_const a b =
  let open Maths in
  if batch_const
  then (
    match List.length (shape b) with
    | 3 -> einsum [ a, "kab"; b, "mbc" ] "kmac"
    | 2 -> einsum [ a, "kab"; b, "mb" ] "kma"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 3 -> einsum [ a, "kmab"; b, "mbc" ] "kmac"
    | 2 -> einsum [ a, "kmab"; b, "mb" ] "kma"
    | _ -> failwith "not batch multipliable")

(* Fv prod, tangent on F and v *)
let bmm_tangent_Fv ~batch_const a b =
  let open Maths in
  if batch_const
  then (
    match List.length (shape b) with
    | 4 -> einsum [ a, "kab"; b, "kmbc" ] "kmac"
    | 3 -> einsum [ a, "kab"; b, "kmb" ] "kma"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 4 -> einsum [ a, "kmab"; b, "kmbc" ] "kmac"
    | 3 -> einsum [ a, "kmab"; b, "kmb" ] "kma"
    | _ -> failwith "not batch multipliable")

(* vF prod, tangent on v *)
let bmm2_tangent_v ~batch_const b a =
  let open Maths in
  if batch_const
  then (
    match List.length (shape a) with
    | 3 -> einsum [ a, "kma"; b, "ab" ] "kmb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 3 -> einsum [ a, "kma"; b, "mab" ] "kmb"
    | _ -> failwith "should not happen")

(* vF prod, tangent on F *)
let bmm2_tangent_F ~batch_const b a =
  let open Maths in
  if batch_const
  then (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b, "kab" ] "kmb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b, "kmab" ] "kmb"
    | _ -> failwith "should not happen")

(* vF prod, tangent on F and v *)
let bmm2_tangent_Fv ~batch_const b a =
  let open Maths in
  if batch_const
  then (
    match List.length (shape a) with
    | 3 -> einsum [ a, "kma"; b, "kab" ] "kmb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 3 -> einsum [ a, "kma"; b, "kmab" ] "kmb"
    | _ -> failwith "should not happen")

(* F^-1v prod, tangent on F *)
let bmm_inv_tangent_F ~batch_const a b =
  let open Maths in
  let a_inv = inv_rectangle a in
  if batch_const
  then (
    match List.length (shape b) with
    | 3 -> einsum [ a_inv, "kab"; b, "mbc" ] "kmac"
    | 2 -> einsum [ a_inv, "kab"; b, "mb" ] "kma"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 3 -> einsum [ a_inv, "kmab"; b, "mbc" ] "kmac"
    | 2 -> einsum [ a_inv, "kmab"; b, "mb" ] "kma"
    | _ -> failwith "not batch multipliable")

(* vF^-1 prod, tangent on F *)
let bmm2_inv_tangent_F ~batch_const a b =
  let open Maths in
  let b_inv = inv_rectangle b in
  if batch_const
  then (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b_inv, "kab" ] "mb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b_inv, "mkab" ] "mb"
    | _ -> failwith "should not happen")

(* F^T v prod, tangent on F *)
let bmm_trans_tangent_F ~batch_const a b =
  let open Maths in
  if batch_const
  then (
    match List.length (shape b) with
    | 3 -> einsum [ a, "kba"; b, "mbc" ] "mkac"
    | 2 -> einsum [ a, "kba"; b, "mb" ] "mka"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 3 -> einsum [ a, "mkba"; b, "mbc" ] "mkac"
    | 2 -> einsum [ a, "mkba"; b, "mb" ] "mka"
    | _ -> failwith "not batch multipliable")

(* vF^T prod, tangent on F *)
let bmm2_trans_tangent_F ~batch_const b a =
  let open Maths in
  if batch_const
  then (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b, "kba" ] "mkb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b, "mkba" ] "mkb"
    | _ -> failwith "should not happen")

(* F^-T v prod, tangent on F. *)
let bmm_inv_trans_tangent_F ~batch_const a b =
  let open Maths in
  let a_inv = inv_rectangle a in
  if batch_const
  then (
    match List.length (shape b) with
    | 3 -> einsum [ a_inv, "kba"; b, "mbc" ] "mac"
    | 2 -> einsum [ a_inv, "kba"; b, "mb" ] "ma"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 3 -> einsum [ a_inv, "mkba"; b, "mbc" ] "mac"
    | 2 -> einsum [ a_inv, "mkba"; b, "mb" ] "ma"
    | _ -> failwith "not batch multipliable")

(* vF^-T prod, tangent on F.*)
let bmm2_inv_trans_tangent_F ~batch_const b a =
  let open Maths in
  let b_inv = inv_rectangle b in
  if batch_const
  then (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b_inv, "kba" ] "mb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b_inv, "mkba" ] "mb"
    | _ -> failwith "should not happen")

let prod_f ~batch_const ~primal_f ~tan_f f : Maths.t Lqr.prod =
  let primal = primal_f ~batch_const (Maths.const (Maths.primal f)) in
  (* tangent on f only *)
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (tan_f ~batch_const (Maths.const df))
  in
  { primal; tangent }

let prod ~batch_const f = prod_f ~batch_const ~primal_f:bmm ~tan_f:bmm_tangent_F f
let prod2 ~batch_const f = prod_f ~batch_const ~primal_f:bmm2 ~tan_f:bmm2_tangent_F f

let prod_inv ~batch_const f =
  prod_f ~batch_const ~primal_f:bmm_inv ~tan_f:bmm_inv_tangent_F f

let prod2_inv ~batch_const f =
  prod_f ~batch_const ~primal_f:bmm2_inv ~tan_f:bmm2_inv_tangent_F f

let prod_trans ~batch_const f =
  prod_f ~batch_const ~primal_f:bmm_trans ~tan_f:bmm_trans_tangent_F f

let prod2_trans ~batch_const f =
  prod_f ~batch_const ~primal_f:bmm2_trans ~tan_f:bmm2_trans_tangent_F f

let prod_inv_trans ~batch_const f =
  prod_f ~batch_const ~primal_f:bmm_inv_trans ~tan_f:bmm_inv_trans_tangent_F f

let prod2_inv_trans ~batch_const f =
  prod_f ~batch_const ~primal_f:bmm2_inv_trans ~tan_f:bmm2_inv_trans_tangent_F f

let prod_tangent ~batch_const f =
  prod_f ~batch_const ~primal_f:bmm_tangent_v ~tan_f:bmm_tangent_Fv f

let prod2_tangent ~batch_const f =
  prod_f ~batch_const ~primal_f:bmm2_tangent_v ~tan_f:bmm2_tangent_Fv f

let tmax = 20
let m = 16
let a_dim = 24
let b_dim = 10

(* -----------------------------------------
   ---- Sample Tensors   ------
   ----------------------------------------- *)

let sample_stable () =
  let a = Mat.gaussian a_dim a_dim in
  let r =
    a |> Linalg.eigvals |> Owl.Dense.Matrix.Z.abs |> Owl.Dense.Matrix.Z.re |> Mat.max'
  in
  Mat.(Float.(0.8 / r) $* a)

let sample_fx_pri ~batch_const =
  let fx_pri =
    if batch_const
    then sample_stable ()
    else
      Array.init m ~f:(fun _ ->
        let a = sample_stable () in
        Arr.reshape a [| 1; a_dim; a_dim |])
      |> Arr.concatenate ~axis:0
  in
  to_device fx_pri

let sample_primal shape = Tensor.randn ~device ~kind shape

let sample_fu ~batch_const =
  if batch_const
  then sample_primal [ b_dim; a_dim ]
  else sample_primal [ m; b_dim; a_dim ]

(* make sure cost matrices are positive definite *)
let pos_sym ~reg d =
  let ell = Mat.gaussian d d in
  Mat.(add_diag (ell *@ transpose ell) reg)

let q_of ~batch_const ~reg d =
  let q =
    if batch_const
    then pos_sym ~reg d
    else
      Array.init m ~f:(fun _ -> Arr.reshape (pos_sym ~reg d) [| 1; d; d |])
      |> Arr.concatenate ~axis:0
  in
  to_device q

let sample_q_xx ~batch_const = q_of ~batch_const ~reg:10. a_dim
let sample_q_uu ~batch_const = q_of ~batch_const ~reg:10. b_dim
let sample_f = sample_primal [ m; a_dim ]
let sample_c_x = sample_primal [ m; a_dim ]
let sample_c_u = sample_primal [ m; b_dim ]

let sample_q_xu ~batch_const =
  if batch_const
  then sample_primal [ a_dim; b_dim ]
  else sample_primal [ m; a_dim; b_dim ]

let map_naive ~batch_const (x : Input.M.t) =
  let irrelevant = Some (fun _ -> assert false) in
  let params =
    List.map x.params ~f:(fun p ->
      Lqr.
        { common =
            { _Fx_prod = Some (bmm ~batch_const p._Fx_prod)
            ; _Fx_prod2 = Some (bmm2 ~batch_const p._Fx_prod)
            ; _Fu_prod = Some (bmm ~batch_const p._Fu_prod)
            ; _Fu_prod2 = Some (bmm2 ~batch_const p._Fu_prod)
            ; _Fx_prod_tangent = irrelevant
            ; _Fx_prod2_tangent = irrelevant
            ; _Fu_prod_tangent = irrelevant
            ; _Fu_prod2_tangent = irrelevant
            ; _Fx_prod_inv = Some (bmm_inv ~batch_const p._Fx_prod)
            ; _Fx_prod2_inv = Some (bmm2_inv ~batch_const p._Fx_prod)
            ; _Fu_prod_trans = Some (bmm_trans ~batch_const p._Fu_prod)
            ; _Fu_prod2_trans = Some (bmm2_trans ~batch_const p._Fu_prod)
            ; _Fx_prod_inv_trans = Some (bmm_inv_trans ~batch_const p._Fx_prod)
            ; _Fx_prod2_inv_trans =
                Some (bmm2_inv_trans ~batch_const p._Fx_prod)
                (*  A A^T to make sure tangents are positive definite; not needed in actual lqr. *)
            ; _Cxx = Some Maths.(p._Cxx *@ btr p._Cxx)
            ; _Cxu = p._Cxu
            ; _Cuu = Some Maths.(p._Cuu *@ btr p._Cuu)
            }
        ; _f = p._f
        ; _cx = p._cx
        ; _cu = p._cu
        })
  in
  Lqr.Params.{ x with params }

let map_implicit ~batch_const (x : Input.M.t) =
  let params =
    List.map x.params ~f:(fun p ->
      Lqr.
        { common =
            { _Fx_prod = Some (prod ~batch_const p._Fx_prod)
            ; _Fx_prod2 = Some (prod2 ~batch_const p._Fx_prod)
            ; _Fu_prod = Some (prod ~batch_const p._Fu_prod)
            ; _Fu_prod2 = Some (prod2 ~batch_const p._Fu_prod)
            ; _Fx_prod_tangent = Some (prod_tangent ~batch_const p._Fx_prod)
            ; _Fx_prod2_tangent = Some (prod2_tangent ~batch_const p._Fx_prod)
            ; _Fu_prod_tangent = Some (prod_tangent ~batch_const p._Fu_prod)
            ; _Fu_prod2_tangent = Some (prod2_tangent ~batch_const p._Fu_prod)
            ; _Fx_prod_inv = Some (prod_inv ~batch_const p._Fx_prod)
            ; _Fx_prod2_inv = Some (prod2_inv ~batch_const p._Fx_prod)
            ; _Fu_prod_trans = Some (prod_trans ~batch_const p._Fu_prod)
            ; _Fu_prod2_trans = Some (prod2_trans ~batch_const p._Fu_prod)
            ; _Fx_prod_inv_trans = Some (prod_inv_trans ~batch_const p._Fx_prod)
            ; _Fx_prod2_inv_trans = Some (prod2_inv_trans ~batch_const p._Fx_prod)
            ; _Cxx = Some Maths.(p._Cxx *@ btr p._Cxx)
            ; _Cxu = p._Cxu
            ; _Cuu = Some Maths.(p._Cuu *@ btr p._Cuu)
            }
        ; _f = p._f
        ; _cx = p._cx
        ; _cu = p._cu
        })
  in
  Lqr.Params.{ x with params }

let f_naive ~batch_const (x : Input.M.t) : Output.M.t =
  let sol, _ = Lqr._solve ~batch_const (map_naive ~batch_const x) in
  sol

let f_implicit ~batch_const (x : Input.M.t) : Output.M.t =
  let sol, _ = Lqr.solve ~batch_const (map_implicit ~batch_const x) in
  sol
