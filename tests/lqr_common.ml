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

let prod ~batch_const f : Maths.t Lqr.prod =
  let primal = bmm ~batch_const (Maths.const (Maths.primal f)) in
  (* tangent on f only *)
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (bmm_tangent_F ~batch_const (Maths.const df))
  in
  { primal; tangent }

let prod_tangent ~batch_const f : Maths.t Lqr.prod =
  (* tangent on v only *)
  let primal = bmm_tangent_v ~batch_const (Maths.const (Maths.primal f)) in
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (bmm_tangent_Fv ~batch_const (Maths.const df))
  in
  { primal; tangent }

let prod2 ~batch_const f : Maths.t Lqr.prod =
  let primal = bmm2 ~batch_const (Maths.const (Maths.primal f)) in
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (bmm2_tangent_F ~batch_const (Maths.const df))
  in
  { primal; tangent }

let prod2_tangent ~batch_const f : Maths.t Lqr.prod =
  let primal = bmm2_tangent_v ~batch_const (Maths.const (Maths.primal f)) in
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (bmm2_tangent_Fv ~batch_const (Maths.const df))
  in
  { primal; tangent }

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
  Lqr._solve ~batch_const (map_naive ~batch_const x)

let f_implicit ~batch_const (x : Input.M.t) : Output.M.t =
  Lqr.solve ~batch_const (map_implicit ~batch_const x)
