open Base
open Torch
open Forward_torch
module Arr = Owl.Arr
module Mat = Owl.Mat
module Linalg = Owl.Linalg.D

let print s = Stdio.print_endline (Sexp.to_string_hum s)
let device = Torch.Device.Cpu
let kind = Torch_core.Kind.(T f64)

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

let bmm a b =
  let open Maths in
  match List.length (shape b) with
  | 3 -> einsum [ a, "mab"; b, "mbc" ] "mac"
  | 2 -> einsum [ a, "mab"; b, "mb" ] "ma"
  | _ -> failwith "not batch multipliable"

let bmm2 b a =
  let open Maths in
  match List.length (shape a) with
  | 2 -> einsum [ a, "ma"; b, "mab" ] "mb"
  | _ -> failwith "should not happen"

(* Fv prod, tangent on v *)
let bmm_tangent_v a b =
  let open Maths in
  match List.length (shape b) with
  | 4 -> einsum [ a, "mab"; b, "kmbc" ] "kmac"
  | 3 -> einsum [ a, "mab"; b, "kmb" ] "kma"
  | _ -> failwith "not batch multipliable"

(* Fv prod, tangent on F *)
let bmm_tangent_F a b =
  let open Maths in
  match List.length (shape b) with
  | 3 -> einsum [ a, "kmab"; b, "mbc" ] "kmac"
  | 2 -> einsum [ a, "kmab"; b, "mb" ] "kma"
  | _ -> failwith "not batch multipliable"

(* Fv prod, tangent on F and v *)
let bmm_tangent_Fv a b =
  let open Maths in
  match List.length (shape b) with
  | 4 -> einsum [ a, "kmab"; b, "kmbc" ] "kmac"
  | 3 -> einsum [ a, "kmab"; b, "kmb" ] "kma"
  | _ -> failwith "not batch multipliable"

(* vF prod, tangent on v *)
let bmm2_tangent_v b a =
  let open Maths in
  match List.length (shape a) with
  | 3 -> einsum [ a, "kma"; b, "mab" ] "kmb"
  | _ -> failwith "should not happen"

(* vF prod, tangent on F *)
let bmm2_tangent_F b a =
  let open Maths in
  match List.length (shape a) with
  | 2 -> einsum [ a, "ma"; b, "kmab" ] "kmb"
  | _ -> failwith "should not happen"

(* vF prod, tangent on F and v *)
let bmm2_tangent_Fv b a =
  let open Maths in
  match List.length (shape a) with
  | 3 -> einsum [ a, "kma"; b, "kmab" ] "kmb"
  | _ -> failwith "should not happen"

let prod f : Maths.t Lqr.prod =
  let primal = bmm (Maths.const (Maths.primal f)) in
  (* tangent on f only *)
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (bmm_tangent_F (Maths.const df))
  in
  { primal; tangent }

let prod_tangent f : Maths.t Lqr.prod =
  (* tangent on v only *)
  let primal = bmm_tangent_v (Maths.const (Maths.primal f)) in
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (bmm_tangent_Fv (Maths.const df))
  in
  { primal; tangent }

let prod2 f : Maths.t Lqr.prod =
  let primal = bmm2 (Maths.const (Maths.primal f)) in
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (bmm2_tangent_F (Maths.const df))
  in
  { primal; tangent }

let prod2_tangent f : Maths.t Lqr.prod =
  let primal = bmm2_tangent_v (Maths.const (Maths.primal f)) in
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (bmm2_tangent_Fv (Maths.const df))
  in
  { primal; tangent }

let tmax = 12
let m = 7
let a_dim = 5
let b_dim = 3

let a () =
  Array.init m ~f:(fun _ ->
    let a = Mat.gaussian a_dim a_dim in
    let r =
      a |> Linalg.eigvals |> Owl.Dense.Matrix.Z.abs |> Owl.Dense.Matrix.Z.re |> Mat.max'
    in
    Arr.reshape Mat.(Float.(0.8 / r) $* a) [| 1; a_dim; a_dim |])
  |> Arr.concatenate ~axis:0
  |> Tensor.of_bigarray ~device

let b () = Arr.gaussian [| m; b_dim; a_dim |] |> Tensor.of_bigarray ~device

let q_of ~reg d =
  Array.init m ~f:(fun _ ->
    let ell = Mat.gaussian d d in
    Arr.reshape Mat.(add_diag (ell *@ transpose ell) reg) [| 1; d; d |])
  |> Arr.concatenate ~axis:0
  |> Tensor.of_bigarray ~device

let q_xx () = q_of ~reg:1. a_dim
let q_uu () = q_of ~reg:1. b_dim
let _f () = Arr.gaussian [| m; a_dim |] |> Tensor.of_bigarray ~device
let _cx () = Arr.gaussian [| m; a_dim |] |> Tensor.of_bigarray ~device
let _cu () = Arr.gaussian [| m; b_dim |] |> Tensor.of_bigarray ~device
let _cxu () = Arr.gaussian [| m; a_dim; b_dim |] |> Tensor.of_bigarray ~device

let map_naive (x : Input.M.t) =
  let irrelevant = Some (fun _ -> assert false) in
  let params =
    List.map x.params ~f:(fun p ->
      Lqr.
        { common =
            { _Fx_prod = Some (bmm p._Fx_prod)
            ; _Fx_prod2 = Some (bmm2 p._Fx_prod)
            ; _Fu_prod = Some (bmm p._Fu_prod)
            ; _Fu_prod2 = Some (bmm2 p._Fu_prod)
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

let map_implicit (x : Input.M.t) =
  let params =
    List.map x.params ~f:(fun p ->
      Lqr.
        { common =
            { _Fx_prod = Some (prod p._Fx_prod)
            ; _Fx_prod2 = Some (prod2 p._Fx_prod)
            ; _Fu_prod = Some (prod p._Fu_prod)
            ; _Fu_prod2 = Some (prod2 p._Fu_prod)
            ; _Fx_prod_tangent = Some (prod_tangent p._Fx_prod)
            ; _Fx_prod2_tangent = Some (prod2_tangent p._Fx_prod)
            ; _Fu_prod_tangent = Some (prod_tangent p._Fu_prod)
            ; _Fu_prod2_tangent = Some (prod2_tangent p._Fu_prod)
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

let f_naive (x : Input.M.t) : Output.M.t = Lqr._solve (map_naive x)
let f_implicit (x : Input.M.t) : Output.M.t = Lqr.solve (map_implicit x)
