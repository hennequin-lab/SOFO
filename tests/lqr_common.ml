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
module Output = Prms.List (Lqr.Solution.Make (O))

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

let tmax = 12
let bs = 7
let n = 5
let m = 3

let a () =
  Array.init bs ~f:(fun _ ->
    let a = Mat.gaussian n n in
    let r =
      a |> Linalg.eigvals |> Owl.Dense.Matrix.Z.abs |> Owl.Dense.Matrix.Z.re |> Mat.max'
    in
    Arr.reshape Mat.(Float.(0.8 / r) $* a) [| 1; n; n |])
  |> Arr.concatenate ~axis:0
  |> Tensor.of_bigarray ~device

let b () = Arr.gaussian [| bs; m; n |] |> Tensor.of_bigarray ~device

let q_of ~reg d =
  Array.init bs ~f:(fun _ ->
    let ell = Mat.gaussian d d in
    Arr.reshape Mat.(add_diag (ell *@ transpose ell) reg) [| 1; d; d |])
  |> Arr.concatenate ~axis:0
  |> Tensor.of_bigarray ~device

let q_xx () = q_of ~reg:1. n
let q_uu () = q_of ~reg:1. m
let _f () = Arr.gaussian [| bs; n |] |> Tensor.of_bigarray ~device
let _cx () = Arr.gaussian [| bs; n |] |> Tensor.of_bigarray ~device
let _cu () = Arr.gaussian [| bs; m |] |> Tensor.of_bigarray ~device
let _cxu () = Arr.gaussian [| bs; n; m |] |> Tensor.of_bigarray ~device
