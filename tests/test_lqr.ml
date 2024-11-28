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

module Input =
  Lqr.Params.Make (Prms.P) (Prms.List (Temp.Make (Prms.P) (Prms.Option (Prms.P))))

module Output = Prms.List (Lqr.Solution.Make (Prms.P))
module L = Lqr.Make (Lqr.MathsOps)

let bmm a b =
  let open Maths in
  match List.length (shape b) with
  | 3 -> einsum [ a, "aij"; b, "ajk" ] "aik"
  | 2 -> einsum [ a, "aij"; b, "aj" ] "ai"
  | _ -> failwith "not batch multipliable"

let bmm2 b a =
  let open Maths in
  match List.length (shape a) with
  | 2 -> einsum [ a, "ai"; b, "aij" ] "aj"
  | _ -> failwith "should not happen"

let f (x : Input.M.t) : Output.M.t =
  let x =
    let params =
      List.map x.params ~f:(fun p ->
        Lqr.
          { _f = p._f
          ; _Fx_prod = bmm p._Fx_prod
          ; _Fx_prod2 = bmm2 p._Fx_prod
          ; _Fu_prod = bmm p._Fu_prod
          ; _Fu_prod2 = bmm2 p._Fu_prod
          ; _cx = p._cx
          ; _cu = p._cu
          ; _Cxx = p._Cxx
          ; _Cxu = p._Cxu
          ; _Cuu = p._Cuu
          })
    in
    Lqr.Params.{ x with params }
  in
  L.solve x

let tmax = 2
let bs = 7
let n = 5
let m = 3

let a () =
  Array.init bs ~f:(fun _ ->
    (*
       let a = Mat.gaussian n n in
       let r =
       a |> Linalg.eigvals |> Owl.Dense.Matrix.Z.abs |> Owl.Dense.Matrix.Z.re |> Mat.max'
       in
       Arr.reshape Mat.(Float.(0.8 / r) $* a) [| 1; n; n |]) *)
    Arr.reshape (Mat.(0.8 $* eye n)) [| 1; n; n |])
  |> Arr.concatenate ~axis:0
  |> Tensor.of_bigarray ~device

let b () = Arr.gaussian [| bs; m; n |] |> Tensor.of_bigarray ~device

let q_of ~reg:_ d =
  Array.init bs ~f:(fun _ ->
    (*
       let ell = Mat.gaussian d d in
       Arr.reshape Mat.(add_diag (ell *@ transpose ell) reg) [| 1; d; d |]) *)
    Arr.reshape Mat.(eye d) [| 1; d; d |])
  |> Arr.concatenate ~axis:0
  |> Tensor.of_bigarray ~device

let q_xx () = q_of ~reg:10. n
let q_uu () = q_of ~reg:10. m
let q_xu () = Arr.gaussian [| bs; n; m |] |> Tensor.of_bigarray ~device
let _f () = Arr.gaussian [| bs; n |] |> Tensor.of_bigarray ~device
let _cx () = Arr.gaussian [| bs; n |] |> Tensor.of_bigarray ~device
let _cu () = Arr.gaussian [| bs; m |] |> Tensor.of_bigarray ~device

let check_grad (x : Input.T.t) =
  let module F = Framework.Make (Input) (Output) in
  F.run x ~f

let _ =
  let x =
    Lqr.Params.
      { x0 = Tensor.randn ~kind ~device [ bs; n ]
      ; params =
          (let tmp () =
             Temp.
               { _f = None
               ; _Fx_prod = a ()
               ; _Fu_prod = b ()
               ; _cx = None
               ; _cu = None
               ; _Cxx = q_xx ()
               ; _Cxu = None
               ; _Cuu = q_uu ()
               }
           in
           List.init 3 ~f:(fun _ -> tmp ()))
      }
  in
  Array.init 100 ~f:(fun _ ->
    let dp1, dp2, e = check_grad x in
    [| dp1; dp2; e |])
  |> Mat.of_arrays
  |> Mat.save_txt ~out:"lqr_test_haha"
