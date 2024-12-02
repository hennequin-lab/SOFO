(* test whether directly differentiating and implicitly differentiating through the lqr agrees. *)
open Base
open Torch
open Forward_torch
module Arr = Owl.Arr
module Mat = Owl.Mat
module Linalg = Owl.Linalg.D
open Lqr_common

let rel_tol = Alcotest.float 1e-4
let n_tests = 20
let k = 12

let x0 =
  let pri = Tensor.rand ~kind ~device [ bs; n ] in
  let tan = Tensor.randn ~kind ~device [ k; bs; n ] in
  Maths.make_dual pri ~t:(Maths.Direct tan)

let fx () =
  let pri = Tensor.randn ~kind ~device [ bs; n; n ] in
  let tan = Tensor.randn ~kind ~device [ k; bs; n; n ] in
  Maths.make_dual pri ~t:(Maths.Direct tan)

let fu () =
  let pri = Tensor.randn ~kind ~device [ bs; m; n ] in
  let tan = Tensor.randn ~kind ~device [ k; bs; m; n ] in
  Maths.make_dual pri ~t:(Maths.Direct tan)

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

let _Fx_prod fx = prod fx
let _Fx_prod2 fx = prod2 fx
let _Fx_prod_tangent fx = prod_tangent fx
let _Fx_prod2_tangent fx = prod2_tangent fx
let _Fu_prod fu = prod fu
let _Fu_prod2 fu = prod2 fu
let _Fu_prod_tangent fu = prod_tangent fu
let _Fu_prod2_tangent fu = prod2_tangent fu

let q_of_tan ~reg d =
  Array.init k ~f:(fun _ ->
    Array.init bs ~f:(fun _ ->
      let ell = Mat.gaussian d d in
      Arr.reshape Mat.(add_diag (ell *@ transpose ell) reg) [| 1; 1; d; d |])
    |> Arr.concatenate ~axis:1)
  |> Arr.concatenate ~axis:0
  |> Tensor.of_bigarray ~device

let dq_xx () = q_of_tan ~reg:0.1 n
let dq_uu () = q_of_tan ~reg:0.1 m

let q_xx () =
  let pri = q_xx () in
  let tan = dq_xx () in
  Maths.make_dual pri ~t:(Maths.Direct tan)

let q_uu () =
  let pri = q_uu () in
  let tan = dq_uu () in
  Maths.make_dual pri ~t:(Maths.Direct tan)

let common_params =
  Lqr.Params.
    { x0 = Some x0
    ; params =
        (let tmp () =
           Temp.
             { _f = None
             ; _Fx_prod = fx ()
             ; _Fu_prod = fu ()
             ; _cx = None
             ; _cu = None
             ; _Cxx = q_xx ()
             ; _Cxu = None
             ; _Cuu = q_uu ()
             }
         in
         List.init (tmax + 1) ~f:(fun _ -> tmp ()))
    }

let f_naive (x : Input.M.t) : Output.M.t =
  let x =
    let params =
      List.map x.params ~f:(fun p ->
        Lqr.
          { common =
              { _Fx_prod = Some (bmm p._Fx_prod)
              ; _Fx_prod2 = Some (bmm2 p._Fx_prod)
              ; _Fu_prod = Some (bmm p._Fu_prod)
              ; _Fu_prod2 = Some (bmm2 p._Fu_prod)
              ; _Fx_prod_tangent = Some (bmm_tangent_v p._Fx_prod)
              ; _Fx_prod2_tangent = Some (bmm2_tangent_v p._Fx_prod)
              ; _Fu_prod_tangent = Some (bmm_tangent_v p._Fu_prod)
              ; _Fu_prod2_tangent = Some (bmm2_tangent_v p._Fu_prod)
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
  in
  Lqr._solve x

let f_implicit (x : Input.M.t) =
  let x =
    let params =
      List.map x.params ~f:(fun p ->
        Lqr.
          { common =
              { _Fx_prod = Some (_Fx_prod p._Fx_prod)
              ; _Fx_prod2 = Some (_Fx_prod2 p._Fx_prod)
              ; _Fu_prod = Some (_Fu_prod p._Fu_prod)
              ; _Fu_prod2 = Some (_Fu_prod2 p._Fu_prod)
              ; _Fx_prod_tangent = Some (_Fx_prod_tangent p._Fx_prod)
              ; _Fx_prod2_tangent = Some (_Fx_prod2_tangent p._Fx_prod)
              ; _Fu_prod_tangent = Some (_Fu_prod_tangent p._Fu_prod)
              ; _Fu_prod2_tangent = Some (_Fu_prod2_tangent p._Fu_prod)
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
  in
  Lqr.solve x

let check_implicit common_params =
  let lqr_naive = f_naive common_params in
  let lqr_implicit = f_implicit common_params in
  let x_error =
    List.fold2_exn lqr_naive lqr_implicit ~init:0. ~f:(fun acc naive implicit ->
      let x_naive = naive.x |> Option.value_exn in
      let x_implicit = implicit.x in
      let error =
        Tensor.(
          norm (Option.value_exn Maths.(tangent (x_naive - x_implicit))) |> to_float0_exn)
      in
      acc +. error)
  in
  x_error

(* test whether the tangents agree *)
let test_LQR () =
  let common_params =
    Lqr.Params.
      { x0 = Some x0
      ; params =
          (let tmp () =
             Temp.
               { _f = None
               ; _Fx_prod = fx ()
               ; _Fu_prod = fu ()
               ; _cx = None
               ; _cu = None
               ; _Cxx = q_xx ()
               ; _Cxu = None
               ; _Cuu = q_uu ()
               }
           in
           List.init (tmax + 1) ~f:(fun _ -> tmp ()))
      }
  in
  let e = check_implicit common_params in
  Alcotest.(check @@ rel_tol) "LQR test" 0.0 e

let () =
  let open Alcotest in
  run
    "LQR tests"
    [ "Check implicit", List.init n_tests ~f:(fun _ -> test_case "Simple" `Quick test_LQR)
    ]
