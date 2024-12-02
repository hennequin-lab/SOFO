open Base
open Torch
open Forward_torch
module Arr = Owl.Arr
module Mat = Owl.Mat
module Linalg = Owl.Linalg.D
open Lqr_common

let f (x : Input.M.t) : Output.M.t =
  let x =
    let params =
      List.map x.params ~f:(fun p ->
        Lqr.
          { common =
              { _Fx_prod = Some (bmm p._Fx_prod)
              ; _Fx_prod2 = Some (bmm2 p._Fx_prod)
              ; _Fu_prod = Some (bmm p._Fu_prod)
              ; _Fu_prod2 = Some (bmm2 p._Fu_prod)
              ; _Fx_prod_tangent = Some (bmm_tangent_F p._Fx_prod)
              ; _Fx_prod2_tangent = Some (bmm2_tangent_F p._Fx_prod)
              ; _Fu_prod_tangent = Some (bmm_tangent_F p._Fu_prod)
              ; _Fu_prod2_tangent = Some (bmm2_tangent_F p._Fu_prod)
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

let rel_tol = Alcotest.float 1e-4
let n_tests = 20

let check_grad (x : Input.T.t) =
  let module F = Framework.Make (Input) (Output) in
  F.run x ~f

let test_LQR () =
  let x =
    Lqr.Params.
      { x0 = Some (Tensor.randn ~kind ~device [ m; a_dim ])
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
           List.init (tmax + 1) ~f:(fun _ -> tmp ()))
      }
  in
  let _, _, e = check_grad x in
  Alcotest.(check @@ rel_tol) "LQR test" 0.0 e

let () =
  let open Alcotest in
  run
    "LQR tests"
    [ "Simple-case", List.init n_tests ~f:(fun _ -> test_case "Simple" `Quick test_LQR) ]
