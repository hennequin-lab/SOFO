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
        let irrelevant = Some (fun _ -> assert false) in
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
  in
  Lqr._solve x

let f' (x : Input.M.t) : Output.M.t =
  let x =
    let params =
      List.map x.params ~f:(fun p ->
        let irrelevant = Some Lqr.{ primal = (fun _ -> assert false); tangent = None } in
        let wrap p = Lqr.{ primal = p; tangent = None } in
        Lqr.
          { common =
              { _Fx_prod = Some (wrap (bmm p._Fx_prod))
              ; _Fx_prod2 = Some (wrap (bmm2 p._Fx_prod))
              ; _Fu_prod = Some (wrap (bmm p._Fu_prod))
              ; _Fu_prod2 = Some (wrap (bmm2 p._Fu_prod))
              ; _Fx_prod_tangent = Some (wrap (_Fx_prod_tangent p._Fx_prod))
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

let rel_tol = Alcotest.float 1e-4
let n_tests = 20

let check_grad ~f (x : Input.T.t) =
  let module F = Framework.Make (Input) (Output) in
  F.run x ~f

let test_LQR f () =
  let x =
    Lqr.Params.
      { x0 = Some (Tensor.randn ~kind ~device [ bs; n ])
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
  let _, _, e = check_grad ~f x in
  Alcotest.(check @@ rel_tol) "LQR test" 0.0 e

let () =
  Alcotest.run
    "LQR tests"
    [ ( "solve_"
      , List.init n_tests ~f:(fun _ -> Alcotest.test_case "Simple" `Quick (test_LQR f)) )
    ; ( "solve"
      , List.init n_tests ~f:(fun _ -> Alcotest.test_case "Simple" `Quick (test_LQR f')) )
    ]
