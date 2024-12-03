open Base
open Torch
open Forward_torch
module Arr = Owl.Arr
module Mat = Owl.Mat
module Linalg = Owl.Linalg.D
open Lqr_common

let rel_tol = Alcotest.float 1e-4
let n_tests = 20

let check_grad ~f (x : Input.T.t) =
  let module F = Framework.Make (Input) (Output) in
  F.run x ~f

let test_LQR f () =
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
  let _, _, e = check_grad ~f x in
  Alcotest.(check @@ rel_tol) "LQR test" 0.0 e

let () =
  Alcotest.run
    "LQR tests"
    [ ( "solve_"
      , List.init n_tests ~f:(fun _ ->
          Alcotest.test_case "Simple" `Quick (test_LQR f_naive)) )
    ; ( "solve"
      , List.init n_tests ~f:(fun _ ->
          Alcotest.test_case "Simple" `Quick (test_LQR f_implicit)) )
    ]
