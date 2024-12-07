open Base
open Torch
module Arr = Owl.Arr
module Mat = Owl.Mat
module Linalg = Owl.Linalg.D
open Lqr_common

let rel_tol = Alcotest.float 1e-4
let n_tests = 10

let check_grad ~f (x : Input.T.t) =
  let module F = Framework.Make (Input) (Output) in
  F.run x ~f

(* let batch_const = false *)

let test_LQR f ~batch_const () =
  let x =
    Lqr.Params.
      { x0 = Some (Tensor.randn ~kind ~device [ m; a_dim ])
      ; params =
          (let tmp () =
             Temp.
               { _f = Some sample_f
               ; _Fx_prod = sample_fx_pri ~batch_const
               ; _Fu_prod = sample_fu ~batch_const
               ; _cx = Some sample_c_x
               ; _cu = Some sample_c_u
               ; _Cxx = sample_q_xx ~batch_const
               ; _Cxu = Some (sample_q_xu ~batch_const)
               ; _Cuu = sample_q_uu ~batch_const
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
    [ ( "naive_true"
      , List.init n_tests ~f:(fun _ ->
          Alcotest.test_case
            "Naive batch const true"
            `Quick
            (test_LQR (f_naive ~batch_const:true) ~batch_const:true)) )
     ; ( "niave_false"
      , List.init n_tests ~f:(fun _ ->
          Alcotest.test_case
            "Naive batch const false"
            `Quick
            (test_LQR (f_naive ~batch_const:false) ~batch_const:false)) ) 
    ; ( "implicit_true"
      , List.init n_tests ~f:(fun _ ->
          Alcotest.test_case
            "Implicit batch const true"
            `Quick
            (test_LQR (f_implicit ~batch_const:true) ~batch_const:true)) )
    ; ( "implicit_false"
      , List.init n_tests ~f:(fun _ ->
          Alcotest.test_case
            "Implicit batch const false"
            `Quick
            (test_LQR (f_implicit ~batch_const:false) ~batch_const:false)) )
    ]
