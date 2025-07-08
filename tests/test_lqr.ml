open Base
open Torch
open Forward_torch
module Arr = Owl.Arr
module Mat = Owl.Mat
module Linalg = Owl.Linalg.D
open Lqr_common

let rel_tol = Alcotest.float 1e-3
let n_tests = 10

module O = Prms.Option (Prms.Single)
module Input = Lqr.Params.Make (O) (Prms.List (Temp.Make (Prms.Single) (O)))
module Output = Prms.List (Lqr.Solution.Make (O))

let check_grad ~f x =
  let module F = Framework.Make (Input) (Output) in
  F.run x ~f

(* let batch_const = false *)

let map_implicit
      (x :
        ( Maths.any Maths.t option
          , (Maths.any Maths.t, Maths.any Maths.t option) Lqr_common.Temp.p list )
          Lqr.Params.p)
      ~batch_const
  =
  let params =
    List.map x.params ~f:(fun p ->
      Lqr.
        { common =
            { _Fx_prod = Some (Lqr_common.prod ~batch_const p._Fx_prod)
            ; _Fx_prod2 = Some (Lqr_common.prod2 ~batch_const p._Fx_prod)
            ; _Fu_prod = Some (Lqr_common.prod ~batch_const p._Fu_prod)
            ; _Fu_prod2 = Some (Lqr_common.prod2 ~batch_const p._Fu_prod)
            ; _Fx_prod_tangent = Some (Lqr_common.prod_tangent ~batch_const p._Fx_prod)
            ; _Fx_prod2_tangent = Some Lqr_common.(prod2_tangent ~batch_const p._Fx_prod)
            ; _Fu_prod_tangent = Some (Lqr_common.prod_tangent ~batch_const p._Fu_prod)
            ; _Fu_prod2_tangent = Some (Lqr_common.prod2_tangent ~batch_const p._Fu_prod)
            ; _Cxx = Some p._Cxx
            ; _Cxu = p._Cxu
            ; _Cuu = Some p._Cuu
            }
        ; _f = p._f
        ; _cx = p._cx
        ; _cu = p._cu
        })
  in
  Lqr.Params.{ x with params }

let map_naive
      (x :
        ( Maths.any Maths.t option
          , (Maths.any Maths.t, Maths.any Maths.t option) Lqr_common.Temp.p list )
          Lqr.Params.p)
      ~batch_const
  =
  let irrelevant = Some (fun _ -> assert false) in
  let params =
    List.map x.params ~f:(fun p ->
      Lqr.
        { common =
            { _Fx_prod = Some (Lqr_common.bmm ~batch_const p._Fx_prod)
            ; _Fx_prod2 = Some (Lqr_common.bmm2 ~batch_const p._Fx_prod)
            ; _Fu_prod = Some (Lqr_common.bmm ~batch_const p._Fu_prod)
            ; _Fu_prod2 = Some (Lqr_common.bmm2 ~batch_const p._Fu_prod)
            ; _Fx_prod_tangent = irrelevant
            ; _Fx_prod2_tangent = irrelevant
            ; _Fu_prod_tangent = irrelevant
            ; _Fu_prod2_tangent = irrelevant
            ; _Cxx = Some p._Cxx
            ; _Cxu = p._Cxu
            ; _Cuu = Some p._Cuu
            }
        ; _f = p._f
        ; _cx = p._cx
        ; _cu = p._cu
        })
  in
  Lqr.Params.{ x with params }

let test_LQR f ~batch_const () =
  let x =
    Lqr.Params.
      { x0 = Some (Maths.of_tensor (Tensor.randn ~kind ~device [ m; a_dim ]))
      ; params =
          (let tmp () =
             Temp.
               { _f = Some (Maths.of_tensor sample_f)
               ; _Fx_prod = Maths.of_tensor (sample_fx_pri ~batch_const)
               ; _Fu_prod = Maths.of_tensor (sample_fu ~batch_const)
               ; _cx = Some (Maths.of_tensor sample_c_x)
               ; _cu = Some (Maths.of_tensor sample_c_u)
               ; _Cxx = Maths.of_tensor (sample_q_xx ~batch_const)
               ; _Cxu = Some (Maths.of_tensor (sample_q_xu ~batch_const))
               ; _Cuu = Maths.of_tensor (sample_q_uu ~batch_const)
               }
           in
           List.init (tmax + 1) ~f:(fun _ -> tmp ()))
      }
  in
  let _, _, e = check_grad ~f x in
  Alcotest.(check @@ rel_tol) "LQR test" 0.0 e

let f_naive ~batch_const (x : Maths.any Input.t) :Maths.any Output.t =
  let sol, _ = Lqr._solve ~batch_const (map_naive ~batch_const x) in
  let sol = List.map sol ~f:(fun s -> Lqr.Solution.{x =Some s.x; u = Some s.u}) in 
  sol

let f_implicit ~batch_const (x : Maths.any Input.t) =
  let sol = Lqr.solve ~batch_const (map_implicit ~batch_const x) in
  (* let sol = List.map sol ~f:(fun x -> Some x) in *)
  let sol = List.map sol ~f:(fun s -> Lqr.Solution.{x =Some s.x; u = Some s.u}) in 

  sol

let () =
  Alcotest.run
    "LQR tests"
    [ ( "naive_true"
      , List.init n_tests ~f:(fun _ ->
          Alcotest.test_case
            "Naive batch const true"
            `Quick
            (test_LQR (f_naive ~batch_const:true) ~batch_const:true)) )
    ; ( "naive_false"
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
