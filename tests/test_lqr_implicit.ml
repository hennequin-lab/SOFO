(* test whether directly differentiating and implicitly differentiating through the lqr agrees. *)
open Base
open Torch
open Forward_torch
module Arr = Owl.Arr
module Mat = Owl.Mat
module Linalg = Owl.Linalg.D
open Lqr_common

let rel_tol = Alcotest.float 1e-4
let n_tests = 10
let k = 8
(* let batch_const = true *)

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

let sample_fx_tan ~batch_const =
  let fx_tan =
    if batch_const
    then
      Array.init k ~f:(fun _ ->
        let a = sample_stable () in
        Arr.reshape a [| 1; a_dim; a_dim |])
      |> Arr.concatenate ~axis:0
    else
      Array.init k ~f:(fun _ ->
        Array.init m ~f:(fun _ ->
          let a = sample_stable () in
          Arr.reshape a [| 1; 1; a_dim; a_dim |])
        |> Arr.concatenate ~axis:1)
      |> Arr.concatenate ~axis:0
  in
  to_device fx_tan

(* make sure cost matrices are positive definite *)
let q_tan_of ~batch_const ~reg d =
  let q_tan =
    if batch_const
    then
      Array.init k ~f:(fun _ ->
        let _q_tan = pos_sym ~reg d in
        Arr.reshape _q_tan [| 1; d; d |])
      |> Arr.concatenate ~axis:0
    else
      Array.init k ~f:(fun _ ->
        Array.init m ~f:(fun _ ->
          let _q_tan = pos_sym ~reg d in
          Arr.reshape _q_tan [| 1; 1; d; d |])
        |> Arr.concatenate ~axis:1)
      |> Arr.concatenate ~axis:0
  in
  to_device q_tan

let sample_q_xx ~batch_const =
  let pri = q_of ~batch_const ~reg:1. a_dim in
  let tan = q_tan_of ~batch_const ~reg:1. a_dim in
  Maths.make_dual pri ~t:(Maths.Direct tan)

let sample_q_uu ~batch_const =
  let pri = q_of ~batch_const ~reg:1. b_dim in
  let tan = q_tan_of ~batch_const ~reg:1. b_dim in
  Maths.make_dual pri ~t:(Maths.Direct tan)

let sample_tangent shape =
  let pri = Tensor.randn ~device ~kind shape in
  let tan = Tensor.randn ~device ~kind (k :: shape) in
  Maths.make_dual pri ~t:(Maths.Direct tan)

let sample_x0 () = sample_tangent [ m; a_dim ]

let sample_fx ~batch_const =
  let pri = sample_fx_pri ~batch_const in
  let tan = sample_fx_tan ~batch_const in
  Maths.make_dual pri ~t:(Maths.Direct tan)

let sample_fu ~batch_const =
  if batch_const
  then sample_tangent [ b_dim; a_dim ]
  else sample_tangent [ m; b_dim; a_dim ]

let sample_q_xu ~batch_const =
  if batch_const
  then sample_tangent [ a_dim; b_dim ]
  else sample_tangent [ m; a_dim; b_dim ]

let sample_c_x () = sample_tangent [ m; a_dim ]
let sample_c_u () = sample_tangent [ m; b_dim ]
let sample_f () = sample_tangent [ m; a_dim ]
let sample_u () = sample_tangent [ m; b_dim ]

let sample_tangent shape =
  let pri = Tensor.randn ~device ~kind shape in
  let tan = Tensor.randn ~device ~kind (k :: shape) in
  Maths.make_dual pri ~t:(Maths.Direct tan)

let sample_x0 () = sample_tangent [ m; a_dim ]

let sample_fx ~batch_const =
  let pri = sample_fx_pri ~batch_const in
  let tan = sample_fx_tan ~batch_const in
  Maths.make_dual pri ~t:(Maths.Direct tan)

let sample_fu ~batch_const =
  if batch_const
  then sample_tangent [ b_dim; a_dim ]
  else sample_tangent [ m; b_dim; a_dim ]

let sample_q_xu ~batch_const =
  if batch_const
  then sample_tangent [ a_dim; b_dim ]
  else sample_tangent [ m; a_dim; b_dim ]

let sample_c_x () = sample_tangent [ m; a_dim ]
let sample_c_u () = sample_tangent [ m; b_dim ]
let sample_f () = sample_tangent [ m; a_dim ]
let sample_u () = sample_tangent [ m; b_dim ]

let check_implicit ~batch_const common_params =
  let lqr_naive = f_naive ~batch_const common_params in
  let lqr_implicit = f_implicit ~batch_const common_params in
  let x_error =
    List.fold2_exn lqr_naive lqr_implicit ~init:0. ~f:(fun acc naive implicit ->
      let x_naive = naive.x in
      let x_implicit = implicit.x in
      let error =
        Tensor.(
          norm (Option.value_exn Maths.(tangent (x_naive - x_implicit))) |> to_float0_exn)
      in
      acc +. error)
  in
  x_error

(* test whether the tangents agree *)
let x0 = sample_x0 ()

let test_same_tangents ~batch_const () =
  let common_params =
    Lqr.Params.
      { x0 = Some x0
      ; params =
          (let tmp () =
             Temp.
               { _f = Some (sample_f ())
               ; _Fx_prod = sample_fx ~batch_const
               ; _Fu_prod = sample_fu ~batch_const
               ; _cx = Some (sample_c_x ())
               ; _cu = Some (sample_c_u ())
               ; _Cxx = sample_q_xx ~batch_const
               ; _Cxu = Some (sample_q_xu ~batch_const)
               ; _Cuu = sample_q_uu ~batch_const
               }
           in
           List.init (tmax + 1) ~f:(fun _ -> tmp ()))
      }
  in
  let e = check_implicit ~batch_const common_params in
  Alcotest.(check @@ rel_tol) "LQR test" 0.0 e

let () =
  let open Alcotest in
  run
    "LQR tests"
    [ ( "Check implicit true"
      , List.init n_tests ~f:(fun _ ->
          test_case
            "Implicit batch const true"
            `Quick
            (test_same_tangents ~batch_const:true)) )
    ; ( "Check implicit false"
      , List.init n_tests ~f:(fun _ ->
          test_case
            "Implicit batch const false"
            `Quick
            (test_same_tangents ~batch_const:false)) )
    ]
