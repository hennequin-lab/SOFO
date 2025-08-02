(* test whether the gradient of MGU agrees with results obtained from finite differences *)
open Base
open Torch
open Forward_torch
module Arr = Owl.Arr
module Mat = Owl.Mat
module Linalg = Owl.Linalg.D

let print s = Stdio.print_endline (Sexp.to_string_hum s)
let epsilon = 1e-3
let rel_tol = Alcotest.float 1e-3
let n_tests = 10

(* state and control dims *)
let n = 20
let m = 5

(* parameters of the MGU model *)
type 'a theta =
  { _U_f : 'a
  ; _U_h : 'a
  ; _b_f : 'a
  ; _b_h : 'a
  ; _W : 'a
  }

(* (1 + e^-x)^{-2} (e^-x) *)
let d_sigmoid x = Tensor.(sigmoid x * (f 1. - sigmoid x))

let soft_relu x =
  let tmp = Tensor.(square x + f 4.) in
  let num = Tensor.(sqrt tmp + x) in
  Tensor.((num / f 2.) - f 1.)

(* 1/2 [1 + (x^2 + 4)^{-1/2} x]*)
let d_soft_relu x =
  let tmp = Tensor.(square x + f 4.) in
  let tmp2 = Tensor.(f 1. / sqrt tmp) in
  Tensor.(((tmp2 * x) + f 1.) / f 2.)

let pre_sig x theta = Tensor.(matmul x theta._U_f + theta._b_f)
let pre_g ~f_t ~x theta = Tensor.(matmul (f_t * x) theta._U_h + theta._b_h)
let x_hat ~pre_g ~u theta = Tensor.(soft_relu pre_g + matmul u theta._W)

let f ~x ~u theta =
  let pre_sig = pre_sig x theta in
  let f_t = Tensor.sigmoid pre_sig in
  let pre_g = pre_g ~f_t ~x theta in
  let x_hat = x_hat ~pre_g ~u theta in
  let new_x = Tensor.(((f 1. - f_t) * x) + (f_t * x_hat)) in
  new_x

(* dim [bs x m x n ] *)
let _Fu ~x theta =
  let pre_sig = pre_sig x theta in
  let f_t = Tensor.sigmoid pre_sig in
  Tensor.einsum ~equation:"ma,ba->mba" [ f_t; theta._W ] ~path:None

(* dim [bs x n x n ] *)
let _Fx ~x ~u theta =
  let pre_sig = pre_sig x theta in
  let f_t = Tensor.sigmoid pre_sig in
  let pre_g = pre_g ~f_t ~x theta in
  let x_hat = x_hat ~pre_g ~u theta in
  let tmp_einsum2 a b = Tensor.einsum ~equation:"ba,ma->mba" [ a; b ] ~path:None in
  let term1 = Tensor.diag_embed Tensor.(f 1. - f_t) ~offset:0 ~dim1:(-2) ~dim2:(-1) in
  let term2 =
    let tmp = Tensor.(d_sigmoid pre_sig * (x_hat - x)) in
    (*  _U_f [b * a], tmp [m x a] , -> [m * b * a] *)
    tmp_einsum2 theta._U_f tmp
  in
  let term3 =
    let tmp1 = tmp_einsum2 theta._U_h (d_soft_relu pre_g) in
    let tmp2 = tmp_einsum2 theta._U_f (d_sigmoid pre_sig) in
    let tmp3 = Tensor.diag_embed f_t ~offset:0 ~dim1:(-2) ~dim2:(-1) in
    let tmp4 = Tensor.(tmp2 + tmp3) in
    let tmp5 = Tensor.einsum ~equation:"mab,mbc->mac" [ tmp4; tmp1 ] ~path:None in
    print [%message (Tensor.shape tmp5 : int list) (Tensor.shape f_t : int list)];
    Tensor.( (unsqueeze ~dim:1 f_t) * tmp5)
  in
  Tensor.(term1 + term2 + term3)

(* example theta *)
let theta_eg =
  let sample_ten ~shape = Tensor.randn shape in
  let _b_f = sample_ten ~shape:[ 1; n ] in
  let _b_h = sample_ten ~shape:[ 1; n ] in
  let _W = sample_ten ~shape:[ m; n ] in
  let _U_f = sample_ten ~shape:[ n; n ] in
  let _U_h = sample_ten ~shape:[ n; n ] in
  { _U_f; _U_h; _b_f; _b_h; _W }

let x = Tensor.ones [1; n ]
let u = Tensor.ones [ 1; m ]

(* test partial f/partial u or partial f/partial x on the idx-th element of u/x *)
let test_Fz ~idx ~test_x () =
  let _Fz_finite =
    (* dz is of shape [1 x n]/[1 x m] where idx-th element is 1. the gradient corresponds 
    to the idx- th col in Jacobian, hence idx-th row in Fx/Fu *)
    let dz = if test_x then Tensor.zeros_like x else Tensor.zeros_like u in
    Tensor.set_float2 dz 0 idx 1.;
    let dz = Tensor.(mul_scalar dz Scalar.(f epsilon)) in
    let fplus =
      if test_x
      then f ~x:Tensor.(x + dz) ~u theta_eg
      else f ~x ~u:Tensor.(u + dz) theta_eg
    in
    let fminus =
      if test_x
      then f ~x:Tensor.(x - dz) ~u theta_eg
      else f ~x ~u:Tensor.(u - dz) theta_eg
    in
    Tensor.(div_scalar (fplus - fminus) Scalar.(f Float.(2. * epsilon)))
  in
  let _Fz_analytic =
    let tmp = if test_x then _Fx ~x ~u theta_eg else _Fu ~x theta_eg in
    Tensor.slice tmp ~dim:1 ~start:(Some idx) ~end_:(Some (idx + 1)) ~step:1
  in
  let diff =
    Tensor.(norm (_Fz_finite - _Fz_analytic) / norm _Fz_finite) |> Tensor.to_float0_exn
  in
  Alcotest.(check @@ rel_tol) "LQR test" 0.0 diff

let () =
  Alcotest.run
    "MGU tests"
    [ ( "fx"
      , List.init n_tests ~f:(fun _ ->
          Alcotest.test_case "Fx" `Quick (test_Fz ~idx:(Random.int n) ~test_x:true)) )
    ; ( "fu"
      , List.init n_tests ~f:(fun _ ->
          Alcotest.test_case "Fu" `Quick (test_Fz ~idx:(Random.int m) ~test_x:false)) )
    ]
