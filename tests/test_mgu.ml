(* test whether the gradient of MGU agrees with results obtained from finite differences *)
open Base
open Torch
open Forward_torch
module Arr = Owl.Arr
module Mat = Owl.Mat
module Linalg = Owl.Linalg.D
open Lqr_common

let epsilon = 1e-3
let rel_tol = Alcotest.float 1e-4
let n_tests = 10

(* state and control dims *)
let a = 10
let b = 8

(* parameters of the MGU model *)
type 'a theta =
  { _U_f : 'a
  ; _U_h : 'a
  ; _b_f : 'a
  ; _b_h : 'a
  ; _W : 'a
  }

(* (1 + e^-x)^{-2} (e^-x)*)
let d_sigmoid x = Tensor.(sigmoid x * (f 1. - sigmoid x))

let soft_relu x =
  let tmp = Tensor.(square x + f 4.) in
  let num = Tensor.(sqrt tmp + x) in
  Tensor.((num / f 2.) - f 1.)

(* 1/2 [1 + (x^2 + 4)^{-1/2} x]*)
let d_soft_relu x =
  let tmp = Tensor.(square x + f 4.) in
  let tmp2 = Tensor.(f 1. / sqrt tmp) in
  Tensor.(div_scalar ((tmp2 * x) + f 1.) (Scalar.f 2.))

let f ~x ~u theta =
  let pre_sig = Tensor.(matmul theta._U_f x + theta._b_f) in
  let f_t = Tensor.sigmoid pre_sig in
  let pre_g = Tensor.(matmul theta._U_h (f_t * x) + theta._b_h) in
  let x_hat = Tensor.(soft_relu pre_g + matmul theta._W u) in
  Tensor.(((f 1. - f_t) * x) + (f_t * x_hat))

let _Fu ~x theta =
  let pre_sig = Tensor.(matmul theta._U_f x + theta._b_f) in
  let f_t = Tensor.sigmoid pre_sig in
  Tensor.einsum ~equation:"a,ab->ab" [ Tensor.squeeze f_t; theta._W ] ~path:None

let _Fx ~x ~u theta =
  let pre_sig = Tensor.(matmul theta._U_f x + theta._b_f) in
  let f_t = Tensor.sigmoid pre_sig in
  let pre_g = Tensor.(matmul theta._U_h (f_t * x) + theta._b_h) in
  let x_hat = Tensor.(soft_relu pre_g + matmul theta._W u) in
  let tmp_einsum2 a b = Tensor.einsum ~equation:"ab,a->ab" [ a; b ] ~path:None in
  (* Tensor.einsum ~equation:"ab,a->ab" [ a; b ] ~path:None in *)
  let term1 =
    Tensor.diag_embed Tensor.(f 1. - squeeze f_t) ~offset:0 ~dim1:(-2) ~dim2:(-1)
  in
  let term2 =
    let tmp = Tensor.(d_sigmoid pre_sig * (x_hat - x)) in
    tmp_einsum2 theta._U_f (Tensor.squeeze tmp)
  in
  let term3 =
    let tmp1 = tmp_einsum2 theta._U_h (Tensor.squeeze (d_soft_relu pre_g)) in
    let tmp2 = tmp_einsum2 theta._U_f (Tensor.squeeze (d_sigmoid pre_sig)) in
    let tmp3 = Tensor.diag_embed Tensor.(squeeze f_t) ~offset:0 ~dim1:(-2) ~dim2:(-1) in
    let tmp4 = Tensor.(tmp2 + tmp3) in
    let tmp5 = Tensor.(matmul tmp1 tmp4) in
    Tensor.(f_t * tmp5)
  in
  Tensor.(term1 + term2 + term3)

(* example theta *)
let theta_eg =
  let sample_ten ~shape = Tensor.randn shape in
  let _b_f = sample_ten ~shape:[ a; 1 ] in
  let _b_h = sample_ten ~shape:[ a; 1 ] in
  let _W = sample_ten ~shape:[ a; b ] in
  let _U_f = sample_ten ~shape:[ a; a ] in
  let _U_h = sample_ten ~shape:[ a; a ] in
  { _U_f; _U_h; _b_f; _b_h; _W }

let x = Tensor.ones [ a; 1 ]
let u = Tensor.ones [ b; 1 ]

(* test partial f/partial u or partial f/partial x on the idx-th element of u/x *)
let test_Fz ~idx ~test_x =
  let _Fz_finite =
    let dz = if test_x then Tensor.zeros_like x else Tensor.zeros_like u in
    Tensor.set_float2 dz idx 0 1.;
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
  Tensor.(print (norm (_Fz_finite - _Fz_analytic) / norm _Fz_finite))

let _ = test_Fz ~idx:6 ~test_x:true
