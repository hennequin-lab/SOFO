(* chekcing lqr recovers the ground truth in the limit of small costs *)
open Printf
open Base
open Forward_torch
open Torch
open Sofo
module Mat = Owl.Dense.Matrix.D

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let n = 24
let m = 10
let o = 48
let tmax = 10
let batch_size = 512
let kind = Torch_core.Kind.(T f64)
let device = Torch.Device.cuda_if_available ()
let base = Optimizer.Config.Base.{ default with kind; ba_kind = Bigarray.float64 }
let x0 = Tensor.zeros ~device ~kind [ batch_size; n ]

let sample_stable n =
  let a = Mat.gaussian n n in
  let r =
    a
    |> Owl.Linalg.D.eigvals
    |> Owl.Dense.Matrix.Z.abs
    |> Owl.Dense.Matrix.Z.re
    |> Mat.max'
  in
  Mat.(Float.(0.8 / r) $* a)

let a = Tensor.of_bigarray ~device (sample_stable n) |> Maths.const
let b = Tensor.randn ~device ~kind [ m; n ] |> Maths.const
let c = Tensor.randn ~device ~kind [ n; o ] |> Maths.const

(* x list goes from 0 to T but o list goes from 1 to T *)
let rollout ~u_list =
  let _, xs, os =
    List.fold u_list ~init:(x0, [], []) ~f:(fun (x_prev, xs, os) u ->
      let x = Tensor.(matmul x_prev (Maths.primal a) + matmul u (Maths.primal b)) in
      let o = Tensor.matmul x (Maths.primal c) in
      x, x :: xs, o :: os)
  in
  List.rev xs, List.rev os

let sample_data () =
  let u_list =
    List.init tmax ~f:(fun _ -> Tensor.randn ~device ~kind [ batch_size; m ])
  in
  let x_list, o_list = rollout ~u_list in
  u_list, x_list, o_list

(* create params for lds from f *)
let params_from_f ~x0 ~o_list
  : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
  =
  let o_list = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
  let _Cxx = Maths.(einsum [ c, "ab"; c, "cb" ] "ac") in
  let _Cuu =
    Tensor.(f 0.001 * of_bigarray ~device:base.device (Mat.eye m)) |> Maths.const
  in
  Lqr.Params.
    { x0 = Some x0
    ; params =
        List.map o_list ~f:(fun o ->
          let _cx = Maths.(neg (einsum [ const o, "mb"; c, "ab" ] "ma")) in
          Lds_data.Temp.
            { _f = None
            ; _Fx_prod = a
            ; _Fu_prod = b
            ; _cx = Some _cx
            ; _cu = None
            ; _Cxx
            ; _Cxu = None
            ; _Cuu
            })
    }

let lqr_pass ~data =
  let _, _, o_list = data in
  let p =
    params_from_f ~x0:(Maths.const x0) ~o_list |> Lds_data.map_naive ~batch_const:true
  in
  let sol, _ = Lqr._solve ~laplace:false ~batch_const:true p in
  Convenience.print [%message (List.length sol : int)];
  ( List.map sol ~f:(fun s -> s.u)
  , List.map sol ~f:(fun s -> s.x)
  , List.map sol ~f:(fun s -> Maths.(s.x *@ c)) )

let ((us, xs, os) as data) = sample_data ()
let us', xs', os' = lqr_pass ~data

let us_err =
  List.fold2_exn us us' ~init:0. ~f:(fun e u u' ->
    let de = Maths.(sum (sqr (const u - u'))) |> Maths.primal |> Tensor.to_float0_exn in
    Float.(e + de))

let os_err =
  List.fold2_exn os os' ~init:0. ~f:(fun e o o' ->
    let de = Maths.(sum (sqr (const o - o'))) |> Maths.primal |> Tensor.to_float0_exn in
    Float.(e + de))

let xs_err =
  List.fold2_exn xs xs' ~init:0. ~f:(fun e x x' ->
    let de = Maths.(sum (sqr (const x - x'))) |> Maths.primal |> Tensor.to_float0_exn in
    Float.(e + de))

let _ = Convenience.print [%message (us_err : float) (os_err : float) (xs_err : float)]
