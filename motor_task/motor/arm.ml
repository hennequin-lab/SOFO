open Base
open Torch
open Forward_torch
module Mat = Owl.Dense.Matrix.S
open Sofo

let base = Optimizer.Config.Base.default

type 'a pair =
  { x1 : 'a
  ; x2 : 'a
  }

(* angular or hand state *)
type 'a state =
  { pos : 'a pair
  ; vel : 'a pair
  }

let map s ~f =
  { pos = { x1 = f s.pos.x1; x2 = f s.pos.x2 }
  ; vel = { x1 = f s.vel.x1; x2 = f s.vel.x2 }
  }

let concat_pairs x =
  let x1 = List.map x ~f:(fun y -> y.x1) |> Tensor.concatenate ~dim:0 in
  let x2 = List.map x ~f:(fun y -> y.x2) |> Tensor.concatenate ~dim:0 in
  { x1; x2 }

(* ----------------------------------------------------------------------------
   ---     CONSTANT PARAMETERS                                              ---
   ---------------------------------------------------------------------------- *)

(* arm lengths *)
let _L1 = 0.3
let _L2 = 0.3

(* moments of inertia *)
let _I1 = 0.025
let _I2 = 0.045

(* masses *)
let _M1 = 1.4
let _M2 = 1.0

(* B matrix *)
let _B11 = 0.05
let _B12 = 0.025
let _B21 = 0.025
let _B22 = 0.05

(* distances *)
let _S1 = 0.11
let _S2 = 0.16

(* initial theta *)
let theta_init bs =
  let a = 20. in
  let theta1 = Float.(a * pi / 180.) in
  let theta2 = Float.((180. - (2. * a)) * pi / 180.) in
  let as_batch x =
    Tensor.(mul_scalar (ones ~device:base.device [ bs; 1 ]) (Scalar.f x))
  in
  { pos =
      { x1 = as_batch theta1 |> Maths.of_tensor; x2 = as_batch theta2 |> Maths.of_tensor }
  ; vel = { x1 = as_batch 0. |> Maths.of_tensor; x2 = as_batch 0. |> Maths.of_tensor }
  }

(* derived constants *)
let _A1 = _I1 +. _I2 +. (_M2 *. Float.square _L1)
let _A2 = _M2 *. _L1 *. _S2
let _A3 = _I2

(* ----------------------------------------------------------------------------
   ---     KINEMATICS MATH FUNCTIONS                                        ---
   ---------------------------------------------------------------------------- *)

(* forward dynamics *)
let theta_dot_dot =
  let open Maths in
  let prod_with_b =
    let b11, b12, b21, b22 = _B11, _B12, _B21, _B22 in
    fun (x1, x2) -> (b11 $* x1) + (b12 $* x2), (b21 $* x1) + (b22 $* x2)
  in
  fun (theta : [< `const | `dual ] Maths.t state) (torque1, torque2) ->
    let prod_with_minv =
      let m11, m12, m21, m22 =
        let z = _A2 $* cos theta.pos.x2 in
        _A1 $+ (2.0 $* z), _A3 $+ z, _A3 $+ z, f _A3
      in
      (* analytical inverse *)
      let d11, d12, d21, d22 =
        let z = 1. $/ (m11 * m22) - (m12 * m21) in
        z * m22, neg (z * m12), neg (z * m21), z * m11
      in
      fun (x1, x2) -> (d11 * x1) + (d12 * x2), (d21 * x1) + (d22 * x2)
    in
    let c1, c2 =
      let z = _A2 $* sin theta.pos.x2 in
      z * neg theta.vel.x2 * ((2.0 $* theta.vel.x1) + theta.vel.x2), z * sqr theta.vel.x1
    in
    let h1, h2 = prod_with_b (theta.vel.x1, theta.vel.x2) in
    prod_with_minv (torque1 - c1 - h1, torque2 - c2 - h2)

(* compute hand position from given angular state *)
let hand_of theta =
  let open Maths in
  let joint_x = _L1 $* cos theta.pos.x1 in
  let joint_y = _L1 $* sin theta.pos.x1 in
  let joint_x_dot = neg (_L1 $* theta.vel.x1 * sin theta.pos.x1) in
  let joint_y_dot = _L1 $* theta.vel.x1 * cos theta.pos.x1 in
  let z = theta.pos.x1 + theta.pos.x2 in
  let zdot = theta.vel.x1 + theta.vel.x2 in
  { pos = { x1 = joint_x + (_L2 $* cos z); x2 = joint_y + (_L2 $* sin z) }
  ; vel =
      { x1 = joint_x_dot - (_L2 $* zdot * sin z)
      ; x2 = joint_y_dot + (_L2 $* zdot * cos z)
      }
  }

(* one-step integration of the arm model given current state and momentary torques *)
let integrate ~dt state torque =
  let open Maths in
  let theta_dot_dot1, theta_dot_dot2 = theta_dot_dot state torque in
  { pos =
      { x1 = state.pos.x1 + (dt $* state.vel.x1)
      ; x2 = state.pos.x2 + (dt $* state.vel.x2)
      }
  ; vel =
      { x1 = state.vel.x1 + (dt $* theta_dot_dot1)
      ; x2 = state.vel.x2 + (dt $* theta_dot_dot2)
      }
  }

let central_spot =
  hand_of (theta_init 1)
  |> map ~f:Maths.to_tensor
  |> map ~f:(fun x -> Tensor.(sum x |> to_float0_exn))

let reach_target ?angle ?radius () =
  let angle =
    match angle with
    | Some a -> a
    | None -> Owl_stats.uniform_rvs ~a:(-36.) ~b:(180. +. 36.)
  in
  let angle = Float.(pi * angle / 180.) in
  let radius =
    match radius with
    | Some r -> r
    | None -> Owl_stats.uniform_rvs ~a:0.2 ~b:0.5
  in
  Float.
    { pos =
        { x1 = central_spot.pos.x1 + (radius * cos angle)
        ; x2 = central_spot.pos.x2 + (radius * sin angle)
        }
    ; vel = { x1 = 0.; x2 = 0. }
    }
