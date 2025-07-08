(* Test whether ilqr is correct with arm model. *)

open Base
open Torch
open Forward_torch
module Mat = Owl.Dense.Matrix.S
open Sofo

let base = Optimizer.Config.Base.default
let batch_size = 5
let tmax = 100
let n = 4
let m = 2
let dt = 0.01
let conv_threshold = 0.001

(* -----------------------------------------
   ----- Utilitiy Functions ------
   ----------------------------------------- *)
let ( +? ) a b =
  match a, b with
  | None, None -> None
  | Some a, None -> Some a
  | None, Some b -> Some b
  | Some a, Some b -> Some Maths.(a + b)

type 'a pair =
  { x1 : 'a
  ; x2 : 'a
  }

(* -----------------------------------------
   ----- Arm Model ------
   ----------------------------------------- *)
(* angular or hand state *)
type 'a state =
  { pos : 'a pair
  ; vel : 'a pair
  }

let map s ~f =
  { pos = { x1 = f s.pos.x1; x2 = f s.pos.x2 }
  ; vel = { x1 = f s.vel.x1; x2 = f s.vel.x2 }
  }

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

let _B =
  let rh =
    Mat.of_array [| _B11; _B12; _B21; _B22 |] 2 2
    |> Tensor.of_bigarray ~device:base.device
  in
  let lh = Tensor.zeros ~device:base.device ~kind:base.kind [ 2; 2 ] in
  Tensor.concat [ lh; rh ] ~dim:1 |> Maths.of_tensor

(* let total = Tensor.concat [ lh; rh ] ~dim:2 in *)
(* List.init batch_size ~f:(fun _ -> total) |> Tensor.concat ~dim:0 |> Maths.of_tensor *)

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
  fun (theta : Maths.any Maths.t state) (torque1, torque2) ->
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

(* let reach_target ?angle ?radius () =
  let angle =
    match angle with
    | Some a -> a
    | None -> Owl_stats.uniform_rvs ~a:(-36.) ~b:(180. +. 36.)
  in
  let angle_ = Float.(pi * angle / 180.) in
  let radius =
    match radius with
    | Some r -> r
    | None -> Owl_stats.uniform_rvs ~a:0.2 ~b:0.5
  in
  let to_tens x =
    x |> Tensor.of_float0 ~device:base.device |> Tensor.reshape ~shape:[ 1; 1 ]
  in
  ( Float.(central_spot.pos.x1 + (radius * cos angle_)) |> to_tens
  , Float.(central_spot.pos.x2 + (radius * sin angle_)) |> to_tens ) *)

(* specify desired angle and reach target directly *)
let reach_target () =
  let a1 =
    (* Owl_stats.uniform_rvs ~a:(-36.) ~b:(180. +. 36.) *)
    Owl_stats.uniform_rvs ~a:0. ~b:45. |> fun x -> Float.(x / 180.)
  in
  let a2 =
    (* Owl_stats.uniform_rvs ~a:(-36.) ~b:(180. +. 36.) *)
    Owl_stats.uniform_rvs ~a:0. ~b:45. |> fun x -> Float.(x / 180.)
  in
  let joint_x = Float.(_L1 * cos a1) in
  let joint_y = Float.(_L1 * sin a1) in
  let z = Float.(a1 + a2) in
  let pos1 = Float.(joint_x + (_L2 * cos z)) in
  let pos2 = Float.(joint_y + (_L2 * sin z)) in
  let to_tens x =
    x |> Tensor.of_float0 ~device:base.device |> Tensor.reshape ~shape:[ 1; 1 ]
  in
  ( a1 |> to_tens
  , a2 |> to_tens
  , Float.(central_spot.pos.x1 + pos1) |> to_tens
  , Float.(central_spot.pos.x2 + pos2) |> to_tens )

(* ----------------------------------------------------------------------------
    ---     PROBLEM DEFINITION                                       ---
    ---------------------------------------------------------------------------- *)

(* initial angle positions and velocities batched; shape [m x 4 ] *)
let x0_batched =
  let central_state = theta_init batch_size in
  Maths.concat
    [ central_state.pos.x1
    ; central_state.pos.x2
    ; central_state.vel.x1
    ; central_state.vel.x2
    ]
    ~dim:1

(* given target positions, calculate target angle positions *)
let target_state (y1, y2) =
  let r_sqr = Tensor.(square y1 + square (y2 - f central_spot.pos.x2)) in
  let x2 =
    let num = Tensor.(r_sqr - f Float.(square _L1 + square _L2)) in
    Tensor.(arccos (num / f Float.(2. * _L1 * _L2))) |> Tensor.to_float0_exn
  in
  let x1 =
    let tmp1 =
      let num = y1 in
      let denom =
        Tensor.(f Float.(sqrt (square _L1 + square _L2 + (2. * _L1 * _L2 * cos x2))))
      in
      Tensor.(arccos (num / denom))
    in
    let tmp2 =
      let num = Tensor.(f Float.(_L1 + (_L2 * cos x2))) in
      let denom = Tensor.f Float.(_L2 * sin x2) in
      Tensor.(arctan (num / denom))
    in
    Tensor.(tmp1 - tmp2)
  in
  x1, Tensor.of_float0 ~device:base.device x2

(* batched targets of shape [m x 4 ] *)
let targets_batched =
  let batched_targets =
    List.init batch_size ~f:(fun _ ->
      let target_a1, target_a2, _, _ = reach_target () in
      target_a1, target_a2)
  in
  let final =
    List.map batched_targets ~f:(fun (x1, x2) ->
      let x1_reshaped = Tensor.reshape x1 ~shape:[ 1; 1 ] in
      let x2_reshaped = Tensor.reshape x2 ~shape:[ 1; 1 ] in
      let vel_targets = Tensor.zeros ~device:base.device ~kind:base.kind [ 1; 1 ] in
      Tensor.concat [ x1_reshaped; x2_reshaped; vel_targets; vel_targets ] ~dim:1
      |> Maths.of_tensor)
  in
  Maths.concat final ~dim:0

let zeros = Tensor.zeros ~device:base.device ~kind:base.kind [ 1; 1 ]
let ones = Tensor.ones ~device:base.device ~kind:base.kind [ 1; 1 ]
let dt_t = Tensor.of_float0 dt ~device:base.device |> Tensor.reshape ~shape:[ 1; 1 ]

(* after decomposition, pos1 etc has shape [m x 1 x 1] *)
let decompose_x x =
  let reshape_tmp = Maths.reshape ~shape:[ -1; 1; 1 ] in
  let pos1 = Maths.slice ~dim:1 ~start:0 ~end_:1 ~step:1 x in
  let pos2 = Maths.slice ~dim:1 ~start:1 ~end_:2 ~step:1 x in
  let vel1 = Maths.slice ~dim:1 ~start:2 ~end_:3 ~step:1 x in
  let vel2 = Maths.slice ~dim:1 ~start:3 ~end_:4 ~step:1 x in
  reshape_tmp pos1, reshape_tmp pos2, reshape_tmp vel1, reshape_tmp vel2

(* after decomposition, pos1 etc has shape [m x 1 x 1] *)
let decompose_u u =
  let reshape_tmp = Maths.reshape ~shape:[ -1; 1; 1 ] in
  let u1 = Maths.slice ~dim:1 ~start:0 ~end_:1 ~step:1 u in
  let u2 = Maths.slice ~dim:1 ~start:1 ~end_:2 ~step:1 u in
  reshape_tmp u1, reshape_tmp u2

(* shape [m x 2 x 2] *)
let _M ~pos2 =
  let common = Maths.(_A2 $* cos pos2) in
  let row1 =
    Maths.concat [ Maths.(_A1 $+ (2. $* common)); Maths.(_A3 $+ common) ] ~dim:2
  in
  let row2 =
    Maths.concat
      [ Maths.(_A3 $+ common)
      ; Maths.(
          of_tensor
            Tensor.(ones ~device:base.device ~kind:base.kind [ batch_size; 1; 1 ] * f _A3))
      ]
      ~dim:2
  in
  Maths.concat [ row1; row2 ] ~dim:1

(* shape [m x 1 x 1] *)
let _P ~pos2 =
  let common = Maths.(_A2 $* cos pos2) in
  Maths.((_A3 $* (_A1 $+ (2. $* common))) - sqr (_A3 $+ (2. $* common)))
  |> Maths.reshape ~shape:[ -1; 1; 1 ]

(* shape [m x 2 x 2] *)
let _M_inv ~pos2 ~_P =
  let common = Maths.(_A2 $* cos pos2) in
  let row1 =
    Maths.concat
      [ Maths.(
          any
            (of_tensor
               Tensor.(
                 ones ~device:base.device ~kind:base.kind [ batch_size; 1; 1 ] * f _A3)))
      ; Maths.(neg (_A3 $+ common))
      ]
      ~dim:2
  in
  let row2 =
    Maths.concat [ Maths.(neg (_A3 $+ common)); Maths.(_A1 $+ (2. $* common)) ] ~dim:2
  in
  let mat = Maths.concat [ row1; row2 ] ~dim:1 in
  Maths.(mat / _P)

(* shape [m x 2 x 1] *)
let _X ~pos2 ~vel1 ~vel2 =
  let common = Maths.(_A2 $* sin pos2) in
  let row1 = Maths.(neg vel2 * ((2. $* vel1) + vel2) * common) in
  let row2 = Maths.(sqr vel1 * common) in
  Maths.concat [ row1; row2 ] ~dim:1

(* _Fu is partial f/ partial u *)
let _Fu ~x =
  match x with
  | Some x ->
    let _, pos2, _, _ = decompose_x x in
    let _P = _P ~pos2 in
    let _M_inv = _M_inv ~pos2 ~_P in
    let zeros =
      Tensor.zeros [ batch_size; 2; 2 ] ~device:base.device ~kind:base.kind
      |> Maths.of_tensor
      |> Maths.any
    in
    (* need to transpose since the convention is x = u Fu + x Fx *)
    Maths.concat [ zeros; _M_inv ] ~dim:1
    |> fun x -> Maths.(dt $* x) |> Maths.transpose ~dims:[ 0; 2;1 ]
  | _ ->
    Tensor.zeros ~device:base.device ~kind:base.kind [ batch_size; 2; 4 ]
    |> Maths.of_tensor
    |> Maths.any

(* _Fx is partial f/ partial x. *)
let _Fx ~(x : Maths.any Maths.t option) ~(u : Maths.any Maths.t option) =
  match x, u with
  | Some x, Some u ->
    let _, pos2, vel1, vel2 = decompose_x x in
    let u1, u2 = decompose_u u in
    let _P = _P ~pos2 in
    let _M_inv = _M_inv ~pos2 ~_P in
    let _X = _X ~pos2 ~vel1 ~vel2 in
    (* first two rows *)
    let row_12 =
      let zeros_tmp =
        Tensor.zeros [ batch_size; 2; 2 ] ~device:base.device ~kind:base.kind
        |> Maths.of_tensor
      in
      let eye_emp =
        List.init batch_size ~f:(fun _ ->
          Tensor.eye ~n:2 ~options:(base.kind, base.device) |> Tensor.unsqueeze ~dim:0)
        |> Tensor.concat ~dim:0
        |> Maths.of_tensor
      in
      Maths.concat [ zeros_tmp; eye_emp ] ~dim:2
    in
    (* decompose x and u into individual components *)
    let h1 =
      let tmp1 = Maths.(_A2 $* sin pos2 * sqr vel1) in
      let tmp2 = Maths.(0.025 $* vel1) in
      let tmp3 = Maths.(0.05 $* vel2) in
      Maths.(u2 - tmp1 - tmp2 - tmp3)
    in
    let h2 =
      let tmp1 = Maths.(_A2 $* sin pos2 * neg vel2 * ((2. $* vel1) + vel2)) in
      let tmp2 = Maths.(0.05 $* vel1) in
      let tmp3 = Maths.(0.025 $* vel2) in
      Maths.(u1 - tmp1 - tmp2 - tmp3)
    in
    let _f = Maths.((_A3 $* h2) - ((_A3 $+ (_A2 $* cos pos2)) * h1)) in
    let row3 =
      let row31 =
        Tensor.zeros ~device:base.device [ batch_size; 1; 1 ]
        |> Maths.of_tensor
        |> Maths.any
      in
      let row32 =
        let tmp1 = Maths.(neg (Float.(_A3 * _A2) $* cos pos2)) in
        let tmp2 = Maths.(_A2 $* sin pos2 * h1) in
        let tmp3 =
          let part1 = Maths.(_A3 $+ (_A2 $* cos pos2)) in
          let part2 = Maths.(_A2 $* cos pos2 * sqr vel1) in
          Maths.(part1 * part2)
        in
        let tmp4 = Maths.(_f * (Float.(2. * square _A2) $* cos vel2 * sin vel2)) in
        Maths.(((tmp1 + tmp2 + tmp3) / _P) - (tmp4 / sqr _P))
      in
      let row33 =
        let tmp1 = Maths.(Float.(_A3 * _A2) $* sin pos2 * vel2) in
        let tmp2 =
          Float.(0.05 * _A3)
          |> Tensor.of_float0 ~device:base.device
          |> Tensor.reshape ~shape:[ 1; 1; 1 ]
          |> Maths.of_tensor
        in
        let tmp3 =
          let part1 = Maths.(_A3 $+ (_A2 $* cos pos2)) in
          let part2 = Maths.((Float.(-2. * _A2) $* sin pos2 * vel1) - f 0.025) in
          Maths.(part1 * part2)
        in
        Maths.((tmp1 - tmp2 - tmp3) / _P)
      in
      let row34 =
        let tmp1 = Maths.(Float.(_A3 * _A2) $* sin pos2 * (2. $* vel1 + vel2)) in
        let tmp2 =
          Float.(0.025 * _A3)
          |> Tensor.of_float0 ~device:base.device
          |> Tensor.reshape ~shape:[ 1; 1; 1 ]
          |> Maths.of_tensor
        in
        let tmp3 = Maths.(0.05 $* (_A3 $+ (_A2 $* cos pos2))) in
        Maths.((tmp1 - tmp2 + tmp3) / _P)
      in
      Maths.concat [ row31; row32; row33; row34 ] ~dim:2
    in
    let _z =
      Maths.(
        ((_A1 $+ (Float.(2. * _A2) $* cos pos2)) * h1) - ((_A3 $+ (_A2 $* cos pos2)) * h2))
    in
    let row4 =
      let row41 =
        Tensor.zeros ~device:base.device [ batch_size; 1; 1 ]
        |> Maths.of_tensor
        |> Maths.any
      in
      let row42 =
        let tmp1 = Maths.((_A2 $* sin pos2) * h2) in
        let tmp2 = Maths.(neg (Float.(2. * _A2) $* sin pos2) * h1) in
        let tmp3 =
          let part1 = Maths.(_A1 $+ (Float.(2. * _A2) $* cos pos2)) in
          let part2 = Maths.(_A2 $* cos pos2 * sqr vel1) in
          Maths.(part1 * part2)
        in
        let tmp4 =
          let part1 = Maths.(_A3 $+ (_A2 $* cos pos2)) in
          let part2 = Maths.(_A2 $* cos pos2 * vel2 * ((2. $* vel1) + vel2)) in
          Maths.(part1 * part2)
        in
        let tmp5 = Maths.(Float.(2. * square _A2) $* _z * cos pos2 * sin pos2) in
        Maths.(((tmp1 + tmp2 + tmp3 - tmp4) / _P) - (tmp5 / sqr _P))
      in
      let row43 =
        let tmp1 =
          Maths.(
            (_A3 $+ (_A2 $* cos pos2)) * ((Float.(2. * _A2) $* sin pos2 * vel2) - f 0.05))
        in
        let tmp2 =
          Maths.(
            (_A1 $+ (Float.(2. * _A2) $* cos pos2))
            * (Float.(neg 2. * _A2) $* (sin pos2 * vel1) - f 0.025))
        in
        Maths.((tmp2 - tmp1) / _P)
      in
      let row44 =
        let tmp1 =
          Maths.(
            (_A3 $+ (_A2 $* cos pos2))
            * ((Float.(2. * _A2) $* sin pos2 * (vel1 + vel2)) - f 0.025))
        in
        let tmp2 = Maths.(neg (0.05 $* (_A1 $+ (Float.(2. * _A2) $* cos pos2)))) in
        Maths.((tmp2 - tmp1) / _P)
      in
      Maths.concat [ row41; row42; row43; row44 ] ~dim:2
    in
    let dx_dot = Maths.concat [ row_12; row3; row4 ] ~dim:1 in
    let eye_ =
      List.init batch_size ~f:(fun _ ->
        Tensor.eye ~n:4 ~options:(base.kind, base.device) |> Tensor.unsqueeze ~dim:0)
      |> Tensor.concat ~dim:0
      |> Maths.of_tensor
      |> Maths.any
    in
    let final = Maths.(eye_ + (dt $* dx_dot)) |> Maths.transpose ~dims:[ 0;  2;1] in
    final
  | _ ->
    Tensor.zeros ~device:base.device ~kind:base.kind [ batch_size; 4; 4 ]
    |> Maths.of_tensor
    |> Maths.any

(* rollout x list under sampled u *)
let rollout_one_step ~x ~u =
  let old_state =
    let pos1 = Maths.slice ~dim:1 ~start:0 ~end_:1 ~step:1 x in
    let pos2 = Maths.slice ~dim:1 ~start:1 ~end_:2 ~step:1 x in
    let vel1 = Maths.slice ~dim:1 ~start:2 ~end_:3 ~step:1 x in
    let vel2 = Maths.slice ~dim:1 ~start:3 ~end_:4 ~step:1 x in
    { pos = { x1 = pos1; x2 = pos2 }; vel = { x1 = vel1; x2 = vel2 } }
  in
  let torque =
    let torq1 = Maths.slice ~dim:1 ~start:0 ~end_:1 ~step:1 u in
    let torq2 = Maths.slice ~dim:1 ~start:1 ~end_:2 ~step:1 u in
    torq1, torq2
  in
  let new_state = integrate ~dt old_state torque in
  let new_x =
    Maths.concat
      [ new_state.pos.x1; new_state.pos.x2; new_state.vel.x1; new_state.vel.x2 ]
      ~dim:1
  in
  new_x

let rollout_sol ~u_list ~x0 =
  let _, x_list =
    List.fold u_list ~init:(x0, []) ~f:(fun (x, accu) u ->
      let new_x = rollout_one_step ~x ~u in
      new_x, Lqr.Solution.{ u = Some u; x = Some new_x } :: accu)
  in
  List.rev x_list

(* artificially add one to tau so it goes from 0 to T *)
let extend_tau_list (tau : Maths.any Maths.t option Lqr.Solution.p list) =
  let u_list = List.map tau ~f:(fun s -> s.u) in
  let x_list = List.map tau ~f:(fun s -> s.x) in
  let u_ext = u_list @ [ None ] in
  let x_ext = Some x0_batched :: x_list in
  List.map2_exn u_ext x_ext ~f:(fun u x -> Lqr.Solution.{ u; x })

(* given a trajectory and parameters, calculate average cost across batch (summed over time) *)
let cost_func
      ~batch_const
      (tau : Maths.any Maths.t option Lqr.Solution.p list)
      (p :
        ( Maths.any Maths.t option
          , ( Maths.any Maths.t
              , Maths.any Maths.t -> Maths.any Maths.t )
              Lqr.momentary_params
              list )
          Lqr.Params.p)
  =
  let tau_extended = extend_tau_list tau in
  let maybe_tmp_einsum_sqr ~batch_const a c b =
    match a, c, b with
    | Some a, Some c, Some b ->
      let c_eqn = if batch_const then "ab" else "mab" in
      Some (Maths.einsum [ a, "ma"; c, c_eqn; b, "mb" ] "m")
    | _ -> None
  in
  let maybe_tmp_einsum ~batch_const a b =
    match a, b with
    | Some a, Some b ->
      let b_eqn = if batch_const then "a" else "ma" in
      Some (Maths.einsum [ a, "ma"; b, b_eqn ] "m")
    | _ -> None
  in
  let cost =
    List.fold2_exn tau_extended p.params ~init:None ~f:(fun accu tau p ->
      let x_sqr_cost = maybe_tmp_einsum_sqr ~batch_const tau.x p.common._Cxx tau.x in
      let u_sqr_cost = maybe_tmp_einsum_sqr ~batch_const tau.u p.common._Cuu tau.u in
      let xu_cost =
        let tmp = maybe_tmp_einsum_sqr ~batch_const tau.x p.common._Cxu tau.u in
        match tmp with
        | None -> None
        | Some x -> Some Maths.(2. $* x)
      in
      let x_cost = maybe_tmp_einsum ~batch_const tau.x p._cx in
      let u_cost = maybe_tmp_einsum ~batch_const tau.u p._cu in
      accu +? (x_sqr_cost +? u_sqr_cost) +? (xu_cost +? (x_cost +? u_cost)))
    |> Option.value_exn
  in
  cost |> Maths.to_tensor |> Tensor.mean |> Tensor.to_float0_exn

let map_implicit
      (x :
        ( Maths.any Maths.t option
          , (Maths.any Maths.t, Maths.any Maths.t option) Lds_data.Temp.p list )
          Lqr.Params.p)
      ~batch_const
  =
  let params =
    List.map x.params ~f:(fun p ->
      Lqr.
        { common =
            { _Fx_prod = Some (Lds_data.prod ~batch_const p._Fx_prod)
            ; _Fx_prod2 = Some (Lds_data.prod2 ~batch_const p._Fx_prod)
            ; _Fu_prod = Some (Lds_data.prod ~batch_const p._Fu_prod)
            ; _Fu_prod2 = Some (Lds_data.prod2 ~batch_const p._Fu_prod)
            ; _Fx_prod_tangent = Some (Lds_data.prod_tangent ~batch_const p._Fx_prod)
            ; _Fx_prod2_tangent = Some Lds_data.(prod2_tangent ~batch_const p._Fx_prod)
            ; _Fu_prod_tangent = Some (Lds_data.prod_tangent ~batch_const p._Fu_prod)
            ; _Fu_prod2_tangent = Some (Lds_data.prod2_tangent ~batch_const p._Fu_prod)
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
          , (Maths.any Maths.t, Maths.any Maths.t option) Lds_data.Temp.p list )
          Lqr.Params.p)
      ~batch_const
  =
  let irrelevant = Some (fun _ -> assert false) in
  let params =
    List.map x.params ~f:(fun p ->
      Lqr.
        { common =
            { _Fx_prod = Some (Lds_data.bmm ~batch_const p._Fx_prod)
            ; _Fx_prod2 = Some (Lds_data.bmm2 ~batch_const p._Fx_prod)
            ; _Fu_prod = Some (Lds_data.bmm ~batch_const p._Fu_prod)
            ; _Fu_prod2 = Some (Lds_data.bmm2 ~batch_const p._Fu_prod)
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

let ilqr ~targets_batched =
  let f_theta = rollout_one_step in
  let params_func (tau : Maths.any Maths.t option Lqr.Solution.p list)
    : ( Maths.any Maths.t option
        , (Maths.any Maths.t, Maths.any Maths.t -> Maths.any Maths.t) Lqr.momentary_params
            list )
        Lqr.Params.p
    =
    let _Cxx = Maths.of_tensor (Tensor.eye ~options:(base.kind, base.device) ~n) in
    let _Cxx_batched =
      List.init batch_size ~f:(fun _ -> Maths.reshape _Cxx ~shape:[ 1; n; n ])
      |> Maths.concat ~dim:0
    in
    let _Cuu_batched =
      List.init batch_size ~f:(fun _ ->
        Maths.reshape
          (Maths.of_tensor Tensor.(f 0.1 * eye ~options:(base.kind, base.device) ~n:2))
          ~shape:[ 1; m; m ])
      |> Maths.concat ~dim:0
    in
    let tau_extended = extend_tau_list tau in
    let _cx = Maths.einsum [ targets_batched, "ma"; _Cxx_batched, "mab" ] "mb" in
    let tmp_list =
      Lqr.Params.
        { x0 = Some x0_batched
        ; params =
            List.mapi tau_extended ~f:(fun i s ->
    

              Lds_data.Temp.
                { _f = None
                ; _Fx_prod = _Fx ~x:s.x ~u:s.u
                ; _Fu_prod = _Fu ~x:s.x
                ; _cx =
                    Some _cx
                    (* (if i = tmax
                     then Some _cx
                     else
                       Some
                         (Maths.of_tensor
                            (Tensor.zeros
                               ~kind:base.kind
                               ~device:base.device
                               [ batch_size; a ]))) *)
                ; _cu = None
                ; _Cxx =
                    _Cxx_batched
                    (* (if i = tmax
                     then _Cxx_batched
                     else
                       Maths.of_tensor
                         (Tensor.zeros
                            ~kind:base.kind
                            ~device:base.device
                            [ batch_size; a; a ])) *)
                ; _Cxu = None
                ; _Cuu = _Cuu_batched
                })
        }
    in
    map_naive tmp_list ~batch_const:false
  in
  let u_init =
    List.init tmax ~f:(fun _ ->
      let rand = Tensor.randn ~device:base.device ~kind:base.kind [ batch_size; 2 ] in
      Maths.any (Maths.of_tensor rand))
  in
  let tau_init = rollout_sol ~u_list:u_init ~x0:x0_batched in
  let sol, _ =
    Ilqr._isolve
      ~f_theta
      ~batch_const:false
      ~gamma:1.
      ~cost_func
      ~params_func
      ~conv_threshold
      ~tau_init
      ~max_iter:200
  in
  sol

let _ =
  let calc_error x =
    let diff = Maths.(x - targets_batched) |> Maths.to_tensor in
    diff |> Tensor.norm |> Tensor.to_float0_exn
  in
  let init_error = calc_error x0_batched in
  Sofo.print [%message (init_error : float)];
  let sol = ilqr ~targets_batched in
  let final_state =
    let last = List.last_exn sol in
    Option.value_exn last.x
  in
  let final_error = calc_error final_state in
  Sofo.print [%message (final_error : float)]
