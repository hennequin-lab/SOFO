(* Test whether ilqr is correct with arm model. *)

open Base
open Torch
open Forward_torch
module Arr = Owl.Dense.Ndarray.S
module Mat = Owl.Dense.Matrix.S
open Sofo

let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default
let bs = 2
let tmax = 5000
let n = 4
let m = 2
let dt = 0.001
let conv_threshold = 1e-4

(* -----------------------------------------
   ----- Arm Model ------
   ----------------------------------------- *)

type 'a pair =
  { x1 : 'a
  ; x2 : 'a
  }

(* angular or hand state *)
type 'a state =
  { pos : 'a pair
  ; vel : 'a pair
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

(* distances *)
let _S1 = 0.11
let _S2 = 0.16

(* derived constants *)
let _A1 = _I1 +. _I2 +. (_M2 *. Float.square _L1)
let _A2 = _M2 *. _L1 *. _S2
let _A3 = _I2

(* initial angular pos and vel *)
let theta_init bs =
  let a = 20. in
  let theta1 = Float.(a * pi / 180.) in
  let theta2 = Float.((180. - (2. * a)) * pi / 180.) in
  let as_batch x = Tensor.(ones ~device:base.device [ bs; 1 ] * f x) in
  { pos =
      { x1 = as_batch theta1 |> Maths.of_tensor |> Maths.any
      ; x2 = as_batch theta2 |> Maths.of_tensor |> Maths.any
      }
  ; vel =
      { x1 = as_batch 0. |> Maths.of_tensor |> Maths.any
      ; x2 = as_batch 0. |> Maths.of_tensor |> Maths.any
      }
  }

(* compute hand pos given angular pos *)
let hand_pos_of pos_x1 pos_x2 =
  let open Maths in
  let joint_x = _L1 $* cos pos_x1 in
  let joint_y = _L1 $* sin pos_x1 in
  let z = pos_x1 + pos_x2 in
  joint_x + (_L2 $* cos z), joint_y + (_L2 $* sin z)

(* compute hand vel given angular pos and angular vel *)
(* let hand_vel_of pos_x1 pos_x2 vel_x1 vel_x2 =
  let open Maths in
  let joint_x_dot = neg (_L1 $* vel_x1 * sin pos_x1) in
  let joint_y_dot = _L1 $* vel_x1 * cos pos_x1 in
  let z = pos_x1 + pos_x2 in
  let zdot = vel_x1 + vel_x2 in
  joint_x_dot - (_L2 $* zdot * sin z), joint_y_dot + (_L2 $* zdot * cos z) *)

(* compute hand pos and velocity given angular state *)
(* let hand_of theta =
  let open Maths in
  let pos_x1, pos_x2 = hand_pos_of theta.pos.x1 theta.pos.x2 in
  let vel_x1, vel_x2 = hand_vel_of theta.pos.x1 theta.pos.x2 theta.vel.x1 theta.vel.x2 in
  pos_x1, pos_x2, vel_x1, vel_x2 *)

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

(* one-step integration of the angular pos and vel given current state and momentary torques *)
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

(* specify target angles and pos directly *)
let reach_target () =
  let a1 =
    Owl_stats.uniform_rvs ~a:(-36.) ~b:(180. +. 36.) |> fun x -> Float.(x * pi / 180.)
  in
  let a2 =
    Owl_stats.uniform_rvs ~a:(-36.) ~b:(180. +. 36.) |> fun x -> Float.(x * pi / 180.)
  in
  let joint_x = Float.(_L1 * cos a1) in
  let joint_y = Float.(_L1 * sin a1) in
  let z = Float.(a1 + a2) in
  let pos1 = Float.(joint_x + (_L2 * cos z)) in
  let pos2 = Float.(joint_y + (_L2 * sin z)) in
  let to_tens x =
    x |> Tensor.of_float0 ~device:base.device |> Tensor.reshape ~shape:[ 1; 1 ]
  in
  a1 |> to_tens, a2 |> to_tens, pos1 |> to_tens, pos2 |> to_tens

(* ----------------------------------------------------------------------------
    ---     PROBLEM DEFINITION                                       ---
    ---------------------------------------------------------------------------- *)
let _Cxx_batched =
  let _Cxx = Maths.(any (of_tensor (Tensor.eye ~options:(base.kind, base.device) ~n))) in
  Maths.broadcast_to _Cxx ~size:[ bs; n; n ]

let _Cuu_batched =
  let _Cuu =
    Maths.(any (of_tensor Tensor.(f 1. * eye ~options:(base.kind, base.device) ~n:m)))
  in
  Maths.broadcast_to _Cuu ~size:[ bs; m; m ]

(* initial angle positions and velocities batched; shape [bs x 4 ] *)
let angles_init = theta_init bs

let x0_batched =
  let central_state = angles_init in
  Maths.concat
    [ central_state.pos.x1
    ; central_state.pos.x2
    ; central_state.vel.x1
    ; central_state.vel.x2
    ]
    ~dim:1

(* batched targets of shape [bs x 4]; angular positions and velocities *)
let targets_batched, targets_hand_pos_batched =
  let batched_targets_handpos =
    List.init bs ~f:(fun _ ->
      let target_a1, target_a2, target_hand_pos1, target_hand_pos2 = reach_target () in
      let reshape_ = Tensor.reshape ~shape:[ 1; 1 ] in
      let target_a1 = reshape_ target_a1
      and target_a2 = reshape_ target_a2
      and target_hand_pos1 = reshape_ target_hand_pos1
      and target_hand_pos2 = reshape_ target_hand_pos2 in
      let vel_targets = Tensor.zeros ~device:base.device ~kind:base.kind [ 1; 1 ] in
      ( Maths.of_tensor
          (Tensor.concat [ target_a1; target_a2; vel_targets; vel_targets ] ~dim:1)
      , Maths.of_tensor (Tensor.concat [ target_hand_pos1; target_hand_pos2 ] ~dim:1) ))
  in
  ( Maths.concat (List.map batched_targets_handpos ~f:fst) ~dim:0
  , Maths.concat (List.map batched_targets_handpos ~f:snd) ~dim:0 )

(* let _ = Tensor.print (Maths.to_tensor x0_batched)
let _ = Tensor.print (Maths.to_tensor targets_batched) *)

(* after decomposition, pos1 etc has shape [bs x 1 x 1] *)
let decompose_x x =
  let reshape_tmp = Maths.reshape ~shape:[ -1; 1; 1 ] in
  let pos1 = Maths.slice ~dim:1 ~start:0 ~end_:1 x in
  let pos2 = Maths.slice ~dim:1 ~start:1 ~end_:2 x in
  let vel1 = Maths.slice ~dim:1 ~start:2 ~end_:3 x in
  let vel2 = Maths.slice ~dim:1 ~start:3 ~end_:4 x in
  reshape_tmp pos1, reshape_tmp pos2, reshape_tmp vel1, reshape_tmp vel2

(* after decomposition, pos1 etc has shape [bs x 1 x 1] *)
let decompose_u u =
  let reshape_tmp = Maths.reshape ~shape:[ -1; 1; 1 ] in
  let u1 = Maths.slice ~dim:1 ~start:0 ~end_:1 u in
  let u2 = Maths.slice ~dim:1 ~start:1 ~end_:2 u in
  reshape_tmp u1, reshape_tmp u2

(* shape [bs x 2 x 2] *)
let _M ~pos2 =
  let common = Maths.(_A2 $* cos pos2) in
  let tmp = Maths.(_A3 $+ common) in
  let row1 = Maths.concat [ Maths.(_A1 $+ (2. $* common)); tmp ] ~dim:2 in
  let row2 =
    Maths.concat
      [ tmp
      ; Maths.(
          any
            (of_tensor
               Tensor.(ones ~device:base.device ~kind:base.kind [ bs; 1; 1 ] * f _A3)))
      ]
      ~dim:2
  in
  Maths.concat [ row1; row2 ] ~dim:1

(* shape [bs x 1 x 1] *)
let _P ~pos2 =
  let common = Maths.(_A2 $* cos pos2) in
  Maths.((_A3 $* (_A1 $+ (2. $* common))) - sqr (_A3 $+ common))
  |> Maths.reshape ~shape:[ -1; 1; 1 ]

(* shape [bs x 2 x 2] *)
let _M_inv ~pos2 ~_P =
  let common = Maths.(_A2 $* cos pos2) in
  let tmp = Maths.(neg (_A3 $+ common)) in
  let row1 =
    Maths.concat
      [ Maths.(
          any
            (of_tensor
               Tensor.(ones ~device:base.device ~kind:base.kind [ bs; 1; 1 ] * f _A3)))
      ; tmp
      ]
      ~dim:2
  in
  let row2 = Maths.concat [ tmp; Maths.(_A1 $+ (2. $* common)) ] ~dim:2 in
  let mat = Maths.concat [ row1; row2 ] ~dim:1 in
  Maths.(mat / _P)

(* shape [bs x 2 x 1] *)
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
      Tensor.zeros [ bs; 2; 2 ] ~device:base.device ~kind:base.kind
      |> Maths.of_tensor
      |> Maths.any
    in
    (* need to transpose since the convention is x = u Fu + x Fx *)
    Maths.concat [ zeros; _M_inv ] ~dim:1
    |> fun x -> Maths.(dt $* x) |> Maths.transpose ~dims:[ 0; 2; 1 ]
  | _ ->
    Tensor.zeros ~device:base.device ~kind:base.kind [ bs; 2; 4 ]
    |> Maths.of_tensor
    |> Maths.any

(* _Fx is partial f/ partial x. *)
let _Fx ~(x : Maths.any Maths.t option) ~(u : Maths.any Maths.t option) =
  match x, u with
  | Some x, Some u ->
    let _, pos2, vel1, vel2 = decompose_x x in
    let u1, u2 = decompose_u u in
    let _P = _P ~pos2 in
    let _M = _M ~pos2 in
    let _M_inv = _M_inv ~pos2 ~_P in
    let _X = _X ~pos2 ~vel1 ~vel2 in
    (* first two rows *)
    let row_12 =
      let zeros_tmp =
        Tensor.zeros [ bs; 2; 2 ] ~device:base.device ~kind:base.kind |> Maths.of_tensor
      in
      let eye_emp =
        List.init bs ~f:(fun _ ->
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
        Tensor.zeros ~device:base.device [ bs; 1; 1 ] |> Maths.of_tensor |> Maths.any
      in
      let row32 =
        let tmp1 =
          Maths.(Float.(_A3 * _A2) $* cos pos2 * vel2 * ((f 2. * vel1) + vel2))
        in
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
        let tmp1 = Maths.(Float.(2. * _A3 * _A2) $* sin pos2 * vel2) in
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
        Tensor.zeros ~device:base.device [ bs; 1; 1 ] |> Maths.of_tensor |> Maths.any
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
          let part2 = Maths.(_A2 $* cos pos2 * neg vel2 * ((2. $* vel1) + vel2)) in
          Maths.(part1 * part2)
        in
        let tmp5 = Maths.(Float.(2. * square _A2) $* _z * cos pos2 * sin pos2) in
        Maths.(((tmp1 + tmp2 - tmp3 + tmp4) / _P) - (tmp5 / sqr _P))
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
      Tensor.eye ~n ~options:(base.kind, base.device)
      |> Tensor.broadcast_to ~size:[ bs; n; n ]
      |> Maths.of_tensor
      |> Maths.any
    in
    let final = Maths.(eye_ + (dt $* dx_dot)) |> Maths.transpose ~dims:[ 0; 2; 1 ] in
    final
  | _ ->
    Tensor.zeros ~device:base.device ~kind:base.kind [ bs; 4; 4 ]
    |> Maths.of_tensor
    |> Maths.any

(* rollout x list under sampled u *)
let rollout_one_step ~x ~u =
  let old_state =
    let pos1 = Maths.slice ~dim:1 ~start:0 ~end_:1 x in
    let pos2 = Maths.slice ~dim:1 ~start:1 ~end_:2 x in
    let vel1 = Maths.slice ~dim:1 ~start:2 ~end_:3 x in
    let vel2 = Maths.slice ~dim:1 ~start:3 ~end_:4 x in
    { pos = { x1 = pos1; x2 = pos2 }; vel = { x1 = vel1; x2 = vel2 } }
  in
  let torque =
    let torq1 = Maths.slice ~dim:1 ~start:0 ~end_:1 u in
    let torq2 = Maths.slice ~dim:1 ~start:1 ~end_:2 u in
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

let opt_const_map x = Option.map x ~f:(fun x -> Maths.(any (const x)))

let map_to_const (tau : Maths.any Maths.t option Lqr.Solution.p list) =
  List.map tau ~f:(fun tau ->
    Lqr.Solution.{ u = opt_const_map tau.u; x = opt_const_map tau.x })

(* given a trajectory, calculate average cost across batch (summed over time) *)
let cost_func (tau : Maths.any Maths.t option Lqr.Solution.p list) =
  let x_list = List.map tau ~f:(fun s -> s.x |> Option.value_exn) in
  let u_list = List.map tau ~f:(fun s -> s.u |> Option.value_exn) in
  let x_cost =
    let x_cost_lst =
      List.map x_list ~f:(fun x ->
        Maths.(
          einsum
            [ x - targets_batched, "ma"; _Cxx_batched, "mab"; x - targets_batched, "mb" ]
            "m"))
    in
    List.fold x_cost_lst ~init:Maths.(any (f 0.)) ~f:(fun accu c -> Maths.(accu + c))
  in
  let u_cost =
    List.fold
      u_list
      ~init:Maths.(any (f 0.))
      ~f:(fun accu u ->
        Maths.(accu + einsum [ u, "ma"; _Cuu_batched, "mab"; u, "mb" ] "m"))
  in
  Maths.(x_cost + u_cost) |> Maths.to_tensor

let ilqr ~targets_batched =
  let f_theta ~i:_ = rollout_one_step in
  let params_func ~no_tangents (tau : Maths.any Maths.t option Lqr.Solution.p list)
    : ( Maths.any Maths.t option
        , (Maths.any Maths.t, Maths.any Maths.t -> Maths.any Maths.t) Lqr.momentary_params
            list )
        Lqr.Params.p
    =
    let tau_extended =
      let tmp = extend_tau_list tau in
      if no_tangents then map_to_const tmp else tmp
    in
    let tmp_list =
      Lqr.Params.
        { x0 = Some x0_batched
        ; params =
            List.map tau_extended ~f:(fun s ->
              let _cu =
                match s.u with
                | None -> None
                | Some u -> Some Maths.(einsum [ u, "ma"; _Cuu_batched, "mab" ] "mb")
              in
              let _cx =
                Maths.(
                  einsum
                    [ Option.value_exn s.x - targets_batched, "ma"; _Cxx_batched, "mab" ]
                    "mb")
              in
              Lds_data.Temp.
                { _f = None
                ; _Fx_prod = _Fx ~x:s.x ~u:s.u
                ; _Fu_prod = _Fu ~x:s.x
                ; _cx = Some _cx
                ; _cu
                ; _Cxx = _Cxx_batched
                ; _Cxu = None
                ; _Cuu = _Cuu_batched
                })
        }
    in
    Lds_data.map_naive tmp_list ~batch_const:false
  in
  let u_init =
    List.init tmax ~f:(fun _ ->
      let rand = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; m ] in
      Maths.any (Maths.of_tensor rand))
  in
  let tau_init = rollout_sol ~u_list:u_init ~x0:x0_batched in
  let sol, _ =
    Ilqr._isolve
      ~linesearch:true
      ~linesearch_bs_avg:false
      ~expected_reduction:true
      ~batch_const:false
      ~f_theta
      ~gamma:0.5
      ~cost_func
      ~params_func
      ~conv_threshold
      ~tau_init
      2000
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
  let to_bigarray x = x |> Maths.to_tensor |> Tensor.to_bigarray ~kind:base.ba_kind in
  let inferred_u_mat =
    let inferred_us = List.map sol ~f:(fun x -> x.u) in
    List.map inferred_us ~f:(fun u -> Arr.reshape (to_bigarray u) [| 1; bs; m |])
    |> List.to_array
    |> Arr.concatenate ~axis:0
  in
  (* initial hand positions; shape [bs x 2] *)
  let x0_pos_batched =
    let hand_pos_x1, hand_pos_x2 = hand_pos_of angles_init.pos.x1 angles_init.pos.x2 in
    Maths.concat [ hand_pos_x1; hand_pos_x2 ] ~dim:1
  in
  let inferred_x_mat =
    let inferred_xs = List.map sol ~f:(fun x -> x.x) in
    List.fold
      inferred_xs
      ~init:[ Maths.unsqueeze ~dim:0 x0_pos_batched ]
      ~f:(fun accu inferred_x ->
        let open Maths in
        let hand_angle_x = Maths.slice inferred_x ~dim:1 ~start:0 ~end_:1 in
        let hand_angle_y = Maths.slice inferred_x ~dim:1 ~start:1 ~end_:2 in
        let pos_x, pos_y = hand_pos_of hand_angle_x hand_angle_y in
        unsqueeze ~dim:0 ((concat [ pos_x; pos_y ]) ~dim:1) :: accu)
    |> List.rev
    |> Maths.concat ~dim:0
    |> to_bigarray
  in
  (* save inferred, start and target hand pos, and inferred control inputs*)
  Arr.(save_npy ~out:(in_dir "inferred_x") inferred_x_mat);
  Mat.(save_npy ~out:(in_dir "targets_x") (to_bigarray targets_hand_pos_batched));
  Mat.(save_npy ~out:(in_dir "inferred_u") inferred_u_mat);
  Mat.(save_npy ~out:(in_dir "x0") (to_bigarray x0_pos_batched));
  let final_state =
    let last = List.last_exn sol in
    last.x
  in
  let final_error = calc_error final_state in
  Sofo.print [%message (final_error : float)]
