(* Test whether ilqr is correct with arm model controlled with MGU2. TODO: this is not correct! *)

open Base
open Torch
open Forward_torch
module Mat = Owl.Dense.Matrix.S
open Sofo

let base = Optimizer.Config.Base.default
let batch_size = 32
let tmax = 10
let a = 4
let b = 2
let dt = 0.01
let conv_threshold = 0.01

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
  Tensor.concat [ lh; rh ] ~dim:1 |> Maths.const

(* let total = Tensor.concat [ lh; rh ] ~dim:2 in *)
(* List.init batch_size ~f:(fun _ -> total) |> Tensor.concat ~dim:0 |> Maths.const *)

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
  { pos = { x1 = as_batch theta1 |> Maths.const; x2 = as_batch theta2 |> Maths.const }
  ; vel = { x1 = as_batch 0. |> Maths.const; x2 = as_batch 0. |> Maths.const }
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
  fun (theta : Maths.t state) (torque1, torque2) ->
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
  |> map ~f:Maths.primal
  |> map ~f:(fun x -> Tensor.(sum x |> to_float0_exn))

let reach_target ?angle ?radius () =
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
  , Float.(central_spot.pos.x2 + (radius * sin angle_)) |> to_tens )

(* ----------------------------------------------------------------------------
    ---     PROBLEM DEFINITION                                       ---
    ---------------------------------------------------------------------------- *)

(* initial angle positions and velocities batched; shape [m x 4 ] *)
let x0_batched =
  let central_state = theta_init batch_size in
  Maths.concat_list
    [ central_state.pos.x1
    ; central_state.pos.x2
    ; central_state.vel.x1
    ; central_state.vel.x2
    ]
    ~dim:1

(* given target positions, calculate target angle positions *)
let target_state (y1, y2) =
  let r_sqr =
    Tensor.(square y1 + square (sub_scalar y2 (Scalar.f central_spot.pos.x2)))
  in
  let x2 =
    let num = Tensor.(sub_scalar r_sqr (Scalar.f Float.(square _L1 + square _L2))) in
    let denom = Tensor.of_float0 ~device:base.device Float.(2. * _L1 * _L2) in
    Tensor.(arccos (num / denom)) |> Tensor.to_float0_exn
  in
  let x1 =
    let tmp1 =
      let num = y1 in
      let denom =
        Tensor.(
          of_float0
            ~device:base.device
            Float.(sqrt (square _L1 + square _L2 + (2. * _L1 * _L2 * cos x2))))
      in
      Tensor.(arccos (num / denom))
    in
    let tmp2 =
      let num = Tensor.(of_float0 ~device:base.device Float.(_L1 + (_L2 * cos x2))) in
      let denom = Tensor.of_float0 ~device:base.device Float.(_L2 * sin x2) in
      Tensor.(arctan (num / denom))
    in
    Tensor.(tmp1 - tmp2)
  in
  x1, Tensor.of_float0 ~device:base.device x2

(* batched targets of shape [m x 4 ] *)
let targets_batched =
  let batched_targets =
    List.init batch_size ~f:(fun _ ->
      let target_pos = reach_target () in
      target_state target_pos)
  in
  let final =
    List.map batched_targets ~f:(fun (x1, x2) ->
      let x1_reshaped = Tensor.reshape x1 ~shape:[ 1; 1 ] in
      let x2_reshaped = Tensor.reshape x2 ~shape:[ 1; 1 ] in
      let vel_targets = Tensor.zeros ~device:base.device ~kind:base.kind [ 1; 1 ] in
      Tensor.concat [ x1_reshaped; x2_reshaped; vel_targets; vel_targets ] ~dim:1
      |> Maths.const)
  in
  Maths.concat_list final ~dim:0

let zeros = Tensor.zeros ~device:base.device ~kind:base.kind [ 1; 1 ]
let ones = Tensor.ones ~device:base.device ~kind:base.kind [ 1; 1 ]
let dt_t = Tensor.of_float0 dt ~device:base.device |> Tensor.reshape ~shape:[ 1; 1 ]

(* f = x(t+1) = P x (t) + Q \dot{x}(t) *)
let _P =
  let row1 = Tensor.concat [ ones; zeros; dt_t; zeros ] ~dim:1 in
  let row2 = Tensor.concat [ zeros; ones; zeros; dt_t ] ~dim:1 in
  let row3 = Tensor.concat [ zeros; zeros; ones; zeros ] ~dim:1 in
  let row4 = Tensor.concat [ zeros; zeros; zeros; ones ] ~dim:1 in
  Tensor.concat [ row1; row2; row3; row4 ] ~dim:0 |> Maths.const

let _Q =
  let row1 = Tensor.concat [ zeros; zeros; zeros; zeros ] ~dim:1 in
  let row2 = row1 in
  let row3 = Tensor.concat [ zeros; zeros; dt_t; zeros ] ~dim:1 in
  let row4 = Tensor.concat [ zeros; zeros; zeros; dt_t ] ~dim:1 in
  Tensor.concat [ row1; row2; row3; row4 ] ~dim:0 |> Maths.const

(* after decomposition, pos1 etc has shape [m x 1 x 1] *)
let decompose_x x =
  let reshape_tmp = Maths.reshape ~shape:[ -1; 1; 1 ] in
  let pos1 = Maths.slice ~dim:1 ~start:(Some 0) ~end_:(Some 1) ~step:1 x in
  let pos2 = Maths.slice ~dim:1 ~start:(Some 1) ~end_:(Some 2) ~step:1 x in
  let vel1 = Maths.slice ~dim:1 ~start:(Some 2) ~end_:(Some 3) ~step:1 x in
  let vel2 = Maths.slice ~dim:1 ~start:(Some 3) ~end_:(Some 4) ~step:1 x in
  reshape_tmp pos1, reshape_tmp pos2, reshape_tmp vel1, reshape_tmp vel2

(* shape [m x 2 x 2] *)
let _M ~pos2 =
  let common = Maths.(_A2 $* cos pos2) in
  let row1 =
    Maths.concat_list [ Maths.(_A1 $+ (2. $* common)); Maths.(_A3 $+ common) ] ~dim:2
  in
  let row2 =
    Maths.concat_list
      [ Maths.(_A3 $+ common)
      ; Maths.(
          const
            Tensor.(
              mul_scalar
                (ones ~device:base.device ~kind:base.kind [ batch_size; 1; 1 ])
                (Scalar.f _A3)))
      ]
      ~dim:2
  in
  Maths.concat_list [ row1; row2 ] ~dim:1

(* shape [m x 2 x 2] *)
let _JM ~pos2 =
  let common = Maths.(neg (_A2 $* sin pos2)) in
  let row1 = Maths.concat_list [ Maths.(2. $* common); common ] ~dim:2 in
  let row2 =
    Maths.concat_list
      [ Maths.(2. $* common)
      ; Maths.(
          const (Tensor.zeros ~device:base.device ~kind:base.kind [ batch_size; 1; 1 ]))
      ]
      ~dim:2
  in
  Maths.concat_list [ row1; row2 ] ~dim:1

(* shape [m x 2 x 4] *)
let _JX ~pos2 ~vel1 ~vel2 =
  let a2_cos_pos2 = Maths.(_A2 $* cos pos2) in
  let a2_sin_pos2 = Maths.(_A2 $* sin pos2) in
  let zeros_tmp =
    Maths.(const (Tensor.zeros ~device:base.device ~kind:base.kind [ batch_size; 1; 1 ]))
  in
  let row1 =
    Maths.concat_list
      [ zeros_tmp
      ; Maths.(neg vel2 * ((2. $* vel1) + vel2) * a2_cos_pos2)
      ; Maths.(neg (2. $* vel2) * a2_sin_pos2)
      ; Maths.(neg (2. $* vel1 + vel2) * a2_sin_pos2)
      ]
      ~dim:2
  in
  let row2 =
    Maths.concat_list
      [ zeros_tmp
      ; Maths.(sqr vel1 * a2_cos_pos2)
      ; Maths.((2. $* vel1) * a2_sin_pos2)
      ; zeros_tmp
      ]
      ~dim:2
  in
  Maths.concat_list [ row1; row2 ] ~dim:1

(* shape [m x 2 x 1] *)
let _X ~pos2 ~vel1 ~vel2 =
  let common = Maths.(_A2 $* sin pos2) in
  let row1 = Maths.(neg vel2 * ((2. $* vel1) + vel2) * common) in
  let row2 = Maths.(sqr vel1 * common) in
  Maths.concat_list [ row1; row2 ] ~dim:1

(* _Fu is partial f/ partial u *)
let _Fu ~x =
  match x with
  | Some x ->
    let _, pos2, vel1, vel2 = decompose_x x in
    let _M = _M ~pos2 in
    let _JM = _JM ~pos2 in
    let _JX = _JX ~pos2 ~vel1 ~vel2 in
    (* TODO: what is the correct way of setting the disturbance? currently I set everything to 0.1 *)
    let _C = Maths.(_M + (0.01 $* _JM)) in
    (* shape [m x 2 x 2] *)
    let _C_inv = Maths.inv_sqr _C in
    (* shape [m x 4 x 2] *)
    let _C_inv_complete =
      let upper =
        Tensor.zeros ~device:base.device ~kind:base.kind [ batch_size; 2; 2 ]
        |> Maths.const
      in
      Maths.concat upper _C_inv ~dim:1
    in
    (* need to transpose since the convention is x = u Fu + x Fx *)
    Maths.(einsum [ _Q, "ab"; _C_inv_complete, "mbc" ] "mac")
    |> Maths.transpose ~dim0:1 ~dim1:2
  | _ ->
    Tensor.zeros ~device:base.device ~kind:base.kind [ batch_size; 2; 4 ] |> Maths.const

(* _Fx is partial f/ partial x *)
let _Fx ~x ~u =
  match x, u with
  | Some x, Some u ->
    let _, pos2, vel1, vel2 = decompose_x x in
    let _M = _M ~pos2 in
    let _JM = _JM ~pos2 in
    let _JX = _JX ~pos2 ~vel1 ~vel2 in
    let _X = _X ~pos2 ~vel1 ~vel2 in
    (* TODO: what is the correct way of setting the disturbance? *)
    let _C = Maths.(_M + (0.01 $* _JM)) in
    (* shape [m x 2 x 2] *)
    let _C_inv = Maths.inv_sqr _C in
    (* shape [m x 4 x 2] *)
    let _C_inv_complete =
      let upper =
        Tensor.zeros ~device:base.device ~kind:base.kind [ batch_size; 2; 2 ]
        |> Maths.const
      in
      Maths.concat upper _C_inv ~dim:1
    in
    let tmp_einsum a b = Maths.einsum [ a, "mab"; b, "mbc" ] "mac" in
    let common =
      Maths.(neg (tmp_einsum _C_inv_complete (Maths.unsqueeze ~dim:0 _B + _JX)))
    in
    let col1 = Maths.slice common ~dim:2 ~start:(Some 0) ~end_:(Some 1) ~step:1 in
    let col3 = Maths.slice common ~dim:2 ~start:(Some 2) ~end_:(Some 3) ~step:1 in
    let col4 = Maths.slice common ~dim:2 ~start:(Some 3) ~end_:(Some 4) ~step:1 in
    let col2 =
      let x_dist =
        Tensor.(
          mul_scalar (ones ~device:base.device ~kind:base.kind [ batch_size; 4; 1 ]))
          (Scalar.f 0.01)
        |> Maths.const
      in
      let _D =
        let _Bx = Maths.einsum [ _B, "ab"; x, "mb" ] "ma" in
        Maths.(unsqueeze ~dim:(-1) (u - _Bx) - (_X + tmp_einsum _JX x_dist))
      in
      let col2_tmp = Maths.slice common ~dim:2 ~start:(Some 1) ~end_:(Some 2) ~step:1 in
      let _JM_complete =
        let left =
          Tensor.zeros ~device:base.device ~kind:base.kind [ batch_size; 2; 2 ]
          |> Maths.const
        in
        Maths.concat left _C_inv ~dim:2
      in
      let tmp1 = Maths.(neg _C_inv_complete *@ _JM_complete *@ _C_inv_complete *@ _D) in
      Maths.(tmp1 + col2_tmp)
    in
    let partial = Maths.concat_list [ col1; col2; col3; col4 ] ~dim:2 in
    Maths.(unsqueeze _P ~dim:0 + einsum [ _Q, "ab"; partial, "mbc" ] "mac")
    |> Maths.transpose ~dim0:1 ~dim1:2
  | _ ->
    Tensor.zeros ~device:base.device ~kind:base.kind [ batch_size; 4; 4 ] |> Maths.const

(* rollout x list under sampled u *)
let rollout_one_step ~x ~u =
  let old_state =
    let pos1 = Maths.slice ~dim:1 ~start:(Some 0) ~end_:(Some 1) ~step:1 x in
    let pos2 = Maths.slice ~dim:1 ~start:(Some 1) ~end_:(Some 2) ~step:1 x in
    let vel1 = Maths.slice ~dim:1 ~start:(Some 2) ~end_:(Some 3) ~step:1 x in
    let vel2 = Maths.slice ~dim:1 ~start:(Some 3) ~end_:(Some 4) ~step:1 x in
    { pos = { x1 = pos1; x2 = pos2 }; vel = { x1 = vel1; x2 = vel2 } }
  in
  let torque =
    let torq1 = Maths.slice ~dim:1 ~start:(Some 0) ~end_:(Some 1) ~step:1 u in
    let torq2 = Maths.slice ~dim:1 ~start:(Some 1) ~end_:(Some 2) ~step:1 u in
    torq1, torq2
  in
  let new_state = integrate ~dt old_state torque in
  let new_x =
    Maths.concat_list
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
let extend_tau_list (tau : Maths.t option Lqr.Solution.p list) =
  let u_list = List.map tau ~f:(fun s -> s.u) in
  let x_list = List.map tau ~f:(fun s -> s.x) in
  let u_ext = u_list @ [ None ] in
  let x_ext = Some x0_batched :: x_list in
  List.map2_exn u_ext x_ext ~f:(fun u x -> Lqr.Solution.{ u; x })

(* given a trajectory and parameters, calculate average cost across batch (summed over time) *)
let cost_func
      ~batch_const
      (tau : Maths.t option Lqr.Solution.p list)
      (p :
        ( Maths.t option
          , (Maths.t, Maths.t -> Maths.t) Lqr.momentary_params list )
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
  cost |> Maths.primal |> Tensor.mean |> Tensor.to_float0_exn

let ilqr ~targets_batched =
  let f_theta = rollout_one_step in
  let params_func (tau : Maths.t option Lqr.Solution.p list)
    : ( Maths.t option
        , (Maths.t, Maths.t -> Maths.t) Lqr.momentary_params list )
        Lqr.Params.p
    =
    let _Cxx = Maths.const (Tensor.eye ~options:(base.kind, base.device) ~n:a) in
    let _Cxx_batched =
      List.init batch_size ~f:(fun _ -> Maths.reshape _Cxx ~shape:[ 1; a; a ])
      |> Maths.concat_list ~dim:0
    in
    let _Cuu_batched =
      List.init batch_size ~f:(fun _ ->
        Maths.reshape
          (Maths.const (Tensor.eye ~options:(base.kind, base.device) ~n:2))
          ~shape:[ 1; b; b ])
      |> Maths.concat_list ~dim:0
    in
    let tau_extended = extend_tau_list tau in
    (* let _cx = Maths.((squeeze ~dim:2 targets_batched) *@ _Cxx) in *)
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
                    (if i = tmax
                     then Some _cx
                     else
                       Some
                         (Maths.const
                            (Tensor.zeros
                               ~kind:base.kind
                               ~device:base.device
                               [ batch_size; a ])))
                ; _cu = None
                ; _Cxx =
                    (if i = tmax
                     then _Cxx_batched
                     else
                       Maths.const
                         (Tensor.zeros
                            ~kind:base.kind
                            ~device:base.device
                            [ batch_size; a; a ]))
                ; _Cxu = None
                ; _Cuu = _Cuu_batched
                })
        }
    in
    Lds_data.map_naive tmp_list ~batch_const:false
  in
  let u_init =
    List.init tmax ~f:(fun _ ->
      let rand = Tensor.randn ~device:base.device ~kind:base.kind [ batch_size; 2 ] in
      Maths.const rand)
  in
  let tau_init = rollout_sol ~u_list:u_init ~x0:x0_batched in
  let sol, _ =
    Ilqr._isolve
      ~laplace:false
      ~f_theta
      ~batch_const:false
      ~cost_func
      ~params_func
      ~conv_threshold
      ~tau_init
  in
  sol

let _ =
  let calc_error x =
    let diff = Maths.(x - targets_batched) |> Maths.primal in
    diff |> Tensor.norm |> Tensor.to_float0_exn
  in
  let init_error = calc_error x0_batched in
  Convenience.print [%message (init_error : float)];
  let sol = ilqr ~targets_batched in
  let final_state =
    let last = List.last_exn sol in
    Option.value_exn last.x
  in
  let error = calc_error final_state in
  Convenience.print [%message (error : float)]