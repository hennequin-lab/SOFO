(* Test whether ilqr is correct with the Lorenz attractor. *)
open Printf
open Base
open Domainslib
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let _ =
  Random.init 1985;
  (* Random.self_init (); *)
  Owl_stats_prng.init 1985;
  Torch_core.Wrapper.manual_seed 1985

let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default
let conv_threshold = 0.00001

(* -----------------------------------------
   -- Generate Lorenz data            ------
   ----------------------------------------- *)

let bs = 4
let tmax = 100
let dt = 0.01
let u_threshold = 0.01

(* state and control dims both 3. *)
let n, m = 3, 3
let sigma, rho, beta = 10., 28., 8. /. 3.

let lorenz_step ?(u = Mat.zeros 1 3) ~dt x =
  let x = Mat.get x 0 0
  and y = Mat.get x 0 1
  and z = Mat.get x 0 2 in
  let u_x = Mat.get u 0 0
  and u_y = Mat.get u 0 1
  and u_z = Mat.get u 0 2 in
  let x_tt = (dt *. ((sigma *. (y -. x)) +. u_x)) +. x
  and y_tt = (dt *. ((x *. (rho -. z)) -. y +. u_y)) +. y
  and z_tt = (dt *. ((x *. y) -. (beta *. z) +. u_z)) +. z in
  Mat.(of_array [| x_tt; y_tt; z_tt |] 1 3)

(* [lorenz_flow] has shape [tmax+1 x bs x 3]*)
let gen_lorenz ~dt x0 u_array =
  let x0_array = Mat.to_rows x0 in
  let lorenz_flow =
    Array.map2_exn x0_array u_array ~f:(fun x0 u ->
      (* x0 has shape [1 x 3], u of shape [tmax x 3] *)
      let u_t_array = Mat.to_rows u in
      let x_list =
        Array.fold u_t_array ~init:[ x0 ] ~f:(fun accu u_t ->
          let x_next = lorenz_step ~u:u_t ~dt (List.hd_exn accu) in
          x_next :: accu)
        |> List.rev
      in
      (* x_mat has shape [(tmax+1) x 3] *)
      let x_mat = x_list |> List.to_array |> Mat.concatenate ~axis:0 in
      Mat.reshape x_mat [| tmax + 1; 1; 3 |])
    |> Arr.concatenate ~axis:1
  in
  lorenz_flow

(* list of length (tmax+1), each tensor of shape [bs x 3] *)
let get_batch data =
  List.init (tmax + 1) ~f:(fun i ->
    Arr.(squeeze (get_slice [ [ i ] ] data)) |> Tensor.of_bigarray ~device:base.device)

(* u_array has length bs, each mat is of shape [tmax x 3] *)
let u_array =
  Array.init bs ~f:(fun _ ->
    let input =
      Array.init (tmax * 3) ~f:(fun _ ->
        if Float.(Random.float 1.0 < u_threshold)
        then Float.(1. * Owl_stats_prng.rand_gaussian ())
        else 0.)
    in
    Arr.of_array input [| tmax; 3 |])

let x0 = Mat.gaussian bs 3

let x0_maths =
  x0 |> Tensor.of_bigarray ~device:base.device |> Maths.of_tensor |> Maths.any

(* generate trajectory *)
let observation =
  gen_lorenz ~dt x0 u_array
  |> get_batch
  |> List.map ~f:(fun x -> x |> Maths.of_tensor |> Maths.any)

(* -----------------------------------------
   -- iLQR params         ------
   ----------------------------------------- *)
let _Cxx_batched =
  let _Cxx = Maths.(any (of_tensor (Tensor.eye ~options:(base.kind, base.device) ~n))) in
  Maths.broadcast_to _Cxx ~size:[ bs; n; n ]

let _Cuu_batched =
  let _Cuu =
    Maths.(any (of_tensor Tensor.(f 1e-4 * eye ~options:(base.kind, base.device) ~n:m)))
  in
  Maths.broadcast_to _Cuu ~size:[ bs; m; m ]

(* _Fu is partial f/ partial u *)
let _Fu =
  Tensor.(f dt * eye ~n ~options:(base.kind, base.device))
  |> Tensor.broadcast_to ~size:[ bs; n; n ]
  |> Maths.of_tensor
  |> Maths.any

(* _Fx is partial f/ partial x. *)
let _Fx ~(x : Maths.any Maths.t option) =
  let x = Option.value_exn x in
  let x_t = Maths.slice ~start:0 ~end_:1 ~dim:1 x
  and y_t = Maths.slice ~start:1 ~end_:2 ~dim:1 x
  and z_t = Maths.slice ~start:2 ~end_:3 ~dim:1 x in
  let row1 =
    Mat.of_array [| Float.(1. - (dt * sigma)); dt *. sigma; 0. |] 1 3
    |> Tensor.of_bigarray ~device:base.device
    |> Tensor.broadcast_to ~size:[ bs; 1; 3 ]
    |> Maths.of_tensor
    |> Maths.any
  in
  let row2 =
    let row21 = Maths.(dt $* f rho - z_t) in
    let row22 =
      Tensor.of_bigarray ~device:base.device (Mat.of_array [| Float.(1. -. dt) |] 1 1)
      |> Maths.of_tensor
      |> Maths.broadcast_to ~size:[ bs; 1 ]
      |> Maths.any
    in
    let row23 = Maths.(neg (dt $* x_t)) in
    Maths.concat ~dim:1 [ row21; row22; row23 ] |> Maths.reshape ~shape:[ bs; 1; 3 ]
  in
  let row3 =
    let row31 = Maths.(dt $* y_t) in
    let row32 = Maths.(dt $* x_t) in
    let row33 =
      Tensor.of_bigarray
        ~device:base.device
        (Mat.of_array [| Float.(1. -. (beta *. dt)) |] 1 1)
      |> Maths.of_tensor
      |> Maths.broadcast_to ~size:[ bs; 1 ]
      |> Maths.any
    in
    Maths.concat ~dim:1 [ row31; row32; row33 ] |> Maths.reshape ~shape:[ bs; 1; 3 ]
  in
  (* transpose since notation follows x = x Fx + u Fu *)
  Maths.concat [ row1; row2; row3 ] ~dim:1 |> Maths.transpose ~dims:[ 0; 2; 1 ]

(* rollout x list under u in batch mode *)
let rollout_one_step ~x ~u =
  let x_t = Maths.slice ~start:0 ~end_:1 ~dim:1 x
  and y_t = Maths.slice ~start:1 ~end_:2 ~dim:1 x
  and z_t = Maths.slice ~start:2 ~end_:3 ~dim:1 x in
  let u_x = Maths.slice ~start:0 ~end_:1 ~dim:1 u
  and u_y = Maths.slice ~start:1 ~end_:2 ~dim:1 u
  and u_z = Maths.slice ~start:2 ~end_:3 ~dim:1 u in
  let x_tt = Maths.((dt $* (f sigma * (y_t - x_t)) + u_x) + x_t)
  and y_tt = Maths.((dt $* (x_t * (f rho - z_t)) - y_t + u_y) + y_t)
  and z_tt = Maths.((dt $* (x_t * y_t) - (f beta * z_t) + u_z) + z_t) in
  Maths.concat [ x_tt; y_tt; z_tt ] ~dim:1

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
  let x_ext = Some x0_maths :: x_list in
  List.map2_exn u_ext x_ext ~f:(fun u x -> Lqr.Solution.{ u; x })

(* given a trajectory calculate average cost across batch (summed over time) *)
let cost_func (tau : Maths.any Maths.t option Lqr.Solution.p list) =
  let x_list = List.map tau ~f:(fun s -> s.x |> Option.value_exn) in
  let u_list = List.map tau ~f:(fun s -> s.u |> Option.value_exn) in
  let x_cost =
    let x_cost_lst =
      List.map2_exn x_list (List.tl_exn observation) ~f:(fun x o ->
        Maths.(einsum [ x - o, "ma"; _Cxx_batched, "mab"; x - o, "mb" ] "m"))
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
  Maths.(x_cost + u_cost) |> Maths.to_tensor |> Tensor.mean |> Tensor.to_float0_exn

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

let ilqr ~observation =
  let f_theta = rollout_one_step in
  let params_func (tau : Maths.any Maths.t option Lqr.Solution.p list)
    : ( Maths.any Maths.t option
        , (Maths.any Maths.t, Maths.any Maths.t -> Maths.any Maths.t) Lqr.momentary_params
            list )
        Lqr.Params.p
    =
    let tau_extended = extend_tau_list tau in
    let tmp_list =
      (* TODO: should we assume x0=0 as in the original paper? *)
      Lqr.Params.
        { x0 =
            (* Some Maths.(any (zeros_like x0_maths)) *)
            Some x0_maths
        ; params =
            List.map2_exn tau_extended observation ~f:(fun tau o ->
              let _cx = Maths.(neg (einsum [ o, "ma"; _Cxx_batched, "mab" ] "mb")) in
              Lds_data.Temp.
                { _f = None
                ; _Fx_prod = _Fx ~x:tau.x
                ; _Fu_prod = _Fu
                ; _cx = Some _cx
                ; _cu = None
                ; _Cxx = _Cxx_batched
                ; _Cxu = None
                ; _Cuu = _Cuu_batched
                })
        }
    in
    map_naive tmp_list ~batch_const:false
  in
  let u_init =
    List.init tmax ~f:(fun _ ->
      let rand = Tensor.(zeros ~device:base.device ~kind:base.kind [ bs; 3 ]) in
      Maths.any (Maths.of_tensor rand))
  in
  let tau_init = rollout_sol ~u_list:u_init ~x0:x0_maths in
  let sol, _ =
    Ilqr._isolve
      ~f_theta
      ~batch_const:false
      ~gamma:0.9
      ~cost_func
      ~params_func
      ~conv_threshold
      ~tau_init
      2000
  in
  sol

(* -----------------------------------------
   -- iLQR pass        ------
   ----------------------------------------- *)
let _ =
  let sol = ilqr ~observation in
  let inferred_us = List.map sol ~f:(fun x -> x.u |> Option.value_exn) in
  (* u_mat and inferred_u_mat shape [tmax x bs x 3] *)
  let u_mat =
    let tmp = Array.map u_array ~f:(fun u_a -> Arr.reshape u_a [| 1; tmax; 3 |]) in
    Arr.concatenate tmp ~axis:0 |> Arr.transpose ~axis:[| 1; 0; 2 |]
  in
  let inferred_u_mat =
    List.map inferred_us ~f:(fun u ->
      let tmp = u |> Maths.to_tensor |> Tensor.to_bigarray ~kind:base.ba_kind in
      Arr.reshape tmp [| 1; bs; 3 |])
    |> List.to_array
    |> Arr.concatenate ~axis:0
  in
  let inferred_x_mat =
    let inferred_xs = List.map sol ~f:(fun x -> x.x |> Option.value_exn) in
    List.map inferred_xs ~f:(fun x ->
      let tmp = x |> Maths.to_tensor |> Tensor.to_bigarray ~kind:base.ba_kind in
      Arr.reshape tmp [| 1; bs; 3 |])
    |> List.to_array
    |> Arr.concatenate ~axis:0
  in
  let x_mat =
    let tmp =
      Maths.concat (List.map (List.tl_exn observation) ~f:(Maths.unsqueeze ~dim:0)) ~dim:0
    in
    tmp |> Maths.to_tensor |> Tensor.to_bigarray ~kind:base.ba_kind
  in
  Arr.(save_npy ~out:(in_dir "inferred_x") inferred_x_mat);
  Mat.(save_npy ~out:(in_dir "x") x_mat);
  Mat.(save_npy ~out:(in_dir "inferred_u") inferred_u_mat);
  Mat.(save_npy ~out:(in_dir "u") u_mat);
  let final_error = Arr.(mean' (sqr (u_mat - inferred_u_mat))) in
  Sofo.print [%message (final_error : float)]
