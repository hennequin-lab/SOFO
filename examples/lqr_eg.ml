(* directly parameterise control matrices for affine dynamics *)
open Printf
open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

let in_dir = Cmdargs.in_dir "-d"

let _ =
  (* Random.init 1999; *)
  Random.self_init ();
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let batch_size = 2
let base = Optimizer.Config.Base.default
let to_device = Tensor.of_bigarray ~device:base.device

(* -----------------------------------------
   -- Define Control Problem          ------
   ----------------------------------------- *)

let with_given_seed_owl seed f =
  (* generate a random key to later restore the state of the RNG *)
  let key = Random.int Int.max_value in
  (* now force the state of the RNG under which f will be evaluated *)
  Owl_stats_prng.init seed;
  let result = f () in
  (* restore the RGN using key *)
  Owl_stats_prng.init key;
  (* return the result *)
  result

let base = Optimizer.Config.Base.default

module Lds_params_dim_tan = struct
  let a = 6
  let b = 1
  let n_steps = 10
  let n_tangents = 3
  let kind = base.kind
  let device = base.device
end

module Data_Tan = Lds_data.Make_LDS_Tan (Lds_params_dim_tan)

(* different costs for each batch and at each timestep; q goes from 1 to T and r goes from 0 to T-1 *)
let control_costs () =
  let q_r_list =
    List.init Lds_params_dim_tan.n_steps ~f:(fun _ ->
      let q_batched =
        let q_array =
          Array.init batch_size ~f:(fun _ ->
            let q_1 = Mat.gaussian Lds_params_dim_tan.a Lds_params_dim_tan.a in
            let qqT = Mat.(q_1 *@ transpose q_1) in
            let q_tmp = Mat.(qqT + eye Lds_params_dim_tan.a) in
            Arr.reshape q_tmp [| 1; Lds_params_dim_tan.a; Lds_params_dim_tan.a |])
        in
        Arr.concatenate q_array ~axis:0
      in
      let r_batched =
        let r_array =
          Array.init batch_size ~f:(fun _ ->
            let r_tmp = Mat.(eye Lds_params_dim_tan.b) in
            Arr.reshape r_tmp [| 1; Lds_params_dim_tan.b; Lds_params_dim_tan.b |])
        in
        Arr.concatenate r_array ~axis:0
      in
      q_batched, r_batched)
  in
  q_r_list

let q_r_list = with_given_seed_owl 1985 control_costs

let q_list =
  List.map
    ~f:(fun q_r ->
      let q = fst q_r in
      Maths.const (to_device q))
    q_r_list

let r_list =
  List.map
    ~f:(fun q_r ->
      let r = snd q_r in
      Maths.const (to_device r))
    q_r_list

(* returns the x0 mat, list of target mat. targets x go from t=1 to t=T and targets u go from t=0 to t=T-1. *)
let sample_data bs =
  let batch_lds_params, x0, x_u_list = Data_Tan.batch_trajectory bs in
  let targets_list = List.map x_u_list ~f:(fun (x, _, _) -> x) in
  let target_controls_list = List.map x_u_list ~f:(fun (_, u, _) -> u) in
  let f_ts_list = List.map x_u_list ~f:(fun (_, _, f_t) -> f_t) in
  batch_lds_params, x0, targets_list, target_controls_list, f_ts_list

let batch_lds_params, x0, targets_list, target_controls_list, f_ts_list =
  sample_data batch_size

let n_steps = List.length targets_list
let repeat_list lst n = List.concat (List.init n ~f:(fun _ -> lst))

(* -----------------------------------------
   -- Maths operations (with tangent)  ------
   ----------------------------------------- *)

let batch_vecmat a b = Maths.(einsum [ a, "mi"; b, "mij" ] "mj")

(* form state params/cost_params/xu_desired *)
let state_params : Forward_torch.Lqr_type.state_params =
  { n_steps = List.length targets_list
  ; x_0 = x0
  ; f_x_list = List.map batch_lds_params ~f:fst
  ; f_u_list = List.map batch_lds_params ~f:snd
  ; f_t_list = Some f_ts_list
  }

let c_x_list =
  let tmp =
    List.map2_exn targets_list q_list ~f:(fun target q ->
      Maths.(batch_vecmat (neg target) q))
  in
  Some
    (Maths.const
       (Tensor.zeros
          [ batch_size; Lds_params_dim_tan.a ]
          ~device:Lds_params_dim_tan.device)
     :: tmp)

let c_u_list =
  let tmp =
    List.map2_exn target_controls_list r_list ~f:(fun target r ->
      Maths.(batch_vecmat (neg target) r))
  in
  Some
    (tmp
     @ [ Maths.const
           (Tensor.zeros
              [ batch_size; Lds_params_dim_tan.b ]
              ~device:Lds_params_dim_tan.device)
       ])

(* form cost parameters, which all go from step 0 to T.*)
let cost_params : Forward_torch.Lqr_type.cost_params =
  { c_xx_list =
      Maths.const
        (Tensor.zeros
           [ batch_size; Lds_params_dim_tan.a; Lds_params_dim_tan.a ]
           ~device:Lds_params_dim_tan.device)
      :: q_list
  ; c_xu_list = None
  ; c_uu_list =
      r_list
      @ [ Maths.const
            (Tensor.zeros
               [ batch_size; Lds_params_dim_tan.b; Lds_params_dim_tan.b ]
               ~device:Lds_params_dim_tan.device)
        ]
  ; c_x_list
  ; c_u_list
  }

let t0 = Unix.gettimeofday ()

(* compare directly using Maths operation and using Tensor operation. *)
let x_list1, u_list1 = Lqr.lqr ~state_params ~cost_params
let x_list3, u_list3 = Lqr.lqr_sep ~state_params ~cost_params

(* let _ =
   List.iter2_exn u_list1 u_list3 ~f:(fun x1 x2 ->
   Tensor.print (Tensor.norm (Option.value_exn Maths.(tangent (x1 - x2))))) *)

let t1 = Unix.gettimeofday ()
let time_elapsed = Float.(t1 - t0)

let _ =
  Convenience.print [%message (Lds_params_dim_tan.n_steps : int) (time_elapsed : float)]
