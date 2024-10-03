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

let batch_size = 256
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

(* -----------------------------------------
   -- Maths operations (with tangent)  ------
   ----------------------------------------- *)
let base = Optimizer.Config.Base.default

module Lds_params_dim_tan = struct
  let a = 24
  let b = 10
  let n_steps = 20
  let n_tangents = 256
  let kind = base.kind
  let device = base.device
end

(* Lds_data.Default_Tan with device = base.device and kind = base.kind and n_tangents = 6 *)
module Data_Tan = Lds_data.Make_LDS_Tan (Lds_params_dim_tan)

let control_costs () =
  let q =
    let q_1 = Mat.gaussian Lds_params_dim_tan.a Lds_params_dim_tan.a in
    let qqT = Mat.(q_1 *@ transpose q_1) in
    Mat.(qqT + eye Lds_params_dim_tan.a)
  in
  let r = Mat.(eye Lds_params_dim_tan.b) in
  q, r

let q, r = with_given_seed_owl 1985 control_costs

(* returns the x0 mat, list of target mat. targets x go from t=1 to t=T and targets u go from t=0 to t=T-1. *)
let sample_data bs =
  let x0, x_u_list = Data_Tan.batch_trajectory bs in
  let targets_list = List.map x_u_list ~f:fst in
  let target_controls_list = List.map x_u_list ~f:snd in
  x0, targets_list, target_controls_list

let x0, targets_list, target_controls_list = sample_data batch_size
let lds_params = Data_Tan.lds_params
let n_steps = List.length targets_list

(* form state params/cost_params/xu_desired *)
let state_params : Forward_torch.Lqr_type.state_params =
  { n_steps = List.length targets_list
  ; x_0 = x0
  ; f_x_list = [ lds_params.a_tot ]
  ; f_u_list = [ lds_params.b_tot ]
  ; f_t_list = [ None ]
  }

let repeat_list lst n = List.concat (List.init n ~f:(fun _ -> lst))

let c_x_list =
  let tmp =
    List.map targets_list ~f:(fun target -> Maths.(neg target *@ const (to_device q)))
  in
  Some (x0 :: tmp)

let c_u_list =
  let tmp =
    List.map target_controls_list ~f:(fun target ->
      Maths.(neg target *@ const (to_device r)))
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
      Maths.const (Tensor.zeros_like (to_device q))
      :: repeat_list [ Maths.const (to_device q) ] n_steps
  ; c_xu_list = None
  ; c_uu_list =
      repeat_list [ Maths.const (to_device r) ] n_steps
      @ [ Maths.const (Tensor.zeros_like (to_device r)) ]
  ; c_x_list
  ; c_u_list
  }

let x_list1, u_list1 = Lqr.lqr ~state_params ~cost_params
