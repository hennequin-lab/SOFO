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

(* returns the x0 mat, list of target mat. targets go from t=1 to t=T. *)
let sample_data_tan bs =
  let x0, x_u_list = Data_Tan.batch_trajectory bs in
  let targets_list = List.map x_u_list ~f:(fun (x, _) -> Some x) in
  let target_controls_list = List.map x_u_list ~f:(fun (_, u) -> Some u) in
  x0, targets_list, target_controls_list

let x0, targets_list, target_controls_list = sample_data_tan batch_size
let lds_params = Data_Tan.lds_params

(* form state params/cost_params/xu_desired *)
let state_params : Forward_torch.Lqt_type.state_params =
  { n_steps = List.length targets_list
  ; a_list = [ lds_params.a_tot ]
  ; b_list = [ lds_params.b_tot ]
  ; f_t_list = [ None ]
  }

let cost_params : Forward_torch.Lqt_type.cost_params =
  let q_full =
    let q_primal = to_device q in
    let q_tangent =
      Maths.Direct
        Tensor.(
          randn
            [ Lds_params_dim_tan.n_tangents; Lds_params_dim_tan.a; Lds_params_dim_tan.a ]
            ~device:Lds_params_dim_tan.device)
    in
    Maths.make_dual q_primal ~t:q_tangent
  in
  let r_full =
    let r_primal = to_device r in
    let r_tangent =
      Maths.Direct
        Tensor.(
          randn
            [ Lds_params_dim_tan.n_tangents; Lds_params_dim_tan.b; Lds_params_dim_tan.b ]
            ~device:Lds_params_dim_tan.device)
    in
    Maths.make_dual r_primal ~t:r_tangent
  in
  { q_list = [ q_full ]; r_list = [ r_full ] }

let x_u_desired : Forward_torch.Lqt_type.x_u_desired =
  { x_0 = x0; x_d_list = targets_list; u_d_list = [ None ] }

let x_list1, u_list1 = Lqt.lqt ~state_params ~x_u_desired ~cost_params
let x_list2, u_list2 = Lqt.sep_primal_tan_lqt ~state_params ~x_u_desired ~cost_params

let _ =
  List.iter2_exn x_list1 x_list2 ~f:(fun x1 x2 ->
    let per_error_tan =
      Tensor.(
        norm (Option.value_exn Maths.(tangent (x2 - x1)))
        / norm (Option.value_exn (Maths.tangent x1)))
    in
    Tensor.print per_error_tan
    (* let per_error_primal =
       Tensor.(norm Maths.(primal (x2 - x1)) / norm (Maths.primal x1))
       in
       Tensor.print per_error_primal *)
    (* Tensor.(print (Option.value_exn Maths.(tangent (x1 - x2)))) *)
    (* Tensor.(print ( Maths.(primal (x1 - x2)))); *))
