(* propagate tangents either naively together with primals or separately to form (K+1) lqr problems. *)
open Base
open Torch
open Forward_torch
open Sofo
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

(* let in_dir = Cmdargs.in_dir "-d" *)

let _ =
  (* Random.init 1999; *)
  Random.self_init ();
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let batch_size = 256
(* let base = Optimizer.Config.Base.default *)
(* let to_device = Tensor.of_bigarray ~device:base.device *)

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
  let a = 24
  let b = 10
  let n_steps = 25
  let n_tangents = 128
  let kind = base.kind
  let device = base.device
end

module Data_Tan = Lds_data.Make_LDS_Tan (Lds_params_dim_tan)

let device_ = Lds_params_dim_tan.device
let kind_ = Lds_params_dim_tan.kind

(* sample a symmetric positive definite matrix of size n *)
let create_sym_pos n =
  let q_1 = Tensor.randn [ n; n ] ~device:device_ in
  let qqT = Tensor.(matmul q_1 (transpose q_1 ~dim0:1 ~dim1:0)) in
  Tensor.(qqT + eye ~n ~options:(kind_, device_))

(* maths object where primal has shape 1 by n by n and tangent has shape k by n by n *)
let create_sym_maths n =
  let q_primal =
    let q_tmp = create_sym_pos n in
    Tensor.reshape q_tmp ~shape:[ 1; n; n ]
  in
  let q_tangent =
    let q_tangent_list =
      List.init Lds_params_dim_tan.n_tangents ~f:(fun _ ->
        let q_tan_tmp = create_sym_pos n in
        Tensor.reshape q_tan_tmp ~shape:[ 1; 1; n; n ])
    in
    Tensor.concat q_tangent_list ~dim:0
  in
  Maths.make_dual q_primal ~t:(Maths.Direct q_tangent)

(* different costs for each batch and at each timestep; q goes from 1 to T and r goes from 0 to T-1; if invariant then use the same q and r across trials and time *)
let control_costs ~invariant () =
  let q_list =
    List.init Lds_params_dim_tan.n_steps ~f:(fun _ ->
      let q_batched =
        let q_inv = create_sym_maths Lds_params_dim_tan.a in
        let q_list =
          List.init batch_size ~f:(fun _ ->
            if invariant then q_inv else create_sym_maths Lds_params_dim_tan.a)
        in
        Maths.concat_list q_list ~dim:0
      in
      q_batched)
  in
  let r_list =
    List.init Lds_params_dim_tan.n_steps ~f:(fun _ ->
      let r_batched =
        let r_inv = create_sym_maths Lds_params_dim_tan.b in
        let r_list =
          List.init batch_size ~f:(fun _ ->
            if invariant then r_inv else create_sym_maths Lds_params_dim_tan.b)
        in
        Maths.concat_list r_list ~dim:0
      in
      r_batched)
  in
  q_list, r_list

let q_list, r_list = with_given_seed_owl 1985 (control_costs ~invariant:true)

(* returns the x0 mat, list of target mat. targets x go from t=1 to t=T and targets u go from t=0 to t=T-1. *)
let sample_data bs =
  let batch_lds_params, x0, x_u_list = Data_Tan.batch_trajectory bs in
  let fx_list, fu_list =
    List.map batch_lds_params ~f:fst, List.map batch_lds_params ~f:snd
  in
  let targets_list, target_controls_list, f_ts_list =
    ( List.map x_u_list ~f:(fun (x, _, _) -> x)
    , List.map x_u_list ~f:(fun (_, u, _) -> u)
    , List.map x_u_list ~f:(fun (_, _, f_t) -> f_t) )
  in
  fx_list, fu_list, batch_lds_params, x0, targets_list, target_controls_list, f_ts_list

let fx_list, fu_list, batch_lds_params, x0, targets_list, target_controls_list, f_ts_list =
  sample_data batch_size

(* -----------------------------------------
   -- Maths operations (with tangent)  ------
   ----------------------------------------- *)
let batch_vecmat a b = Maths.(einsum [ a, "mi"; b, "mij" ] "mj")

let params : Maths.t Forward_torch.Lqr.params =
  let momentary_params =
    List.init Lds_params_dim_tan.n_steps ~f:(fun t ->
      let curr_params : Maths.t Forward_torch.Lqr.momentary_params =
        let _f = Some (List.nth_exn f_ts_list t) in
        let _Fx_prod x =
          let fx = List.nth_exn fx_list t in
          Maths.(fx *@ x)
        in
        (* simple LDS, prod is same as prod2 *)
        let _Fx_prod2 x =
          let fx = List.nth_exn fx_list t in
          Maths.(x *@ fx)
        in
        let _Fu_prod x =
          let fu = List.nth_exn fu_list t in
          Maths.(fu *@ x)
        in
        (* simple LDS, prod is same as prod2 *)
        let _Fu_prod2 x =
          let fu = List.nth_exn fu_list t in
          Maths.(x *@ fu)
        in
        let _cx =
          if t = 0
          then
            Some
              (Maths.const
                 (Tensor.zeros
                    [ batch_size; Lds_params_dim_tan.a ]
                    ~device:Lds_params_dim_tan.device))
          else (
            let target = List.nth_exn targets_list Int.(t - 1) in
            let q = List.nth_exn q_list Int.(t - 1) in
            Some Maths.(batch_vecmat (neg target) q))
        in
        let _cu =
          if t = Int.(Lds_params_dim_tan.n_steps - 1)
          then
            Some
              (Maths.const
                 (Tensor.zeros
                    [ batch_size; Lds_params_dim_tan.b ]
                    ~device:Lds_params_dim_tan.device))
          else (
            let target = List.nth_exn targets_list t in
            let r = List.nth_exn r_list t in
            Some Maths.(batch_vecmat (neg target) r))
        in
        let _Cxx =
          if t = 0
          then
            Maths.const
              (Tensor.zeros
                 [ batch_size; Lds_params_dim_tan.a; Lds_params_dim_tan.a ]
                 ~device:Lds_params_dim_tan.device)
          else List.nth_exn q_list t
        in
        let _Cxu =
          let primal =
            Tensor.zeros
              [ batch_size; Lds_params_dim_tan.a; Lds_params_dim_tan.b ]
              ~device:Lds_params_dim_tan.device
          in
          let tangent =
            Tensor.zeros
              [ Lds_params_dim_tan.n_tangents
              ; batch_size
              ; Lds_params_dim_tan.a
              ; Lds_params_dim_tan.b
              ]
              ~device:Lds_params_dim_tan.device
          in
          Maths.make_dual primal ~t:(Maths.Direct tangent)
        in
        let _Cuu =
          if t = Int.(Lds_params_dim_tan.n_steps - 1)
          then
            Maths.const
              (Tensor.zeros
                 [ batch_size; Lds_params_dim_tan.b; Lds_params_dim_tan.b ]
                 ~device:Lds_params_dim_tan.device)
          else List.nth_exn q_list t
        in
        { _f; _Fx_prod; _Fx_prod2; _Fu_prod; _Fu_prod2; _cx; _cu; _Cxx; _Cxu; _Cuu }
      in
      curr_params)
  in
  { x0; params = momentary_params }

let t0 = Unix.gettimeofday ()

(* compare directly using Maths operation and using Tensor operation. *)
let _ = Convenience.print [%message "start lqr"]

module LqrSolver = Lqr.Make (Lqr.MathsOps)

let sol = LqrSolver.solve params

let x_list, u_list =
  List.map sol ~f:(fun sol -> sol.x), List.map sol ~f:(fun sol -> sol.u)

let t1 = Unix.gettimeofday ()
let time_elapsed = Float.(t1 - t0)

let _ =
  Convenience.print [%message (Lds_params_dim_tan.n_steps : int) (time_elapsed : float)]
