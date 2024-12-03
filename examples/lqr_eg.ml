open Base
open Torch
open Forward_torch
open Sofo
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

(* -----------------------------------------
   -- Define Control Problem          ------
   ----------------------------------------- *)
module Lds_params_dim = struct
  let a = 5
  let b = 3
  let tmax = 10
  let m = 3
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.cuda_if_available ()
end

module Data = Lds_data.Make_LDS (Lds_params_dim)

let q_id_of d =
  List.init Lds_params_dim.m ~f:(fun _ ->
    let ell = Tensor.eye ~n:d ~options:(Lds_params_dim.kind, Lds_params_dim.device) in
    Tensor.reshape ell ~shape:[ 1; d; d ])
  |> Tensor.concat ~dim:0

(* state cost set as identity *)
let _Cxx =
  let pri = q_id_of Lds_params_dim.a in
  Maths.(const pri)

(* control cost set by alpha *)
let alpha = 0.01

let _Cuu =
  let pri = q_id_of Lds_params_dim.b in
  Maths.(alpha $* const pri)

(* sample params first to rollout traj for _cx calculation *)
let x0 = Data.sample_x0 ()

(* need to sample these first to get the trajectory *)
let f_list : Maths.t Lds_data.f_params list =
  List.init (Lds_params_dim.tmax + 1) ~f:(fun _ ->
    Lds_data.
      { _Fx_prod = Data.sample_fx ()
      ; _Fu_prod = Data.sample_fu ()
      ; _f = Some (Data.sample_f ())
      })

let u_targets, x_targets = Data.traj_rollout ~x0 ~f_list

let params : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
  =
  Lqr.Params.
    { x0 = Some x0
    ; params =
        List.map2_exn f_list x_targets ~f:(fun params x ->
          Lds_data.Temp.
            { _f = params._f
            ; _Fx_prod = params._Fx_prod
            ; _Fu_prod = params._Fu_prod
            ; _cx = Some Maths.(neg x)
            ; _cu = None
            ; _Cxx
            ; _Cxu = None
            ; _Cuu
            })
    }

(* -----------------------------------------
   -- Check quality of LQR results    ------
   ----------------------------------------- *)
let f_naive params = Lqr._solve (Data.naive_params params)
let f_implicit params = Lqr.solve (Data.implicit_params params)

let check_quality common_params u_targets =
  let naive_result = f_naive common_params in
  let implicit_result = f_implicit common_params in
  let u_error (result : Maths.t Lqr.Solution.p list) =
    List.fold2_exn result u_targets ~init:0. ~f:(fun acc res u_target ->
      let u_res = res.u in
      let error =
        Tensor.(
          norm Maths.(primal (u_res - u_target)) / norm Maths.(primal u_target)
          |> to_float0_exn)
      in
      acc +. error)
  in
  let naive_error = u_error naive_result in
  let implicit_error = u_error implicit_result in
  Convenience.print [%message (naive_error : float)];
  Convenience.print [%message (implicit_error : float)]

let _ = check_quality params u_targets
