open Base
open Torch
open Forward_torch
open Sofo
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

(* let in_dir = Cmdargs.in_dir "-d" *)

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

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
   -- Define Control Problem          ------
   ----------------------------------------- *)
module Lds_params_dim = struct
  let a = 5
  let b = 3
  let tmax = 10
  let m = 7
  let k = 12
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
let alpha = 0.1

let _Cuu =
  let pri = q_id_of Lds_params_dim.b in
  Maths.(alpha $* const pri)

(* sample params first to rollout traj for _cx calculation *)
let x0 = Data.sample_x0 ()

(* need to sample these first to get the trajectory *)
let f_list : Maths.t Lds_typ.f_params list =
  List.init (Lds_params_dim.tmax + 1) ~f:(fun _ ->
    Lds_typ.
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

let check_quality common_params u_targets =
  (* let result = Lqr._solve (Data.naive_params common_params) in *)
  let result = Lqr.solve (Data.implicit_params common_params) in
  let u_error =
    List.fold2_exn result u_targets ~init:0. ~f:(fun acc res u_target ->
      let u_res = res.u in
      (* let u_res = res.u |> Option.value_exn in *)
      let error = Tensor.(norm Maths.(primal (u_res - u_target)) |> to_float0_exn) in
      acc +. error)
  in
  u_error

let error = check_quality params u_targets
let _ = Convenience.print [%message (error : float)]
