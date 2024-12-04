(* memory profiling of lqr naive vs implicit *)
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
  let a = 24
  let b = 10
  let tmax = 10
  let m = 64
  let k = 64
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.cuda_if_available ()
end

module Data = Lds_data.Make_LDS (Lds_params_dim)

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

(* for the purpose of memory profiling only everything has tangents *)
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
            ; _cu = Some (Data.sample_c_u ())
            ; _Cxx = Data.sample_q_xx ()
            ; _Cxu = Some (Data.sample_q_xu ())
            ; _Cuu = Data.sample_q_uu ()
            })
    }

(* -----------------------------------------
   ---- Memory Profiling   ------
   ----------------------------------------- *)
let f_naive params = Lqr._solve (Data.naive_params params)
let f_implicit params = Lqr.solve (Data.implicit_params params)
let t0 = Unix.gettimeofday ()
let naive_result = f_implicit params
let t1 = Unix.gettimeofday ()
let _ = Convenience.print [%message "implicit" (t1 -. t0 : float)]
(* let t0 = Unix.gettimeofday ()
let naive_result = f_naive params
let t1 = Unix.gettimeofday ()
let _ = Convenience.print [%message "naive" (t1 -. t0 : float)] *)
