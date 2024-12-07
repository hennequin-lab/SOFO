(* Linear Gaussian Dynamics, with same state/control/cost parameters across trial (i.e. batch_const=true) *)
open Base
open Forward_torch
open Sofo
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

(* let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000) *)

(* -----------------------------------------
   -- Define Control Problem          ------
   ----------------------------------------- *)
module Lds_params_dim = struct
  let a = 6
  let b = 3
  let tmax = 10
  let m = 64
  let k = 64
  let batch_const = true
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.cuda_if_available ()
end

module Data = Lds_data.Make_LDS (Lds_params_dim)

(* sample params first to rollout traj for _cx calculation *)
let x0 = Data.sample_x0 ()

(* need to sample these first to get the trajectory *)
let f_list : Maths.t Lds_data.f_params list =
  let a = Data.sample_fx () in
  let b = Data.sample_fu () in
  List.init (Lds_params_dim.tmax + 1) ~f:(fun _ ->
    Lds_data.{ _Fx_prod = a; _Fu_prod = b; _f = Some (Data.sample_f ()) })

let _, x_targets = Data.traj_rollout ~x0 ~f_list

(* for the purpose of memory profiling only everything has tangents *)
let params : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
  =
  let _Cxx = Data.sample_q_xx () in
  let _Cxu = Some (Data.sample_q_xu ()) in
  let _Cuu = Data.sample_q_uu () in
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
            ; _Cxx
            ; _Cxu
            ; _Cuu
            })
    }

(* -----------------------------------------
   ---- Memory Profiling   ------
   ----------------------------------------- *)

let time_this ~label f =
  Stdlib.Gc.compact ();
  let t0 = Unix.gettimeofday () in
  let result =
    Array.init 20 ~f:(fun i ->
      Convenience.print [%message (i : int)];
      ignore (f ()))
  in
  let dt = Unix.gettimeofday () -. t0 in
  Convenience.print [%message (label : string) (dt : float)];
  result

let _ =
  match Cmdargs.get_string "-method" with
  | Some "implicit" ->
    let p = Data.implicit_params params in
    time_this ~label:"implicit" (fun _ ->
      Lqr.solve ~batch_const:Lds_params_dim.batch_const p)
    |> ignore
  | Some "naive" ->
    let p = Data.naive_params params in
    time_this ~label:"naive" (fun _ ->
      Lqr._solve ~batch_const:Lds_params_dim.batch_const p)
    |> ignore
  | _ -> failwith "use cmdline arg '-method {naive | implicit}'"
