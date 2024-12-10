(* memory profiling of lqr naive vs implicit *)
open Base
open Forward_torch
open Sofo

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

(* -----------------------------------------
   -- Define Control Problem          ------
   ----------------------------------------- *)
module Dims = struct
  let a = 24
  let b = 10
  let o = 0
  let tmax = 10
  let m = 64
  let k = 64
  let batch_const = false
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.cuda_if_available ()
end

module Data = Lds_data.Make_LDS (Dims)

(* sample params first to rollout traj for _cx calculation *)
let x0 = Data.sample_x0 ()

(* need to sample these first to get the trajectory *)
let f_list : Maths.t Lds_data.f_params list =
  let _Fx = Data.sample_fx () in
  let _Fu = Data.sample_fu () in
  List.init (Dims.tmax + 1) ~f:(fun _ ->
    Lds_data.
      { _Fx_prod = _Fx
      ; _Fu_prod = _Fu
      ; _f = Some (Data.sample_f ())
      ; _c = None
      ; _b = None
      ; _cov = None
      })

let u_list = Data.sample_u_list ()
let x_list, _ = Data.traj_rollout ~x0 ~f_list ~u_list

(* for the purpose of memory profiling only everything has tangents *)
let params : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
  =
  let _Cxx = Data.sample_q_xx () in
  let _Cxu = Some (Data.sample_q_xu ()) in
  let _Cuu = Data.sample_q_uu () in
  Lqr.Params.
    { x0 = Some x0
    ; params =
        List.map2_exn f_list x_list ~f:(fun params x ->
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
    Array.init 200 ~f:(fun i ->
      Convenience.print [%message (i : int)];
      ignore (f ()))
  in
  let dt = Unix.gettimeofday () -. t0 in
  Convenience.print [%message (label : string) (dt : float)];
  result

let _ =
  match Cmdargs.get_string "-method" with
  | Some "implicit" ->
    let p = Lds_data.map_implicit ~batch_const:Dims.batch_const params in
    time_this ~label:"implicit" (fun _ -> Lqr.solve ~batch_const:Dims.batch_const p)
    |> ignore
  | Some "naive" ->
    let p = Lds_data.map_naive ~batch_const:Dims.batch_const params in
    time_this ~label:"naive" (fun _ -> Lqr._solve ~batch_const:Dims.batch_const p)
    |> ignore
  | _ -> failwith "use cmdline arg '-method {naive | implicit}'"
