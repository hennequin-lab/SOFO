(* propagate tangents either naively together with primals or separately to form (K+1) lqr problems. *)
open Base
open Torch
open Forward_torch
open Sofo
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

let base = Optimizer.Config.Base.default

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

(* -----------------------------------------
   -- Define Control Problem          ------
   ----------------------------------------- *)

module Dims = struct
  let n = 5
  let m = 3
  let o = 15
  let tmax = 10
  let bs = 8
  let k = 12
  let batch_const = false
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.cuda_if_available ()
end

module Data = Lds_data.Make_LDS (Dims)

let q_id_of d =
  let ell = Tensor.eye ~n:d ~options:(Dims.kind, Dims.device) in
  if Dims.batch_const
  then ell
  else (
    let ell_tmp = Tensor.unsqueeze ell ~dim:0 in
    List.init Dims.bs ~f:(fun _ -> ell_tmp) |> Tensor.concat ~dim:0)

(* Q set as identity *)
let _Cxx =
  let pri = q_id_of Dims.n in
  Maths.(of_tensor pri)

(* control cost set by alpha *)
let alpha = 1e-5

let _Cuu =
  let pri = q_id_of Dims.m in
  Maths.(alpha $* of_tensor pri)

let x0 = Data.sample_x0 () |> Maths.any

let map_implicit
      (x :
        ( Maths.any Maths.t option
          , (Maths.any Maths.t, Maths.any Maths.t option) Lds_data.Temp.p list )
          Lqr.Params.p)
      ~batch_const
  =
  let params =
    List.map x.params ~f:(fun p ->
      Lqr.
        { common =
            { _Fx_prod = Some (Lds_data.prod ~batch_const p._Fx_prod)
            ; _Fx_prod2 = Some (Lds_data.prod2 ~batch_const p._Fx_prod)
            ; _Fu_prod = Some (Lds_data.prod ~batch_const p._Fu_prod)
            ; _Fu_prod2 = Some (Lds_data.prod2 ~batch_const p._Fu_prod)
            ; _Fx_prod_tangent = Some (Lds_data.prod_tangent ~batch_const p._Fx_prod)
            ; _Fx_prod2_tangent = Some Lds_data.(prod2_tangent ~batch_const p._Fx_prod)
            ; _Fu_prod_tangent = Some (Lds_data.prod_tangent ~batch_const p._Fu_prod)
            ; _Fu_prod2_tangent = Some (Lds_data.prod2_tangent ~batch_const p._Fu_prod)
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

(* need to sample params first to get the trajectory *)
let f_list : Maths.any Maths.t Lds_data.f_params list =
  let _Fx = Data.sample_fx ~target_sa:0.8 in
  let _Fu = Data.sample_fu () in
  List.init (Dims.tmax + 1) ~f:(fun _ ->
    Lds_data.
      { _Fx_prod = Maths.any _Fx
      ; _Fu_prod = Maths.any _Fu
      ; _f = None
      ; _c = None
      ; _b = None
      ; _cov = None
      })

let u_targets = List.map (Data.sample_u_list ()) ~f:Maths.any
let x_list, _ = Data.traj_rollout ~x0 ~f_list ~u_list:u_targets

let params
  : ( Maths.any Maths.t option
      , (Maths.any Maths.t, Maths.any Maths.t option) Lds_data.Temp.p list )
      Lqr.Params.p
  =
  Lqr.Params.
    { x0 = Some x0
    ; params =
        List.map2_exn f_list x_list ~f:(fun params x ->
          Lds_data.Temp.
            { _f = params._f
            ; _Fx_prod = params._Fx_prod
            ; _Fu_prod = params._Fu_prod
            ; _cx = Some Maths.(neg x)
            ; _cu = None
            ; _Cxx = Maths.any _Cxx
            ; _Cxu = None
            ; _Cuu = Maths.any _Cuu
            })
    }

let check_quality common_params u_targets =
  let p = map_naive ~batch_const:Dims.batch_const common_params in
  let result, _ = Lqr._solve ~batch_const:Dims.batch_const p in
  (* let p = map_implicit ~batch_const:Dims.batch_const common_params in
  let result = Lqr.solve ~batch_const:Dims.batch_const p in *)
  let u_error =
    List.fold2_exn result u_targets ~init:0. ~f:(fun acc res u_target ->
      let u_res = res.u in
      (* let u_res = res.u |> Option.value_exn in *)
      let error =
        Tensor.(
          norm
            (mean_dim
               ~dtype:base.kind
               ~dim:(Some [ 0 ])
               ~keepdim:false
               Maths.(to_tensor (u_res - u_target)))
          |> to_float0_exn)
      in
      acc +. error)
  in
  u_error

let error = check_quality params u_targets
let _ = Sofo.print [%message (error : float)]
