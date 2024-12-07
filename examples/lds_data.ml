(* In the simple linear Gaussian case, we assume Fx, Fu, Q_xx, Q_uu and f are the same across all trials in the batch. *)
open Base
open Torch
open Forward_torch
open Lqr
module Mat = Owl.Dense.Matrix.D
module Arr = Owl.Dense.Ndarray.D
module Linalg = Owl.Linalg.D

type 'a f_params =
  { _Fx_prod : 'a
  ; _Fu_prod : 'a
  ; _f : 'a option
  }

(* -----------------------------------------
   -- Some utility functions         ------
   ----------------------------------------- *)

(* Fv prod; if batch_const, F does not have a leading batch dimension. *)
let bmm ~batch_const a b =
  let open Maths in
  if batch_const
  then (
    match List.length (shape b) with
    | 3 -> einsum [ a, "ab"; b, "mbc" ] "mac"
    | 2 -> einsum [ a, "ab"; b, "mb" ] "ma"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 3 -> einsum [ a, "mab"; b, "mbc" ] "mac"
    | 2 -> einsum [ a, "mab"; b, "mb" ] "ma"
    | _ -> failwith "not batch multipliable")

(* vF prod *)
let bmm2 ~batch_const b a =
  let open Maths in
  if batch_const
  then (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b, "ab" ] "mb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b, "mab" ] "mb"
    | _ -> failwith "should not happen")

(* Fv prod, tangent on v *)
let bmm_tangent_v ~batch_const a b =
  let open Maths in
  if batch_const
  then (
    match List.length (shape b) with
    | 4 -> einsum [ a, "ab"; b, "kmbc" ] "kmac"
    | 3 -> einsum [ a, "ab"; b, "kmb" ] "kma"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 4 -> einsum [ a, "mab"; b, "kmbc" ] "kmac"
    | 3 -> einsum [ a, "mab"; b, "kmb" ] "kma"
    | _ -> failwith "not batch multipliable")

(* Fv prod, tangent on F *)
let bmm_tangent_F ~batch_const a b =
  let open Maths in
  if batch_const
  then (
    match List.length (shape b) with
    | 3 -> einsum [ a, "kab"; b, "mbc" ] "kmac"
    | 2 -> einsum [ a, "kab"; b, "mb" ] "kma"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 3 -> einsum [ a, "kmab"; b, "mbc" ] "kmac"
    | 2 -> einsum [ a, "kmab"; b, "mb" ] "kma"
    | _ -> failwith "not batch multipliable")

(* Fv prod, tangent on F and v *)
let bmm_tangent_Fv ~batch_const a b =
  let open Maths in
  if batch_const
  then (
    match List.length (shape b) with
    | 4 -> einsum [ a, "kab"; b, "kmbc" ] "kmac"
    | 3 -> einsum [ a, "kab"; b, "kmb" ] "kma"
    | _ -> failwith "not batch multipliable")
  else (
    match List.length (shape b) with
    | 4 -> einsum [ a, "kmab"; b, "kmbc" ] "kmac"
    | 3 -> einsum [ a, "kmab"; b, "kmb" ] "kma"
    | _ -> failwith "not batch multipliable")

(* vF prod, tangent on v *)
let bmm2_tangent_v ~batch_const b a =
  let open Maths in
  if batch_const
  then (
    match List.length (shape a) with
    | 3 -> einsum [ a, "kma"; b, "ab" ] "kmb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 3 -> einsum [ a, "kma"; b, "mab" ] "kmb"
    | _ -> failwith "should not happen")

(* vF prod, tangent on F *)
let bmm2_tangent_F ~batch_const b a =
  let open Maths in
  if batch_const
  then (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b, "kab" ] "kmb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 2 -> einsum [ a, "ma"; b, "kmab" ] "kmb"
    | _ -> failwith "should not happen")

(* vF prod, tangent on F and v *)
let bmm2_tangent_Fv ~batch_const b a =
  let open Maths in
  if batch_const
  then (
    match List.length (shape a) with
    | 3 -> einsum [ a, "kma"; b, "kab" ] "kmb"
    | _ -> failwith "should not happen")
  else (
    match List.length (shape a) with
    | 3 -> einsum [ a, "kma"; b, "kmab" ] "kmb"
    | _ -> failwith "should not happen")

let prod ~batch_const f : Maths.t Lqr.prod =
  let primal = bmm ~batch_const (Maths.const (Maths.primal f)) in
  (* tangent on f only *)
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (bmm_tangent_F ~batch_const (Maths.const df))
  in
  { primal; tangent }

let prod_tangent ~batch_const f : Maths.t Lqr.prod =
  (* tangent on v only *)
  let primal = bmm_tangent_v ~batch_const (Maths.const (Maths.primal f)) in
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (bmm_tangent_Fv ~batch_const (Maths.const df))
  in
  { primal; tangent }

let prod2 ~batch_const f : Maths.t Lqr.prod =
  let primal = bmm2 ~batch_const (Maths.const (Maths.primal f)) in
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (bmm2_tangent_F ~batch_const (Maths.const df))
  in
  { primal; tangent }

let prod2_tangent ~batch_const f : Maths.t Lqr.prod =
  let primal = bmm2_tangent_v ~batch_const (Maths.const (Maths.primal f)) in
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (bmm2_tangent_Fv ~batch_const (Maths.const df))
  in
  { primal; tangent }

let maybe_add a b =
  match a, b with
  | None, None -> None
  | Some a, None -> Some a
  | None, Some b -> Some b
  | Some a, Some b -> Some Maths.(a + b)

let maybe_prod f v =
  match f, v with
  | Some f, Some v -> Some (f v)
  | _ -> None

(* -----------------------------------------
   -- Parameter specification     ------
   ----------------------------------------- *)

module Temp = struct
  type ('a, 'o) p =
    { _f : 'o
    ; _Fx_prod : 'a
    ; _Fu_prod : 'a
    ; _cx : 'o
    ; _cu : 'o
    ; _Cxx : 'a
    ; _Cxu : 'o
    ; _Cuu : 'a
    }
  [@@deriving prms]
end

module O = Prms.Option (Prms.P)
module Input = Lqr.Params.Make (O) (Prms.List (Temp.Make (Prms.P) (O)))
module Output = Prms.List (Lqr.Solution.Make (O))

(* a is the state dimension, b is the control dimension, tmax is the horizon length, m is the batch size and k is number of tangents. if batch_constant, F_x, F_u, C_uu, C_xx and C_xu are all the same across trials and hence do not have a leading batch dimension. *)
module Default = struct
  let a = 5
  let b = 3
  let tmax = 10
  let m = 7
  let k = 512
  let batch_const = false
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.cuda_if_available ()
end

(* -----------------------------------------
   -- Main LDS Module    ------
   ----------------------------------------- *)
module Make_LDS (X : module type of Default) = struct
  let to_device = Tensor.of_bigarray ~device:X.device
  let device = X.device
  let kind = X.kind

  (* make sure fx is stable *)
  let sample_stable () =
    let a = Mat.gaussian X.a X.a in
    let r =
      a |> Linalg.eigvals |> Owl.Dense.Matrix.Z.abs |> Owl.Dense.Matrix.Z.re |> Mat.max'
    in
    Mat.(Float.(0.8 / r) $* a)

  let sample_fx_pri () =
    let fx_pri =
      if X.batch_const
      then sample_stable ()
      else
        Array.init X.m ~f:(fun _ ->
          let a = sample_stable () in
          Arr.reshape a [| 1; X.a; X.a |])
        |> Arr.concatenate ~axis:0
    in
    to_device fx_pri

  let sample_fx_tan () =
    let fx_tan =
      if X.batch_const
      then
        Array.init X.k ~f:(fun _ ->
          let a = sample_stable () in
          Arr.reshape a [| 1; X.a; X.a |])
        |> Arr.concatenate ~axis:0
      else
        Array.init X.k ~f:(fun _ ->
          Array.init X.m ~f:(fun _ ->
            let a = sample_stable () in
            Arr.reshape a [| 1; 1; X.a; X.a |])
          |> Arr.concatenate ~axis:1)
        |> Arr.concatenate ~axis:0
    in
    to_device fx_tan

  (* make sure cost matrices are positive definite *)
  let pos_sym ~reg d =
    let ell = Mat.gaussian d d in
    Mat.(add_diag (ell *@ transpose ell) reg)

  let q_of ~reg d =
    let q =
      if X.batch_const
      then pos_sym ~reg d
      else
        Array.init X.m ~f:(fun _ -> Arr.reshape (pos_sym ~reg d) [| 1; d; d |])
        |> Arr.concatenate ~axis:0
    in
    to_device q

  let q_tan_of ~reg d =
    let q_tan =
      if X.batch_const
      then
        Array.init X.k ~f:(fun _ ->
          let _q_tan = pos_sym ~reg d in
          Arr.reshape _q_tan [| 1; d; d |])
        |> Arr.concatenate ~axis:0
      else
        Array.init X.k ~f:(fun _ ->
          Array.init X.m ~f:(fun _ ->
            let _q_tan = pos_sym ~reg d in
            Arr.reshape _q_tan [| 1; 1; d; d |])
          |> Arr.concatenate ~axis:1)
        |> Arr.concatenate ~axis:0
    in
    to_device q_tan

  let sample_q_xx () =
    let pri = q_of ~reg:1. X.a in
    let tan = q_tan_of ~reg:1. X.a in
    Maths.make_dual pri ~t:(Maths.Direct tan)

  let sample_q_uu () =
    let pri = q_of ~reg:1. X.b in
    let tan = q_tan_of ~reg:1. X.b in
    Maths.make_dual pri ~t:(Maths.Direct tan)

  let sample_tangent shape =
    let pri = Tensor.randn ~device ~kind shape in
    let tan = Tensor.randn ~device ~kind (X.k :: shape) in
    Maths.make_dual pri ~t:(Maths.Direct tan)

  let sample_x0 () = sample_tangent [ X.m; X.a ]

  let sample_fx () =
    let pri = sample_fx_pri () in
    let tan = sample_fx_tan () in
    Maths.make_dual pri ~t:(Maths.Direct tan)

  let sample_fu () =
    if X.batch_const
    then sample_tangent [ X.b; X.a ]
    else sample_tangent [ X.m; X.b; X.a ]

  let sample_q_xu () =
    if X.batch_const
    then sample_tangent [ X.a; X.b ]
    else sample_tangent [ X.m; X.a; X.b ]

  let sample_c_x () = sample_tangent [ X.m; X.a ]
  let sample_c_u () = sample_tangent [ X.m; X.b ]
  let sample_f () = sample_tangent [ X.m; X.a ]
  let sample_u () = sample_tangent [ X.m; X.b ]

  let params : (Maths.t option, (Maths.t, Maths.t option) Temp.p list) Params.p =
    Lqr.Params.
      { x0 = Some (sample_x0 ())
      ; params =
          (let tmp =
             Temp.
               { _f = Some (sample_f ())
               ; _Fx_prod = sample_fx ()
               ; _Fu_prod = sample_fu ()
               ; _cx = Some (sample_c_x ())
               ; _cu = Some (sample_c_u ())
               ; _Cxx = sample_q_xx ()
               ; _Cxu = Some (sample_q_xu ())
               ; _Cuu = sample_q_uu ()
               }
           in
           List.init (X.tmax + 1) ~f:(fun _ -> tmp))
      }

  let print s = Stdio.print_endline (Sexp.to_string_hum s)

  (* given parameters such as f_x, f_u and f, returns u list and x list; u list goes from 0 to T-1 and x_targets list goes from 0 to T *)
  let traj_rollout ~x0 ~(f_list : Maths.t f_params list) =
    (* randomly sample u *)
    let tmax = List.length f_list - 1 in
    let u_list = List.init tmax ~f:(fun _ -> sample_u ()) in
    let f_list_except_last = List.drop_last_exn f_list in
    let tmp_einsum a b =
      if X.batch_const
      then Maths.einsum [ a, "ma"; b, "ab" ] "mb"
      else Maths.einsum [ a, "ma"; b, "mab" ] "mb"
    in
    let _, x_list =
      List.fold2_exn
        f_list_except_last
        u_list
        ~init:(x0, [ x0 ])
        ~f:(fun (x, x_list) f_p u ->
          let new_x =
            let common = Maths.(tmp_einsum x f_p._Fx_prod + tmp_einsum u f_p._Fu_prod) in
            match f_p._f with
            | None -> common
            | Some _f -> Maths.(_f + common)
          in
          new_x, new_x :: x_list)
    in
    u_list, List.rev x_list

  let naive_params (x : Input.M.t) =
    let irrelevant = Some (fun _ -> assert false) in
    let params =
      let p0 = List.hd_exn x.params in
      let _Cxx = Some Maths.(p0._Cxx *@ btr p0._Cxx) in
      let _Cuu = Some Maths.(p0._Cuu *@ btr p0._Cuu) in
      List.map x.params ~f:(fun p ->
        Lqr.
          { common =
              { _Fx_prod = Some (bmm ~batch_const:X.batch_const p0._Fx_prod)
              ; _Fx_prod2 = Some (bmm2 ~batch_const:X.batch_const p0._Fx_prod)
              ; _Fu_prod = Some (bmm ~batch_const:X.batch_const p0._Fu_prod)
              ; _Fu_prod2 = Some (bmm2 ~batch_const:X.batch_const p0._Fu_prod)
              ; _Fx_prod_tangent = irrelevant
              ; _Fx_prod2_tangent = irrelevant
              ; _Fu_prod_tangent = irrelevant
              ; _Fu_prod2_tangent = irrelevant
              ; _Cxx
              ; _Cxu = p0._Cxu
              ; _Cuu
              }
          ; _f = p._f
          ; _cx = p._cx
          ; _cu = p._cu
          })
    in
    Lqr.Params.{ x with params }

  let implicit_params (x : Input.M.t) =
    let params =
      let p0 = List.hd_exn x.params in
      let _Cxx = Some Maths.(p0._Cxx *@ btr p0._Cxx) in
      let _Cuu = Some Maths.(p0._Cuu *@ btr p0._Cuu) in
      List.map x.params ~f:(fun p ->
        Lqr.
          { common =
              { _Fx_prod = Some (prod ~batch_const:X.batch_const p0._Fx_prod)
              ; _Fx_prod2 = Some (prod2 ~batch_const:X.batch_const p0._Fx_prod)
              ; _Fu_prod = Some (prod ~batch_const:X.batch_const p0._Fu_prod)
              ; _Fu_prod2 = Some (prod2 ~batch_const:X.batch_const p0._Fu_prod)
              ; _Fx_prod_tangent =
                  Some (prod_tangent ~batch_const:X.batch_const p._Fx_prod)
              ; _Fx_prod2_tangent =
                  Some (prod2_tangent ~batch_const:X.batch_const p._Fx_prod)
              ; _Fu_prod_tangent =
                  Some (prod_tangent ~batch_const:X.batch_const p._Fu_prod)
              ; _Fu_prod2_tangent =
                  Some (prod2_tangent ~batch_const:X.batch_const p._Fu_prod)
              ; _Cxx
              ; _Cxu = p0._Cxu
              ; _Cuu
              }
          ; _f = p._f
          ; _cx = p._cx
          ; _cu = p._cu
          })
    in
    Lqr.Params.{ x with params }
end
