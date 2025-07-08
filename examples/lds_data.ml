(* In the simple linear Gaussian case, we assume Fx, Fu, Q_xx, Q_uu and f are the same across all trials in the batch. *)
open Base
open Torch
open Forward_torch
open Lqr
module Mat = Owl.Dense.Matrix.D
module Arr = Owl.Dense.Ndarray.D
module Linalg = Owl.Linalg.D

let print s = Stdio.print_endline (Sexp.to_string_hum s)

(* -----------------------------------------
   -- Some utility functions         ------
   ----------------------------------------- *)

(* Fv prod; if batch_const, F does not have a leading batch dimension. Assume b always has a leading dimension *)
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

(* -----------------------------------------
   -- If tangents      ------
   ----------------------------------------- *)

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

let prod_f ~batch_const ~primal_f ~tan_f f : Maths.any Maths.t Lqr.prod =
  let primal = primal_f ~batch_const (Maths.any (Maths.of_tensor (Maths.to_tensor f))) in
  (* tangent on f only *)
  let tangent =
    match Maths.tangent f with
    | None -> None
    | Some df -> Some (tan_f ~batch_const (Maths.any (Maths.const df)))
  in
  { primal; tangent }

let prod ~batch_const f = prod_f ~batch_const ~primal_f:bmm ~tan_f:bmm_tangent_F f
let prod2 ~batch_const f = prod_f ~batch_const ~primal_f:bmm2 ~tan_f:bmm2_tangent_F f

let prod_tangent ~batch_const f =
  prod_f ~batch_const ~primal_f:bmm_tangent_v ~tan_f:bmm_tangent_Fv f

let prod2_tangent ~batch_const f =
  prod_f ~batch_const ~primal_f:bmm2_tangent_v ~tan_f:bmm2_tangent_Fv f

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

type 'a f_params =
  { _Fx_prod : 'a
  ; _Fu_prod : 'a
  ; _f : 'a option
  ; _c : 'a option (* output/emission parameters *)
  ; _b : 'a option
  ; _cov : 'a option
  }

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

module O = Prms.Option (Prms.Single)
module Input = Lqr.Params.Make (O) (Prms.List (Temp.Make (Prms.Single) (O)))
module Output = Prms.List (Lqr.Solution.Make (O))

(* n is the state dimension, m is the control dimension, o is the output dimension, tmax is the horizon length and m is the batch size. if batch_constant, F_x, F_u, C_uu, C_xx and C_xu are all the same across trials and hence do not have a leading batch dimension. *)
module Default = struct
  let n = 5
  let m = 3
  let o = 4
  let tmax = 10
  let bs = 7
  let k = 512
  let batch_const = false
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.cuda_if_available ()
end

let within x (a, b) = Float.(a < x) && Float.(x < b)

(* make sure fx is stable *)
let sample_stable ~target_sa ~a =
  let w =
    let tmp = Mat.gaussian a a in
    let r = tmp |> Linalg.eigvals |> Owl.Dense.Matrix.Z.re |> Mat.max' in
    Mat.(Float.(target_sa / r) $* tmp)
  in
  let w_i = Mat.((w - eye a) *$ 0.1) in
  Owl.Linalg.Generic.expm w_i

let sample_fx_pri ~target_sa ~batch_const ~m ~a =
  if batch_const
  then sample_stable ~target_sa ~a
  else
    Array.init m ~f:(fun _ ->
      let a_tmp = sample_stable ~target_sa ~a in
      Arr.reshape a_tmp [| 1; a; a |])
    |> Arr.concatenate ~axis:0

(* make sure cost matrices are positive definite *)
let pos_sym ~reg d =
  let ell = Mat.gaussian d d in
  Mat.(add_diag (ell *@ transpose ell) reg)

let q_of ~batch_const ~m ~reg d =
  if batch_const
  then pos_sym ~reg d
  else
    Array.init m ~f:(fun _ -> Arr.reshape (pos_sym ~reg d) [| 1; d; d |])
    |> Arr.concatenate ~axis:0

(* let map_naive (x) ~batch_const =
  let irrelevant = Some (fun _ -> assert false) in
  let params =
    List.map x.params ~f:(fun p ->
      Lqr.
        { common =
            { _Fx_prod = Some (bmm ~batch_const p._Fx_prod)
            ; _Fx_prod2 = Some (bmm2 ~batch_const p._Fx_prod)
            ; _Fu_prod = Some (bmm ~batch_const p._Fu_prod)
            ; _Fu_prod2 = Some (bmm2 ~batch_const p._Fu_prod)
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

let map_implicit (x : Input.M.t) ~batch_const =
  let params =
    List.map x.params ~f:(fun p ->
      Lqr.
        { common =
            { _Fx_prod = Some (prod ~batch_const p._Fx_prod)
            ; _Fx_prod2 = Some (prod2 ~batch_const p._Fx_prod)
            ; _Fu_prod = Some (prod ~batch_const p._Fu_prod)
            ; _Fu_prod2 = Some (prod2 ~batch_const p._Fu_prod)
            ; _Fx_prod_tangent = Some (prod_tangent ~batch_const p._Fx_prod)
            ; _Fx_prod2_tangent = Some (prod2_tangent ~batch_const p._Fx_prod)
            ; _Fu_prod_tangent = Some (prod_tangent ~batch_const p._Fu_prod)
            ; _Fu_prod2_tangent = Some (prod2_tangent ~batch_const p._Fu_prod)
            ; _Cxx = Some p._Cxx
            ; _Cxu = p._Cxu
            ; _Cuu = Some p._Cuu
            }
        ; _f = p._f
        ; _cx = p._cx
        ; _cu = p._cu
        })
  in
  Lqr.Params.{ x with params } *)

(* -----------------------------------------
   -- LDS Module with tangents    ------
   ----------------------------------------- *)
(* this should only be used for memory profiling *)
module Make_LDS (X : module type of Default) = struct
  let to_device = Tensor.of_bigarray ~device:X.device

  let sample_fx_tan ~target_sa =
    let fx_tan =
      if X.batch_const
      then
        Array.init X.k ~f:(fun _ ->
          let a = sample_stable ~target_sa ~a:X.n in
          Arr.reshape a [| 1; X.n; X.n |])
        |> Arr.concatenate ~axis:0
      else
        Array.init X.k ~f:(fun _ ->
          Array.init X.bs ~f:(fun _ ->
            let a = sample_stable ~target_sa ~a:X.n in
            Arr.reshape a [| 1; 1; X.n; X.n |])
          |> Arr.concatenate ~axis:1)
        |> Arr.concatenate ~axis:0
    in
    to_device fx_tan

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
          Array.init X.bs ~f:(fun _ ->
            let _q_tan = pos_sym ~reg d in
            Arr.reshape _q_tan [| 1; 1; d; d |])
          |> Arr.concatenate ~axis:1)
        |> Arr.concatenate ~axis:0
    in
    to_device q_tan

  let sample_q_xx () =
    let pri = q_of ~batch_const:X.batch_const ~m:X.bs ~reg:1. X.n |> to_device in
    let tan = q_tan_of ~reg:1. X.n in
    Maths.dual (Maths.of_tensor pri) ~tangent:(Maths.of_tensor tan)

  let sample_q_uu () =
    let pri = q_of ~batch_const:X.batch_const ~m:X.bs ~reg:1. X.m |> to_device in
    let tan = q_tan_of ~reg:1. X.m in
    Maths.dual (Maths.of_tensor pri) ~tangent:(Maths.of_tensor tan)

  let sample_tangent shape =
    let pri = Tensor.randn ~device:X.device ~kind:X.kind shape in
    let tan = Tensor.randn ~device:X.device ~kind:X.kind (X.k :: shape) in
    Maths.dual (Maths.of_tensor pri) ~tangent:(Maths.of_tensor tan)

  let sample_x0 () = sample_tangent [ X.bs; X.n ]

  let sample_fx ~target_sa =
    let pri =
      sample_fx_pri ~batch_const:X.batch_const ~m:X.bs ~a:X.n ~target_sa |> to_device
    in
    let tan = sample_fx_tan ~target_sa in
    Maths.dual (Maths.of_tensor pri) ~tangent:(Maths.of_tensor tan)

  let sample_fu () =
    if X.batch_const
    then sample_tangent [ X.m; X.n ]
    else sample_tangent [ X.bs; X.m; X.n ]

  let sample_q_xu () =
    if X.batch_const
    then sample_tangent [ X.n; X.m ]
    else sample_tangent [ X.bs; X.n; X.m ]

  let sample_c_x () = sample_tangent [ X.bs; X.n ]
  let sample_c_u () = sample_tangent [ X.bs; X.m ]
  let sample_f () = sample_tangent [ X.bs; X.n ]
  let sample_u () = sample_tangent [ X.bs; X.m ]
  let sample_u_list () = List.init X.tmax ~f:(fun _ -> sample_u ())

  (* output follows a Gaussian with mean= z c + b and diagonal cov *)
  let sample_c () =
    if X.batch_const
    then sample_tangent [ X.n; X.o ]
    else sample_tangent [ X.bs; X.n; X.o ]

  let sample_b () =
    if X.batch_const then sample_tangent [ X.o ] else sample_tangent [ X.bs; X.o ]

  (* sqrt of output covariance *)
  let sample_output_cov () =
    let pri =
      if X.batch_const
      then Tensor.(abs (Tensor.randn ~device:X.device ~kind:X.kind [ X.o ]))
      else Tensor.(abs (Tensor.randn ~device:X.device ~kind:X.kind [ X.bs; X.o ]))
    in
    let tan =
      if X.batch_const
      then Tensor.(abs (Tensor.randn ~device:X.device ~kind:X.kind [ X.k; X.o ]))
      else Tensor.(abs (Tensor.randn ~device:X.device ~kind:X.kind [ X.k; X.bs; X.o ]))
    in
    Maths.dual (Maths.of_tensor pri) ~tangent:(Maths.of_tensor tan)

  let sample_gaussian ~(cov : Maths.const Maths.t) =
    let eps = Maths.any (sample_tangent [ X.bs; X.o ]) in
    let cov_sqrt = Maths.sqrt cov in
    if X.batch_const
    then Maths.einsum [ eps, "ma"; Maths.any cov_sqrt, "ab" ] "mb"
    else Maths.einsum [ eps, "ma"; Maths.any cov_sqrt, "mab" ] "mb"

  (* given parameters such as f_x, f_u and f, returns u list and x list; u list goes from 0 to T-1, x_list goes from 0 to T and o list goes from 1 to T. *)
  let traj_rollout ~x0 ~(f_list : Maths.any Maths.t f_params list) ~u_list =
    let f_list_except_last = List.drop_last_exn f_list in
    let tmp_einsum a b =
      if X.batch_const
      then Maths.einsum [ a, "ma"; b, "ab" ] "mb"
      else Maths.einsum [ a, "ma"; b, "mab" ] "mb"
    in
    let _, x_list, o_list =
      List.fold2_exn
        f_list_except_last
        u_list
        ~init:(x0, [ x0 ], [])
        ~f:(fun (x, x_list, o_list) f_p u ->
          let new_x =
            let common = Maths.(tmp_einsum x f_p._Fx_prod + tmp_einsum u f_p._Fu_prod) in
            match f_p._f with
            | None -> common
            | Some _f -> Maths.(_f + common)
          in
          let new_o =
            match f_p._cov with
            | None -> None
            | Some _cov ->
              let _cov = Maths.const _cov in
              let noise = sample_gaussian ~cov:_cov in
              let b = Option.value_exn f_p._b in
              let b_reshaped = if X.batch_const then Maths.unsqueeze b ~dim:0 else b in
              Some Maths.(tmp_einsum new_x (Option.value_exn f_p._c) + b_reshaped + noise)
          in
          new_x, new_x :: x_list, new_o :: o_list)
    in
    List.rev x_list, List.rev o_list
end

(* -----------------------------------------
   -- LDS Module but with tensors only  ------
   ----------------------------------------- *)
module Default_Tensor = struct
  let n = 5
  let m = 3
  let o = 4
  let tmax = 10
  let bs = 7
  let batch_const = false
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.cuda_if_available ()
end

module Make_LDS_Tensor (X : module type of Default_Tensor) = struct
  let to_device = Tensor.of_bigarray ~device:X.device
  let sample_q_xx () = q_of ~batch_const:X.batch_const ~m:X.bs ~reg:1. X.n |> to_device
  let sample_q_uu () = q_of ~batch_const:X.batch_const ~m:X.bs ~reg:1. X.m |> to_device

  let sample_tensor _shape =
    let n = List.hd_exn _shape in
    Tensor.(f Float.(1. /. sqrt (of_int n)) * randn ~device:X.device ~kind:X.kind _shape)

  let sample_x1 () = sample_tensor [ X.bs; X.n ]

  let sample_fx ~target_sa =
    sample_fx_pri ~target_sa ~batch_const:X.batch_const ~m:X.bs ~a:X.n |> to_device

  let sample_fu () =
    if X.batch_const then sample_tensor [ X.m; X.n ] else sample_tensor [ X.bs; X.m; X.n ]

  let sample_q_xu () =
    if X.batch_const then sample_tensor [ X.n; X.m ] else sample_tensor [ X.bs; X.n; X.m ]

  let sample_c_x () = sample_tensor [ X.bs; X.n ]
  let sample_c_u () = sample_tensor [ X.bs; X.m ]
  let sample_f () = sample_tensor [ X.bs; X.n ]
  let sample_u_list = List.init X.tmax ~f:(fun _ -> sample_tensor [ X.bs; X.m ])

  (* output follows a Gaussian with mean = z c + b and diagonal cov *)
  let sample_c () =
    if X.batch_const then sample_tensor [ X.n; X.o ] else sample_tensor [ X.bs; X.n; X.o ]

  let sample_b () =
    if X.batch_const then sample_tensor [ X.o ] else sample_tensor [ X.bs; X.o ]

  let sample_output_cov () =
    if X.batch_const
    then Tensor.(abs (Tensor.randn ~device:X.device ~kind:X.kind [ X.o ]))
    else Tensor.(abs (Tensor.randn ~device:X.device ~kind:X.kind [ X.bs; X.o ]))

  (* given parameters such as f_x, f_u and f, returns u list and x list; u list goes from 1 to T-1 and x_list goes from 1 to T and o list goes from 1 to T. *)
  let traj_rollout ~x1 ~(f_list : Tensor.t f_params list) ~u_list =
    let tmp_einsum a b =
      let eqn = if X.batch_const then "ma,ab->mb" else "ma,mab->mb" in
      Tensor.einsum ~equation:eqn [ a; b ] ~path:None
    in
    let _, x_list, o_list =
      List.fold2_exn
        f_list
        u_list
        ~init:(x1, [ x1 ], [])
        ~f:(fun (x, x_list, o_list) f_p u ->
          let o =
            match f_p._cov with
            | None -> None
            | Some _cov ->
              let noise =
                let eps = sample_tensor [ X.bs; X.o ] in
                let cov_sqrt = Tensor.sqrt _cov in
                let eqn = if X.batch_const then "ma,a->ma" else "ma,ma->ma" in
                Tensor.einsum [ eps; cov_sqrt ] ~equation:eqn ~path:None
              in
              let with_emission =
                match f_p._c with
                | Some c -> Tensor.(noise + tmp_einsum x c)
                | None -> Tensor.(noise + x)
              in
              let with_b =
                match f_p._b with
                | None -> with_emission
                | Some b ->
                  let b_reshaped =
                    if X.batch_const then Tensor.unsqueeze b ~dim:0 else b
                  in
                  Tensor.(with_emission + b_reshaped)
              in
              Some with_b
          in
          let new_x =
            let common = Tensor.(tmp_einsum x f_p._Fx_prod + tmp_einsum u f_p._Fu_prod) in
            match f_p._f with
            | None -> common
            | Some _f -> Tensor.(_f + common)
          in
          new_x, new_x :: x_list, o :: o_list)
    in
    List.rev x_list, List.rev o_list
end
