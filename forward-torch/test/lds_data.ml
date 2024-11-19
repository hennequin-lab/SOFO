open Base
open Torch
open Forward_torch
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S
module C = Owl.Dense.Matrix.C

let within x (a, b) = Float.(a < x) && Float.(x < b)

module Complex = Stdlib.Complex

(* make sure eigvals of a are within 0.1 and 0.9. sample from a truncated exponential *)
let rec sample_a size =
  let q, r, _ = Owl.Linalg.S.qr Mat.(gaussian size size) in
  let q = Mat.(q * signum (diag r)) in
  let d =
    Mat.init_2d 1 size (fun _ _ -> 0.1 +. Owl_stats.exponential_rvs ~lambda:(1. /. 3.))
  in
  let d = Mat.diagm d in
  let a = Mat.(transpose (sqrt d) * q * sqrt (reci (d +$ 1.))) in
  (* calculate eigvals of a *)
  let a_eigvals = Owl.Linalg.S.eigvals a in
  let a_eigvals_norm_1 = Complex.(norm (C.get a_eigvals 0 0)) in
  let a_eigvals_norm_2 = Complex.(norm (C.get a_eigvals 0 1)) in
  (* reject if eigvals too big or too small. *)
  if within a_eigvals_norm_1 (0.1, 0.5) && within a_eigvals_norm_2 (0.1, 0.5)
     (* global covariance set to I + D *)
  then a
  else sample_a size

type 'a state = { x_curr : 'a }

type 'a accu =
  { lds_params : 'a array * 'a array
  ; us : 'a array
  ; xs : 'a array
  ; f_ts : 'a array
  }

(* a is the state dimension, b is the control dimension, n_steps is the horizon length. *)
module Default = struct
  let a = 4
  let b = 1
  let n_steps = 10
  let kind = Torch_core.Kind.(T f32)
  let device = Torch.Device.cuda_if_available ()
end

module Make_LDS (X : module type of Default) = struct
  let to_device = Tensor.of_bigarray ~device:X.device

  (* construct a_tot from a_lower_tri *)
  let a_tot_from_a a_lower_tri =
    let id_a_half = Tensor.eye ~n:Int.(X.a / 2) ~options:(X.kind, X.device) in
    let zero_a_half = Tensor.zeros [ Int.(X.a / 2); Int.(X.a / 2) ] ~device:X.device in
    let a_upper = Tensor.concat [ zero_a_half; id_a_half ] ~dim:1 in
    let a_lower = Tensor.concat [ id_a_half; a_lower_tri ] ~dim:1 in
    Tensor.concat [ a_upper; a_lower ] ~dim:0

  (* generate parameters for lds *)
  let sample_lds_params () =
    (* if coupling layer dynamics *)
    let a_tot_list =
      Array.init X.n_steps ~f:(fun _ ->
        let a_lower_tri = to_device (sample_a Int.(X.a / 2)) in
        a_tot_from_a a_lower_tri)
    in
    let b_tot_list =
      Array.init X.n_steps ~f:(fun _ ->
        let b_upper =
          Tensor.(
            mul_scalar
              (randn [ Int.(X.a / 2); X.b ] ~device:X.device)
              (Scalar.f Float.(1. / of_int X.a)))
        in
        let b_lower = Tensor.zeros [ Int.(X.a / 2); X.b ] ~device:X.device in
        Tensor.concat [ b_upper; b_lower ] ~dim:0)
    in
    (* let a_tot = sample_a X.a in
       let b_tot = Mat.gaussian ~mu:0. ~sigma:Float.(1. / of_int X.a) X.a X.b in *)
    a_tot_list, b_tot_list

  (* u goes from u_0 to u_{T-1}, x goes from x_0 to x_{T} *)
  let sample_traj =
    let lds_params = sample_lds_params () in
    (* initialise control at each step randomly *)
    let u = Array.init X.n_steps ~f:(fun _ -> to_device (Arr.gaussian [| X.b; 1 |])) in
    (* initialise constant at each step randomly *)
    let f_t = Array.init X.n_steps ~f:(fun _ -> to_device (Arr.gaussian [| X.a; 1 |])) in
    (* initialise x0 *)
    let x0 = to_device (Arr.gaussian [| X.a; 1 |]) in
    let a_tot_array, b_tot_array = fst lds_params, snd lds_params in
    let rec iter t state tot_traj =
      if t = X.n_steps
      then List.rev tot_traj
      else (
        let x_next =
          Tensor.(
            matmul a_tot_array.(t) state.x_curr + matmul b_tot_array.(t) u.(t) + f_t.(t))
        in
        iter (t + 1) { x_curr = x_next } (x_next :: tot_traj))
    in
    let xs = iter 0 { x_curr = x0 } [ x0 ] in
    { lds_params; us = u; xs = Array.of_list xs; f_ts = f_t }

  (* refactor from batch array of accus to x0 and a time list of x and u. *)
  let minibatch_as_data minibatch =
    let bs = Array.length minibatch in
    let x_0 =
      let x_0_list =
        List.init bs ~f:(fun i -> Tensor.(view minibatch.(i).xs.(0) ~size:[ 1; -1 ]))
      in
      Tensor.concat x_0_list ~dim:0
    in
    (* x_u_list contains x1 to x_T and u_0 to u_T-1. *)
    let x_u_list =
      List.init X.n_steps ~f:(fun t ->
        let x =
          let x_list =
            List.init bs ~f:(fun i ->
              Tensor.(view minibatch.(i).xs.(Int.(t + 1)) ~size:[ 1; -1 ]))
          in
          Tensor.concat x_list ~dim:0
        in
        let u =
          let u_list =
            List.init bs ~f:(fun i -> Tensor.(view minibatch.(i).us.(t) ~size:[ 1; -1 ]))
          in
          Tensor.concat u_list ~dim:0
        in
        let f_t =
          let f_t_list =
            List.init bs ~f:(fun i ->
              Tensor.(view minibatch.(i).f_ts.(t) ~size:[ 1; -1 ]))
          in
          Tensor.concat f_t_list ~dim:0
        in
        x, u, f_t)
    in
    (* a time list of (a_tot, b_tot), each a_tot and each b_tot is batched *)
    let batch_lds_params =
      List.init X.n_steps ~f:(fun t ->
        let a_tot =
          let a_tot_list =
            List.init bs ~f:(fun i ->
              Tensor.view (fst minibatch.(i).lds_params).(t) ~size:[ 1; X.a; X.a ])
          in
          Tensor.concat a_tot_list ~dim:0
        in
        let b_tot =
          let b_tot_list =
            List.init bs ~f:(fun i ->
              Tensor.view (snd minibatch.(i).lds_params).(t) ~size:[ 1; X.a; X.b ])
          in
          Tensor.concat b_tot_list ~dim:0
        in
        a_tot, b_tot)
    in
    batch_lds_params, x_0, x_u_list

  let batch_trajectory bs =
    let batched = Array.init bs ~f:(fun _ -> sample_traj) in
    minibatch_as_data batched
end

(* a is the state dimension, b is the control dimension, n_steps is the horizon length. *)
module Default_Tan = struct
  let a = 4
  let b = 1
  let n_steps = 10
  let n_tangents = 5
  let kind = Torch_core.Kind.(T f32)
  let device = Torch.Device.cuda_if_available ()
end

module Make_LDS_Tan (X : module type of Default_Tan) = struct
  let to_device = Tensor.of_bigarray ~device:X.device

  (* construct a_tot from a_lower_tri *)
  let a_tot_from_a a_lower_tri =
    let id_a_half =
      Tensor.eye ~n:Int.(X.a / 2) ~options:(X.kind, X.device) |> Maths.const
    in
    let zero_a_half =
      Tensor.zeros [ Int.(X.a / 2); Int.(X.a / 2) ] ~device:X.device |> Maths.const
    in
    let a_upper = Maths.concat zero_a_half id_a_half ~dim:1 in
    let a_lower = Maths.concat id_a_half a_lower_tri ~dim:1 in
    Maths.concat a_upper a_lower ~dim:0

  (* generate parameters for lds *)
  let sample_lds_params () =
    (* if coupling layer dynamics *)
    let a_tot_list =
      Array.init X.n_steps ~f:(fun _ ->
        let a_lower_tri =
          let a_primal = to_device (sample_a Int.(X.a / 2)) in
          let a_tangent =
            Maths.Direct
              (to_device (Arr.gaussian [| X.n_tangents; Int.(X.a / 2); Int.(X.a / 2) |]))
          in
          Maths.make_dual a_primal ~t:a_tangent
        in
        a_tot_from_a a_lower_tri)
    in
    let b_tot_list =
      Array.init X.n_steps ~f:(fun _ ->
        let b_upper =
          let b_primal =
            Tensor.(
              mul_scalar
                (randn [ Int.(X.a / 2); X.b ] ~device:X.device)
                (Scalar.f Float.(1. / of_int X.a)))
          in
          let b_tangent =
            Maths.Direct
              Tensor.(
                mul_scalar
                  (randn [ X.n_tangents; Int.(X.a / 2); X.b ] ~device:X.device)
                  (Scalar.f Float.(1. / of_int X.a)))
          in
          Maths.make_dual b_primal ~t:b_tangent
        in
        let b_lower =
          Tensor.zeros [ Int.(X.a / 2); X.b ] ~device:X.device |> Maths.const
        in
        Maths.concat b_upper b_lower ~dim:0)
    in
    (* let a_tot = sample_a X.a in
       let b_tot = Mat.gaussian ~mu:0. ~sigma:Float.(1. / of_int X.a) X.a X.b in *)
    a_tot_list, b_tot_list

  (* u goes from u_0 to u_{T-1}, x goes from x_0 to x_{T} *)
  let sample_traj =
    let lds_params = sample_lds_params () in
    (* initialise control at each step randomly *)
    let u =
      Array.init X.n_steps ~f:(fun _ ->
        let b_primal = to_device (Arr.gaussian [| X.b; 1 |]) in
        let b_tangent =
          Maths.Direct (to_device (Arr.gaussian [| X.n_tangents; X.b; 1 |]))
        in
        Maths.make_dual b_primal ~t:b_tangent)
    in
    (* initialise constant at each step randomly *)
    let f_t =
      Array.init X.n_steps ~f:(fun _ ->
        let f_t_primal = to_device (Arr.gaussian [| X.a; 1 |]) in
        let f_t_tangent =
          Maths.Direct (to_device (Arr.gaussian [| X.n_tangents; X.a; 1 |]))
        in
        Maths.make_dual f_t_primal ~t:f_t_tangent)
    in
    (* initialise x0 *)
    let x0 =
      let x0_primal = to_device (Arr.gaussian [| X.a; 1 |]) in
      let x0_tangent =
        Maths.Direct (to_device (Arr.gaussian [| X.n_tangents; X.a; 1 |]))
      in
      Maths.make_dual x0_primal ~t:x0_tangent
    in
    let a_tot_array, b_tot_array = fst lds_params, snd lds_params in
    let rec iter t state tot_traj =
      if t = X.n_steps
      then List.rev tot_traj
      else (
        let x_next =
          Maths.((a_tot_array.(t) *@ state.x_curr) + (b_tot_array.(t) *@ u.(t)) + f_t.(t))
        in
        iter (t + 1) { x_curr = x_next } (x_next :: tot_traj))
    in
    let xs = iter 0 { x_curr = x0 } [ x0 ] in
    { lds_params; us = u; xs = Array.of_list xs; f_ts = f_t }

  (* refactor from batch array of accus to x0 and a time list of x and u. *)
  let minibatch_as_data minibatch =
    let bs = Array.length minibatch in
    let x_0 =
      let x_0_list =
        List.init bs ~f:(fun i -> Maths.(view minibatch.(i).xs.(0) ~size:[ 1; -1 ]))
      in
      Maths.concat_list x_0_list ~dim:0
    in
    (* x_u_list contains x1 to x_T and u_0 to u_T-1. *)
    let x_u_list =
      List.init X.n_steps ~f:(fun t ->
        let x =
          let x_list =
            List.init bs ~f:(fun i ->
              Maths.(view minibatch.(i).xs.(Int.(t + 1)) ~size:[ 1; -1 ]))
          in
          Maths.concat_list x_list ~dim:0
        in
        let u =
          let u_list =
            List.init bs ~f:(fun i -> Maths.(view minibatch.(i).us.(t) ~size:[ 1; -1 ]))
          in
          Maths.concat_list u_list ~dim:0
        in
        let f_t =
          let f_t_list =
            List.init bs ~f:(fun i -> Maths.(view minibatch.(i).f_ts.(t) ~size:[ 1; -1 ]))
          in
          Maths.concat_list f_t_list ~dim:0
        in
        x, u, f_t)
    in
    (* a time list of (a_tot, b_tot), each a_tot and each b_tot is batched *)
    let batch_lds_params =
      List.init X.n_steps ~f:(fun t ->
        let a_tot =
          let a_tot_list =
            List.init bs ~f:(fun i ->
              Maths.view (fst minibatch.(i).lds_params).(t) ~size:[ 1; X.a; X.a ])
          in
          Maths.concat_list a_tot_list ~dim:0
        in
        let b_tot =
          let b_tot_list =
            List.init bs ~f:(fun i ->
              Maths.view (snd minibatch.(i).lds_params).(t) ~size:[ 1; X.a; X.b ])
          in
          Maths.concat_list b_tot_list ~dim:0
        in
        a_tot, b_tot)
    in
    batch_lds_params, x_0, x_u_list

  let batch_trajectory bs =
    let batched = Array.init bs ~f:(fun _ -> sample_traj) in
    minibatch_as_data batched
end
