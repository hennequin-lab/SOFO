open Base
open Torch
include Lqr_type
include Maths

let print s = Stdio.printf "%s\n%!" (Base.Sexp.to_string_hum s)

(* extract t-th element from list; if list is length 1 then use the same element *)
let extract_list list t =
  if List.length list = 1 then List.hd_exn list else List.nth_exn list t

let extract_list_opt t list ~shape ~device =
  match list with
  | None -> Maths.const (Tensor.zeros shape ~device)
  | Some list -> List.nth_exn list t

let extract_list_opt_tensor t list ~shape ~device =
  match list with
  | None -> Tensor.zeros shape ~device
  | Some list -> List.nth_exn list t

let trans_2d x = Maths.transpose x ~dim0:1 ~dim1:0
let trans_2d_tensor x = Tensor.transpose x ~dim0:1 ~dim1:0

(* Transpose a list of lists *)
let rec transpose lst =
  match lst with
  | [] -> []
  | [] :: _ -> []
  | _ -> List.map ~f:List.hd_exn lst :: transpose (List.map ~f:List.tl_exn lst)

(* linear quadratic regulator; everything here is a Maths.t object *)
let lqr ~(state_params : state_params) ~(cost_params : cost_params) =
  let n_steps = state_params.n_steps in
  let x_0 = state_params.x_0 in
  let f_x_list = state_params.f_x_list in
  let f_u_list = state_params.f_u_list in
  let f_t_list = state_params.f_t_list in
  let c_xx_list = cost_params.c_xx_list in
  let c_xu_list = cost_params.c_xu_list in
  let c_uu_list = cost_params.c_uu_list in
  let c_x_list = cost_params.c_x_list in
  let c_u_list = cost_params.c_u_list in
  let f_u_eg = Maths.primal (List.hd_exn f_u_list) in
  (* batch size *)
  let m = List.hd_exn (Tensor.shape (Maths.primal x_0)) in
  (* state dim *)
  let a_dim = List.hd_exn (Tensor.shape f_u_eg) in
  (* control dim *)
  let b_dim = List.nth_exn (Tensor.shape f_u_eg) 1 in
  let device = Tensor.device f_u_eg in
  (* step 1: backward pass to calculate K_t and k_t *)
  let v_mat_final = List.last_exn c_xx_list in
  let v_vec_final = extract_list_opt n_steps c_x_list ~shape:[ m; a_dim ] ~device in
  let backward t v_mat_next v_vec_next =
    let c_uu_curr = extract_list c_uu_list t in
    let c_xx_curr = extract_list c_xx_list t in
    let c_xu_curr = extract_list_opt t c_xu_list ~shape:[ a_dim; b_dim ] ~device in
    let c_x_curr = extract_list_opt t c_x_list ~shape:[ m; a_dim ] ~device in
    let c_u_curr = extract_list_opt t c_u_list ~shape:[ m; b_dim ] ~device in
    let f_x_curr = extract_list f_x_list t in
    let f_u_curr = extract_list f_u_list t in
    let f_t_curr = extract_list_opt t f_t_list ~shape:[ m; a_dim ] ~device in
    let q_uu_curr = Maths.(c_uu_curr + (trans_2d f_u_curr *@ v_mat_next *@ f_u_curr)) in
    let q_xx_curr = Maths.(c_xx_curr + (trans_2d f_x_curr *@ v_mat_next *@ f_x_curr)) in
    let q_xu_curr = Maths.(c_xu_curr + (trans_2d f_x_curr *@ v_mat_next *@ f_u_curr)) in
    let q_u_curr =
      Maths.(c_u_curr + (v_vec_next *@ f_u_curr) + (f_t_curr *@ v_mat_next *@ f_u_curr))
    in
    let q_x_curr =
      Maths.(c_x_curr + (v_vec_next *@ f_x_curr) + (f_t_curr *@ v_mat_next *@ f_x_curr))
    in
    let k_mat_curr =
      Maths.linsolve q_uu_curr (Maths.neg (trans_2d q_xu_curr)) ~left:true
    in
    let k_vec_curr =
      Maths.linsolve q_uu_curr (Maths.neg (trans_2d q_u_curr)) ~left:true |> trans_2d
    in
    let v_mat_curr = Maths.(q_xx_curr + (q_xu_curr *@ k_mat_curr)) in
    let v_vec_curr = Maths.(q_x_curr + (q_u_curr *@ k_mat_curr)) in
    v_mat_curr, v_vec_curr, k_mat_curr, k_vec_curr
  in
  (* k_mat and k_vec go from 0 to T-1 *)
  let k_mat_list, k_vec_list =
    let rec backward_pass t v_mat_next v_vec_next k_mat_accu k_vec_accu =
      if t = -1
      then k_mat_accu, k_vec_accu
      else (
        let v_mat_curr, v_vec_curr, k_mat_curr, k_vec_curr =
          backward t v_mat_next v_vec_next
        in
        backward_pass
          Int.(t - 1)
          v_mat_curr
          v_vec_curr
          (k_mat_curr :: k_mat_accu)
          (k_vec_curr :: k_vec_accu))
    in
    backward_pass Int.(n_steps - 1) v_mat_final v_vec_final [] []
  in
  (* step 2: forward pass to obtain controls and states. *)
  let forward t x_curr =
    let f_x_curr = extract_list f_x_list t in
    let f_u_curr = extract_list f_u_list t in
    let k_mat_curr = List.nth_exn k_mat_list t in
    let k_vec_curr = List.nth_exn k_vec_list t in
    let u_curr = Maths.((x_curr *@ trans_2d k_mat_curr) + k_vec_curr) in
    let x_next = Maths.((x_curr *@ trans_2d f_x_curr) + (u_curr *@ trans_2d f_u_curr)) in
    x_next, u_curr
  in
  (* x goes from 0 to T and u goes from 1 to T *)
  let x_list, u_list =
    let rec forward_pass t x_curr x_accu u_accu =
      if t = n_steps
      then List.rev x_accu, List.rev u_accu
      else (
        let x_next, u_curr = forward t x_curr in
        forward_pass Int.(t + 1) x_next (x_next :: x_accu) (u_curr :: u_accu))
    in
    forward_pass 0 x_0 [ x_0 ] []
  in
  x_list, u_list

(* linear quadratic regulator; everything here is a Tensor object *)
let lqr_tensor ~(state_params : state_params_tensor) ~(cost_params : cost_params_tensor) =
  let n_steps = state_params.n_steps in
  let x_0 = state_params.x_0 in
  let f_x_list = state_params.f_x_list in
  let f_u_list = state_params.f_u_list in
  let f_t_list = state_params.f_t_list in
  let c_xx_list = cost_params.c_xx_list in
  let c_xu_list = cost_params.c_xu_list in
  let c_uu_list = cost_params.c_uu_list in
  let c_x_list = cost_params.c_x_list in
  let c_u_list = cost_params.c_u_list in
  let f_u_eg = List.hd_exn f_u_list in
  (* batch size *)
  let m = List.hd_exn (Tensor.shape x_0) in
  (* state dim *)
  let a_dim = List.hd_exn (Tensor.shape f_u_eg) in
  (* control dim *)
  let b_dim = List.nth_exn (Tensor.shape f_u_eg) 1 in
  let device = Tensor.device f_u_eg in
  (* step 1: backward pass to calculate K_t and k_t *)
  let v_mat_final = List.last_exn c_xx_list in
  let v_vec_final =
    extract_list_opt_tensor n_steps c_x_list ~shape:[ m; a_dim ] ~device
  in
  let backward t v_mat_next v_vec_next =
    let c_uu_curr = extract_list c_uu_list t in
    let c_xx_curr = extract_list c_xx_list t in
    let c_xu_curr = extract_list_opt_tensor t c_xu_list ~shape:[ a_dim; b_dim ] ~device in
    let c_x_curr = extract_list_opt_tensor t c_x_list ~shape:[ m; a_dim ] ~device in
    let c_u_curr = extract_list_opt_tensor t c_u_list ~shape:[ m; b_dim ] ~device in
    let f_x_curr = extract_list f_x_list t in
    let f_u_curr = extract_list f_u_list t in
    let f_t_curr = extract_list_opt_tensor t f_t_list ~shape:[ m; a_dim ] ~device in
    let q_uu_curr =
      Tensor.(c_uu_curr + matmul (matmul (trans_2d_tensor f_u_curr) v_mat_next) f_u_curr)
    in
    let q_xx_curr =
      Tensor.(c_xx_curr + matmul (matmul (trans_2d_tensor f_x_curr) v_mat_next) f_x_curr)
    in
    let q_xu_curr =
      Tensor.(c_xu_curr + matmul (matmul (trans_2d_tensor f_x_curr) v_mat_next) f_u_curr)
    in
    let q_u_curr =
      Tensor.(
        c_u_curr
        + matmul v_vec_next f_u_curr
        + matmul (matmul f_t_curr v_mat_next) f_u_curr)
    in
    let q_x_curr =
      Tensor.(
        c_x_curr
        + matmul v_vec_next f_x_curr
        + matmul (matmul f_t_curr v_mat_next) f_x_curr)
    in
    let k_mat_curr =
      Tensor.linalg_solve
        ~a:q_uu_curr
        ~b:(Tensor.neg (trans_2d_tensor q_xu_curr))
        ~left:true
    in
    let k_vec_curr =
      Tensor.linalg_solve
        ~a:q_uu_curr
        ~b:(Tensor.neg (trans_2d_tensor q_u_curr))
        ~left:true
      |> trans_2d_tensor
    in
    let v_mat_curr = Tensor.(q_xx_curr + matmul q_xu_curr k_mat_curr) in
    let v_vec_curr = Tensor.(q_x_curr + matmul q_u_curr k_mat_curr) in
    v_mat_curr, v_vec_curr, k_mat_curr, k_vec_curr
  in
  (* k_mat and k_vec go from 0 to T-1 *)
  let k_mat_list, k_vec_list =
    let rec backward_pass t v_mat_next v_vec_next k_mat_accu k_vec_accu =
      if t = -1
      then k_mat_accu, k_vec_accu
      else (
        let v_mat_curr, v_vec_curr, k_mat_curr, k_vec_curr =
          backward t v_mat_next v_vec_next
        in
        backward_pass
          Int.(t - 1)
          v_mat_curr
          v_vec_curr
          (k_mat_curr :: k_mat_accu)
          (k_vec_curr :: k_vec_accu))
    in
    backward_pass Int.(n_steps - 1) v_mat_final v_vec_final [] []
  in
  (* step 2: forward pass to obtain controls and states. *)
  let forward t x_curr =
    let f_x_curr = extract_list f_x_list t in
    let f_u_curr = extract_list f_u_list t in
    let k_mat_curr = List.nth_exn k_mat_list t in
    let k_vec_curr = List.nth_exn k_vec_list t in
    let u_curr = Tensor.(matmul x_curr (trans_2d_tensor k_mat_curr) + k_vec_curr) in
    let x_next =
      Tensor.(
        matmul x_curr (trans_2d_tensor f_x_curr)
        + matmul u_curr (trans_2d_tensor f_u_curr))
    in
    x_next, u_curr
  in
  (* x goes from 0 to T and u goes from 1 to T *)
  let x_list, u_list =
    let rec forward_pass t x_curr x_accu u_accu =
      if t = n_steps
      then List.rev x_accu, List.rev u_accu
      else (
        let x_next, u_curr = forward t x_curr in
        forward_pass Int.(t + 1) x_next (x_next :: x_accu) (u_curr :: u_accu))
    in
    forward_pass 0 x_0 [ x_0 ] []
  in
  x_list, u_list

(* separate primal and tangents and perform a total of (K+1) lqrs *)
(* let sep_primal_tan_lqr ~(state_params : state_params) ~(cost_params : cost_params) =
   let n_steps = state_params.n_steps in
   let x_0 = state_params.x_0 in
   let f_x_list = state_params.f_x_list in
   let f_u_list = state_params.f_u_list in
   let f_t_list = state_params.f_t_list in
   let c_xx_list = cost_params.c_xx_list in
   let c_xu_list = cost_params.c_xu_list in
   let c_uu_list = cost_params.c_uu_list in
   let c_x_list = cost_params.c_x_list in
   let c_u_list = cost_params.c_u_list in
   let f_u_eg = Maths.primal (List.hd_exn f_u_list) in
   (* batch size *)
   let m = List.hd_exn (Tensor.shape (Maths.primal x_0)) in
   (* state dim *)
   let a_dim = List.hd_exn (Tensor.shape f_u_eg) in
   (* control dim *)
   let b_dim = List.nth_exn (Tensor.shape f_u_eg) 1 in
   let device = Tensor.device f_u_eg in
   (* step 1: lqr on the primal *)
   let extract_primal_opt list =
   List.map list ~f:(fun i -> Option.map i ~f:Maths.primal)
   in
   let extract_primal list = List.map list ~f:Maths.primal in
   let extract_primal_list_opt list = Option.value_map list ~f:(fun x -> Some (List.map x ~f:Maths.primal)) ~default:None  in

   let x_0_primal = Maths.primal x_0 in
   let f_x_primal_list = extract_primal f_x_list in
   let f_u_primal_list = extract_primal f_u_list in
   let f_t_primal_list = extract_primal_list_opt f_t_list in
   let c_xx_primal_list = cost_params.c_xx_list in
   let c_xu_primal_list = extract_primal_list_opt c_xu_list in
   let c_uu_primal_list = cost_params.c_uu_list in
   let c_x_primal_list = extract_primal_list_opt c_x_list in
   let c_u_primal_list = extract_primal_list_opt c_u_list in *)
