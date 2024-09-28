open Base
open Torch
include Lqt_type
include Maths

(* extract t-th element from list; if list is length 1 then use the same element *)
let extract list t =
  if List.length list = 1 then List.hd_exn list else List.nth_exn list t

let trans_2d x = Maths.transpose x ~dim0:1 ~dim1:0
let trans_2d_tensor x = Tensor.transpose x ~dim0:1 ~dim1:0

(* linear quadratic trajectory tracking; everything here is a Maths.t object *)
let lqt
  ~(state_params : state_params)
  ~(x_u_desired : x_u_desired)
  ~(cost_params : cost_params)
  =
  let x_0 = x_u_desired.x_0 in
  let x_d_list = x_u_desired.x_d_list in
  let u_d_list = x_u_desired.u_d_list in
  let a_list = state_params.a_list in
  let b_list = state_params.b_list in
  let q_list = cost_params.q_list in
  let r_list = cost_params.r_list in
  let f_t_list = state_params.f_t_list in
  let b_eg = Maths.primal (List.hd_exn b_list) in
  (* batch size *)
  let m = List.hd_exn (Tensor.shape (Maths.primal x_0)) in
  (* state dim *)
  let a_dim = List.hd_exn (Tensor.shape b_eg) in
  (* control dim *)
  let b_dim = List.nth_exn (Tensor.shape b_eg) 1 in
  let device = Tensor.device b_eg in
  let kind = Tensor.type_ b_eg in
  let eye_a_dim = Tensor.eye ~n:a_dim ~options:(kind, device) |> Maths.const in
  let eye_b_dim = Tensor.eye ~n:b_dim ~options:(kind, device) |> Maths.const in
  (* iterate the matrix P and the vector p simultaneously *)
  let p_mat_vec_iter ~p_mat_next ~p_vec_next t =
    let x_d_t = if t = 0 then Some x_0 else extract x_d_list Int.(t - 1) in
    let u_d_t = extract u_d_list t in
    let f_t = extract f_t_list t in
    let q_t = extract q_list Int.(t - 1) in
    let r_t = extract r_list t in
    let a_t, b_t = extract a_list t, extract b_list t in
    let a_t_trans, b_t_trans = trans_2d a_t, trans_2d b_t in
    let lambda = Maths.(r_t + (b_t_trans *@ p_mat_next *@ b_t)) in
    let lambda_inv = Maths.inv_sqr lambda in
    let common = Maths.(p_mat_next *@ b_t *@ lambda_inv *@ b_t_trans) in
    let p_mat_curr =
      Maths.(q_t + (a_t_trans *@ (p_mat_next - (common *@ p_mat_next)) *@ a_t))
    in
    let p_vec_curr =
      let tmp1 =
        match u_d_t, f_t with
        | None, None -> p_vec_next
        | Some u_d_t, None -> Maths.(p_vec_next - (u_d_t *@ b_t_trans *@ p_mat_next))
        | Some u_d_t, Some f_t ->
          Maths.(p_vec_next - (((u_d_t *@ b_t_trans) + f_t) *@ p_mat_next))
        | None, Some f_t -> Maths.(p_vec_next - (f_t *@ p_mat_next))
      in
      let tmp2 =
        Maths.((eye_a_dim - (p_mat_next *@ b_t *@ lambda_inv *@ b_t_trans)) *@ a_t)
      in
      match x_d_t with
      | None -> Maths.(tmp1 *@ tmp2)
      | Some x_d_t -> Maths.((x_d_t *@ q_t) + (tmp1 *@ tmp2))
    in
    p_mat_curr, p_vec_curr
  in
  (* get a list of P matrices and p vectors form t = 0 to t = T - 1 *)
  let p_mat_list, p_vec_list =
    let q_T = List.last_exn q_list in
    (* if no x_d_T set p_vec_T to zeros *)
    let p_vec_final =
      let x_T = List.last_exn x_d_list in
      match x_T with
      | None -> Tensor.zeros [ m; a_dim ] ~device |> Maths.const
      | Some x_T -> Maths.(x_T *@ q_T)
    in
    let rec backwards_p_mat t p_mat_next p_vec_next p_mat_accu p_vec_accu =
      if t = 0
      then p_mat_accu, p_vec_accu
      else (
        let p_mat_curr, p_vec_curr = p_mat_vec_iter ~p_mat_next ~p_vec_next t in
        backwards_p_mat
          Int.(t - 1)
          p_mat_curr
          p_vec_curr
          (p_mat_curr :: p_mat_accu)
          (p_vec_curr :: p_vec_accu))
    in
    backwards_p_mat Int.(state_params.n_steps - 1) q_T p_vec_final [ q_T ] [ p_vec_final ]
  in
  let calc_u ~p_mat_next ~p_vec_next ~x_curr ~f_curr ~a_curr ~b_curr ~r_curr ~u_d_curr =
    let r_curr_inv = Maths.inv_sqr r_curr in
    let sigma =
      Maths.(eye_b_dim + (r_curr_inv *@ trans_2d b_curr *@ p_mat_next *@ b_curr))
    in
    let sigma_inv = Maths.inv_sqr sigma in
    let tmp1 =
      let common = Maths.(x_curr *@ trans_2d a_curr *@ p_mat_next) in
      match f_curr with
      | None -> Maths.(common - p_vec_next)
      | Some f_curr -> Maths.(common + (f_curr *@ p_mat_next) - p_vec_next)
    in
    let tmp_mult = Maths.(neg tmp1 *@ b_curr *@ r_curr_inv *@ sigma_inv) in
    let u_curr =
      match u_d_curr with
      | None -> tmp_mult
      | Some u_d_curr -> Maths.(tmp_mult + (u_d_curr *@ sigma_inv))
    in
    u_curr
  in
  (* forward pass to obtain optimal control and resulting states *)
  let x_list, u_list =
    let rec forward x_curr x_accu u_accu t =
      if t = state_params.n_steps
      then List.rev x_accu, List.rev u_accu
      else (
        let p_mat_next = List.nth_exn p_mat_list t in
        let p_vec_next = List.nth_exn p_vec_list t in
        let f_curr = extract f_t_list t in
        let a_curr, b_curr = extract a_list t, extract b_list t in
        let r_curr = extract r_list t in
        let u_d_curr = extract u_d_list t in
        let u_curr =
          calc_u ~p_mat_next ~p_vec_next ~x_curr ~f_curr ~a_curr ~b_curr ~r_curr ~u_d_curr
        in
        let x_next =
          let common =
            Maths.((x_curr *@ trans_2d a_curr) + (u_curr *@ trans_2d b_curr))
          in
          match f_curr with
          | None -> common
          | Some f_curr -> Maths.(common + f_curr)
        in
        forward x_next (x_next :: x_accu) (u_curr :: u_accu) Int.(t + 1))
    in
    forward x_0 [ x_0 ] [] 0
  in
  x_list, u_list

(* linear quadratic trajectory tracking; everything here is a tensor object *)
let lqt_tensor
  ~(state_params : state_params_tensor)
  ~(x_u_desired : x_u_desired_tensor)
  ~(cost_params : cost_params_tensor)
  =
  let x_0 = x_u_desired.x_0 in
  let x_d_list = x_u_desired.x_d_list in
  let u_d_list = x_u_desired.u_d_list in
  let a_list = state_params.a_list in
  let b_list = state_params.b_list in
  let q_list = cost_params.q_list in
  let r_list = cost_params.r_list in
  let f_t_list = state_params.f_t_list in
  let b_eg = List.hd_exn b_list in
  (* batch size *)
  let m = List.hd_exn (Tensor.shape x_0) in
  (* state dim *)
  let a_dim = List.hd_exn (Tensor.shape b_eg) in
  (* control dim *)
  let b_dim = List.nth_exn (Tensor.shape b_eg) 1 in
  let device = Tensor.device b_eg in
  let kind = Tensor.type_ b_eg in
  let eye_a_dim = Tensor.eye ~n:a_dim ~options:(kind, device) in
  let eye_b_dim = Tensor.eye ~n:b_dim ~options:(kind, device) in
  (* iterate the matrix P and the vector p simultaneously *)
  let p_mat_vec_iter ~p_mat_next ~p_vec_next t =
    let x_d_t = if t = 0 then Some x_0 else extract x_d_list Int.(t - 1) in
    let u_d_t = extract u_d_list t in
    let f_t = extract f_t_list t in
    let q_t = extract q_list Int.(t - 1) in
    let r_t = extract r_list t in
    let a_t, b_t = extract a_list t, extract b_list t in
    let a_t_trans, b_t_trans = trans_2d_tensor a_t, trans_2d_tensor b_t in
    let lambda = Tensor.(r_t + matmul b_t_trans (matmul p_mat_next b_t)) in
    let lambda_inv = Tensor.inverse lambda in
    let common = Tensor.(matmul p_mat_next (matmul b_t (matmul lambda_inv b_t_trans))) in
    let p_mat_curr =
      Tensor.(
        q_t + matmul a_t_trans (p_mat_next - (matmul (matmul common p_mat_next)) a_t))
    in
    let p_vec_curr =
      let tmp1 =
        match u_d_t, f_t with
        | None, None -> p_vec_next
        | Some u_d_t, None ->
          Tensor.(p_vec_next - matmul u_d_t (matmul b_t_trans p_mat_next))
        | Some u_d_t, Some f_t ->
          Tensor.(p_vec_next - matmul (matmul u_d_t b_t_trans + f_t) p_mat_next)
        | None, Some f_t -> Tensor.(p_vec_next - matmul f_t p_mat_next)
      in
      let tmp2 =
        let tmp = Tensor.(matmul p_mat_next (matmul b_t (matmul lambda_inv b_t_trans))) in
        Tensor.(matmul (eye_a_dim - tmp) a_t)
      in
      match x_d_t with
      | None -> Tensor.(matmul tmp1 tmp2)
      | Some x_d_t -> Tensor.(matmul x_d_t q_t + matmul tmp1 tmp2)
    in
    p_mat_curr, p_vec_curr
  in
  (* get a list of P matrices and p vectors form t = 0 to t = T - 1 *)
  let p_mat_list, p_vec_list =
    let q_T = List.last_exn q_list in
    (* if no x_d_T set p_vec_T to zeros *)
    let p_vec_final =
      let x_T = List.last_exn x_d_list in
      match x_T with
      | None -> Tensor.zeros [ m; a_dim ] ~device
      | Some x_T -> Tensor.(matmul x_T q_T)
    in
    let rec backwards_p_mat t p_mat_next p_vec_next p_mat_accu p_vec_accu =
      if t = 0
      then p_mat_accu, p_vec_accu
      else (
        let p_mat_curr, p_vec_curr = p_mat_vec_iter ~p_mat_next ~p_vec_next t in
        backwards_p_mat
          Int.(t - 1)
          p_mat_curr
          p_vec_curr
          (p_mat_curr :: p_mat_accu)
          (p_vec_curr :: p_vec_accu))
    in
    backwards_p_mat Int.(state_params.n_steps - 1) q_T p_vec_final [ q_T ] [ p_vec_final ]
  in
  let calc_u ~p_mat_next ~p_vec_next ~x_curr ~f_curr ~a_curr ~b_curr ~r_curr ~u_d_curr =
    let r_curr_inv = Tensor.inverse r_curr in
    let sigma =
      let tmp =
        Tensor.(
          matmul r_curr_inv (matmul (trans_2d_tensor b_curr) (matmul p_mat_next b_curr)))
      in
      Tensor.(eye_b_dim + tmp)
    in
    let sigma_inv = Tensor.inverse sigma in
    let tmp1 =
      let common = Tensor.(matmul x_curr (matmul (trans_2d_tensor a_curr) p_mat_next)) in
      match f_curr with
      | None -> Tensor.(common - p_vec_next)
      | Some f_curr -> Tensor.(common + matmul f_curr p_mat_next - p_vec_next)
    in
    let tmp_mult =
      Tensor.(neg (matmul tmp1 (matmul b_curr (matmul r_curr_inv sigma_inv))))
    in
    let u_curr =
      match u_d_curr with
      | None -> tmp_mult
      | Some u_d_curr -> Tensor.(tmp_mult + matmul u_d_curr sigma_inv)
    in
    u_curr
  in
  (* forward pass to obtain optimal control and resulting states *)
  let x_list, u_list =
    let rec forward x_curr x_accu u_accu t =
      if t = state_params.n_steps
      then List.rev x_accu, List.rev u_accu
      else (
        let p_mat_next = List.nth_exn p_mat_list t in
        let p_vec_next = List.nth_exn p_vec_list t in
        let f_curr = extract f_t_list t in
        let a_curr, b_curr = extract a_list t, extract b_list t in
        let r_curr = extract r_list t in
        let u_d_curr = extract u_d_list t in
        let u_curr =
          calc_u ~p_mat_next ~p_vec_next ~x_curr ~f_curr ~a_curr ~b_curr ~r_curr ~u_d_curr
        in
        let x_next =
          let common =
            Tensor.(
              matmul x_curr (trans_2d_tensor a_curr)
              + matmul u_curr (trans_2d_tensor b_curr))
          in
          match f_curr with
          | None -> common
          | Some f_curr -> Tensor.(common + f_curr)
        in
        forward x_next (x_next :: x_accu) (u_curr :: u_accu) Int.(t + 1))
    in
    forward x_0 [ x_0 ] [] 0
  in
  x_list, u_list
