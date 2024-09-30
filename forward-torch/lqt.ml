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
      Tensor.(q_t + matmul a_t_trans (matmul (p_mat_next - matmul common p_mat_next) a_t))
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

(* Transpose a list of lists *)
let rec transpose lst =
  match lst with
  | [] -> []
  | [] :: _ -> []
  | _ -> List.map ~f:List.hd_exn lst :: transpose (List.map ~f:List.tl_exn lst)

(* seperate primal and tangents and perform a total of (K+1) lqts *)
let sep_primal_tan_lqt
  ~(state_params : state_params)
  ~(x_u_desired : x_u_desired)
  ~(cost_params : cost_params)
  =
  (* step 0: extract the parameters *)
  let x_0 = x_u_desired.x_0 in
  let x_d_list = x_u_desired.x_d_list in
  let u_d_list = x_u_desired.u_d_list in
  let a_list = state_params.a_list in
  let b_list = state_params.b_list in
  let q_list = cost_params.q_list in
  let r_list = cost_params.r_list in
  let f_t_list = state_params.f_t_list in
  let b_eg = Maths.primal (List.hd_exn b_list) in
  let device = Tensor.device b_eg in
  let n_tangents = List.hd_exn (Tensor.shape (Option.value_exn (Maths.tangent x_0))) in
  (* step 1: lqt on the primal *)
  let extract_primal_opt list =
    List.map list ~f:(fun i -> Option.map i ~f:Maths.primal)
  in
  let extract_primal list = List.map list ~f:Maths.primal in
  let x_0_primal = Maths.primal x_0 in
  let x_d_primal_list = extract_primal_opt x_d_list in
  let u_d_primal_list = extract_primal_opt u_d_list in
  let f_t_primal_list = extract_primal_opt f_t_list in
  let a_primal_list = extract_primal a_list in
  let b_primal_list = extract_primal b_list in
  let q_primal_list = extract_primal q_list in
  let r_primal_list = extract_primal r_list in
  let state_params_tensor =
    { n_steps = state_params.n_steps
    ; a_list = a_primal_list
    ; b_list = b_primal_list
    ; f_t_list = f_t_primal_list
    }
  in
  let x_u_desired_tensor =
    { x_0 = x_0_primal; x_d_list = x_d_primal_list; u_d_list = u_d_primal_list }
  in
  let cost_params_tensor = { q_list = q_primal_list; r_list = r_primal_list } in
  let x_list_primal, u_list_primal =
    lqt_tensor
      ~state_params:state_params_tensor
      ~x_u_desired:x_u_desired_tensor
      ~cost_params:cost_params_tensor
  in
  (* step 2: for each tangent, calculate the new targets *)
  let extract_kth_tan ~list ~k tan_shape =
    List.map list ~f:(fun x ->
      let tangent = Maths.tangent x in
      let reshaped_tangent =
        Option.value_map tangent ~default:(Tensor.zeros tan_shape ~device) ~f:(fun tan ->
          let k_th_tan =
            Tensor.slice tan ~dim:0 ~start:(Some k) ~end_:(Some Int.(k + 1)) ~step:1
          in
          Tensor.reshape k_th_tan ~shape:tan_shape)
      in
      Some reshaped_tangent)
  in
  let extract_and_reshape ~list k =
    let shape = Tensor.shape (Maths.primal (List.hd_exn list)) in
    extract_kth_tan ~list ~k shape
  in
  (* iterate to extract lambda obtained from the primal lqt *)
  let lambda_list =
    let lambda_final =
      let q_final = List.last_exn q_primal_list in
      let x_final = List.last_exn x_list_primal in
      let x_d_final =
        let x_d_final_opt = List.last_exn x_d_primal_list in
        Option.value_map x_d_final_opt ~default:(Tensor.zeros_like x_final) ~f:(fun x ->
          x)
      in
      Tensor.(matmul (x_final - x_d_final) q_final)
    in
    let rec lambda_rec t lambda_next lambda_accu =
      if t = 0
      then List.rev lambda_accu
      else (
        let a_curr = extract a_primal_list t in
        let q_curr = extract q_primal_list Int.(t - 1) in
        let x_curr = List.nth_exn x_list_primal t in
        (* if no x_d, set to zero tensor *)
        let x_d_curr =
          let x_d_curr_opt = extract x_d_primal_list Int.(t - 1) in
          Option.value_map x_d_curr_opt ~default:(Tensor.zeros_like x_curr) ~f:(fun x ->
            x)
        in
        let lambda =
          Tensor.(
            matmul lambda_next a_curr
            + matmul (x_curr - x_d_curr) (trans_2d_tensor q_curr))
        in
        lambda_rec Int.(t - 1) lambda (lambda :: lambda_accu))
    in
    lambda_rec state_params.n_steps lambda_final [ lambda_final ]
  in
  (* list of n_tangents, each element a tuple of x_list, u_list, each of length n_steps. *)
  let tangents_lqt =
    List.init n_tangents ~f:(fun k ->
      (* extract k-th tangent for Q, R, A, B *)
      let q_kth_tan_list = extract_and_reshape ~list:q_list k in
      let r_kth_tan_list = extract_and_reshape ~list:r_list k in
      let a_kth_tan_list = extract_and_reshape ~list:a_list k in
      let b_kth_tan_list = extract_and_reshape ~list:b_list k in
      (* iterate to extract new x_targets, which go from 1 to T *)
      let x_d_kth_tan_list =
        List.init state_params.n_steps ~f:(fun i ->
          let t = Int.(i + 1) in
          let q = extract q_primal_list Int.(t - 1) in
          let q_inv = Tensor.inverse q in
          let x = List.nth_exn x_list_primal t in
          (* if no x_d, set to zero tensor *)
          let x_d =
            let x_d_opt = extract x_d_primal_list Int.(t - 1) in
            Option.value_map x_d_opt ~default:(Tensor.zeros_like x) ~f:(fun x -> x)
          in
          let dq_kth =
            let dq_opt = extract q_kth_tan_list Int.(t - 1) in
            Option.value_map
              dq_opt
              ~default:(Tensor.zeros_like (List.hd_exn q_primal_list))
              ~f:(fun x -> x)
          in
          let da_kth =
            let da_opt = extract a_kth_tan_list t in
            Option.value_map
              da_opt
              ~default:(Tensor.zeros_like (List.hd_exn a_primal_list))
              ~f:(fun x -> x)
          in
          let final =
            let common = Tensor.(matmul (x - x_d) (trans_2d_tensor dq_kth)) in
            if t = state_params.n_steps
            then Tensor.(neg (matmul common q_inv))
            else (
              let lambda_next = List.nth_exn lambda_list t in
              Tensor.(neg (matmul (common + matmul lambda_next da_kth) q_inv)))
          in
          Some final)
      in
      (* iterate to extract new u_targets, which go from 0 to T-1 *)
      let u_d_kth_tan_list =
        List.init state_params.n_steps ~f:(fun i ->
          let lambda_next = List.nth_exn lambda_list i in
          let r = extract r_primal_list i in
          let r_inv = Tensor.inverse r in
          let u = List.nth_exn u_list_primal i in
          (* if no u_d, set to zero tensor *)
          let u_d =
            let u_d_opt = extract u_d_primal_list i in
            Option.value_map u_d_opt ~default:(Tensor.zeros_like u) ~f:(fun x -> x)
          in
          let dr_kth =
            let dr_opt = extract r_kth_tan_list i in
            Option.value_map
              dr_opt
              ~default:(Tensor.zeros_like (List.hd_exn r_primal_list))
              ~f:(fun x -> x)
          in
          let db_kth =
            let db_opt = extract b_kth_tan_list i in
            Option.value_map
              db_opt
              ~default:(Tensor.zeros_like (List.hd_exn b_primal_list))
              ~f:(fun x -> x)
          in
          Some
            Tensor.(
              neg
                (matmul
                   (matmul (u - u_d) (trans_2d_tensor dr_kth) + matmul lambda_next db_kth)
                   r_inv)))
      in
      let x_0_kth_tan =
        Option.value_map
          (Maths.tangent x_0)
          ~default:(Tensor.zeros_like x_0_primal)
          ~f:(fun tan ->
            let k_th_tan =
              Tensor.slice tan ~dim:0 ~start:(Some k) ~end_:(Some Int.(k + 1)) ~step:1
            in
            Tensor.reshape k_th_tan ~shape:(Tensor.shape x_0_primal))
      in
      let x_u_desired_kth_tan =
        { x_0 = x_0_kth_tan; x_d_list = x_d_kth_tan_list; u_d_list = u_d_kth_tan_list }
      in
      lqt_tensor
        ~state_params:state_params_tensor
        ~x_u_desired:x_u_desired_kth_tan
        ~cost_params:cost_params_tensor)
  in
  (* step 3: merge primal and tangents for x and u. *)
  let merge_primal_tan primal_list tangents_lqt =
    List.map2_exn primal_list tangents_lqt ~f:(fun primal tan_list ->
      let tangents =
        List.map tan_list ~f:(fun tan ->
          Tensor.reshape tan ~shape:(1 :: Tensor.shape primal))
      in
      let tangents_full = Maths.Direct (Tensor.concat tangents ~dim:0) in
      Maths.make_dual primal ~t:tangents_full)
  in
  let x_tangents_lqt = List.map tangents_lqt ~f:fst |> transpose in
  let u_tangents_lqt = List.map tangents_lqt ~f:snd |> transpose in
  let final_x_list = merge_primal_tan x_list_primal x_tangents_lqt in
  let final_u_list = merge_primal_tan u_list_primal u_tangents_lqt in
  final_x_list, final_u_list
