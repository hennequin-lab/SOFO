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
    let x_next =
      let common =
        Maths.((x_curr *@ trans_2d f_x_curr) + (u_curr *@ trans_2d f_u_curr))
      in
      match f_t_list with
      | None -> common
      | Some f_t_list -> Maths.(common + extract_list f_t_list t)
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
      let common =
        Tensor.(
          matmul x_curr (trans_2d_tensor f_x_curr)
          + matmul u_curr (trans_2d_tensor f_u_curr))
      in
      match f_t_list with
      | None -> common
      | Some f_t_list -> Tensor.(common + extract_list f_t_list t)
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
let lqr_sep ~(state_params : state_params) ~(cost_params : cost_params) =
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
  let n_tangents = List.hd_exn (Tensor.shape (Option.value_exn (Maths.tangent x_0))) in
  let device = Tensor.device f_u_eg in
  (* step 1: lqr on the primal *)
  let extract_primal list = List.map list ~f:Maths.primal in
  let extract_primal_list_opt list =
    match list with
    | None -> None
    | Some list -> Some (List.map list ~f:Maths.primal)
  in
  let x_0_primal = Maths.primal x_0 in
  let f_x_primal_list = extract_primal f_x_list in
  let f_u_primal_list = extract_primal f_u_list in
  let f_t_primal_list = extract_primal_list_opt f_t_list in
  let c_xx_primal_list = extract_primal c_xx_list in
  let c_xu_primal_list = extract_primal_list_opt c_xu_list in
  let c_uu_primal_list = extract_primal c_uu_list in
  let c_x_primal_list = extract_primal_list_opt c_x_list in
  let c_u_primal_list = extract_primal_list_opt c_u_list in
  let state_params_tensor_primal =
    { n_steps
    ; x_0 = x_0_primal
    ; f_x_list = f_x_primal_list
    ; f_u_list = f_u_primal_list
    ; f_t_list = f_t_primal_list
    }
  in
  let cost_params_tensor_primal =
    { c_xx_list = c_xx_primal_list
    ; c_xu_list = c_xu_primal_list
    ; c_uu_list = c_uu_primal_list
    ; c_x_list = c_x_primal_list
    ; c_u_list = c_u_primal_list
    }
  in
  let x_primal_list, u_primal_list =
    lqr_tensor
      ~state_params:state_params_tensor_primal
      ~cost_params:cost_params_tensor_primal
  in
  Stdlib.Gc.major ();
  (* step 2: lqr on tangents *)
  let extract_kth_tan list k =
    let tan_shape = Tensor.shape (Maths.primal (List.hd_exn list)) in
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
  (* extract kth tangent and reshape it to the corresponding primal for the each Maths.t element in a list, the list can be optional. *)
  let extract_kth_tan_opt list k =
    match list with
    | None -> None
    | Some list ->
      let tan_shape = Tensor.shape (Maths.primal (List.hd_exn list)) in
      Some
        (List.map list ~f:(fun x ->
           let tangent = Maths.tangent x in
           let reshaped_tangent =
             Option.value_map
               tangent
               ~default:(Tensor.zeros tan_shape ~device)
               ~f:(fun tan ->
                 let k_th_tan =
                   Tensor.slice
                     tan
                     ~dim:0
                     ~start:(Some k)
                     ~end_:(Some Int.(k + 1))
                     ~step:1
                 in
                 Tensor.reshape k_th_tan ~shape:tan_shape)
           in
           Some reshaped_tangent))
  in
  let tangents_lqr =
    List.init n_tangents ~f:(fun k ->
      Stdlib.Gc.major ();
      let x_0_tangent =
        Option.value_map
          (Maths.tangent x_0)
          ~default:(Tensor.zeros_like x_0_primal)
          ~f:(fun tan ->
            let k_th_tan =
              Tensor.slice tan ~dim:0 ~start:(Some k) ~end_:(Some Int.(k + 1)) ~step:1
            in
            Tensor.reshape k_th_tan ~shape:(Tensor.shape x_0_primal))
      in
      let f_x_tangent_list = extract_kth_tan f_x_list k in
      let f_u_tangent_list = extract_kth_tan f_u_list k in
      let f_t_tangent_list = extract_kth_tan_opt f_t_list k in
      let c_xx_tangent_list = extract_kth_tan c_xx_list k in
      let c_xu_tangent_list = extract_kth_tan_opt c_xu_list k in
      let c_uu_tangent_list = extract_kth_tan c_uu_list k in
      let c_x_tangent_list = extract_kth_tan_opt c_x_list k in
      let c_u_tangent_list = extract_kth_tan_opt c_u_list k in
      (* step 3: form new parameters from tangents *)
      (* f_t goes from t=0 to t=T-1 *)
      let new_f_t_list =
        (* drop x_T so list goes from t=0 to t=T-1. *)
        let x_primal_list_tmp = List.drop_last_exn x_primal_list in
        let tmp1 =
          let x_dfx df_x x =
            match df_x with
            | None -> Tensor.zeros_like x_0_primal
            | Some df_x -> Tensor.(matmul x (trans_2d_tensor df_x))
          in
          if List.length f_x_tangent_list > 1
          then
            List.map2_exn f_x_tangent_list x_primal_list_tmp ~f:(fun df_x x ->
              x_dfx df_x x)
          else (
            let df_x = List.hd_exn f_x_tangent_list in
            List.map x_primal_list_tmp ~f:(fun x -> x_dfx df_x x))
        in
        let tmp2 =
          let u_dfu df_u u =
            match df_u with
            | None -> Tensor.zeros_like x_0_primal
            | Some df_u -> Tensor.(matmul u (trans_2d_tensor df_u))
          in
          if List.length f_u_tangent_list > 1
          then
            List.map2_exn f_u_tangent_list u_primal_list ~f:(fun df_u u -> u_dfu df_u u)
          else (
            let df_u = List.hd_exn f_u_tangent_list in
            List.map u_primal_list ~f:(fun u -> u_dfu df_u u))
        in
        let tmp12 = List.map2_exn tmp1 tmp2 ~f:(fun tm1 tm2 -> Tensor.(tm1 + tm2)) in
        let final =
          match f_t_tangent_list with
          | None -> tmp12
          | Some f_t_tangent_list ->
            List.map2_exn tmp12 f_t_tangent_list ~f:(fun tmp12 df_t ->
              match df_t with
              | None -> tmp12
              | Some df_t -> Tensor.(tmp12 + df_t))
        in
        Some final
      in
      (* this list goes from 0 to T-1 *)
      let n_steps_list = List.range 0 Int.(n_steps) in
      let lambda_T =
        let common =
          Tensor.(matmul (List.last_exn x_primal_list) (List.last_exn c_xx_primal_list))
        in
        match c_x_primal_list with
        | None -> common
        | Some c_x_primal_list -> Tensor.(common + List.last_exn c_x_primal_list)
      in
      let c_x_T =
        let tmp2 =
          let dc_xx_last = List.last_exn c_xx_tangent_list in
          match dc_xx_last with
          | None -> Tensor.zeros_like x_0_primal
          | Some dc_xx_last -> Tensor.(matmul (List.last_exn x_primal_list) dc_xx_last)
        in
        match c_x_tangent_list with
        | None -> tmp2
        | Some c_x_tangent_list ->
          let c_x_tangent_last = List.last_exn c_x_tangent_list in
          (match c_x_tangent_last with
           | None -> tmp2
           | Some c_x_tangent_last -> Tensor.(tmp2 + c_x_tangent_last))
      in
      let c_u_T = Tensor.zeros [ m; b_dim ] ~device in
      let _, new_c_x_list, new_c_u_list =
        List.fold_right
          n_steps_list
          ~init:(lambda_T, [ c_x_T ], [ c_u_T ])
          ~f:(fun t (lambda_next, c_x_list, c_u_list) ->
            let lambda_curr =
              let f_x_curr = extract_list f_x_primal_list t in
              let x_curr = List.nth_exn x_primal_list t in
              let c_xx_curr = extract_list c_xx_primal_list t in
              let u_curr = List.nth_exn u_primal_list t in
              let c_x_curr =
                extract_list_opt_tensor t c_x_primal_list ~shape:[ m; a_dim ] ~device
              in
              let final =
                let common =
                  Tensor.(
                    matmul lambda_next f_x_curr + matmul x_curr c_xx_curr + c_x_curr)
                in
                match c_xu_primal_list with
                | None -> common
                | Some c_xu_primal_list ->
                  let c_xu_curr = extract_list c_xu_primal_list t in
                  Tensor.(common + matmul u_curr (trans_2d_tensor c_xu_curr))
              in
              final
            in
            let c_x_t =
              let tmp1 =
                match c_xu_tangent_list with
                | None -> Tensor.zeros_like c_x_T
                | Some c_xu_tangent_list ->
                  let c_xu_tangent_curr = List.nth_exn c_xu_tangent_list t in
                  (match c_xu_tangent_curr with
                   | None -> Tensor.zeros_like c_x_T
                   | Some c_xu_tangent_curr ->
                     Tensor.(
                       matmul
                         (List.nth_exn u_primal_list t)
                         (trans_2d_tensor c_xu_tangent_curr)))
              in
              let tmp2 =
                let dc_xx = List.nth_exn c_xx_tangent_list t in
                match dc_xx with
                | None -> Tensor.zeros_like c_x_T
                | Some dc_xx -> Tensor.(matmul (List.nth_exn x_primal_list t) dc_xx)
              in
              let tmp3 =
                let f_x_primal_curr =
                  if List.length f_x_primal_list = 1
                  then List.hd_exn f_x_primal_list
                  else List.nth_exn f_x_primal_list t
                in
                Tensor.matmul lambda_next f_x_primal_curr
              in
              let final =
                let common = Tensor.(tmp1 + tmp2 + tmp3) in
                match c_x_tangent_list with
                | None -> common
                | Some c_x_tangent_list ->
                  let c_x_tangent_curr = List.nth_exn c_x_tangent_list t in
                  (match c_x_tangent_curr with
                   | None -> common
                   | Some c_x_tangent_curr -> Tensor.(common + c_x_tangent_curr))
              in
              final
            in
            let c_u_t =
              let tmp1 =
                match c_xu_tangent_list with
                | None -> Tensor.zeros_like c_u_T
                | Some c_xu_tangent_list ->
                  let c_xu_tangent_curr = List.nth_exn c_xu_tangent_list t in
                  (match c_xu_tangent_curr with
                   | None -> Tensor.zeros_like c_u_T
                   | Some c_xu_tangent_curr ->
                     Tensor.(matmul (List.nth_exn x_primal_list t) c_xu_tangent_curr))
              in
              let tmp2 =
                let dc_uu = List.nth_exn c_uu_tangent_list t in
                match dc_uu with
                | None -> Tensor.zeros_like c_u_T
                | Some dc_uu -> Tensor.(matmul (List.nth_exn u_primal_list t) dc_uu)
              in
              let tmp3 =
                let f_u_primal_curr =
                  if List.length f_u_primal_list = 1
                  then List.hd_exn f_u_primal_list
                  else List.nth_exn f_u_primal_list t
                in
                Tensor.matmul lambda_next f_u_primal_curr
              in
              let final =
                let common = Tensor.(tmp1 + tmp2 + tmp3) in
                match c_u_tangent_list with
                | None -> common
                | Some c_u_tangent_list ->
                  let c_u_tangent_curr = List.nth_exn c_u_tangent_list t in
                  (match c_u_tangent_curr with
                   | None -> common
                   | Some c_u_tangent_curr -> Tensor.(common + c_u_tangent_curr))
              in
              final
            in
            let c_x_t_final = if t = 0 then Tensor.zeros_like x_0_primal else c_x_t in
            lambda_curr, c_x_t_final :: c_x_list, c_u_t :: c_u_list)
      in
      let state_params_tensor_tangent =
        { n_steps
        ; x_0 = x_0_tangent
        ; f_x_list = f_x_primal_list
        ; f_u_list = f_u_primal_list
        ; f_t_list = new_f_t_list
        }
      in
      let cost_params_tensor_tangent =
        { c_xx_list = c_xx_primal_list
        ; c_xu_list = c_xu_primal_list
        ; c_uu_list = c_uu_primal_list
        ; c_x_list = Some new_c_x_list
        ; c_u_list = Some new_c_u_list
        }
      in
      lqr_tensor
        ~state_params:state_params_tensor_tangent
        ~cost_params:cost_params_tensor_tangent)
  in
  Stdlib.Gc.major ();
  (* step 3: merge primal and tangents for x and u. *)
  let merge_primal_tan primal_list tangents_lqr =
    List.map2_exn primal_list tangents_lqr ~f:(fun primal tan_list ->
      let tangents =
        List.map tan_list ~f:(fun tan ->
          Tensor.reshape tan ~shape:(1 :: Tensor.shape primal))
      in
      let tangents_full = Maths.Direct (Tensor.concat tangents ~dim:0) in
      Maths.make_dual primal ~t:tangents_full)
  in
  (* tangents of timesteps to timesteps of tangents *)
  let x_tangents_lqr = List.map tangents_lqr ~f:fst |> transpose in
  let u_tangents_lqr = List.map tangents_lqr ~f:snd |> transpose in
  let final_x_list = merge_primal_tan x_primal_list x_tangents_lqr in
  let final_u_list = merge_primal_tan u_primal_list u_tangents_lqr in
  final_x_list, final_u_list
