open Base
open Torch
include Lqr_type
include Maths


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


(* A B, where a is size [m x i x j] and b is size [m x j x k] in batch mode *)
let batch_matmul a b = Maths.(einsum [ a, "mij"; b, "mjk" ] "mik")
let batch_matmul_tensor a b = Tensor.einsum ~equation:"mij,mjk->mik" [ a; b ] ~path:None

(* A B^T *)
let batch_matmul_trans a b = Maths.(einsum [ a, "mij"; b, "mkj" ] "mik")

let batch_matmul_trans_tensor a b =
  Tensor.einsum ~equation:"mij,mkj->mik" [ a; b ] ~path:None

(* A^T B *)
let batch_trans_matmul a b = Maths.(einsum [ a, "mij"; b, "mik" ] "mjk")

let batch_trans_matmul_tensor a b =
  Tensor.einsum ~equation:"mij,mik->mjk" [ a; b ] ~path:None

(* a B, where a is a vector and B is a matrix *)
let batch_vecmat a b = Maths.(einsum [ a, "mi"; b, "mij" ] "mj")
let batch_vecmat_tensor a b = Tensor.einsum ~equation:"mi,mij->mj" [ a; b ] ~path:None

(* a dB , where a has shape [m x i] and dB has shape [k x m x i x j]*)
let batch_vec_tanmat_tensor a b =
  Tensor.einsum ~equation:"mi,kmij->kmj" [ a; b ] ~path:None

(* a B^T *)
let batch_vecmat_trans a b = Maths.(einsum [ a, "mi"; b, "mji" ] "mj")

let batch_vecmat_trans_tensor a b =
  Tensor.einsum ~equation:"mi,mji->mj" [ a; b ] ~path:None

(* a dB^T, where a has shape [m x i] and dB has shape [k x m x j x i] *)
let batch_vec_tanmat_trans_tensor a b =
  Tensor.einsum ~equation:"mi,kmji->kmj" [ a; b ] ~path:None

(* da B, where a has shape [k x m x i] and B has shape [m x i x j] *)
let batch_tanvec_mat_tensor a b =
  Tensor.einsum ~equation:"kmi,mij->kmj" [ a; b ] ~path:None
(* da B^T, where a has shape [k x m x i] and B has shape [m x i x j] *)

let batch_tanvec_mat_trans_tensor a b =
  Tensor.einsum ~equation:"kmi,mji->kmj" [ a; b ] ~path:None

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
  let a_dim = List.nth_exn (Tensor.shape f_u_eg) 1 in
  (* control dim *)
  let b_dim = List.nth_exn (Tensor.shape f_u_eg) 2 in
  let device = Tensor.device f_u_eg in
  (* step 1: backward pass to calculate K_t and k_t *)
  let v_mat_final = List.last_exn c_xx_list in
  let v_vec_final = extract_list_opt n_steps c_x_list ~shape:[ m; a_dim ] ~device in
  let backward t v_mat_next v_vec_next =
    let c_uu_curr = extract_list c_uu_list t in
    let c_xx_curr = extract_list c_xx_list t in
    let c_xu_curr = extract_list_opt t c_xu_list ~shape:[ m; a_dim; b_dim ] ~device in
    let c_x_curr = extract_list_opt t c_x_list ~shape:[ m; a_dim ] ~device in
    let c_u_curr = extract_list_opt t c_u_list ~shape:[ m; b_dim ] ~device in
    let f_x_curr = extract_list f_x_list t in
    let f_u_curr = extract_list f_u_list t in
    let f_t_curr = extract_list_opt t f_t_list ~shape:[ m; a_dim ] ~device in
    let q_uu_curr =
      Maths.(c_uu_curr + batch_trans_matmul f_u_curr (batch_matmul v_mat_next f_u_curr))
    in
    let q_xx_curr =
      Maths.(c_xx_curr + batch_trans_matmul f_x_curr (batch_matmul v_mat_next f_x_curr))
    in
    let q_xu_curr =
      Maths.(c_xu_curr + batch_trans_matmul f_x_curr (batch_matmul v_mat_next f_u_curr))
    in
    let q_u_curr =
      Maths.(
        c_u_curr
        + batch_vecmat v_vec_next f_u_curr
        + batch_vecmat f_t_curr (batch_matmul v_mat_next f_u_curr))
    in
    let q_x_curr =
      Maths.(
        c_x_curr
        + batch_vecmat v_vec_next f_x_curr
        + batch_vecmat f_t_curr (batch_matmul v_mat_next f_x_curr))
    in
    let k_mat_curr =
      Maths.linsolve
        q_uu_curr
        (Maths.neg (transpose q_xu_curr ~dim0:2 ~dim1:1))
        ~left:true
    in
    let k_vec_curr = Maths.linsolve q_uu_curr (Maths.neg q_u_curr) ~left:true in
    let v_mat_curr = Maths.(q_xx_curr + batch_matmul q_xu_curr k_mat_curr) in
    let v_vec_curr = Maths.(q_x_curr + batch_vecmat q_u_curr k_mat_curr) in
    v_mat_curr, v_vec_curr, k_mat_curr, k_vec_curr
  in
  (* k_mat and k_vec go from 0 to T-1 *)
  let k_mat_list, k_vec_list =
    let rec backward_pass t v_mat_next v_vec_next k_mat_accu k_vec_accu =
      if t = -1
      then k_mat_accu, k_vec_accu
      else (
        Stdlib.Gc.major ();
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
  Stdlib.Gc.major ();
  (* step 2: forward pass to obtain controls and states. *)
  let forward t x_curr =
    let f_x_curr = extract_list f_x_list t in
    let f_u_curr = extract_list f_u_list t in
    let k_mat_curr = List.nth_exn k_mat_list t in
    let k_vec_curr = List.nth_exn k_vec_list t in
    let u_curr = Maths.(batch_vecmat_trans x_curr k_mat_curr + k_vec_curr) in
    let x_next =
      let common =
        Maths.(batch_vecmat_trans x_curr f_x_curr + batch_vecmat_trans u_curr f_u_curr)
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
        Stdlib.Gc.major ();
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
  (* if use lqr to solve a batch of k tangents of m problems, f_t has size *)
  let tangent_batched =
    match c_x_list with
    | None -> false
    | Some c_x_list ->
      if List.length (Tensor.shape (List.hd_exn c_x_list)) = 3 then true else false
  in
  (* batch size *)
  let m = List.hd_exn (Tensor.shape f_u_eg) in
  (* state dim *)
  let a_dim = List.nth_exn (Tensor.shape f_u_eg) 1 in
  (* control dim *)
  let b_dim = List.nth_exn (Tensor.shape f_u_eg) 2 in
  let device = Tensor.device f_u_eg in
  (* step 1: backward pass to calculate K_t and k_t *)
  let v_mat_final = List.last_exn c_xx_list in
  let v_vec_final =
    extract_list_opt_tensor n_steps c_x_list ~shape:[ m; a_dim ] ~device
  in
  let backward t v_mat_next v_vec_next =
    let c_uu_curr = extract_list c_uu_list t in
    let c_xx_curr = extract_list c_xx_list t in
    let c_xu_curr =
      extract_list_opt_tensor t c_xu_list ~shape:[ m; a_dim; b_dim ] ~device
    in
    let c_x_curr = extract_list_opt_tensor t c_x_list ~shape:[ m; a_dim ] ~device in
    let c_u_curr = extract_list_opt_tensor t c_u_list ~shape:[ m; b_dim ] ~device in
    let f_x_curr = extract_list f_x_list t in
    let f_u_curr = extract_list f_u_list t in
    let f_t_curr = extract_list_opt_tensor t f_t_list ~shape:[ m; a_dim ] ~device in
    let q_uu_curr =
      Tensor.(
        c_uu_curr
        + batch_trans_matmul_tensor f_u_curr (batch_matmul_tensor v_mat_next f_u_curr))
    in
    let q_xx_curr =
      Tensor.(
        c_xx_curr
        + batch_trans_matmul_tensor f_x_curr (batch_matmul_tensor v_mat_next f_x_curr))
    in
    let q_xu_curr =
      Tensor.(
        c_xu_curr
        + batch_trans_matmul_tensor f_x_curr (batch_matmul_tensor v_mat_next f_u_curr))
    in
    let q_u_curr =
      if tangent_batched
      then
        Tensor.(
          c_u_curr
          + batch_tanvec_mat_tensor v_vec_next f_u_curr
          + batch_tanvec_mat_tensor f_t_curr (batch_matmul_tensor v_mat_next f_u_curr))
      else
        Tensor.(
          c_u_curr
          + batch_vecmat_tensor v_vec_next f_u_curr
          + batch_vecmat_tensor f_t_curr (batch_matmul_tensor v_mat_next f_u_curr))
    in
    let q_x_curr =
      if tangent_batched
      then
        Tensor.(
          c_x_curr
          + batch_tanvec_mat_tensor v_vec_next f_x_curr
          + batch_tanvec_mat_tensor f_t_curr (batch_matmul_tensor v_mat_next f_x_curr))
      else
        Tensor.(
          c_x_curr
          + batch_vecmat_tensor v_vec_next f_x_curr
          + batch_vecmat_tensor f_t_curr (batch_matmul_tensor v_mat_next f_x_curr))
    in
    let k_mat_curr =
      Tensor.linalg_solve
        ~a:q_uu_curr
        ~b:Tensor.(neg (transpose q_xu_curr ~dim0:2 ~dim1:1))
        ~left:true
    in
    let k_vec_curr =
      let final =
        let a, b =
          if tangent_batched
          then (
            let k = List.hd_exn (Tensor.shape x_0) in
            let q_uu_expanded =
              Tensor.expand q_uu_curr ~size:(k :: Tensor.shape q_uu_curr) ~implicit:true
            in
            ( Tensor.reshape q_uu_expanded ~shape:[ -1; b_dim; b_dim ]
            , Tensor.(neg (reshape q_u_curr ~shape:[ -1; b_dim ])) ))
          else q_uu_curr, Tensor.neg q_u_curr
        in
        Tensor.linalg_solve ~a ~b ~left:true
      in
      if tangent_batched then Tensor.reshape final ~shape:[ -1; m; b_dim ] else final
    in
    let v_mat_curr = Tensor.(q_xx_curr + batch_matmul_tensor q_xu_curr k_mat_curr) in
    let v_vec_curr =
      if tangent_batched
      then Tensor.(q_x_curr + batch_tanvec_mat_tensor q_u_curr k_mat_curr)
      else Tensor.(q_x_curr + batch_vecmat_tensor q_u_curr k_mat_curr)
    in
    v_mat_curr, v_vec_curr, k_mat_curr, k_vec_curr
  in
  (* k_mat and k_vec go from 0 to T-1 *)
  let k_mat_list, k_vec_list =
    let rec backward_pass t v_mat_next v_vec_next k_mat_accu k_vec_accu =
      if t = -1
      then k_mat_accu, k_vec_accu
      else (
        Stdlib.Gc.major ();
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
  Stdlib.Gc.major ();
  (* step 2: forward pass to obtain controls and states. *)
  let forward t x_curr =
    let f_x_curr = extract_list f_x_list t in
    let f_u_curr = extract_list f_u_list t in
    let k_mat_curr = List.nth_exn k_mat_list t in
    let k_vec_curr = List.nth_exn k_vec_list t in
    let u_curr =
      if tangent_batched
      then Tensor.(batch_tanvec_mat_trans_tensor x_curr k_mat_curr + k_vec_curr)
      else Tensor.(batch_vecmat_trans_tensor x_curr k_mat_curr + k_vec_curr)
    in
    let x_next =
      let common =
        if tangent_batched
        then
          Tensor.(
            batch_tanvec_mat_trans_tensor x_curr f_x_curr
            + batch_tanvec_mat_trans_tensor u_curr f_u_curr)
        else
          Tensor.(
            batch_vecmat_trans_tensor x_curr f_x_curr
            + batch_vecmat_trans_tensor u_curr f_u_curr)
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
        Stdlib.Gc.major ();
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
  let m = List.hd_exn (Tensor.shape f_u_eg) in
  (* state dim *)
  let a_dim = List.nth_exn (Tensor.shape f_u_eg) 1 in
  (* control dim *)
  let b_dim = List.nth_exn (Tensor.shape f_u_eg) 2 in
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
  let extract_tangent list = List.map list ~f:Maths.tangent in
  let extract_tangent_list_opt list =
    match list with
    | None -> None
    | Some list -> Some (List.map list ~f:Maths.tangent)
  in
  let x_0_tangent = Maths.tangent x_0 in
  let f_x_tangent_list = extract_tangent f_x_list in
  let f_u_tangent_list = extract_tangent f_u_list in
  let f_t_tangent_list = extract_tangent_list_opt f_t_list in
  let c_xx_tangent_list = extract_tangent c_xx_list in
  let c_xu_tangent_list = extract_tangent_list_opt c_xu_list in
  let c_uu_tangent_list = extract_tangent c_uu_list in
  let c_x_tangent_list = extract_tangent_list_opt c_x_list in
  let c_u_tangent_list = extract_tangent_list_opt c_u_list in
  (* step 3: create new f_t, c_x and c_u lists *)
  let new_f_t_list =
    List.init n_steps ~f:(fun t ->
      Stdlib.Gc.major ();
      let x = List.nth_exn x_primal_list t in
      let df_x = List.nth_exn f_x_tangent_list t in
      let u = List.nth_exn u_primal_list t in
      let df_u = List.nth_exn f_u_tangent_list t in
      let tmp1 =
        match df_x with
        | None -> Tensor.zeros [ n_tangents; m; a_dim ] ~device
        | Some df_x -> batch_vec_tanmat_trans_tensor x df_x
      in
      let tmp2 =
        match df_u with
        | None -> Tensor.zeros [ n_tangents; m; a_dim ] ~device
        | Some df_u -> batch_vec_tanmat_trans_tensor u df_u
      in
      let tmp12 = Tensor.(tmp1 + tmp2) in
      match f_t_tangent_list with
      | None -> tmp12
      | Some f_t_tangent_list ->
        let df_t = List.nth_exn f_t_tangent_list t in
        (match df_t with
         | None -> tmp12
         | Some df_t -> Tensor.(tmp12 + df_t)))
  in
  let lambda_T =
    let common =
     
        batch_vecmat_tensor (List.last_exn x_primal_list) (List.last_exn c_xx_primal_list)
    in
    match c_x_primal_list with
    | None -> common
    | Some c_x_primal_list -> Tensor.(common + List.last_exn c_x_primal_list)
  in
  let n_steps_list = List.range 0 (n_steps) in
  let new_c_u_T = Tensor.zeros [ n_tangents; m; b_dim ] ~device in
  let new_c_x_T =
    let tmp2 =
      let dc_xx_T = List.last_exn c_xx_tangent_list in
      match dc_xx_T with
      | None -> Tensor.zeros [ n_tangents; m; a_dim ] ~device
      | Some dc_xx_T -> batch_vec_tanmat_tensor (List.last_exn x_primal_list) dc_xx_T
    in
    match c_x_tangent_list with
    | None -> tmp2
    | Some c_x_tangent_list ->
      let dc_x_T = List.last_exn c_x_tangent_list in
      (match dc_x_T with
       | None -> tmp2
       | Some dc_x_T -> Tensor.(tmp2 + dc_x_T))
  in
  let new_c_x_list, new_c_u_list, _ =
    List.fold_right
      n_steps_list
      ~init:([ new_c_x_T ], [ new_c_u_T ], lambda_T)
      ~f:(fun t (c_x_accu, c_u_accu, lambda_next) ->
        let u_t, x_t = List.nth_exn u_primal_list t, List.nth_exn x_primal_list t in
        let c_x =
          if t = 0
          then Tensor.zeros [ n_tangents; m; a_dim ] ~device
          else (
            let tmp1 =
              match c_xu_tangent_list with
              | None -> Tensor.zeros [ n_tangents; m; a_dim ] ~device
              | Some c_xu_tangent_list ->
                let dc_xu = List.nth_exn c_xu_tangent_list t in
                (match dc_xu with
                 | None -> Tensor.zeros [ n_tangents; m; a_dim ] ~device
                 | Some dc_xu -> batch_vec_tanmat_trans_tensor u_t dc_xu)
            in
            let tmp2 =
              let dc_xx = List.nth_exn c_xx_tangent_list t in
              match dc_xx with
              | None -> tmp1
              | Some dc_xx -> batch_vec_tanmat_tensor x_t dc_xx
            in
            let tmp3 = batch_vecmat_tensor lambda_next (List.nth_exn f_x_primal_list t) in
            let final =
              let common = Tensor.(tmp1 + tmp2 + tmp3) in
              match c_x_tangent_list with
              | None -> common
              | Some c_x_tangent_list ->
                let dc_x = List.nth_exn c_x_tangent_list t in
                (match dc_x with
                 | None -> common
                 | Some dc_x -> Tensor.(common + dc_x))
            in
            final)
        in
        let c_u =
          let tmp1 =
            match c_xu_tangent_list with
            | None -> Tensor.zeros [ n_tangents; m; b_dim ] ~device
            | Some c_xu_tangent_list ->
              let dc_xu = List.nth_exn c_xu_tangent_list t in
              (match dc_xu with
               | None -> Tensor.zeros [ n_tangents; m; b_dim ] ~device
               | Some dc_xu -> batch_vec_tanmat_trans_tensor x_t dc_xu)
          in
          let tmp2 =
            let dc_uu = List.nth_exn c_uu_tangent_list t in
            match dc_uu with
            | None -> tmp1
            | Some dc_uu -> batch_vec_tanmat_tensor u_t dc_uu
          in
          let tmp3 = batch_vecmat_tensor lambda_next (List.nth_exn f_u_primal_list t) in
          let final =
            let common = Tensor.(tmp1 + tmp2 + tmp3) in
            match c_u_tangent_list with
            | None -> common
            | Some c_u_tangent_list ->
              let dc_u = List.nth_exn c_u_tangent_list t in
              (match dc_u with
               | None -> common
               | Some dc_u -> Tensor.(common + dc_u))
          in
          final
        in
        let lambda_curr =
          let tmp1 = batch_vecmat_tensor lambda_next (List.nth_exn f_x_primal_list t) in
          let tmp2 = batch_vecmat_tensor x_t (List.nth_exn c_xx_primal_list t) in
          let tmp3 =
            match c_xu_primal_list with
            | None -> Tensor.zeros [ m; a_dim ] ~device
            | Some c_xu_primal_list ->
              let c_xu = List.nth_exn c_xu_primal_list t in
              batch_vecmat_trans_tensor u_t c_xu
          in
          let tmp4 =
            match c_x_primal_list with
            | None -> Tensor.zeros [ m; a_dim ] ~device
            | Some c_x_primal_list -> List.nth_exn c_x_primal_list t
          in
          Tensor.(tmp1 + tmp2 + tmp3 + tmp4)
        in
        c_x :: c_x_accu, c_u :: c_u_accu, lambda_curr)
  in
  let state_params_tensor_tangent =
    { n_steps
    ; x_0 = Option.value_exn x_0_tangent
    ; f_x_list = f_x_primal_list
    ; f_u_list = f_u_primal_list
    ; f_t_list = Some new_f_t_list
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
  let x_tangent_list, u_tangent_list =
    lqr_tensor
      ~state_params:state_params_tensor_tangent
      ~cost_params:cost_params_tensor_tangent
  in
  Stdlib.Gc.major ();
  (* step 3: merge primal and tangents for x and u. *)
  let merge_primal_tan primal_list tangents_lqr =
    List.map2_exn primal_list tangents_lqr ~f:(fun primal tan ->
      Maths.make_dual primal ~t:(Maths.Direct tan))
  in
  let final_x_list = merge_primal_tan x_primal_list x_tangent_list in
  let final_u_list = merge_primal_tan u_primal_list u_tangent_list in
  final_x_list, final_u_list
