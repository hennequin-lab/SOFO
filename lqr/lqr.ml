open Base
open Forward_torch
include Lqr_typ
open Maths

let cleanup () = Stdlib.Gc.major ()
let print s = Stdio.print_endline (Sexp.to_string_hum s)
let zeros_like a = Torch.Tensor.zeros_like (Maths.primal a) |> Maths.const

let ( *@ ) a b =
  match List.length (shape b) with
  | 3 -> einsum [ a, "mab"; b, "mbc" ] "mac"
  | 2 -> einsum [ a, "mab"; b, "mb" ] "ma"
  | _ -> failwith "not batch multipliable"

let maybe_btr = Option.map ~f:btr
let maybe_inv_sqr = Option.map ~f:inv_sqr

let maybe_batch_matmul a b ~batch_const =
  if batch_const
  then (
    match a, b with
    | Some a, Some b -> Some (einsum [ a, "ab"; b, "bc" ] "ac")
    | _ -> None)
  else (
    match a, b with
    | Some a, Some b -> Some (einsum [ a, "mab"; b, "mbc" ] "mac")
    | _ -> None)

let ( +? ) a b =
  match a, b with
  | None, None -> None
  | Some a, None -> Some a
  | None, Some b -> Some b
  | Some a, Some b -> Some (a + b)

let ( -? ) a b =
  match a, b with
  | None, None -> None
  | Some a, None -> Some a
  | None, Some b -> Some b
  | Some a, Some b -> Some (a - b)

let ( *? ) f v =
  match f, v with
  | Some f, Some v -> Some (f v)
  | _ -> None

let maybe_unsqueeze ~dim a =
  match a with
  | None -> None
  | Some a -> Some (unsqueeze a ~dim)

let maybe_squeeze ~dim a =
  match a with
  | None -> None
  | Some a -> Some (squeeze a ~dim)

(* a + m *@ b *)
let maybe_force a m b =
  match a, m, b with
  | Some a, Some m, Some b -> Some (a + (m *@ b))
  | Some a, _, _ -> Some a
  | _, Some m, Some b -> Some (m *@ b)
  | _ -> None

let maybe_einsum (a, opsA) (b, opsB) opsC =
  match a, b with
  | Some a, Some b -> Some (einsum [ a, opsA; b, opsB ] opsC)
  | _ -> None

let neg_inv_symm ~is_vector _b (ell, ell_T) =
  match ell, ell_T with
  | Some ell, Some ell_T ->
    let _b =
      if not is_vector
      then _b
      else Option.map _b ~f:(fun x -> reshape x ~shape:(shape x @ [ 1 ]))
    in
    let _y = Option.map _b ~f:(linsolve_triangular ~left:true ~upper:false ell) in
    Option.map _y ~f:(fun x -> linsolve_triangular ~left:true ~upper:true ell_T x |> neg)
  | _ -> None

(* backward recursion: all the (most expensive) common bits. common goes from 0 to T *)
let backward_common
      ~laplace
      ~batch_const
      (common : (t, t -> t) momentary_params_common list)
  =
  let _V =
    match List.last common with
    | None -> failwith "LQR needs a time horizon >= 1"
    | Some z -> z._Cxx
  in
  (* info goes from 0 to t-1 *)
  let _, info =
    let common_except_last = List.drop_last_exn common in
    List.fold_right common_except_last ~init:(_V, []) ~f:(fun z (_V, info_list) ->
      cleanup ();
      let _Quu, _Qxx, _Qux =
        if batch_const
        then (
          let _V_unsqueezed = maybe_unsqueeze _V ~dim:0 in
          let tmp = z._Fx_prod *? _V_unsqueezed in
          let _Quu =
            let tmp2 =
              z._Fu_prod *? maybe_btr (z._Fu_prod *? _V_unsqueezed)
              |> maybe_squeeze ~dim:0
            in
            z._Cuu +? tmp2
          in
          let _Qxx =
            let tmp2 = z._Fx_prod *? maybe_btr tmp |> maybe_squeeze ~dim:0 in
            z._Cxx +? tmp2
          in
          let _Qux =
            let tmp2 = z._Fu_prod *? maybe_btr tmp |> maybe_squeeze ~dim:0 in
            maybe_btr z._Cxu +? tmp2
          in
          _Quu, _Qxx, _Qux)
        else (
          let tmp = z._Fx_prod *? _V in
          ( z._Cuu +? (z._Fu_prod *? maybe_btr (z._Fu_prod *? _V))
          , z._Cxx +? (z._Fx_prod *? maybe_btr tmp)
          , maybe_btr z._Cxu +? (z._Fu_prod *? maybe_btr tmp) ))
      in
      (* compute LQR gain parameters to be used in the subsequent fwd pass *)
      (* Torch.Tensor.print (Maths.primal (Option.value_exn _Quu));  *)
      let _Quu_inv = Option.map _Quu ~f:inv_sqr in
      let _Quu_chol = Option.map _Quu ~f:cholesky in
      let _Quu_chol_T = maybe_btr _Quu_chol in
      let _K = neg_inv_symm ~is_vector:false _Qux (_Quu_chol, _Quu_chol_T) in
      (* update the value function *)
      let _V_new =
        let tmp2 =
          if batch_const
          then maybe_einsum (_Qux, "ab") (_K, "ac") "bc"
          else maybe_einsum (_Qux, "mab") (_K, "mac") "mbc"
        in
        _Qxx +? tmp2
      in
      (* only save Quu_inv if laplace *)
      _V_new, { _Quu_chol; _Quu_chol_T; _V; _K; _Quu_inv } :: info_list)
  in
  info

let _k ~tangent ~batch_const ~(z : backward_common_info) ~_f _qu =
  match tangent, batch_const with
  | true, true ->
    (match _f with
     | Some _f ->
       (* from [k x m x b] to [b x km ] *)
       let _qu_tmp = Option.value_exn _qu in
       let m, b =
         List.nth_exn (Maths.shape _qu_tmp) 1, List.nth_exn (Maths.shape _qu_tmp) 2
       in
       let _qu_squeezed =
         _qu_tmp |> Maths.reshape ~shape:[ -1; b ] |> Maths.permute ~dims:[ 1; 0 ] |> Some
       in
       neg_inv_symm ~is_vector:false _qu_squeezed (z._Quu_chol, z._Quu_chol_T)
       (* from [b x km ] to [k x m x b] *)
       |> Option.value_exn
       |> Maths.reshape ~shape:[ b; -1; m ]
       |> Maths.permute ~dims:[ 1; 2; 0 ]
       |> Some
     | None -> None)
  | true, false ->
    (match _f with
     | Some _f ->
       (* from [k x m x b] to [m x b x k] *)
       let _qu_swapped =
         Option.value_exn _qu |> Maths.permute ~dims:[ 1; 2; 0 ] |> Some
       in
       neg_inv_symm ~is_vector:false _qu_swapped (z._Quu_chol, z._Quu_chol_T)
       (* from [m x b x k] to [k x m x b] *)
       |> Option.value_exn
       |> Maths.permute ~dims:[ 2; 0; 1 ]
       |> Some
     | None -> None)
  | false, true ->
    (match _qu with
     | None -> None
     | Some _qu ->
       let _qu_swapped = Some (Maths.permute ~dims:[ 1; 0 ] _qu) in
       neg_inv_symm ~is_vector:false _qu_swapped (z._Quu_chol, z._Quu_chol_T)
       |> Option.value_exn
       |> Maths.permute ~dims:[ 1; 0 ]
       |> Some)
  | false, false ->
    let _k_unsqueezed = neg_inv_symm ~is_vector:true _qu (z._Quu_chol, z._Quu_chol_T) in
    (match _k_unsqueezed with
     | None -> None
     | Some _k_unsqueezed ->
       Some (reshape _k_unsqueezed ~shape:(List.take (shape _k_unsqueezed) 2)))

let _qu_qx ~tangent ~batch_const ~z ~_cu ~_cx ~_f ~_Fu_prod ~_Fx_prod _v =
  let tmp =
    let tmp2 =
      match tangent, batch_const with
      | true, true -> maybe_einsum (z._V, "ab") (_f, "kmb") "kma"
      | true, false -> maybe_einsum (z._V, "mab") (_f, "kmb") "kma"
      | false, true -> maybe_einsum (z._V, "ab") (_f, "mb") "ma"
      | false, false -> maybe_einsum (z._V, "mab") (_f, "mb") "ma"
    in
    _v +? tmp2
  in
  _cu +? (_Fu_prod *? tmp), _cx +? (_Fx_prod *? tmp)

let v ~tangent ~batch_const ~_K ~_qx ~_qu =
  let tmp =
    match tangent, batch_const with
    | true, true -> maybe_einsum (_K, "ba") (_qu, "kmb") "kma"
    | true, false -> maybe_einsum (_K, "mba") (_qu, "kmb") "kma"
    | false, true -> maybe_einsum (_K, "ba") (_qu, "mb") "ma"
    | false, false -> maybe_einsum (_K, "mba") (_qu, "mb") "ma"
  in
  _qx +? tmp

(* backward recursion; assumes all the common (expensive) stuff has already been
   computed *)
let backward
      ?(tangent = false)
      ~batch_const
      common_info
      (params : (t option, (t, t -> t) momentary_params list) Params.p)
  =
  let _v =
    match List.last params.params with
    | None -> failwith "LQR needs a time horizon >= 1"
    | Some z -> z._cx
  in
  (* info (k/K) goes from 0 to T-1 *)
  let _, info =
    let params_except_last = List.drop_last_exn params.params in
    List.fold2_exn
      (List.rev common_info)
      (List.rev params_except_last)
      ~init:(_v, [])
      ~f:(fun (_v, info_list) z p ->
        cleanup ();
        let _qu, _qx =
          let _Fu_prod, _Fx_prod =
            if tangent
            then p.common._Fu_prod_tangent, p.common._Fx_prod_tangent
            else p.common._Fu_prod, p.common._Fx_prod
          in
          _qu_qx
            ~tangent
            ~batch_const
            ~z
            ~_cu:p._cu
            ~_cx:p._cx
            ~_f:p._f
            ~_Fu_prod
            ~_Fx_prod
            _v
        in
        (* compute LQR gain parameters to be used in the subsequent fwd pass *)
        let _k = _k ~tangent ~batch_const ~z ~_f:p._f _qu in
        (* update the value function *)
        let _v = v ~tangent ~batch_const ~_K:z._K ~_qx ~_qu in
        _v, { _k; _K = z._K; _Quu_chol = z._Quu_chol } :: info_list)
  in
  info

let _u ~tangent ~batch_const ~_k ~_K ~x =
  let tmp =
    match tangent, batch_const with
    | true, true -> maybe_einsum (x, "kma") (_K, "ba") "kmb"
    | true, false -> maybe_einsum (x, "kma") (_K, "mba") "kmb"
    | false, true -> maybe_einsum (x, "ma") (_K, "ba") "mb"
    | false, false -> maybe_einsum (x, "ma") (_K, "mba") "mb"
  in
  _k +? tmp

let _x ~_f ~_Fx_prod2 ~_Fu_prod2 ~x ~u = _f +? (_Fx_prod2 *? x) +? (_Fu_prod2 *? u)

let forward ?(tangent = false) ~batch_const params (backward_info : backward_info list) =
  let open Params in
  (* u goes from 0 to T-1, x goes from 1 to T *)
  let _, solution =
    let params_except_last = List.drop_last_exn params.params in
    List.fold2_exn
      params_except_last
      backward_info
      ~init:(params.x0, [])
      ~f:(fun (x, accu) p b ->
        cleanup ();
        (* compute the optimal control inputs *)
        let u = _u ~tangent ~batch_const ~_k:b._k ~_K:b._K ~x in
        (* fold them into the state update *)
        let x =
          let _Fx_prod2, _Fu_prod2 =
            if tangent
            then p.common._Fx_prod2_tangent, p.common._Fu_prod2_tangent
            else p.common._Fx_prod2, p.common._Fu_prod2
          in
          _x ~_f:p._f ~_Fx_prod2 ~_Fu_prod2 ~x ~u
        in
        x, Solution.{ u; x } :: accu)
  in
  List.rev solution

(* TODO: covariances over u only *)
let covariances
      ?(batch_const = false)
      ~(common_info : backward_common_info list)
      (p : (t option, (t, t -> t) momentary_params list) Params.p)
  =
  let common_info_eg = List.hd_exn common_info in
  let _P_init = Some (zeros_like (Option.value_exn common_info_eg._V)) in
  (* both params and backward_common need to go from 0 to T-1 *)
  let params = List.drop_last_exn p.params in
  (* The sigma list goes from 0 to T-1 *)
  let _, _Sigma_uu_list =
    List.fold2_exn
      params
      common_info
      ~init:(_P_init, [])
      ~f:(fun (_P, _Sigma_uu_list) params common ->
        let _Sigma_xx = maybe_inv_sqr (_P +? common._V) in
        let _Sigma_uu =
          let tmp1 = maybe_batch_matmul common._K _Sigma_xx ~batch_const in
          let tmp2 = maybe_batch_matmul tmp1 (maybe_btr common._K) ~batch_const in
          let _Quu_inv =
            let _Quu =
              maybe_batch_matmul common._Quu_chol common._Quu_chol_T ~batch_const
            in
            maybe_inv_sqr _Quu
          in
          tmp2 +? _Quu_inv
        in
        let _Px =
          (* since producted with _F_prod always has a leading batch dim, need to artificially add on some tensors *)
          let tmp1 = _P +? params.common._Cxx in
          let tmp1_unsqueezed =
            if batch_const then maybe_unsqueeze tmp1 ~dim:0 else tmp1
          in
          let tmp3 = params.common._Fx_prod_inv *? tmp1_unsqueezed in
          let tmp4 = params.common._Fx_prod2_inv_trans *? tmp3 in
          if batch_const then maybe_squeeze tmp4 ~dim:0 else tmp4
        in
        let _Pu =
          let _Px_tmp = if batch_const then maybe_unsqueeze _Px ~dim:0 else _Px in
          let tmp1 = params.common._Fu_prod *? _Px_tmp in
          let tmp2 = params.common._Fu_prod2_trans *? tmp1 in
          let tmp2_squeezed = if batch_const then maybe_squeeze tmp2 ~dim:0 else tmp2 in
          params.common._Cuu +? tmp2_squeezed
        in
        let _P =
          let _Px_tmp = if batch_const then maybe_unsqueeze _Px ~dim:0 else _Px in
          let tmp1 = params.common._Fu_prod2_trans *? _Px_tmp in
          let _Pu_inv = maybe_inv_sqr _Pu in
          let _Pu_inv_tmp =
            if batch_const then maybe_unsqueeze _Pu_inv ~dim:0 else _Pu_inv
          in
          let tmp2 = maybe_batch_matmul tmp1 _Pu_inv_tmp ~batch_const:false in
          let tmp3 = params.common._Fu_prod *? _Px_tmp in
          let tmp4 = maybe_batch_matmul tmp2 tmp3 ~batch_const:false in
          let tmp5 = if batch_const then maybe_squeeze tmp4 ~dim:0 else tmp4 in
          _Px -? tmp5
        in
        _P, _Sigma_uu :: _Sigma_uu_list)
  in
  List.rev _Sigma_uu_list

(* when batch_const is true, _Fx_prods, _Fu_prods, _Cxx, _Cxu, _Cuu has no leading batch dimension and special care needs to be taken to deal with these *)
let _solve ?(batch_const = false) ?(laplace = false) p =
  let common_info =
    backward_common
      ~batch_const
      ~laplace
      (List.map p.Params.params ~f:(fun x -> x.common))
  in
  cleanup ();
  let bck = backward ~batch_const common_info p in
  cleanup ();
  let sol =
    bck
    |> forward ~batch_const p
    |> List.map ~f:(fun s ->
      Solution.{ x = Option.value_exn s.x; u = Option.value_exn s.u })
  in
  if laplace
  then (
    let covariances = covariances ~batch_const ~common_info p in
    sol, bck, Some (List.map covariances ~f:(fun x -> Option.value_exn x)))
  else sol, bck, None

(* backward pass and forward pass with surrogate rhs for the tangent problem;
   s is the solution obtained from lqr through the primals
   and p is the full set of parameters *)
let tangent_solve
      ~batch_const
      common_info
      (s : t option Solution.p list)
      (p : (t option, (t, t prod) momentary_params list) Params.p)
  : t option Solution.p list
  =
  let _p_implicit_primal = Option.map ~f:(fun x -> x.primal) in
  let _p_implicit_tangent x =
    let tmp = Option.map x ~f:(fun x -> x.tangent) in
    match tmp with
    | Some a -> a
    | None -> None
  in
  let _p_primal = Option.map ~f:(fun x -> Maths.const (Maths.primal x)) in
  let _p_tangent x =
    match x with
    | None -> None
    | Some x -> Option.map (Maths.tangent x) ~f:Maths.const
  in
  (* calculate terminal conditions *)
  let params_T = List.last_exn p.params in
  let x_T = (List.last_exn s).x in
  let lambda_T =
    let _Cxx_curr = _p_primal params_T.common._Cxx in
    let _cx_curr = _p_primal params_T._cx in
    let _Cxx_str = if batch_const then "ba" else "mba" in
    _cx_curr +? maybe_einsum (x_T, "mb") (_Cxx_curr, _Cxx_str) "ma"
    (* else maybe_force _cx_curr _Cxx_curr x_T *)
  in
  let _cx_surro_T =
    let _dCxx_T = _p_tangent params_T.common._Cxx in
    let _dcx_T = _p_tangent params_T._cx in
    let tmp1 =
      let _dCxx_str = if batch_const then "kba" else "kmba" in
      maybe_einsum (x_T, "mb") (_dCxx_T, _dCxx_str) "kma"
    in
    tmp1 +? _dcx_T
  in
  let params_except_last = List.drop_last_exn p.params in
  (* this list goes from 0 to T-1 *)
  let params_xu_list =
    let x_list = _p_primal p.x0 :: List.map s ~f:(fun s -> s.x) |> List.drop_last_exn in
    let u_list = List.map s ~f:(fun s -> s.u) in
    let xu_list = List.map2_exn x_list u_list ~f:(fun x u -> x, u) in
    List.map2_exn params_except_last xu_list ~f:(fun params (x, u) -> { params; x; u })
  in
  (* backward pass with surrogate params constructed on the fly *)
  let _, _, info_list =
    List.fold2_exn
      (List.rev common_info)
      (List.rev params_xu_list)
      ~init:(lambda_T, _cx_surro_T, [])
      ~f:(fun (lambda_next, _v, info_list) z p ->
        cleanup ();
        let _dCxx_curr = _p_tangent p.params.common._Cxx in
        let _dCxu_curr = _p_tangent p.params.common._Cxu in
        let _dcx_curr = _p_tangent p.params._cx in
        let _dCuu_curr = _p_tangent p.params.common._Cuu in
        let _dcu_curr = _p_tangent p.params._cu in
        let _cx_surro_curr =
          let tmp1 =
            let _dCxx_str = if batch_const then "kba" else "kmba" in
            maybe_einsum (p.x, "mb") (_dCxx_curr, _dCxx_str) "kma"
          in
          let tmp2 =
            let _dCxu_str = if batch_const then "kab" else "kmab" in
            maybe_einsum (p.u, "mb") (_dCxu_curr, _dCxu_str) "kma"
          in
          let common = tmp1 +? tmp2 +? _dcx_curr in
          let tmp3 = _p_implicit_tangent p.params.common._Fx_prod *? lambda_next in
          common +? tmp3
        in
        Stdlib.Gc.major ();
        let _cu_surro_curr =
          let tmp1 =
            let _dCuu_str = if batch_const then "kab" else "kmab" in
            maybe_einsum (p.u, "ma") (_dCuu_curr, _dCuu_str) "kmb"
          in
          let tmp2 =
            let _dCxu_str = if batch_const then "kab" else "kmab" in
            maybe_einsum (p.x, "ma") (_dCxu_curr, _dCxu_str) "kmb"
          in
          let common = tmp1 +? tmp2 +? _dcu_curr in
          let tmp3 = _p_implicit_tangent p.params.common._Fu_prod *? lambda_next in
          common +? tmp3
        in
        let _f_surro_curr =
          let tmp1 = _p_implicit_tangent p.params.common._Fx_prod2 *? p.x in
          let tmp2 = _p_implicit_tangent p.params.common._Fu_prod2 *? p.u in
          tmp1 +? tmp2 +? _p_tangent p.params._f
        in
        let lambda_curr =
          let common =
            let _Cxx_curr = _p_primal p.params.common._Cxx in
            let _cx_curr = _p_primal p.params._cx in
            let _Cxx_str = if batch_const then "ba" else "mba" in
            _cx_curr +? maybe_einsum (p.x, "mb") (_Cxx_curr, _Cxx_str) "ma"
          in
          let tmp2 =
            let _Cxu_str = if batch_const then "ab" else "mab" in
            let _Cxu_curr = _p_primal p.params.common._Cxu in
            maybe_einsum (p.u, "mb") (_Cxu_curr, _Cxu_str) "ma"
          in
          let tmp3 = _p_implicit_primal p.params.common._Fx_prod *? lambda_next in
          common +? tmp2 +? tmp3
        in
        (* backward pass *)
        let _qu, _qx =
          let _Fu_prod, _Fx_prod =
            ( _p_implicit_primal p.params.common._Fu_prod_tangent
            , _p_implicit_primal p.params.common._Fx_prod_tangent )
          in
          _qu_qx
            ~tangent:true
            ~batch_const
            ~z
            ~_cu:_cu_surro_curr
            ~_cx:_cx_surro_curr
            ~_f:_f_surro_curr
            ~_Fu_prod
            ~_Fx_prod
            _v
        in
        (* compute LQR gain parameters to be used in the subsequent fwd pass *)
        let _k = _k ~tangent:true ~batch_const ~z ~_f:_f_surro_curr _qu in
        (* update the value function *)
        let _v = v ~tangent:true ~batch_const ~_K:z._K ~_qx ~_qu in
        lambda_curr, _v, { _k; _K = z._K; _f = _f_surro_curr } :: info_list)
  in
  (* forward pass *)
  let _, solution =
    List.fold2_exn
      params_except_last
      info_list
      ~init:(_p_tangent p.x0, [])
      ~f:(fun (x, accu) p b ->
        cleanup ();
        (* compute the optimal control inputs *)
        let u = _u ~tangent:true ~batch_const ~_k:b._k ~_K:b._K ~x in
        (* fold them into the state update *)
        let x =
          let _Fx_prod2, _Fu_prod2 =
            ( _p_implicit_primal p.common._Fx_prod2_tangent
            , _p_implicit_primal p.common._Fu_prod2_tangent )
          in
          _x ~_f:b._f ~_Fx_prod2 ~_Fu_prod2 ~x ~u
        in
        x, Solution.{ u; x } :: accu)
  in
  List.rev solution

let solve ?(batch_const = false) ?(laplace = false) p =
  (* solve the primal problem first *)
  let _p = Option.map ~f:(fun x -> Maths.const (Maths.primal x)) in
  let _p_implicit = Option.map ~f:(fun x -> x.primal) in
  let p_primal =
    Params.
      { x0 = _p p.x0
      ; params =
          List.map p.params ~f:(fun p ->
            { common =
                { _Fx_prod = _p_implicit p.common._Fx_prod
                ; _Fx_prod2 = _p_implicit p.common._Fx_prod2
                ; _Fu_prod = _p_implicit p.common._Fu_prod
                ; _Fu_prod2 = _p_implicit p.common._Fu_prod2
                ; _Fx_prod_tangent = _p_implicit p.common._Fx_prod_tangent
                ; _Fx_prod2_tangent = _p_implicit p.common._Fx_prod2_tangent
                ; _Fu_prod_tangent = _p_implicit p.common._Fu_prod_tangent
                ; _Fu_prod2_tangent = _p_implicit p.common._Fu_prod2_tangent
                ; _Fx_prod_inv = _p_implicit p.common._Fx_prod_inv
                ; _Fx_prod2_inv = _p_implicit p.common._Fx_prod2_inv
                ; _Fu_prod_trans = _p_implicit p.common._Fu_prod_trans
                ; _Fu_prod2_trans = _p_implicit p.common._Fu_prod2_trans
                ; _Fx_prod_inv_trans = _p_implicit p.common._Fx_prod_inv_trans
                ; _Fx_prod2_inv_trans = _p_implicit p.common._Fx_prod2_inv_trans
                ; _Cxx = _p p.common._Cxx
                ; _Cuu = _p p.common._Cuu
                ; _Cxu = _p p.common._Cxu
                }
            ; _f = _p p._f
            ; _cx = _p p._cx
            ; _cu = _p p._cu
            })
      }
  in
  let common_info =
    backward_common
      ~batch_const
      ~laplace
      (List.map p_primal.Params.params ~f:(fun x -> x.common))
  in
  cleanup ();
  (* SOLVE THE PRIMAL PROBLEM *)
  let s =
    let bck = backward common_info ~batch_const p_primal in
    cleanup ();
    bck |> forward ~batch_const p_primal
  in
  cleanup ();
  (* SOLVE THE TANGENT PROBLEM, reusing what's common *)
  let s_tangents = tangent_solve ~batch_const common_info s p in
  cleanup ();
  (* MANUALLY PAIR UP PRIMAL AND TANGENTS OF THE SOLUTION *)
  let sol =
    List.map2_exn s s_tangents ~f:(fun s st ->
      let zip a at =
        match a with
        | Some a ->
          (match at with
           | None -> Maths.const
           | Some at -> Maths.make_dual ~t:(Direct (primal at)))
            (Maths.primal a)
        | None -> failwith "for some reason, no solution in there"
      in
      Solution.{ u = zip s.u st.u; x = zip s.x st.x })
  in
  (* TODO: return backward info list for smoothed posterior covariance over u *)
  if laplace
  then (
    (* TODO: if we need covariances to carry tangents we need to compute everything with tangents, hence cannot be used in this separable method. For now we assume we do not need the covariances to carry tangents i.e. do not differentiate through the samples *)
    let covariances = covariances ~batch_const ~common_info p_primal in
    sol, Some (List.map covariances ~f:(fun x -> Option.value_exn x)))
  else sol, None
