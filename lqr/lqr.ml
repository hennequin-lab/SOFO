open Base
open Forward_torch
include Lqr_typ
open Maths

let cleanup () = Stdlib.Gc.major ()

let ( *@ ) a b =
  match List.length (shape b) with
  | 3 -> einsum [ a, "mab"; b, "mbc" ] "mac"
  | 2 -> einsum [ a, "mab"; b, "mb" ] "ma"
  | _ -> failwith "not batch multipliable"

let maybe_btr = Option.map ~f:btr

let ( +? ) a b =
  match a, b with
  | None, None -> None
  | Some a, None -> Some a
  | None, Some b -> Some b
  | Some a, Some b -> Some (a + b)

let ( *? ) f v =
  match f, v with
  | Some f, Some v -> Some (f v)
  | _ -> None

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

(* backward recursion: all the (most expensive) common bits *)
let backward_common (common : (t, t -> t) momentary_params_common list) =
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
        let tmp = z._Fx_prod *? _V in
        ( z._Cuu +? (z._Fu_prod *? maybe_btr (z._Fu_prod *? _V))
        , z._Cxx +? (z._Fx_prod *? maybe_btr tmp)
        , maybe_btr z._Cxu +? (z._Fu_prod *? maybe_btr tmp) )
      in
      (* compute LQR gain parameters to be used in the subsequent fwd pass *)
      let _Quu_chol = Option.map _Quu ~f:cholesky in
      let _Quu_chol_T = maybe_btr _Quu_chol in
      let _K = neg_inv_symm ~is_vector:false _Qux (_Quu_chol, _Quu_chol_T) in
      (* update the value function *)
      let _V_new = _Qxx +? maybe_einsum (_Qux, "mab") (_K, "mac") "mbc" in
      _V_new, { _Quu_chol; _Quu_chol_T; _V; _K } :: info_list)
  in
  info

let _k ~tangent ~z ~_f _qu =
  if tangent
  then (
    match _f with
    | Some _f ->
      (* from [k x m x b] to [m x b x k] *)
      let _qu_swapped =
        Option.value_exn _qu
        |> Maths.transpose ~dim0:0 ~dim1:1
        |> Maths.transpose ~dim0:1 ~dim1:2
        |> Some
      in
      let _k_swapped =
        neg_inv_symm ~is_vector:false _qu_swapped (z._Quu_chol, z._Quu_chol_T)
      in
      (* from [m x b x k] to [k x m x b] *)
      Maths.transpose ~dim0:1 ~dim1:2 (Option.value_exn _k_swapped)
      |> Maths.transpose ~dim0:0 ~dim1:1
      |> Some
    | None -> None)
  else (
    let _k_unsqueezed = neg_inv_symm ~is_vector:true _qu (z._Quu_chol, z._Quu_chol_T) in
    match _k_unsqueezed with
    | None -> None
    | Some _k_unsqueezed ->
      Some (reshape _k_unsqueezed ~shape:(List.take (shape _k_unsqueezed) 2)))

let _qu_qx ~tangent ~z ~_cu ~_cx ~_f ~_Fu_prod ~_Fx_prod _v =
  let tmp =
    if tangent
    then _v +? maybe_einsum (z._V, "mab") (_f, "kmb") "kma"
    else maybe_force _v z._V _f
  in
  _cu +? (_Fu_prod *? tmp), _cx +? (_Fx_prod *? tmp)

let v ~tangent ~_K ~_qx ~_qu =
  if tangent
  then _qx +? maybe_einsum (_K, "mba") (_qu, "kmb") "kma"
  else _qx +? maybe_einsum (_K, "mba") (_qu, "mb") "ma"

(* backward recursion; assumes all the common (expensive) stuff has already been
   computed *)
let backward
  ?(tangent = false)
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
          _qu_qx ~tangent ~z ~_cu:p._cu ~_cx:p._cx ~_f:p._f ~_Fu_prod ~_Fx_prod _v
        in
        (* compute LQR gain parameters to be used in the subsequent fwd pass *)
        let _k = _k ~tangent ~z ~_f:p._f _qu in
        (* update the value function *)
        let _v = v ~tangent ~_K:z._K ~_qx ~_qu in
        _v, { _k; _K = z._K } :: info_list)
  in
  info

let _u ~tangent ~_k ~_K ~x =
  if tangent
  then _k +? maybe_einsum (x, "kma") (_K, "mba") "kmb"
  else _k +? maybe_einsum (x, "ma") (_K, "mba") "mb"

let _x ~_f ~_Fx_prod2 ~_Fu_prod2 ~x ~u = _f +? (_Fx_prod2 *? x) +? (_Fu_prod2 *? u)

let forward ?(tangent = false) params (backward_info : backward_info list) =
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
        let u = _u ~tangent ~_k:b._k ~_K:b._K ~x in
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

let _solve p =
  let common_info = backward_common (List.map p.Params.params ~f:(fun x -> x.common)) in
  cleanup ();
  let bck = backward common_info p in
  cleanup ();
  bck
  |> forward p
  |> List.map ~f:(fun s ->
    Solution.{ x = Option.value_exn s.x; u = Option.value_exn s.u })

(* backward pass and forward pass with surrogate rhs for the tangent problem; s is the solution obtained from lqr through the primals and p is the full set of parameters *)
let tangent_solve
  common_info
  (s : t option Solution.p list)
  (p : (t option, (t, t prod) momentary_params list) Params.p)
    (* : t option Solution.p list *)
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
    maybe_force _cx_curr _Cxx_curr x_T
  in
  let _cx_surro_T =
    let _dCxx_T = _p_tangent params_T.common._Cxx in
    let _dcx_T = _p_tangent params_T._cx in
    let tmp1 = maybe_einsum (x_T, "mb") (_dCxx_T, "kmba") "kma" in
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
          let tmp1 = maybe_einsum (p.x, "mb") (_dCxx_curr, "kmba") "kma" in
          let tmp2 = maybe_einsum (p.u, "mb") (_dCxu_curr, "kmab") "kma" in
          let common = tmp1 +? tmp2 +? _dcx_curr in
          let tmp3 = _p_implicit_tangent p.params.common._Fx_prod *? lambda_next in
          common +? tmp3
        in
        Stdlib.Gc.major ();
        let _cu_surro_curr =
          let tmp1 = maybe_einsum (p.u, "ma") (_dCuu_curr, "kmab") "kmb" in
          let tmp2 = maybe_einsum (p.x, "ma") (_dCxu_curr, "kmab") "kmb" in
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
            maybe_force _cx_curr _Cxx_curr p.x
          in
          let tmp2 =
            let _Cxu_curr = _p_primal p.params.common._Cxu in
            maybe_einsum (p.u, "mb") (_Cxu_curr, "mab") "ma"
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
            ~z
            ~_cu:_cu_surro_curr
            ~_cx:_cx_surro_curr
            ~_f:_f_surro_curr
            ~_Fu_prod
            ~_Fx_prod
            _v
        in
        (* compute LQR gain parameters to be used in the subsequent fwd pass *)
        let _k = _k ~tangent:true ~z ~_f:_f_surro_curr _qu in
        (* update the value function *)
        let _v = v ~tangent:true ~_K:z._K ~_qx ~_qu in
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
        let u = _u ~tangent:true ~_k:b._k ~_K:b._K ~x in
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

let solve p =
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
    backward_common (List.map p_primal.Params.params ~f:(fun x -> x.common))
  in
  cleanup ();
  (* SOLVE THE PRIMAL PROBLEM *)
  let s =
    let bck = backward common_info p_primal in
    cleanup ();
    bck |> forward p_primal
  in
  cleanup ();
  (* SOLVE THE TANGENT PROBLEM, reusing what's common *)
  let s_tangents = tangent_solve common_info s p in
  cleanup ();
  (* MANUALLY PAIR UP PRIMAL AND TANGENTS OF THE SOLUTION *)
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
