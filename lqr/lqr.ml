open Base
open Torch
open Forward_torch
include Lqr_typ

let print s = Stdio.printf "%s\n%!" (Base.Sexp.to_string_hum s)

open Maths

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

(* reshape a from [shape] to [1, shape] *)
let maybe_expand a =
  match a with
  | Some a -> Some (unsqueeze a ~dim:0)
  | None -> None

let maybe_shape a =
  match a with
  | Some a -> shape a
  | None -> failwith "shape of a None object cannot be determined"

let maybe_reshape a ~shape =
  match a with
  | Some a -> Some (reshape a ~shape)
  | None -> failwith "shape of a None object cannot be determined"

type backward_common_info =
  { _Quu_chol : t option
  ; _Quu_chol_T : t option
  ; _V : t option
  ; _K : t option
  }

type backward_info =
  { _K : t option
  ; _k : t option
  }

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

(* backward recursion; assumes all the common (expensive) stuff has already been
   computed *)
let backward common_info (params : (t option, (t, t -> t) momentary_params list) Params.p)
  =
  let _v =
    match List.last params.params with
    | None -> failwith "LQR needs a time horizon >= 1"
    | Some z -> z._cx
  in
  (* info (k/K) goes from 0 to T-1 *)
  let _, info =
    let params_except_last = List.drop_last_exn params.params in
    List.fold_right2_exn
      common_info
      params_except_last
      ~init:(_v, [])
      ~f:(fun z p (_v, info_list) ->
        let _qu, _qx =
          let tmp = maybe_force _v z._V p._f in
          p._cu +? (p.common._Fu_prod *? tmp), p._cx +? (p.common._Fx_prod *? tmp)
        in
        (* compute LQR gain parameters to be used in the subsequent fwd pass *)
        let _k =
          let _k_unsqueezed =
            neg_inv_symm ~is_vector:true _qu (z._Quu_chol, z._Quu_chol_T)
          in
          match _k_unsqueezed with
          | None -> None
          | Some _k_unsqueezed ->
            Some (reshape _k_unsqueezed ~shape:(List.take (shape _k_unsqueezed) 2))
        in
        (* update the value function *)
        let _v = _qx +? maybe_einsum (z._K, "mba") (_qu, "mb") "ma" in
        _v, { _k; _K = z._K } :: info_list)
  in
  info

(* backward with some parameters having the tangent dimension *)
let backward_tangents
  common_info
  (params : (t option, (t, t -> t) momentary_params list) Params.p)
  =
  let _v =
    match List.last params.params with
    | None -> failwith "LQR needs a time horizon >= 1"
    | Some z -> z._cx
  in
  let _, info =
    let params_except_last = List.drop_last_exn params.params in
    List.fold_right2_exn
      common_info
      params_except_last
      ~init:(_v, [])
      ~f:(fun z p (_v, info_list) ->
        let _qu, _qx =
          let tmp = _v +? maybe_einsum (z._V, "mab") (p._f, "kmb") "kma" in
          ( p._cu +? (p.common._Fu_prod_tangent *? tmp)
          , p._cx +? (p.common._Fx_prod_tangent *? tmp) )
        in
        (* compute LQR gain parameters to be used in the subsequent fwd pass *)
        let _k =
          match p._f with
          | Some _f ->
            let n_tangents, bs = List.hd_exn (shape _f), List.nth_exn (shape _f) 1 in
            (* shape [km x b x b ]*)
            let _quu_chol_expanded =
              List.init n_tangents ~f:(fun _ ->
                unsqueeze (Option.value_exn z._Quu_chol) ~dim:0)
              |> concat_list ~dim:0
            in
            let _quu_chol_squeezed =
              Some
                (reshape
                   _quu_chol_expanded
                   ~shape:
                     (Int.(n_tangents * bs)
                      :: [ List.last_exn (shape _quu_chol_expanded)
                         ; List.last_exn (shape _quu_chol_expanded)
                         ]))
            in
            let _quu_chol_squeezed_T = maybe_btr _quu_chol_squeezed in
            (* shape [km x b] *)
            let _qu_squeezed = maybe_reshape _qu ~shape:[ Int.(n_tangents * bs); -1 ] in
            let _k_squeezed =
              neg_inv_symm
                ~is_vector:true
                _qu_squeezed
                (_quu_chol_squeezed, _quu_chol_squeezed_T)
            in
            maybe_reshape _k_squeezed ~shape:[ n_tangents; bs; -1 ]
          | None -> None
        in
        (* update the value function *)
        let _v = _qx +? maybe_einsum (z._K, "mba") (_qu, "kmb") "kma" in
        _v, { _k; _K = z._K } :: info_list)
  in
  info

let forward params backward_info =
  let open Params in
  (* u goes from 0 to T-1, x goes from 1 to T *)
  let _, solution =
    let params_except_last = List.drop_last_exn params.params in
    List.fold2_exn
      params_except_last
      backward_info
      ~init:(params.x0, [])
      ~f:(fun (x, accu) p b ->
        (* compute the optimal control inputs *)
        let u = b._k +? maybe_einsum (x, "ma") (b._K, "mba") "mb" in
        (* fold them into the state update *)
        let x = p._f +? (p.common._Fx_prod2 *? x) +? (p.common._Fu_prod2 *? u) in
        x, Solution.{ u; x } :: accu)
  in
  List.rev solution

let forward_tangents params backward_info =
  let open Params in
  let _, solution =
    let params_except_last = List.drop_last_exn params.params in
    List.fold2_exn
      params_except_last
      backward_info
      ~init:(params.x0, [])
      ~f:(fun (x, accu) p b ->
        (* compute the optimal control inputs *)
        let u = b._k +? maybe_einsum (x, "kma") (b._K, "mba") "kmb" in
        (* fold them into the state update *)
        let x =
          p._f +? (p.common._Fx_prod2_tangent *? x) +? (p.common._Fu_prod2_tangent *? u)
        in
        x, Solution.{ u; x } :: accu)
  in
  List.rev solution

let _solve p =
  let common_info = backward_common (List.map p.Params.params ~f:(fun x -> x.common)) in
  backward common_info p
  |> forward p
  |> List.map ~f:(fun s ->
    Solution.{ x = Option.value_exn s.x; u = Option.value_exn s.u })

(* surrogate rhs for the tangent problem; s is the solution obtained from lqr through the primals and p is the full set
   of parameters *)
let surrogate_rhs
  (s : t option Solution.p list)
  (p : (t option, (t, t prod) momentary_params list) Params.p)
  : (t option, (t, t -> t) momentary_params list) Params.p
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
    | Some x ->
      let x_tangent = Maths.tangent x in
      (match x_tangent with
       | None -> None
       | Some tangent -> Some (Maths.const tangent))
  in
  (* create new list and foldi is necessary since u goes from 0 to T-1 but x goes from 1 to T in solution *)
  let n_steps = Int.(List.length p.params - 1) in
  let n_steps_list = List.range 0 Int.(n_steps + 1) in
  (* this list goes from 0 to T *)
  let _cx_cu_surro, _ =
    List.fold_right
      n_steps_list
      ~init:([], None)
      ~f:(fun i (_cx_cu_surro_accu, lambda_next) ->
        let x_curr =
          if i = 0
          then _p_primal p.x0
          else (
            let tmp = List.nth_exn s Int.(i - 1) in
            tmp.x)
        in
        let u_curr =
          if i = n_steps
          then None
          else (
            let tmp = List.nth_exn s i in
            tmp.u)
        in
        let params_curr = List.nth_exn p.params i in
        let _dCxx_curr = _p_tangent params_curr.common._Cxx in
        let _dCxu_curr = _p_tangent params_curr.common._Cxu in
        let _dcx_curr = _p_tangent params_curr._cx in
        let _dCuu_curr = _p_tangent params_curr.common._Cuu in
        let _dcu_curr = _p_tangent params_curr._cu in
        let _cx_surro_curr =
          if i = 0
          then None
          else (
            let tmp1 = maybe_einsum (x_curr, "mb") (_dCxx_curr, "kmba") "kma" in
            let tmp2 = maybe_einsum (u_curr, "mb") (_dCxu_curr, "kmab") "kma" in
            let common = tmp1 +? tmp2 +? _dcx_curr in
            match lambda_next with
            | None -> common
            | Some lambda_next ->
              let tmp3 = _p_implicit_tangent params_curr.common._Fx_prod *? lambda_next in
              common +? tmp3)
        in
        let _cu_surro_curr =
          if i = n_steps
          then None
          else (
            let tmp1 = maybe_einsum (u_curr, "ma") (_dCuu_curr, "kmab") "kmb" in
            let tmp2 = maybe_einsum (x_curr, "ma") (_dCxu_curr, "kmab") "kmb" in
            let common = tmp1 +? tmp2 +? _dcu_curr in
            match lambda_next with
            | None -> common
            | Some lambda_next ->
              let tmp3 = _p_implicit_tangent params_curr.common._Fu_prod *? lambda_next in
              common +? tmp3)
        in
        let lambda_curr =
          let common =
            let _Cxx_curr = _p_primal params_curr.common._Cxx in
            let _cx_curr = _p_primal params_curr._cx in
            maybe_force _cx_curr _Cxx_curr x_curr
          in
          match lambda_next with
          | None -> common
          | Some lambda_next ->
            let tmp2 =
              let _Cxu_curr = _p_primal params_curr.common._Cxu in
              maybe_einsum (u_curr, "mb") (_Cxu_curr, "mab") "ma"
            in
            let tmp3 = _p_implicit_primal params_curr.common._Fx_prod *? lambda_next in
            common +? tmp2 +? tmp3
        in
        (_cx_surro_curr, _cu_surro_curr) :: _cx_cu_surro_accu, Some lambda_curr)
  in
  let _f_surro =
    List.mapi p.params ~f:(fun i params_curr ->
      let x_curr =
        if i = 0
        then _p_primal p.x0
        else (
          let s_curr = List.nth_exn s Int.(i - 1) in
          s_curr.x)
      in
      let u_curr =
        if i = n_steps
        then None
        else (
          let s_curr = List.nth_exn s i in
          s_curr.u)
      in
      let tmp1 = _p_implicit_tangent params_curr.common._Fx_prod2 *? x_curr in
      let tmp2 = _p_implicit_tangent params_curr.common._Fu_prod2 *? u_curr in
      if i = n_steps then None else tmp1 +? tmp2 +? _p_tangent params_curr._f)
  in
  let p_tangent =
    Params.
      { x0 = _p_tangent p.x0
      ; params =
          List.map2_exn
            p.params
            (List.zip_exn _f_surro _cx_cu_surro)
            ~f:(fun p (_f_surro, (_cx_surro, _cu_surro)) ->
              { common =
                  { _Fx_prod = _p_implicit_primal p.common._Fx_prod
                  ; _Fx_prod2 = _p_implicit_primal p.common._Fx_prod2
                  ; _Fu_prod = _p_implicit_primal p.common._Fu_prod
                  ; _Fu_prod2 = _p_implicit_primal p.common._Fu_prod2
                  ; _Fx_prod_tangent = _p_implicit_primal p.common._Fx_prod_tangent
                  ; _Fx_prod2_tangent = _p_implicit_primal p.common._Fx_prod2_tangent
                  ; _Fu_prod_tangent = _p_implicit_primal p.common._Fu_prod_tangent
                  ; _Fu_prod2_tangent = _p_implicit_primal p.common._Fu_prod2_tangent
                  ; _Cxx = _p_primal p.common._Cxx
                  ; _Cuu = _p_primal p.common._Cuu
                  ; _Cxu = _p_primal p.common._Cxu
                  }
              ; _f = _f_surro
              ; _cx = _cx_surro
              ; _cu = _cu_surro
              })
      }
  in
  p_tangent

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
  (* SOLVE THE PRIMAL PROBLEM *)
  let s = backward common_info p_primal |> forward p_primal in
  (* SOLVE THE TANGENT PROBLEM, reusing what's common *)
  let surrogate = surrogate_rhs s p in
  let s_tangents =
    backward_tangents common_info surrogate |> forward_tangents surrogate
  in
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
