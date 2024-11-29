open Base
open Torch
open Forward_torch
include Lqr_typ

let print s = Stdio.printf "%s\n%!" (Base.Sexp.to_string_hum s)

open Maths

let ( *@ ) a b =
  match List.length (shape b) with
  | 3 -> einsum [ a, "aij"; b, "ajk" ] "aik"
  | 2 -> einsum [ a, "aij"; b, "aj" ] "ai"
  | _ -> failwith "not batch multipliable"

let maybe_btr = Option.map ~f:btr

let maybe_add a b =
  match a, b with
  | None, None -> None
  | Some a, None -> Some a
  | None, Some b -> Some b
  | Some a, Some b -> Some (a + b)

let maybe_prod f v =
  match f, v with
  | Some f, Some v -> Some (f v)
  | _ -> None

(* let maybe_prod_tan f v =
   match f, v with
   | Some f, Some v ->
   let g x = f in
   let x_tan = Maths.tangent x in

   Some (f v)
   | _ -> None *)

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
    Option.map _y ~f:(linsolve_triangular ~left:true ~upper:true ell_T)
  | _ -> None

(* backward recursion: all the (most expensive) common bits *)
let backward_common (common : (t, t -> t) momentary_params_common list) =
  let _V =
    match List.last common with
    | None -> failwith "LQR needs a time horizon >= 1"
    | Some z -> z._Cxx
  in
  let _, info =
    List.fold_right common ~init:(_V, []) ~f:(fun z (_V, info_list) ->
      let _Quu, _Qxx, _Qux =
        let tmp = maybe_prod z._Fx_prod _V in
        ( maybe_add z._Cuu (maybe_prod z._Fu_prod (maybe_btr (maybe_prod z._Fu_prod _V)))
        , maybe_add z._Cxx (maybe_prod z._Fx_prod (maybe_btr tmp))
        , maybe_add z._Cxu (maybe_prod z._Fu_prod (maybe_btr tmp)) )
      in
      (* compute LQR gain parameters to be used in the subsequent fwd pass *)
      let _Quu_chol = Option.map _Quu ~f:cholesky in
      let _Quu_chol_T = maybe_btr _Quu_chol in
      let _K = neg_inv_symm ~is_vector:false _Qux (_Quu_chol, _Quu_chol_T) in
      (* update the value function *)
      let _V_new = maybe_add _Qxx (maybe_einsum (_Qux, "auy") (_K, "auz") "ayz") in
      _V_new, { _Quu_chol; _Quu_chol_T; _V; _K } :: info_list)
  in
  info (* that's now → in time *)

(* backward recursion; assumes all the common (expensive) stuff has already been
   computed *)
let backward common_info (params : (t option, (t, t -> t) momentary_params list) Params.p)
  =
  let _v =
    match List.last params.params with
    | None -> failwith "LQR needs a time horizon >= 1"
    | Some z -> z._cx
  in
  let _, info =
    List.fold_right2_exn
      common_info
      params.params
      ~init:(_v, [])
      ~f:(fun z p (_v, info_list) ->
        let _qu, _qx =
          let tmp = maybe_force _v z._V p._f in
          ( maybe_add p._cu (maybe_prod p.common._Fu_prod tmp)
          , maybe_add p._cx (maybe_prod p.common._Fx_prod tmp) )
        in
        (* compute LQR gain parameters to be used in the subsequent fwd pass *)
        let _k = neg_inv_symm ~is_vector:true _qu (z._Quu_chol, z._Quu_chol_T) in
        (* update the value function *)
        let _v = maybe_add _qx (maybe_einsum (z._K, "azu") (_qu, "az") "au") in
        _v, { _k; _K = z._K } :: info_list)
  in
  info
(* TODO: we should not need to reverse here *)

let forward params backward_info =
  let open Params in
  let _, solution =
    List.fold2_exn
      params.params
      backward_info
      ~init:(params.x0, [])
      ~f:(fun (x, accu) p b ->
        (* compute the optimal control inputs *)
        let u = maybe_add b._k (maybe_einsum (x, "au") (b._K, "ayu") "ay") in
        (* fold them into the state update *)
        let x =
          maybe_add
            p._f
            (maybe_add
               (maybe_prod p.common._Fx_prod2 x)
               (maybe_prod p.common._Fu_prod2 u))
        in
        x, Solution.{ u; x } :: accu)
  in
  List.rev solution

let _solve p =
  let common_info = backward_common (List.map p.Params.params ~f:(fun x -> x.common)) in
  backward common_info p |> forward p

(* surrogate rhs for the tangent problem; s is the solution obtained from lqr through the primals and p is the full set
   of parameters *)
let surrogate_rhs
  (s : t option Solution.p list)
  (p : (t option, (t, t prod) momentary_params list) Params.p)
  : (t option, (t, t -> t) momentary_params list) Params.p
  =
  let _p_implicit_primal = Option.map ~f:(fun x -> x.primal) in
  let _p_implicit_tangent = Option.map ~f:(fun x -> x.tangent) in
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
  (* initialise lambda *)
  let x_final, u_final =
    let tmp = List.last_exn s in
    tmp.x, tmp.u
  in
  let params_final = List.last_exn p.params in
  let lambda_final =
    let _Cxx_final = _p_primal params_final.common._Cxx in
    let _Cx_final = _p_primal params_final._cx in
    maybe_force _Cx_final x_final _Cxx_final
  in
  (* initialise _cx and _cu*)
  let _cx_surro_final, _cu_surro_final =
    let _dCxx_final = _p_tangent params_final.common._Cxx in
    let _dCxu_final = _p_tangent params_final.common._Cxu in
    let _dcx_final = _p_tangent params_final._cx in
    let _dCuu_final = _p_tangent params_final.common._Cuu in
    let _dcu_final = _p_tangent params_final._cu in
    let _cx_surro_final =
      let tmp1 = maybe_einsum (x_final, "ma") (_dCxx_final, "kmab") "kmb" in
      let tmp2 = maybe_einsum (u_final, "ma") (_dCxu_final, "kmba") "kmb" in
      maybe_add (maybe_add tmp1 tmp2) _dcx_final
    in
    let _cu_surro_final =
      let tmp1 = maybe_einsum (u_final, "ma") (_dCuu_final, "kmab") "kmb" in
      let tmp2 = maybe_einsum (x_final, "ma") (_dCxu_final, "kmab") "kmb" in
      maybe_add (maybe_add tmp1 tmp2) _dcu_final
    in
    _cx_surro_final, _cu_surro_final
  in
  let n_steps = List.length p.params in
  let n_steps_list = List.init n_steps ~f:(fun i -> i) in
  let _cx_cu_surro, _ =
    List.fold_right
      n_steps_list
      ~init:([ _cx_surro_final, _cu_surro_final ], lambda_final)
      ~f:(fun t (_cx_cu_surro_accu, lambda_next) ->
        let params_curr = List.nth_exn p.params t in
        let x_curr, u_curr =
          let tmp = List.nth_exn s t in
          tmp.x, tmp.u
        in
        let _dCxx_curr = _p_tangent params_curr.common._Cxx in
        let _dCxu_curr = _p_tangent params_curr.common._Cxu in
        let _dcx_curr = _p_tangent params_curr._cx in
        let _dCuu_curr = _p_tangent params_curr.common._Cuu in
        let _dcu_curr = _p_tangent params_curr._cu in
        let _cx_surro_curr =
          let tmp1 = maybe_einsum (x_curr, "ma") (_dCxx_curr, "kmab") "kmb" in
          let tmp2 = maybe_einsum (u_curr, "ma") (_dCxu_curr, "kmba") "kmb" in
          let tmp3 =
            maybe_prod (_p_implicit_primal params_curr.common._Fx_prod) lambda_next
          in
          maybe_add (maybe_add (maybe_add tmp1 tmp2) tmp3) _dcx_curr
        in
        let _cu_surro_curr =
          let tmp1 = maybe_einsum (u_curr, "ma") (_dCuu_curr, "kmab") "kmb" in
          let tmp2 = maybe_einsum (x_curr, "ma") (_dCxu_curr, "kmab") "kmb" in
          let tmp3 =
            maybe_prod (_p_implicit_primal params_curr.common._Fu_prod) lambda_next
          in
          maybe_add (maybe_add (maybe_add tmp1 tmp2) tmp3) _dcu_curr
        in
        let lambda_curr =
          let tmp1 =
            let _Cxx_curr = _p_primal params_curr.common._Cxx in
            let _Cx_curr = _p_primal params_curr._cx in
            maybe_force _Cx_curr x_curr _Cxx_curr
          in
          let tmp2 =
            let _Cxu_curr = _p_primal params_curr.common._Cxu in
            maybe_einsum (u_curr, "mb") (_Cxu_curr, "mab") "ma"
          in
          let tmp3 =
            maybe_prod (_p_implicit_primal params_curr.common._Fx_prod2) lambda_next
          in
          maybe_add (maybe_add tmp1 tmp2) tmp3
        in
        (_cx_surro_curr, _cu_surro_curr) :: _cx_cu_surro_accu, lambda_curr)
  in
  let _f_surro =
    List.map2_exn p.params s ~f:(fun params_curr s_curr ->
      let x_curr, u_curr = s_curr.x, s_curr.u in
      let tmp1 = maybe_prod (_p_implicit_tangent params_curr.common._Fx_prod2) x_curr in
      let tmp2 = maybe_prod (_p_implicit_tangent params_curr.common._Fu_prod2) u_curr in
      maybe_add (maybe_add tmp1 tmp2) (_p_primal params_curr._f))
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
                  ; _Fu_prod2 = _p_implicit_primal p.common._Fu_prod
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
                ; _Fu_prod2 = _p_implicit p.common._Fu_prod
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
  let s_tangents = _solve surrogate in
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
