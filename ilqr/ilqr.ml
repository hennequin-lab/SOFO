open Base
open Forward_torch
open Torch
open Lqr
open Maths

let cleanup () = Stdlib.Gc.major ()
let print s = Stdio.print_endline (Sexp.to_string_hum s)

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

(* optionally f v *)
let ( *? ) f v =
  match f, v with
  | Some f, Some v -> Some (f v)
  | _ -> None

let maybe_mul a b =
  match a, b with
  | Some a, Some b -> Some Maths.(a * b)
  | _ -> None

let maybe_einsum (a, opsA) (b, opsB) opsC =
  match a, b with
  | Some a, Some b -> Some (einsum [ a, opsA; b, opsB ] opsC)
  | _ -> None

(* artificially add one to tau so it goes from 0 to T *)
let extend_tau_list ~x0 (tau : Maths.any Maths.t option Lqr.Solution.p list) =
  let u_list = List.map tau ~f:(fun s -> s.u) in
  let x_list = List.map tau ~f:(fun s -> s.x) in
  let u_ext = u_list @ [ None ] in
  let x_ext = x0 :: x_list in
  List.map2_exn u_ext x_ext ~f:(fun u x -> Lqr.Solution.{ u; x })

let _u ~tangent ~batch_const ~_k ~_K ~x ~u ~alpha =
  let tmp =
    match tangent, batch_const with
    | true, true -> maybe_einsum (x, "kma") (_K, "ba") "kmb"
    | true, false -> maybe_einsum (x, "kma") (_K, "mba") "kmb"
    | false, true -> maybe_einsum (x, "ma") (_K, "ba") "mb"
    | false, false -> maybe_einsum (x, "ma") (_K, "mba") "mb"
  in
  (* maybe_scalar_mul _k alpha +? tmp +? u *)
  (maybe_einsum (_k, "ma")) (Some Maths.(any (of_tensor alpha)), "m") "ma" +? tmp +? u

(* calculate the new u w.r.t. the difference between x_t and x_opt_t *)
let forward
      ?(beta = 0.1)
      ~batch_const
      ~linesearch
      ~linesearch_bs_avg
      ~gamma
      ~cost_func
      ~f_theta
      ~(p : (any t option, (any t, any t -> any t) momentary_params list) Params.p)
      ~(tau_opt : any t option Solution.p list)
      ~_dC1
      ~_dC2
      (bck : backward_info list)
  =
  let cost_init = cost_func tau_opt in
  let rec fwd_loop ~stop ~i ~alpha ~tau_prev =
    if stop
    then tau_prev
    else (
      let x0 = p.x0 in
      let alpha_c = Maths.of_tensor alpha |> Maths.any in
      (* let u0 = maybe_scalar_mul (List.hd_exn bck)._k alpha +? (List.hd_exn tau_opt).u in *)
      let u0 =
        let tmp =
          (maybe_einsum ((List.hd_exn bck)._k, "ma"))
            (Some alpha_c, if linesearch_bs_avg then "d" else "m")
            "ma"
        in
        tmp +? (List.hd_exn tau_opt).u
      in
      (* In tau_opt x goes from 1 to T but u goes from 0 to T-1. bck goes from 0 to T-1.
        In tau_opt_trunc and bck_trunc x, u and bck_trunc goes from 1 to T-1 *)
      let tau_opt_trunc =
        let x_opt = List.map tau_opt ~f:(fun tau -> tau.x) in
        let u_opt = List.map tau_opt ~f:(fun tau -> tau.u) in
        let x_opt_trunc = List.drop_last_exn x_opt in
        let u_opt_trunc = List.tl_exn u_opt in
        List.map2_exn x_opt_trunc u_opt_trunc ~f:(fun x u -> Solution.{ x; u })
      in
      let bck_trunc = List.tl_exn bck in
      let bck_tau = List.map2_exn bck_trunc tau_opt_trunc ~f:(fun b tau -> b, tau) in
      let x_f, u_f, solution =
        List.foldi bck_tau ~init:(x0, u0, []) ~f:(fun i (x_prev, u_prev, accu) (b, tau) ->
          cleanup ();
          (* calculate x_t and u_t and fold into the state update *)
          let x_new =
            f_theta ~i ~x:(Option.value_exn x_prev) ~u:(Option.value_exn u_prev)
          in
          let u_new =
            let x_opt = tau.x in
            _u
              ~tangent:false
              ~batch_const
              ~_k:b._k
              ~_K:b._K
              ~x:(Some x_new -? x_opt)
              ~u:tau.u
              ~alpha
          in
          Some x_new, u_new, Solution.{ u = u_prev; x = Some x_new } :: accu)
      in
      (* append x_T and u_T-1 at the back *)
      let tau_curr =
        List.rev solution
        @ [ Solution.
              { u = u_f
              ; x =
                  Some
                    (f_theta
                       ~i:(List.length solution)
                       ~x:(Option.value_exn x_f)
                       ~u:(Option.value_exn u_f))
              }
          ]
      in
      let cost_curr = cost_func tau_curr in
      cleanup ();
      let lower_than_init =
        let cost_change = Tensor.(cost_curr - cost_init) in
        let lower_than_init_bool =
          match _dC1, _dC2 with
          | None, None -> Tensor.(lt cost_change (Scalar.f 0.))
          | Some _dC1, Some _dC2 ->
            (* TODO: how to regularize Quu in batch *)
            let _dV_f =
              let _dV = Maths.(alpha_c * _dC1) + (f 0.5 * alpha_c * alpha_c * _dC2) in
              if linesearch_bs_avg
              then _dV |> Maths.mean ~keepdim:false |> Maths.to_tensor
              else _dV |> Maths.to_tensor
            in
            if linesearch_bs_avg
            then (
              let criterion =
                Float.(
                  Tensor.(to_float0_exn (mean cost_change))
                  <= neg beta * Tensor.to_float0_exn _dV_f)
              in
              Tensor.(f (if criterion then 1. else 0.) * ones_like cost_curr))
            else Tensor.(le_tensor cost_change (neg (f beta * _dV_f)))
          | _ -> failwith "only dC1/dC2 defined (but not both)"
        in
        Tensor.to_kind lower_than_init_bool ~kind:(Tensor.kind cost_curr)
      in
      (* check if any alpha value is smaller than 1e-10. *)
      let alpha_not_converged =
        let bool_tensor = Tensor.lt alpha (Scalar.f 1e-10) in
        let any_less_than_one = Tensor.any bool_tensor in
        Tensor.to_int0_exn any_less_than_one
      in
      if Int.(alpha_not_converged = 1) then failwith "linesearch did not converge";
      let all_trials_converged =
        if Float.(Tensor.(to_float0_exn (mean lower_than_init)) = 1.) then true else false
      in
      let stop = (not linesearch) || all_trials_converged in
      let new_alpha =
        let _scale = Tensor.(lower_than_init + (f gamma * (f 1. - lower_than_init))) in
        Tensor.(_scale * alpha)
      in
      fwd_loop ~stop ~i:Int.(i + 1) ~alpha:new_alpha ~tau_prev:tau_curr)
  in
  (* start with alpha set to 1 *)
  let alpha =
    if linesearch_bs_avg
    then Tensor.ones ~device:(Tensor.device cost_init) ~kind:(Tensor.kind cost_init) [ 1 ]
    else Tensor.ones_like cost_init
  in
  fwd_loop ~stop:false ~i:0 ~alpha ~tau_prev:tau_opt

let ilqr_loop
      ~linesearch
      ~linesearch_bs_avg
      ~expected_reduction
      ~batch_const
      ~gamma
      ~conv_threshold
      ~max_iter
      ~params_func
      ~cost_func
      ~f_theta
      ~cost_init
      ~(tau_init : any t option Solution.p list)
  =
  let rec loop i tau_prev cost_prev =
    let p_curr = params_func tau_prev in
    let common_info =
      List.map p_curr.Params.params ~f:(fun p -> p.common) |> backward_common ~batch_const
    in
    cleanup ();
    let bck, _dC1, _dC2 =
      backward ~ilqr_expected_reduction:expected_reduction ~batch_const common_info p_curr
    in
    let tau_curr =
      forward
        ~batch_const
        ~linesearch
        ~linesearch_bs_avg
        ~gamma
        ~cost_func
        ~f_theta
        ~p:p_curr
        ~tau_opt:tau_prev
        ~_dC1
        ~_dC2
        bck
    in
    let cost_curr = cost_func tau_curr |> Tensor.mean |> Tensor.to_float0_exn in
    let pct_change = Float.(abs (cost_curr - cost_prev) / cost_prev) in
    let stop = Float.(pct_change < conv_threshold) || i = max_iter in
    cleanup ();
    if stop
    then (
      print [%message "no. of iterations of ilqr:" (i : int)];
      tau_curr, Some common_info, Some bck)
    else loop Int.(i + 1) tau_curr cost_curr
  in
  loop 0 tau_init cost_init

(* when batch_const is true, _Fx_prods, _Fu_prods, _Cxx, _Cxu, _Cuu has no leading batch dimension and special care needs to be taken to deal with these *)
let _isolve
      ?(linesearch = true)
      ?(linesearch_bs_avg = true)
      ?(expected_reduction = false)
      ?(batch_const = false)
      ~gamma
      ~f_theta
      ~cost_func
      ~params_func
      ~conv_threshold
      ~tau_init
      max_iter
  =
  (* [expected_reduction] only true iff using linesearch. *)
  if not linesearch then assert (not expected_reduction);
  (* step 1: init params and cost *)
  let cost_init = cost_func tau_init |> Tensor.mean |> Tensor.to_float0_exn in
  (*step 2: loop to find the best controls and states *)
  let tau_best, _, info_best =
    ilqr_loop
      ~linesearch
      ~linesearch_bs_avg
      ~expected_reduction
      ~batch_const
      ~gamma
      ~conv_threshold
      ~max_iter
      ~params_func:(params_func ~no_tangents:true)
      ~cost_func
      ~f_theta
      ~tau_init
      ~cost_init
  in
  cleanup ();
  (* List.map tau_best ~f:(fun tau ->
    Lqr.Solution.{ x = Option.value_exn tau.x; u = Option.value_exn tau.u }), Option.value_exn info_best *)
  (* step 3: final lqr pass with modified cost parameters *)
  let params_func_final =
    let params_func_best = params_func ~no_tangents:false tau_best in
    let tau_best_extended = extend_tau_list ~x0:params_func_best.x0 tau_best in
    let params_tau_list =
      List.map2_exn params_func_best.params tau_best_extended ~f:(fun p tau -> p, tau)
    in
    let params =
      List.mapi params_tau_list ~f:(fun i (p, tau) ->
        let _cx_new =
          p._cx
          -? (maybe_einsum
                (tau.x, "ma")
                (p.common._Cxx, if batch_const then "ab" else "mab")
                "mb"
              +? maybe_einsum
                   (tau.u, "ma")
                   (p.common._Cxu, if batch_const then "ba" else "mba")
                   "mb")
        in
        let _cu_new =
          p._cu
          -? (maybe_einsum
                (tau.x, "ma")
                (p.common._Cxu, if batch_const then "ab" else "mab")
                "mb"
              +? maybe_einsum
                   (tau.u, "ma")
                   (p.common._Cuu, if batch_const then "ab" else "mab")
                   "mb")
        in
        let _f_new =
          let next =
            if Int.(i = List.length params_tau_list - 1)
            then None
            else Some (f_theta ~i ~x:(Option.value_exn tau.x) ~u:(Option.value_exn tau.u))
          in
          next -? ((p.common._Fx_prod2 *? tau.x) +? (p.common._Fu_prod2 *? tau.u))
        in
        Lqr.{ common = p.common; _f = _f_new; _cx = _cx_new; _cu = _cu_new })
    in
    Lqr.Params.{ x0 = params_func_best.x0; params }
  in
  let tau_final, info_final =
    _solve ~ilqr_expected_reduction:false ~batch_const params_func_final
  in
  tau_final, info_final
