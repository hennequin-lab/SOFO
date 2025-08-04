open Base
open Forward_torch
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

let maybe_scalar_mul a s =
  match a with
  | None -> None
  | Some a -> Some Maths.(s $* a)

let maybe_einsum (a, opsA) (b, opsB) opsC =
  match a, b with
  | Some a, Some b -> Some (einsum [ a, opsA; b, opsB ] opsC)
  | _ -> None

let _u ~tangent ~batch_const ~_k ~_K ~x ~u ~alpha =
  let tmp =
    match tangent, batch_const with
    | true, true -> maybe_einsum (x, "kma") (_K, "ba") "kmb"
    | true, false -> maybe_einsum (x, "kma") (_K, "mba") "kmb"
    | false, true -> maybe_einsum (x, "ma") (_K, "ba") "mb"
    | false, false -> maybe_einsum (x, "ma") (_K, "mba") "mb"
  in
  maybe_scalar_mul _k alpha +? tmp +? u

(* calculate the new u w.r.t. the difference between x_t and x_opt_t *)
let forward
      ?(beta = 0.1)
      ~batch_const
      ~linesearch
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
      let u0 = maybe_scalar_mul (List.hd_exn bck)._k alpha +? (List.hd_exn tau_opt).u in
      (* in tau_opt x goes from 1 to T but u goes from 0 to T-1. bck goes from 0 to T-1. *)
      (* in tau_opt_trunc and bck_trunc x,u and bck_trunc goes from 1 to T-1 *)
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
        let cost_change = Float.(cost_curr - cost_init) in
        match _dC1, _dC2 with
        | None, None -> Float.(cost_change <= 0.)
        | Some _dC1, Some _dC2 ->
          (* TODO: how to regularize Quu in batch *)
          (* let _dV = Maths.((f alpha * _dC1) + (f Float.(0.5 * alpha * alpha) * _dC2)) in *)
          let _dV = Maths.((f alpha * _dC2) + (f Float.(0.5 * alpha * alpha) * _dC1)) in
          let _dV_f =
            _dV |> Maths.mean ~keepdim:false |> Maths.const |> Maths.to_float_exn
          in
          (* print [%message (cost_change : float) (_dV_f : float)]; *)
          Float.(cost_change <= neg beta * _dV_f)
        | _ -> failwith "only dC1/dC2 defined (but not both)"
      in
      if Float.(alpha < 1e-10) then failwith "linesearch did not converge";
      let stop = (not linesearch) || lower_than_init in
      fwd_loop ~stop ~i:Int.(i + 1) ~alpha:(alpha *. gamma) ~tau_prev:tau_curr)
  in
  (* start with alpha set to 1 *)
  let alpha = 1. in
  fwd_loop ~stop:false ~i:0 ~alpha ~tau_prev:tau_opt

let ilqr_loop
      ~linesearch
      ~ex_reduction
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
      backward ~ilqr_ex_reduction:ex_reduction ~batch_const common_info p_curr
    in
    let tau_curr =
      forward
        ~batch_const
        ~linesearch
        ~gamma
        ~cost_func
        ~f_theta
        ~p:p_curr
        ~tau_opt:tau_prev
        ~_dC1
        ~_dC2
        bck
    in
    let cost_curr = cost_func tau_curr in
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
      ?(ex_reduction = false)
      ?(batch_const = false)
      ~gamma
      ~f_theta
      ~cost_func
      ~params_func
      ~conv_threshold
      ~tau_init
      max_iter
  =
  (* [ex_reduction] only true iff using linesearch. *)
  if not linesearch then assert (not ex_reduction);
  (* step 1: init params and cost *)
  let cost_init = cost_func tau_init in
  (*step 2: loop to find the best controls and states *)
  let tau_final, _, info_final =
    ilqr_loop
      ~linesearch
      ~ex_reduction
      ~batch_const
      ~gamma
      ~conv_threshold
      ~max_iter
      ~params_func
      ~cost_func
      ~f_theta
      ~tau_init
      ~cost_init
  in
  cleanup ();
  tau_final, info_final
