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

let ( *? ) f v =
  match f, v with
  | Some f, Some v -> Some (f v)
  | _ -> None

let maybe_scalar_mul a s =
  match a with
  | None -> None
  | Some a -> Some Maths.(s $* a)

let maybe_einsum (a, opsA) (b, opsB) opsC =
  match a, b with
  | Some a, Some b -> Some (einsum [ a, opsA; b, opsB ] opsC)
  | _ -> None

let _u ~tangent ~batch_const ~_k ~_K ~x ~alpha =
  let tmp =
    match tangent, batch_const with
    | true, true -> maybe_einsum (x, "kma") (_K, "ba") "kmb"
    | true, false -> maybe_einsum (x, "kma") (_K, "mba") "kmb"
    | false, true -> maybe_einsum (x, "ma") (_K, "ba") "mb"
    | false, false -> maybe_einsum (x, "ma") (_K, "mba") "mb"
  in
  maybe_scalar_mul _k alpha +? tmp

(* calculate the new u w.r.t. the difference between x_t and x_opt_t *)

let forward
      ~batch_const
      ~cost_func
      ~(p : (t option, (t, t -> t) momentary_params list) Params.p)
      ~(tau_opt : t option Solution.p list)
      ~(bck : backward_info list)
      ~conv_threshold
  =
  let rec fwd_loop ~stop ~alpha ~tau_prev ~cost_prev =
    if stop
    then tau_prev
    else (
      let x0 = p.x0 in
      let u0 =
        let _k0 =
          let tmp = List.hd_exn bck in
          tmp._k
        in
        maybe_scalar_mul _k0 alpha
      in
      let _, _, solution =
        let params_except_last = List.drop_last_exn p.params in
        let params_bck_list = List.map2_exn params_except_last bck ~f:(fun p b -> p, b) in
        List.fold2_exn
          params_bck_list
          tau_opt
          ~init:(x0, u0, [])
          ~f:(fun (x, u, accu) (p, b) tau ->
            cleanup ();
            (* fold into the state update *)
            let x_new =
              let _Fx_prod2, _Fu_prod2 = p.common._Fx_prod2, p.common._Fu_prod2 in
              p._f +? (_Fx_prod2 *? x) +? (_Fu_prod2 *? u)
            in
            let u_new =
              let x_opt = tau.x in
              _u ~tangent:false ~batch_const ~_k:b._k ~_K:b._K ~x:(x_new -? x_opt) ~alpha
            in
            x, u_new, Solution.{ u; x = x_new } :: accu)
      in
      let tau_curr = List.rev solution in
      let cost_curr = cost_func ~batch_const tau_curr p in
      let pct_change = Float.(abs (cost_curr - cost_prev) / cost_prev) in
      let stop = Float.(pct_change < conv_threshold) in
      fwd_loop ~stop ~alpha ~tau_prev:(Some tau_curr) ~cost_prev:cost_curr)
  in
  let alpha = 1. in
  fwd_loop ~stop:false ~alpha ~tau_prev:None ~cost_prev:0. |> Option.value_exn

let rec ilqr_loop
          ~batch_const
          ~laplace
          ~conv_threshold
          ~stop
          ~cost_prev
          ~params_func
          ~cost_func
          ~(tau_prev : t option Solution.p list)
          ~common_info_prev
  =
  if stop
  then tau_prev, common_info_prev
  else (
    let p_curr : (t option, (t, t -> t) momentary_params list) Params.p =
      params_func tau_prev
    in
    (* batch const is false since fx and fu now has batch dimension in front *)
    let common_info =
      backward_common
        ~batch_const
        ~laplace
        (List.map p_curr.Params.params ~f:(fun x -> x.common))
    in
    cleanup ();
    let bck = backward ~batch_const common_info p_curr in
    let tau_curr =
      forward ~batch_const ~cost_func ~p:p_curr ~tau_opt:tau_prev ~bck ~conv_threshold
    in
    let cost_curr = cost_func ~batch_const tau_curr p_curr in
    let pct_change = Float.(abs (cost_curr - cost_prev) / cost_prev) in
    let stop = Float.(pct_change < conv_threshold) in
    ilqr_loop
      ~batch_const
      ~laplace
      ~conv_threshold
      ~stop
      ~params_func
      ~cost_func
      ~tau_prev:tau_curr
      ~cost_prev:cost_curr
      ~common_info_prev:(Some common_info))

let rollout ~u_init ~(p_init : (t option, (t, t -> t) momentary_params list) Params.p) =
  (* x0 is 0; u goes from 0 to T-1, x goes from 1 to T *)
  let _, solution =
    let params_except_last = List.drop_last_exn p_init.params in
    List.fold2_exn
      params_except_last
      u_init
      ~init:(p_init.x0, [])
      ~f:(fun (x, accu) p u ->
        cleanup ();
        (* fold into the state update *)
        let x =
          let _Fx_prod2, _Fu_prod2 = p.common._Fx_prod2, p.common._Fu_prod2 in
          p._f +? (_Fx_prod2 *? x) +? (_Fu_prod2 *? u)
        in
        x, Solution.{ u; x } :: accu)
  in
  List.rev solution

(* when batch_const is true, _Fx_prods, _Fu_prods, _Cxx, _Cxu, _Cuu has no leading batch dimension and special care needs to be taken to deal with these *)
let _isolve
      ?(batch_const = false)
      ?(laplace = false)
      ~cost_func
      ~params_func
      (* ~u_init *)
      ~conv_threshold
      (* ~(p_init : (t option, (t, t -> t) momentary_params list) Params.p) *)
      ~tau_init
  =
  (* step 1: rollout u *)
  (* let tau_init = rollout ~u_init ~p_init in *)
  let p_init = params_func tau_init in
  let cost_init = cost_func ~batch_const tau_init p_init in
  (*step 2: loop to find the best controls and states *)
  let tau_final, common_info_final =
    ilqr_loop
      ~batch_const
      ~laplace
      ~conv_threshold
      ~stop:false
      ~params_func
      ~cost_func
      ~tau_prev:tau_init
      ~cost_prev:cost_init
      ~common_info_prev:None
  in
  (* step 3: calculate covariances if required *)
  if laplace
  then (
    let covariances =
      covariances
        ~batch_const
        ~common_info:(Option.value_exn common_info_final)
        (params_func tau_final)
    in
    tau_final, Some (List.map covariances ~f:(fun x -> Option.value_exn x)))
  else tau_final, None
