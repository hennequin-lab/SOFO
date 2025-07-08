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

(* TODO: need Guillaume to proof-read this *)
(* calculate the new u w.r.t. the difference between x_t and x_opt_t *)
let forward
      ~batch_const
      ~gamma
      ~cost_func
      ~f_theta
      ~(p : (any t option, (any t, any t -> any t) momentary_params list) Params.p)
      ~(tau_opt : any t option Solution.p list)
      ~(bck : backward_info list)
      ~conv_threshold
      ~max_iter
  =
  let rec fwd_loop ~stop ~i ~alpha ~tau_prev ~cost_prev =
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
        List.fold2_exn bck tau_opt ~init:(x0, u0, []) ~f:(fun (x, u, accu) b tau ->
          cleanup ();
          (* fold into the state update *)
          let x_new = f_theta ~x:(Option.value_exn x) ~u:(Option.value_exn u) in
          let u_new =
            let x_opt = tau.x in
            _u
              ~tangent:false
              ~batch_const
              ~_k:b._k
              ~_K:b._K
              ~x:(Some x_new -? x_opt)
              ~alpha
          in
          Some x_new, u_new, Solution.{ u; x = Some x_new } :: accu)
      in
      let alpha = gamma *. alpha in
      let tau_curr = List.rev solution in
      let cost_curr = cost_func ~batch_const tau_curr p in
      let pct_change = Float.(abs (cost_curr - cost_prev) / cost_prev) in
      if Float.(is_nan cost_curr)
      then failwith "current cost value is nan"
      else (
        cleanup ();
        let stop = Float.(pct_change < conv_threshold) || i = max_iter in
        fwd_loop ~stop ~i:Int.(i + 1) ~alpha ~tau_prev:tau_curr ~cost_prev:cost_curr))
  in
  let alpha = 1. in
  let cost_init = cost_func ~batch_const tau_opt p in
  fwd_loop ~stop:false ~i:0 ~alpha ~tau_prev:tau_opt ~cost_prev:cost_init

let ilqr_loop
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
  let rec ilqr_rec ~stop ~i ~tau_prev ~cost_prev ~common_info_prev ~info_prev =
    if stop
    then tau_prev, common_info_prev, info_prev
    else (
      let p_curr : (any t option, (any t, any t -> any t) momentary_params list) Params.p =
        params_func tau_prev
      in
      (* batch const is false since fx and fu now has batch dimension in front *)
      let common_info =
        backward_common
          ~batch_const
          (List.map p_curr.Params.params ~f:(fun x -> x.common))
      in
      cleanup ();
      let bck = backward ~batch_const common_info p_curr in
      let tau_curr =
        forward
          ~batch_const
          ~gamma
          ~cost_func
          ~f_theta
          ~p:p_curr
          ~tau_opt:tau_prev
          ~bck
          ~conv_threshold
          ~max_iter
      in
      let cost_curr = cost_func ~batch_const tau_curr p_curr in
      let pct_change = Float.(abs (cost_curr - cost_prev) / cost_prev) in
      let stop = Float.(pct_change < conv_threshold) || i = max_iter in
      cleanup ();
      ilqr_rec
        ~stop
        ~i:Int.(i + 1)
        ~tau_prev:tau_curr
        ~cost_prev:cost_curr
        ~common_info_prev:(Some common_info)
        ~info_prev:(Some bck))
  in
  ilqr_rec
    ~stop:false
    ~i:0
    ~tau_prev:tau_init
    ~cost_prev:cost_init
    ~common_info_prev:None
    ~info_prev:None

let rollout
      ~u_init
      ~(p_init : (any t option, (any t, any t -> any t) momentary_params list) Params.p)
      ~f_theta
  =
  (* x0 is 0; u goes from 0 to T-1, x goes from 1 to T *)
  let _, solution =
    List.fold u_init ~init:(p_init.x0, []) ~f:(fun (x, accu) u ->
      cleanup ();
      (* fold into the state update *)
      let x_new = f_theta ~x:(Option.value_exn x) ~u:(Option.value_exn u) in
      Some x_new, Solution.{ u; x = Some x_new } :: accu)
  in
  List.rev solution

(* when batch_const is true, _Fx_prods, _Fu_prods, _Cxx, _Cxu, _Cuu has no leading batch dimension and special care needs to be taken to deal with these *)
let _isolve
      ?(batch_const = false)
      ~gamma
      ~f_theta
      ~cost_func
      ~params_func
      ~conv_threshold
      ~tau_init
      ~max_iter
  =
  (* step 1: init params and cost *)
  let p_init = params_func tau_init in
  let cost_init = cost_func ~batch_const tau_init p_init in
  (*step 2: loop to find the best controls and states *)
  let tau_final, _, info_final =
    ilqr_loop
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
