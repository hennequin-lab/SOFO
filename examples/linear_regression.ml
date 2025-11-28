(* Example a simple linear regression task to learn W where y = W x. *)
open Base
open Forward_torch
open Maths
open Sofo

let _ =
  Random.init 1985;
  (* Random.self_init (); *)
  Owl_stats_prng.init 1985;
  Torch_core.Wrapper.manual_seed 1985

let print s = Stdio.print_endline (Sexp.to_string_hum s)
let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)
let d_in, d_out = 100, 3

module P = Prms.Single 

let theta_init () : const P.t =
  let sigma = Float.(1. / sqrt (of_int d_in)) in
  sigma $* randn ~kind:base.kind ~device:base.device [ d_in; d_out ]

let theta_0 = theta_init ()

(* save params *)
let _ = P.C.save theta_0 ~kind:base.ba_kind ~out:(in_dir "theta_0")

module Model = struct
  module P = P

  let f ~(theta : _ some P.t) input = Maths.(input *@ theta)
  let init : P.param = P.free theta_0
end

(* -----------------------------------------
   -- Utility functions
   ----------------------------------------- *)
(* self define mask *)
let mask_p ~p theta =
  P.map theta ~f:(fun x ->
    (* value 1 with probability [p] *)
    let x_t = to_tensor x in
    let mask = Torch.Tensor.bernoulli_float_ x_t ~p in
    of_tensor mask)

let flatten (x : _ some P.t) =
  P.fold x ~init:[] ~f:(fun accu (x, _) ->
    let x_reshaped = Torch.Tensor.reshape (to_tensor x) ~shape:[ -1; 1 ] in
    x_reshaped :: accu)
  |> Torch.Tensor.concat ~dim:0

let count_remaining mask =
  let mask_flattened = flatten mask in
  (* masked_select to pick only entries where prev_mask == 1 *)
  Torch.Tensor.masked_select mask_flattened ~mask:mask_flattened
  |> Torch.Tensor.reshape ~shape:[ -1; 1 ]
  |> Torch.Tensor.shape
  |> List.hd_exn

let pruning_mask ?(n_surviving_min = 10) ~p ~mask_prev (theta : _ some P.t) : const P.t =
  let open Torch in
  let theta_flattened_abs = P.map ~f:Tensor.abs (flatten theta) in
  (* If previous mask present, build a flattened mask tensor and select only surviving values *)
  let surviving_values =
    match mask_prev with
    | None -> theta_flattened_abs
    | Some prev_mask ->
      (* flatten and concat prev_mask to a single mask tensor *)
      let prev_mask_t = flatten prev_mask in
      (* masked_select to pick only entries where prev_mask == 1 *)
      Tensor.masked_select theta_flattened_abs ~mask:prev_mask_t
      |> Tensor.reshape ~shape:[ -1; 1 ]
  in
  (* Number of surviving parameters *)
  let n_surviving = Tensor.shape surviving_values |> List.hd_exn in
  let theta_sorted, _ = Tensor.sort surviving_values ~dim:0 ~descending:false in
  let idx =
    let idx = Int.(of_float Float.(p * of_int n_surviving)) in
    (* clamp index into valid range [0, n_surviving - 1] *)
    if idx >= n_surviving then Int.(n_surviving - 1) else idx
  in
  let threshold = Tensor.get_float2 theta_sorted idx 0 in
  (* Build new mask: keep weights if |weights| > threshold *)
  let new_mask =
    P.map theta ~f:(fun w ->
      (* mapped to 1 if greater than threshold *)
      let m = Tensor.gt (Tensor.abs (to_tensor w)) (Scalar.f threshold) in
      of_tensor m)
  in
  let n_surviving_new = count_remaining new_mask in
  print [%message (n_surviving : int)];
  if n_surviving_new < n_surviving_min
  then (
    (* do not prune anymore *)
    match mask_prev with
    | Some prev -> prev
    | None -> P.map theta ~f:Maths.ones_like)
  else (
    (* combine with previous mask if exists: final = prev AND new_mask *)
    match mask_prev with
    | None -> new_mask
    | Some prev ->
      P.map2 prev new_mask ~f:(fun m_prev m_new ->
        Tensor.logical_and (to_tensor m_prev) (to_tensor m_new) |> of_tensor))

module O = Optimizer.SOFO (Model.P)

let config =
  Optimizer.Config.SOFO.
    { base; learning_rate = Some 0.1; n_tangents = 10; damping = `none }

(* -----------------------------------------
   -- Generate linear regression data.
   ----------------------------------------- *)

let data_minibatch =
  let teacher = theta_init () in
  let input_cov_sqrt =
    let u, _ = C.qr (randn ~device:base.device [ d_in; d_in ]) in
    let lambda =
      Array.init d_in ~f:(fun i -> Float.(1. / (1. + square (of_int Int.(i + 1)))))
      |> of_array ~device:base.device ~shape:[ d_in; 1 ]
      |> fun x -> C.(x / mean x)
    in
    C.(sqrt lambda * u)
  in
  fun bs ->
    let x = C.(randn ~device:base.device [ bs; d_in ] *@ input_cov_sqrt) in
    let y = Model.f ~theta:teacher x in
    x, y

(* -----------------------------------------
   -- Optimization loop       ------
   ----------------------------------------- *)

let batch_size = 512
let max_iter = 200
let max_prune_iter = 10

(* remove p at each round *)
let p = 0.2

let train ~mask ~prune_iter =
  let rec loop ~t ~state =
    Stdlib.Gc.major ();
    let x, y = data_minibatch batch_size in
    let theta, tangents = O.prepare ?mask ~config state in
    let y_pred = Model.f ~theta x in
    let loss = Loss.mse ~output_dims:[ 1 ] (y - y_pred) in
    let ggn = Loss.mse_ggn ~output_dims:[ 1 ] (const y) ~vtgt:(tangent_exn y_pred) in
    let new_state = O.step ~config ~info:{ loss; ggn; tangents; mask } state in
    if t % 100 = 0
    then (
      let loss = to_float_exn (const loss) in
      print [%message (t : int) (loss : float)];
      (* save params *)
      O.P.C.save
        (O.P.value (O.params new_state))
        ~kind:base.ba_kind
        ~out:(in_dir (Printf.sprintf "theta_final_%d" prune_iter));
      Owl.Mat.(
        save_txt
          ~append:true
          ~out:(in_dir (Printf.sprintf "loss_%d" prune_iter))
          (of_array [| Float.of_int t; loss |] 1 2)));
    if t < max_iter then loop ~t:Int.(t + 1) ~state:new_state else new_state
  in
  loop ~t:0 ~state:(O.init Model.init)

(* Start training and pruning loop *)
let traine_prune ~p =
  (* first train the network with no mask *)
  let state_0 = train ~mask:None ~prune_iter:0 in
  let rec pruning_loop ~prune_iter ~state ~mask =
    let mask_new = pruning_mask ~p ~mask_prev:mask (O.P.value (O.params state)) in
    let state_new = train ~mask:(Some mask_new) ~prune_iter in
    if prune_iter < max_prune_iter
    then
      pruning_loop ~prune_iter:Int.(prune_iter + 1) ~state:state_new ~mask:(Some mask_new)
    else state_new
  in
  pruning_loop ~prune_iter:1 ~state:state_0 ~mask:None

let _ = traine_prune ~p
