open Base
open Owl
open Torch
open Forward_torch
open Maths
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let cast = Dense.Matrix.Generic.cast_d2s
let base = Optimizer.Config.Base.default

let cifar =
  match Cmdargs.(get_string "-dataset") with
  | Some "mnist" -> false
  | Some "cifar" -> true
  | _ -> failwith "-dataset [mnist | cifar]"

let output_dim = 10
let batch_size = 256
let input_size = if cifar then Int.(32 * 32 * 3) else Int.(28 * 28)
let full_batch_size = 60_000
let layer_sizes = [ 128; output_dim ]
let num_epochs_to_run = 200
let max_iter = Int.(full_batch_size * num_epochs_to_run / batch_size)
let epoch_of t = Float.(of_int t * of_int batch_size / of_int full_batch_size)
let max_prune_iter = 50

(* remove p at each round *)
let p = 0.1

module MLP_Layer = struct
  type 'a t =
    { id : int
    ; w : 'a
    }
  [@@deriving prms]
end

module P = Prms.List (MLP_Layer.Make (Prms.Single))

(* neural network *)
module MLP = struct
  module P = P

  type input = Tensor.t

  let forward ~(theta : _ Maths.some P.t) ~(input : input) =
    let bs = Tensor.shape input |> List.hd_exn in
    List.foldi
      theta
      ~init:Maths.(any (of_tensor input))
      ~f:(fun i accu wb ->
        let open MLP_Layer in
        let accu =
          Maths.concat
            [ accu
            ; Maths.(
                any
                  (of_tensor (Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ])))
            ]
            ~dim:1
        in
        let pre_activation = Maths.(accu *@ wb.w) in
        if i = Int.(List.length layer_sizes - 1)
        then pre_activation
        else Maths.relu pre_activation)

  let f ~data:(input, labels) theta =
    let pred = forward ~theta ~input in
    let ell =
      Loss.cross_entropy ~output_dims:[ 1 ] ~labels:(Maths.of_tensor labels) pred
    in
    let ggn =
      Loss.cross_entropy_ggn
        ~output_dims:[ 1 ]
        (Maths.const pred)
        ~vtgt:(Maths.tangent_exn pred)
    in
    ell, ggn

  let f_adam ~data:(input, labels) theta =
    let pred = forward ~theta ~input in
    let ell =
      Loss.cross_entropy ~output_dims:[ 1 ] ~labels:(Maths.of_tensor labels) pred
    in
    ell

  let init () : P.param =
    let open MLP_Layer in
    List.mapi layer_sizes ~f:(fun i n_o ->
      let n_i = if i = 0 then input_size else List.nth_exn layer_sizes Int.(i - 1) in
      let w =
        Sofo.gaussian_tensor_normed
          ~kind:base.kind
          ~device:base.device
          ~sigma:1.
          [ Int.(n_i + 1); n_o ]
        |> Maths.of_tensor
        |> Prms.Single.free
      in
      { id = i; w })
end

(* -----------------------------------------
   -- Read in data. ------
   ----------------------------------------- *)

let train_set, test_set =
  if cifar
  then (
    (* train data has size [10000 x 32 x 32 x 3 ]*)
    let x_train, y_train =
      List.fold
        (List.init 5 ~f:(fun id -> id))
        ~init:None
        ~f:(fun accu i ->
          let x_train, _, y_train = Owl.Dataset.load_cifar_train_data Int.(i + 1) in
          match accu with
          | None -> Some (x_train, y_train)
          | Some (x_train_accu, y_train_accu) ->
            let new_x_train_accu = Arr.concatenate [| x_train_accu; x_train |] ~axis:0 in
            let new_y_train_accu = Arr.concatenate [| y_train_accu; y_train |] ~axis:0 in
            Some (new_x_train_accu, new_y_train_accu))
      |> Option.value_exn
    in
    let x_test, _, y_test = Owl.Dataset.load_cifar_test_data () in
    (* from torchvision *)
    let mu = cast (Owl.Arr.of_array [| 0.4914; 0.4822; 0.4465 |] [| 1; 1; 1; 3 |])
    and sigma = cast (Owl.Arr.of_array [| 0.2023; 0.1994; 0.2010 |] [| 1; 1; 1; 3 |]) in
    (* cifar10 from Owl dataset already scaled between 0 and 1! *)
    let x_train = Owl.Arr.((x_train - mu) / sigma) in
    let x_test = Owl.Arr.((x_test - mu) / sigma) in
    (x_train, y_train), (x_test, y_test))
  else (
    let dataset_mnist typ =
      let suffix =
        match typ with
        | `train -> "train"
        | `test -> "test"
      in
      let x = Owl.Arr.load_npy ("_data/x_" ^ suffix ^ ".npy") in
      let y =
        Owl.Arr.load_npy ("_data/t_" ^ suffix ^ ".npy") |> Owl.Arr.one_hot output_dim
      in
      let mu = 0.13062754273414612
      and sigma = 0.30810779333114624 in
      let x = Owl.Arr.(((x /$ 255.) -$ mu) /$ sigma) in
      cast x, cast y
    in
    dataset_mnist `train, dataset_mnist `test)

let sample_data (set_x, set_y) =
  let a = (Arr.shape set_x).(0) in
  fun batch_size ->
    if batch_size < 0
    then
      ( Tensor.of_bigarray ~device:base.device set_x
        |> Tensor.reshape ~shape:[ -1; input_size ]
      , Tensor.of_bigarray ~device:base.device set_y )
    else (
      let ids = List.init batch_size ~f:(fun _ -> Random.int a) in
      let xs = Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_x) in
      let ys = Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_y) in
      Tensor.reshape ~shape:[ batch_size; input_size ] xs, ys)

let test_eval ~train_data theta =
  let x, y =
    match train_data with
    | None -> sample_data test_set (-1)
    | Some train_data -> train_data
  in
  let logits = MLP.forward ~theta ~input:x |> Maths.to_tensor in
  let _, max_y = Tensor.max_dim ~keepdim:false ~dim:1 y in
  let _, max_ypred = Tensor.max_dim ~keepdim:false ~dim:1 logits in
  Tensor.eq_tensor max_y max_ypred
  |> Tensor.to_dtype ~dtype:base.kind ~non_blocking:false ~copy:false
  |> Tensor.mean
  |> Tensor.to_float0_exn

(* -----------------------------------------
   -- Pruning functions
   ----------------------------------------- *)
module Prune = struct
  open Torch

  (* self define mask *)
  let mask_p ~p theta =
    P.map theta ~f:(fun x ->
      (* value 1 with probability [p] *)
      let x_t = to_tensor x in
      let mask = Torch.Tensor.bernoulli_float_ x_t ~p in
      of_tensor mask)

  (* Flatten a P.t into a single vector *)
  let flatten (x : _ some P.t) =
    P.fold x ~init:[] ~f:(fun accu (x, _) ->
      let x_reshaped = Torch.Tensor.reshape (to_tensor x) ~shape:[ -1; 1 ] in
      x_reshaped :: accu)
    |> Torch.Tensor.concat ~dim:0

  let count_mask mask =
    Torch.Tensor.masked_select mask ~mask
    |> Torch.Tensor.reshape ~shape:[ -1; 1 ]
    |> Torch.Tensor.shape
    |> List.hd_exn

  let count_remaining mask =
    let mask_flattened = flatten mask in
    (* masked_select to pick only entries where prev_mask == 1 *)
    count_mask mask_flattened

  let n_params x =
    let x_shape = Maths.shape x in
    List.fold x_shape ~init:1 ~f:(fun accu x -> Int.(accu * x))

  (* Select only entries that are 1 in prev mask, or pass through unchanged *)
  let surviving_values ~mask_prev flat_values =
    match mask_prev with
    | None -> flat_values
    | Some prev ->
      let prev_flat = flatten prev in
      Tensor.masked_select flat_values ~mask:prev_flat |> Tensor.reshape ~shape:[ -1; 1 ]

  (* Find the pth smallest absolute value in a Tensor *)
  let threshold_of_values ~p surviving_values =
    let v = Tensor.reshape surviving_values ~shape:[ -1; 1 ] in
    let n = List.hd_exn (Tensor.shape v) in
    let sorted, _ = Tensor.sort (Tensor.abs v) ~dim:0 ~descending:false in
    let idx = Int.(clamp_exn (of_float Float.(p *. of_int n))) ~min:0 ~max:Int.(n - 1) in
    Tensor.get_float2 sorted idx 0

  (* Build a per-tensor mask given a global threshold *)
  let mask_from_threshold theta threshold =
    P.map theta ~f:(fun w ->
      Tensor.abs (to_tensor w) |> fun a -> Tensor.gt a (Scalar.f threshold) |> of_tensor)

  (* Combine with previous mask *)
  let combine mask_prev new_mask =
    match mask_prev with
    | None -> new_mask
    | Some prev ->
      P.map2 prev new_mask ~f:(fun m_prev m_new ->
        Tensor.logical_and (to_tensor m_prev) (to_tensor m_new) |> of_tensor)
end

let pruning_mask_layerwise ?(p_surviving_min = 0.01) ~p ~mask_prev (theta : _ some P.t)
  : const P.t
  =
  let open Torch in
  let open Prune in
  let prune_tensor theta mask_prev_opt =
    let theta_t = to_tensor theta in
    (* Current surviving values *)
    let curr_values =
      match mask_prev_opt with
      | None -> Tensor.reshape theta_t ~shape:[ -1; 1 ]
      | Some m_prev ->
        Tensor.masked_select theta_t ~mask:(to_tensor m_prev)
        |> Tensor.reshape ~shape:[ -1; 1 ]
    in
    let threshold = threshold_of_values ~p curr_values in
    let new_mask_t = Tensor.gt (Tensor.abs theta_t) (Scalar.f threshold) in
    let new_mask = of_tensor new_mask_t in
    let n_new = count_mask new_mask_t in
    let min_required = Int.(of_float Float.(p_surviving_min * of_int (n_params theta))) in
    match mask_prev_opt with
    | None -> new_mask
    | Some prev ->
      if n_new < min_required
      then prev
      else Tensor.logical_and (to_tensor prev) new_mask_t |> of_tensor
  in
  match mask_prev with
  | None -> P.map theta ~f:(fun t -> prune_tensor t None)
  | Some prev -> P.map2 theta prev ~f:(fun t m -> prune_tensor t (Some m))

let pruning_mask_global ?(n_surviving_min = 200) ~p ~mask_prev (theta : _ some P.t)
  : const P.t
  =
  let open Prune in
  (* Global surviving values *)
  let surviving = surviving_values ~mask_prev (flatten theta) in
  (* Threshold for global pruning *)
  let threshold = threshold_of_values ~p surviving in
  (* Build new masks using this threshold *)
  let new_mask = mask_from_threshold theta threshold in
  let n_new = count_remaining new_mask in
  (* only use new mask if n_new is larger than n_surviving_min *)
  if n_new < n_surviving_min
  then (
    match mask_prev with
    | Some prev -> prev
    | None -> P.map theta ~f:Maths.ones_like)
  else combine mask_prev new_mask

(* -----------------------------------------
   -- Optimization with SOFO    ------
   ----------------------------------------- *)

module O = Optimizer.SOFO (MLP.P)

let config =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some 1e-3
    ; n_tangents = 128
    ; damping = `relative_from_top 1e-3
    }

let init_params = MLP.init ()

let _ =
  O.P.C.save (MLP.P.value init_params) ~kind:base.ba_kind ~out:(in_dir "init_params")

let train ~init_params ~mask ~append =
  let rec loop ~t ~state running_avg =
    Stdlib.Gc.major ();
    let data = sample_data train_set batch_size in
    (* CHECKED masking is correct. *)
    let theta, tangents = O.prepare ?mask ~config state in
    let loss, ggn = MLP.f ~data theta in
    let new_state = O.step ~config ~info:{ loss; ggn; tangents; mask } state in
    let loss = Maths.to_float_exn (Maths.const loss) in
    let running_avg =
      let loss_avg =
        match running_avg with
        | [] -> loss
        | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
      in
      if t % 10 = 0
      then (
        O.P.C.save
          (MLP.P.value (O.params state))
          ~kind:base.ba_kind
          ~out:(in_dir ("sofo_params_" ^ append));
        (* save loss & acc *)
        let e = epoch_of t in
        let test_acc =
          test_eval ~train_data:None MLP.P.(const (value (O.params new_state)))
        in
        let train_acc =
          test_eval ~train_data:(Some data) MLP.P.(const (value (O.params new_state)))
        in
        print [%message (e : float) (loss_avg : float)];
        Owl.Mat.(
          save_txt
            ~append:true
            ~out:(in_dir ("loss_" ^ append))
            (of_array [| Float.of_int t; loss_avg; test_acc; train_acc |] 1 4)));
      []
    in
    if t < max_iter
    then loop ~t:Int.(t + 1) ~state:new_state (loss :: running_avg)
    else new_state
  in
  loop ~t:0 ~state:(O.init init_params) []

(* -----------------------------------------
   -- Optimization with Adam    ------
   ----------------------------------------- *)

(* module O = Optimizer.Adam (MLP.P)

let config =
  Optimizer.Config.Adam.
    { base
    ; beta_1 = 0.9
    ; beta_2 = 0.99
    ; eps = 1e-4
    ; learning_rate = Some 0.01
    ; weight_decay = None
    ; debias = false
    }

let init_params = MLP.init ()

let _ =
  O.P.C.save (MLP.P.value init_params) ~kind:base.ba_kind ~out:(in_dir "init_params")

let train ~init_params ~mask ~append =
  let rec loop ~t ~state running_avg =
    Stdlib.Gc.major ();
    let data = sample_data train_set batch_size in
    let theta = O.params state in
    let theta_ = O.P.value theta in
    (* mask theta_ with current mask. CHECKED it is correct. *)
    let theta_ =
      match mask with
      | None -> theta_
      | Some mask -> O.P.map2 theta_ mask ~f:(fun x m -> Maths.C.(x * m))
    in
    let theta_dual =
      O.P.map theta_ ~f:(fun x ->
        let x =
          x |> Maths.to_tensor |> Tensor.copy |> Tensor.to_device ~device:base.device
        in
        let x = Tensor.set_requires_grad x ~r:true in
        Tensor.zero_grad x;
        Maths.of_tensor x)
    in
    let loss, true_g =
      let loss = MLP.f_adam ~data (P.map theta_dual ~f:Maths.any) in
      let loss = Maths.to_tensor loss in
      Tensor.backward loss;
      ( Tensor.to_float0_exn loss
      , O.P.map2 theta (O.P.to_tensor theta_dual) ~f:(fun tagged p ->
          match tagged with
          | Prms.Pinned _ -> Maths.(f 0.)
          | _ -> Maths.of_tensor (Tensor.grad p)) )
    in
    let new_state = O.step ~config ~info:{ g = true_g; mask } state in
    let running_avg =
      let loss_avg =
        match running_avg with
        | [] -> loss
        | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
      in
      if t % 10 = 0
      then (
        O.P.C.save
          (MLP.P.value (O.params state))
          ~kind:base.ba_kind
          ~out:(in_dir ("sofo_params_" ^ append));
        (* save loss & acc *)
        let e = epoch_of t in
        let test_acc =
          test_eval ~train_data:None MLP.P.(const (value (O.params new_state)))
        in
        let train_acc =
          test_eval ~train_data:(Some data) MLP.P.(const (value (O.params new_state)))
        in
        print [%message (e : float) (loss_avg : float)];
        Owl.Mat.(
          save_txt
            ~append:true
            ~out:(in_dir ("loss_" ^ append))
            (of_array [| Float.of_int t; loss_avg; test_acc; train_acc |] 1 4)));
      []
    in
    if t < max_iter
    then loop ~t:Int.(t + 1) ~state:new_state (loss :: running_avg)
    else new_state
  in
  loop ~t:0 ~state:(O.init init_params) [] *)

(* Start training and pruning loop *)
let train_prune ~p =
  (* first train the network with no mask *)
  let state_0 = train ~init_params ~mask:None ~append:(Int.to_string 0) in
  (* iteratively prune, then train *)
  let rec pruning_loop ~prune_iter ~state ~mask =
    let mask_new = pruning_mask_global ~p ~mask_prev:mask (O.P.value (O.params state)) in
    (* let mask_new =
      pruning_mask_layerwise ~p ~mask_prev:mask (O.P.value (O.params state))
    in *)
    let mask_to_save =
      O.P.map mask_new ~f:(fun x ->
        let x_f = Tensor.to_type (to_tensor x) ~type_:base.kind in
        of_tensor x_f)
    in
    O.P.C.save
      mask_to_save
      ~kind:base.ba_kind
      ~out:(in_dir (Printf.sprintf "mask_%d" prune_iter));
    let state_new =
      train ~init_params ~mask:(Some mask_new) ~append:(Int.to_string prune_iter)
    in
    if prune_iter < max_prune_iter
    then
      pruning_loop ~prune_iter:Int.(prune_iter + 1) ~state:state_new ~mask:(Some mask_new)
    else state_new
  in
  pruning_loop ~prune_iter:1 ~state:state_0 ~mask:None

let _ = train_prune ~p

(* Test performance if using a particular mask, but initialise parameters randomly *)
let test_performance ~prune_iter =
  let mask =
    MLP.P.C.load ~device:base.device (in_dir (Printf.sprintf "mask_%d" prune_iter))
  in
  let re_init_params = MLP.init () in
  let state_new =
    train
      ~init_params:re_init_params
      ~mask:(Some mask)
      ~append:(Printf.sprintf "retrain_%s_%d" "mask" prune_iter)
  in
  state_new
