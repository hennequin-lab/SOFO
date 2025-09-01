(** Meta learning a kalman filter with a vanilla rnn. *)
open Base

open Utils
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

(* state dimension *)
let s = 1

module Data = Kalman_data.Make (Kalman_data.Default)

let tmax = 500
let batch_size = 256
let max_iter = 4000
(* let n_trials_simulation = 500 *)
(* let n_trials_baseline = 1000 *)

let _ =
  Random.init 1999;
  (* Random.self_init (); *)
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default

(* generate some random data to get baseline performance of kalman *)
(* let kalman_baseline =
   let open Data in
   Array.init 2 ~f:(fun _ ->
   let data = minibatch ~tmax n_trials_baseline in
   (* save [x; y; b; kalman_x]*)
   let _ =
   Mat.save_txt ~out:(in_dir "example_session") (to_save ~random:false data.(0))
   in
   mse ~filter_fun:(kalman_filter ~random:false) data)
   |> Stats.mean

   let kalman_random_baseline =
   let open Data in
   Array.init 2 ~f:(fun _ ->
   let data = minibatch ~tmax n_trials_baseline in
   (* save [x; y; b; kalman_random_x]*)
   let _ =
   Mat.save_txt ~out:(in_dir "example_session_random") (to_save ~random:true data.(0))
   in
   mse ~filter_fun:(kalman_filter ~random:true) data)
   |> Stats.mean *)

(* let kalman_baseline = 0.4896232
let kalman_random_baseline = 1.3612788 *)

module RNN_P = struct
  type 'a p =
    { c : 'a
    ; b : 'a
    ; o : 'a
    }
  [@@deriving prms]
end

module P = RNN_P.Make (Prms.Single)

let n = 100
let alpha = 0.25

(* neural network *)
module RNN = struct
  module P = P

  let phi x = Maths.relu x

  (* input is the noisy observation and z is the internal state *)
  let forward ~(theta : _ Maths.some P.t) ~input z =
    let bs = Tensor.shape input |> List.hd_exn in
    let input =
      Maths.C.concat
        [ Maths.of_tensor input
        ; Maths.of_tensor (Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ])
        ]
        ~dim:1
    in
    match z with
    | Some z ->
      let leak = Maths.(Float.(1. - alpha) $* z) in
      Maths.(leak + (alpha $* phi ((z *@ theta.c) + (input *@ theta.b))))
    | None ->
      (* this is the equivalent of initialising at z = 0 *)
      Maths.(phi (input *@ theta.b))

  let prediction ~(theta : _ Maths.some P.t) z = Maths.(z *@ theta.o)

  let init : P.param =
    let c =
      Sofo.gaussian_tensor_normed ~kind:base.kind ~device:base.device ~sigma:1.0 [ n; n ]
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    let b =
      Sofo.gaussian_tensor_normed
        ~kind:base.kind
        ~device:base.device
        ~sigma:1.0
        [ s + 1; n ]
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    (* initialise to repeat observation *)
    let o =
      let b =
        Prms.value b
        |> Maths.to_tensor
        |> Tensor.slice ~dim:0 ~start:(Some 0) ~end_:(Some s) ~step:1
      in
      if s = 1
      then (
        let b2 = Tensor.(square_ (norm b)) in
        Tensor.(div_ (transpose ~dim0:1 ~dim1:0 b) b2)
        |> Maths.of_tensor
        |> Prms.Single.free)
      else Tensor.pinverse b ~rcond:0. |> Maths.of_tensor |> Prms.Single.free
    in
    { c; b; o }

  (* here data is a list of (x_t, optional labels). labels is x_t. *)
  let f ~data theta =
    let scaling = Float.(1. / of_int tmax) in
    let result, _ =
      List.foldi data ~init:(None, None) ~f:(fun t (accu, z) (input, labels) ->
        if t % 1 = 0 then Stdlib.Gc.major ();
        let z = forward ~theta ~input z in
        let pred = prediction ~theta z in
        let accu =
          match labels with
          | None -> accu
          | Some labels ->
            let delta_ell =
              Maths.(
                scaling $* Loss.mse ~output_dims:[ 1 ] Maths.(of_tensor labels - pred))
            in
            let delta_ggn =
              Maths.C.(
                scaling
                $* Loss.mse_ggn
                     ~output_dims:[ 1 ]
                     (Maths.const pred)
                     ~vtgt:(Maths.tangent_exn pred))
            in
            (match accu with
             | None -> Some (delta_ell, delta_ggn)
             | Some accu ->
               let ell_accu, ggn_accu = accu in
               Some (Maths.(ell_accu + delta_ell), Maths.C.(ggn_accu + delta_ggn)))
        in
        accu, Some z)
    in
    Option.value_exn result

  (* TODO: make sure the batch size is 1 here, otherwise nonsensical results. Simulate one sample trajectory *)
  let simulate_1d (data : (float, Kalman_data.lds) Kalman_data.state array) theta =
    Array.foldi data ~init:([], None) ~f:(fun t (accu, z) datum ->
      if t % 1 = 0 then Stdlib.Gc.major ();
      let input =
        Tensor.of_bigarray
          ~device:base.device
          Mat.(of_array [| datum.Kalman_data.y |] 1 1)
      in
      let z = forward ~theta ~input z in
      let pred =
        prediction ~theta z |> Maths.to_tensor |> fun x -> Tensor.get_float2 x 0 0
      in
      let accu =
        Mat.of_array
          [| datum.Kalman_data.lds.tau
           ; datum.Kalman_data.lds.b
           ; datum.Kalman_data.lds.beta
           ; datum.Kalman_data.lds.sigma_eps
           ; datum.Kalman_data.y
           ; datum.Kalman_data.x
           ; pred
          |]
          1
          7
        :: accu
      in
      accu, Some z)
    |> fst
    |> List.rev
    |> Array.of_list
    |> Mat.concatenate ~axis:0
end

let sample_data bs =
  Data.minibatch ~tmax bs
  |> Data.minibatch_as_data
  |> List.map ~f:(fun datum ->
    let to_device = Tensor.of_bigarray ~device:base.device in
    to_device datum.Kalman_data.y, Some (to_device datum.Kalman_data.x))

let simulate_1d ~f_name n_trials =
  let module Data = Kalman_data.Make (Kalman_data.Default) in
  let n_list = List.range 0 n_trials in
  List.iter n_list ~f:(fun i ->
    let data = (Data.minibatch ~tmax 1).(0) in
    (* 1d *)
    let kf_prediction = Mat.of_array (Data.kalman_filter ~random:false data) (-1) 1 in
    let kf_random_prediction =
      Mat.of_array (Data.kalman_filter ~random:true data) (-1) 1
    in
    let model_params =
      let params_ba = RNN.P.C.load (in_dir f_name ^ "_params") in
      RNN.P.map params_ba ~f:(fun x -> x |> Maths.const)
    in
    let model_prediction = RNN.simulate_1d data model_params in
    (* the columns are tau, b, beta, sigma_eps, observed_y, x, model prediction, kalman prediction and kalman random prediction *)
    let to_save = Mat.((model_prediction @|| kf_prediction) @|| kf_random_prediction) in
    Mat.(
      save_txt to_save ~out:(in_dir f_name ^ "_" ^ Int.to_string i ^ "_model_prediction")))

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
let _K = 128

module RNN_Spec = struct
  type param_name =
    | C
    | B
    | O
  [@@deriving compare, sexp]

  let all = [ C; B; O ]

  let shape = function
    | C -> [ n; n ]
    | B -> [ s + 1; n ]
    | O -> [ n; s ]

  let n_params = function
    | C -> Int.(_K - 4)
    | B -> 2
    | O -> 2

  let n_params_list = [ Int.(_K - 4); 2; 2 ]
  let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
end

module RNN_Aux = struct
  type 'a p =
    { c_left : 'a
    ; c_right : 'a
    ; b_left : 'a
    ; b_right : 'a
    ; o_left : 'a
    ; o_right : 'a
    }
  [@@deriving prms]
end

module A = RNN_Aux.Make (Prms.Single)

module GGN : Auxiliary with module P = P = struct
  module P = P
  module A = A

  let init_sampling_state () = 0

  let zero_params ~shape _K =
    Tensor.zeros ~device:base.device ~kind:base.kind (_K :: shape)

  let random_params ~shape _K =
    Tensor.randn ~device:base.device ~kind:base.kind (_K :: shape)

  let get_total_n_params param_name =
    let list_prod l = List.fold l ~init:1 ~f:(fun accu i -> accu * i) in
    list_prod (RNN_Spec.shape param_name)

  let get_n_params_before_after param_name =
    let n_params_prefix_suffix_sums = prefix_suffix_sums RNN_Spec.n_params_list in
    let param_idx =
      match param_name with
      | RNN_Spec.C -> 0
      | RNN_Spec.B -> 1
      | RNN_Spec.O -> 2
    in
    List.nth_exn n_params_prefix_suffix_sums param_idx

  let g12v ~(lambda : ([< `const | `dual ] as 'a) A.t) (v : 'a P.t) : Maths.any P.t =
    let open Maths in
    let tmp_einsum left right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    let c = tmp_einsum lambda.c_left lambda.c_right v.c in
    let b = tmp_einsum lambda.b_left lambda.b_right v.b in
    let o = tmp_einsum lambda.o_left lambda.o_right v.o in
    { c; b; o }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~(param_name : RNN_Spec.param_name) ~n_per_param v =
    let zero name =
      Tensor.zeros ~device:base.device ~kind:base.kind (n_per_param :: RNN_Spec.shape name)
    in
    let params_tmp = RNN_P.{ c = zero C; b = zero B; o = zero O } in
    match param_name with
    | C -> { params_tmp with c = v }
    | B -> { params_tmp with b = v }
    | O -> { params_tmp with o = v }

  (* set tangents = zero for other parameters but v for this parameter *)

  let random_localised_vs () =
    let random_localised_param_name param_name =
      let w_shape = RNN_Spec.shape param_name in
      let before, after = get_n_params_before_after param_name in
      let w = random_params ~shape:w_shape (RNN_Spec.n_params param_name) in
      let zeros_before = zero_params ~shape:w_shape before in
      let zeros_after = zero_params ~shape:w_shape after in
      let final = Tensor.concat [ zeros_before; w; zeros_after ] ~dim:0 in
      Maths.of_tensor final
    in
    RNN_P.
      { c = random_localised_param_name C
      ; b = random_localised_param_name B
      ; o = random_localised_param_name O
      }

  (* compute sorted eigenvalues, u_left and u_right. *)
  let eigenvectors_for_params ~(lambda : ([< `const | `dual ] as 'a) A.t) ~param_name =
    let left, right =
      match param_name with
      | RNN_Spec.C -> lambda.c_left, lambda.c_right
      | RNN_Spec.B -> lambda.b_left, lambda.b_right
      | RNN_Spec.O -> lambda.o_left, lambda.o_right
    in
    get_svals_u_left_right left right

  (* cache storage with a ref to memoize computed results *)
  let s_u_cache = ref []

  (* given param name, get eigenvalues and eigenvectors. *)
  let get_s_u ~lambda ~param_name =
    match List.Assoc.find !s_u_cache param_name ~equal:RNN_Spec.equal_param_name with
    | Some s -> s
    | None ->
      let s = eigenvectors_for_params ~lambda ~param_name in
      s_u_cache := (param_name, s) :: !s_u_cache;
      s

  let extract_local_vs ~s_all ~param_name ~u_left ~u_right ~selection =
    let n_per_param = RNN_Spec.n_params param_name in
    let local_vs = get_local_vs ~selection ~s_all ~u_left ~u_right in
    localise ~param_name ~n_per_param local_vs

  let eigenvectors_for_each_param ~lambda ~param_name ~sampling_state:_ =
    let n_per_param = RNN_Spec.n_params param_name in
    let n_params = get_total_n_params param_name in
    let s_all, u_left, u_right = get_s_u ~lambda ~param_name in
    let selection =
      (* List.init n_per_param ~f:(fun i ->
          ((sampling_state * n_per_param) + i) % n_params) *)
      List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    extract_local_vs ~s_all ~param_name ~u_left ~u_right ~selection

  let eigenvectors ~(lambda : _ Maths.some A.t) ~switch_to_learn t (_K : int) =
    let eigenvectors_each =
      List.map RNN_Spec.all ~f:(fun param_name ->
        eigenvectors_for_each_param ~lambda ~param_name ~sampling_state:t)
    in
    let vs =
      List.fold eigenvectors_each ~init:None ~f:(fun accu local_vs ->
        match accu with
        | None -> Some local_vs
        | Some a -> Some (P.map2 a local_vs ~f:(fun x y -> Tensor.concat ~dim:0 [ x; y ])))
    in
    let vs = Option.map vs ~f:(fun v -> P.map v ~f:Maths.of_tensor) in
    (* reset s_u_cache and set sampling state to 0 if learn again *)
    if switch_to_learn then s_u_cache := [];
    let new_sampling_state = if switch_to_learn then 0 else t + 1 in
    Option.value_exn vs, new_sampling_state

  let init () =
    let init_eye size =
      Maths.(0.1 $* eye ~device:base.device ~kind:base.kind size) |> Prms.Single.free
    in
    RNN_Aux.
      { c_left = init_eye n
      ; c_right = init_eye n
      ; b_left = init_eye (s + 1)
      ; b_right = init_eye n
      ; o_left = init_eye n
      ; o_right = init_eye s
      }
end

(* -----------------------------------------
   -- Optimization with SOFO    ------
   ----------------------------------------- *)

module O = Optimizer.SOFO (RNN.P) (GGN)

let config =
  let aux =
    Optimizer.Config.SOFO.
      { (default_aux (in_dir "aux")) with
        config =
          Optimizer.Config.Adam.
            { default with base; learning_rate = Some 1e-3; eps = 1e-8 }
      ; steps = 500
      ; learn_steps = 10
      ; exploit_steps = 10
      }
  in
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some 0.1
    ; n_tangents = _K
    ; damping = `relative_from_top 1e-3
    ; aux = Some aux
    }

let rec loop ~t ~out ~state running_avg =
  Stdlib.Gc.major ();
  let data = sample_data batch_size in
  let theta, tangents, new_sampling_state = O.prepare ~config state in
  let loss, ggn = RNN.f ~data theta in
  let new_state =
    O.step
      ~config
      ~info:{ loss; ggn; tangents; sampling_state = new_sampling_state }
      state
  in
  let loss = Maths.to_float_exn (Maths.const loss) in
  let running_avg =
    let loss_avg =
      match running_avg with
      | [] -> loss
      | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
    in
    if t % 10 = 0
    then (
      (* save params *)
      O.P.C.save
        (RNN.P.value (O.params new_state))
        ~kind:base.ba_kind
        ~out:(in_dir "sofo_params");
      print [%message (t : int) (loss_avg : float)];
      Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss_avg |] 1 2)));
    []
  in
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state (loss :: running_avg)

(* Start the sofo loop. *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init RNN.init) []

(* let _ =
      let f_name = "sofo" in
      simulate_1d ~f_name n_trials_simulation *)
