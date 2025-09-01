(** Meta learning a Settings.classification with randomised labels across different sessions with a vanilla rnn *)
open Base

open Utils
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let batch_size = 256
let max_iter = 200000

let _ =
  Random.init 1999;
  (* Random.self_init (); *)
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run
let base = Optimizer.Config.Base.default
let r = 100
let c' = 400

module Settings = struct
  (* number of Settings.classes *)
  let cl = 3

  (* number of timesteps where an input sample is held fixed *)
  let t_per_sample = 10
  let t_per_sample_exploit = 10

  (* global mean and std *)
  let mu, sigma = 1., 0.1

  (* learn_t = exploit_t *)

  let learn_t = cl * t_per_sample
  let exploit_t = cl * t_per_sample_exploit
  let t_is_exploit t = t > learn_t

  (* only calculate loss if last sample in the t_per_sample_exploit samples *)
  let is_last_sample t = Int.((t - (t_per_sample_exploit - 1)) % t_per_sample_exploit) = 0
end

module RNN_P = struct
  type 'a p =
    { w : 'a
    ; c : 'a
    ; b : 'a
    ; a : 'a
    ; o : 'a
    }
  [@@deriving prms]
end

module P = RNN_P.Make (Prms.Single)

(* neural network *)
module RNN = struct
  module P = P

  let phi x = Maths.relu x

  (* input is the (input, labels, cue) and z is the internal state *)
  let forward ~(theta : _ Maths.some P.t) ~input z =
    let bs = Tensor.shape input |> List.hd_exn in
    let input =
      Maths.concat
        [ Maths.of_tensor input
        ; Maths.of_tensor (Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ])
        ]
        ~dim:1
    in
    match z with
    | Some z ->
      let z_tmp =
        Maths.concat
          [ z
          ; Maths.(
              any (of_tensor (Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ])))
          ]
          ~dim:1
      in
      Maths.((z *@ theta.a) + (phi (z_tmp *@ theta.c) *@ theta.w) + (input *@ theta.b))
    | None ->
      (* this is the equivalent of initialising at z = 0 *)
      Maths.(input *@ theta.b)

  let prediction ~(theta : _ Maths.some P.t) z = Maths.(relu (z *@ theta.o))

  let init ~r ~c' : P.param =
    let to_param x = x |> Maths.of_tensor |> Prms.Single.free in
    let w =
      Sofo.gaussian_tensor_normed ~kind:base.kind ~device:base.device ~sigma:1.0 [ c'; r ]
      |> to_param
    in
    let c =
      Sofo.gaussian_tensor_normed
        ~kind:base.kind
        ~device:base.device
        ~sigma:1.0
        [ r + 1; c' ]
      |> to_param
    in
    let b =
      Sofo.gaussian_tensor_normed
        ~kind:base.kind
        ~device:base.device
        ~sigma:1.0
        [ Int.((2 * Settings.cl) + 2); r ]
      |> to_param
    in
    let o =
      Sofo.gaussian_tensor_normed
        ~kind:base.kind
        ~device:base.device
        ~sigma:1.0
        [ r; Settings.cl ]
      |> to_param
    in
    let a = Tensor.zeros ~kind:base.kind ~device:base.device [ r; r ] |> to_param in
    { w; c; b; a; o }

  let f ~data theta =
    let scaling = Float.(1. / of_int (List.length data)) in
    let result, _ =
      List.fold data ~init:(None, None) ~f:(fun (accu, z) (input, labels) ->
        let z = forward ~theta ~input z in
        let accu =
          match labels with
          | None -> accu
          | Some labels ->
            let pred = prediction ~theta z in
            let delta_ell =
              Maths.(
                scaling
                $* Loss.cross_entropy
                     ~output_dims:[ 1 ]
                     ~labels:(Maths.of_tensor labels)
                     pred)
            in
            let delta_ggn =
              Maths.(
                scaling
                $* Loss.cross_entropy_ggn
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

  let simulate ~(data : (Tensor.t * Tensor.t option) list) theta =
    let total_acc =
      List.fold data ~init:(0., None) ~f:(fun (accu, z) datum ->
        let input, labels = datum in
        let z = forward ~theta ~input z in
        let accu =
          match labels with
          | None -> accu
          | Some labels ->
            let pred = prediction ~theta z in
            let _, max_y = Tensor.max_dim ~keepdim:false ~dim:1 labels in
            let _, max_ypred =
              Tensor.max_dim ~keepdim:false ~dim:1 (Maths.to_tensor pred)
            in
            let accuracy =
              Tensor.eq_tensor max_y max_ypred
              |> Tensor.to_dtype ~dtype:base.kind ~non_blocking:false ~copy:false
              |> Tensor.mean
              |> Tensor.to_float0_exn
            in
            Float.(accu + accuracy)
        in
        accu, Some z)
      |> fst
    in
    (* normalise by the number of exploitations *)
    Float.(total_acc / of_int Settings.cl)
end

(* instantiate labels array into a block of shape [ct x c], each t rows contains the one-hot encoding *)
let labels_block ~label_array t =
  let data_labels_array =
    Array.map label_array ~f:(fun i ->
      let zeros = Mat.zeros t Settings.cl in
      Mat.set_slice [ []; [ i ] ] zeros (Mat.ones t 1);
      zeros)
  in
  Mat.concatenate data_labels_array ~axis:0

(* generate data for input block *)
let input_block ~t_per_sample label_array =
  let data_mu = Mat.(labels_block ~label_array t_per_sample *$ Settings.mu) in
  let data_noise =
    let data_noise_array =
      Array.init Settings.cl ~f:(fun _ ->
        let row_array =
          Array.init Settings.cl ~f:(fun _ ->
            Mat.(
              ones t_per_sample 1 *$ Owl.Stats.gaussian_rvs ~mu:0. ~sigma:Settings.sigma))
        in
        Mat.concatenate row_array ~axis:1)
    in
    Mat.concatenate data_noise_array ~axis:0
  in
  Mat.(data_mu + data_noise)

let to_device = Tensor.of_bigarray ~device:base.device

(* sample inputs and labels for a single batch *)
let sample () =
  (* sample classes to be used for this session *)
  let classes_array = Array.init Settings.cl ~f:(fun x -> x) in
  Array.permute classes_array;
  (* learn phase; inputs always in sequence. *)
  let data_learn_inputs =
    input_block
      ~t_per_sample:Settings.t_per_sample
      (Array.init Settings.cl ~f:(fun x -> x))
  in
  (* labels follow the sample classes_array *)
  let data_learn_labels = labels_block ~label_array:classes_array Settings.t_per_sample in
  let data_learn_cue = Mat.ones Settings.learn_t 1 in
  let data_learn =
    Mat.concatenate [| data_learn_inputs; data_learn_labels; data_learn_cue |] ~axis:1
  in
  (* exploit phase *)
  (* randomised sequence of observed data *)
  let data_exploit_random_seq = Array.init Settings.cl ~f:(fun x -> x) in
  Array.permute data_exploit_random_seq;
  let data_exploit_inputs =
    input_block ~t_per_sample:Settings.t_per_sample_exploit data_exploit_random_seq
  in
  (* no labels in exploit phase *)
  let data_exploit_labels = Mat.zeros Settings.exploit_t Settings.cl in
  let data_exploit_cue = Mat.zeros Settings.exploit_t 1 in
  let data_exploit =
    Mat.concatenate
      [| data_exploit_inputs; data_exploit_labels; data_exploit_cue |]
      ~axis:1
  in
  let data_learn_exploit = Mat.concat_vertical data_learn data_exploit in
  (* targets *)
  (* remap randomised data exploit seq according to sample classes_array. *)
  let data_exploit_actual_label_sequence =
    Array.map data_exploit_random_seq ~f:(fun rand_seq -> classes_array.(rand_seq))
  in
  (* each row is the target *)
  let targets = labels_block ~label_array:data_exploit_actual_label_sequence 1 in
  data_learn_exploit, targets

let sample_data batch_size =
  let data_minibatch = Array.init batch_size ~f:(fun _ -> sample ()) in
  let t_max = Mat.row_num (fst data_minibatch.(0)) in
  (* list of time-indexed mat, where each row is one batch. *)
  let data_list =
    List.init t_max ~f:(fun t ->
      let datum_array =
        Array.map data_minibatch ~f:(fun (datum_mat, _) ->
          Mat.get_fancy [ I t; R [] ] datum_mat)
      in
      Mat.concatenate datum_array ~axis:0)
  in
  (* targets only available in exploitation phase for the last sample *)
  let targets_list =
    List.init (Settings.learn_t + Settings.exploit_t) ~f:(fun t ->
      if Settings.t_is_exploit t && Settings.is_last_sample t
      then (
        let target_array =
          let n_th_target =
            Int.(
              (t - Settings.learn_t - Settings.t_per_sample_exploit)
              / Settings.t_per_sample_exploit)
          in
          Array.map data_minibatch ~f:(fun (_, target_mat) ->
            Mat.get_fancy [ I n_th_target; R [] ] target_mat)
        in
        Some (Mat.concatenate target_array ~axis:0))
      else None)
  in
  List.map2_exn data_list targets_list ~f:(fun datum target ->
    ( to_device datum
    , Option.value_map target ~default:None ~f:(fun target -> Some (to_device target)) ))

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
let _K = 256

module RNN_Spec = struct
  type param_name =
    | W
    | C
    | B
    | A
    | O
  [@@deriving compare, sexp]

  let all = [ W; C; A ]

  let shape = function
    | W -> [ c'; r ]
    | C -> [ r + 1; c' ]
    | B -> [ Int.((2 * Settings.cl) + 2); r ]
    | A -> [ r; r ]
    | O -> [ r; Settings.cl ]

  let n_params = function
    | W -> 100
    | C -> 100
    | B -> 5
    | A -> Int.(_K - 210)
    | O -> 5

  let n_params_list = [ 100; 100; 5; Int.(_K - 210); 5 ]
  let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
end

module RNN_Aux = struct
  type 'a p =
    { w_left : 'a
    ; w_right : 'a
    ; c_left : 'a
    ; c_right : 'a
    ; b_left : 'a
    ; b_right : 'a
    ; a_left : 'a
    ; a_right : 'a
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
      | RNN_Spec.W -> 0
      | RNN_Spec.C -> 1
      | RNN_Spec.B -> 2
      | RNN_Spec.A -> 3
      | RNN_Spec.O -> 4
    in
    List.nth_exn n_params_prefix_suffix_sums param_idx

  let g12v ~(lambda : ([< `const | `dual ] as 'a) A.t) (v : 'a P.t) : Maths.any P.t =
    let open Maths in
    let tmp_einsum left right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    let w = tmp_einsum lambda.w_left lambda.w_right v.w in
    let c = tmp_einsum lambda.c_left lambda.c_right v.c in
    let b = tmp_einsum lambda.b_left lambda.b_right v.b in
    let a = tmp_einsum lambda.a_left lambda.a_right v.a in
    let o = tmp_einsum lambda.o_left lambda.o_right v.o in
    { w; c; b; a; o }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~(param_name : RNN_Spec.param_name) ~n_per_param v =
    let zero name =
      Tensor.zeros ~device:base.device ~kind:base.kind (n_per_param :: RNN_Spec.shape name)
    in
    let params_tmp =
      RNN_P.{ w = zero W; c = zero C; b = zero B; a = zero A; o = zero O }
    in
    match param_name with
    | W -> { params_tmp with w = v }
    | C -> { params_tmp with c = v }
    | B -> { params_tmp with b = v }
    | A -> { params_tmp with a = v }
    | O -> { params_tmp with o = v }

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
      { w = random_localised_param_name W
      ; c = random_localised_param_name C
      ; b = random_localised_param_name B
      ; a = random_localised_param_name A
      ; o = random_localised_param_name O
      }

  (* compute sorted eigenvalues, u_left and u_right. *)
  let eigenvectors_for_params ~(lambda : ([< `const | `dual ] as 'a) A.t) ~param_name =
    let left, right =
      match param_name with
      | RNN_Spec.W -> lambda.w_left, lambda.w_right
      | RNN_Spec.C -> lambda.c_left, lambda.c_right
      | RNN_Spec.B -> lambda.b_left, lambda.b_right
      | RNN_Spec.A -> lambda.a_left, lambda.a_right
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
      { w_left = init_eye c'
      ; w_right = init_eye r
      ; c_left = init_eye (r + 1)
      ; c_right = init_eye c'
      ; b_left = init_eye Int.((2 * Settings.cl) + 2)
      ; b_right = init_eye r
      ; a_left = init_eye r
      ; a_right = init_eye r
      ; o_left = init_eye r
      ; o_right = init_eye Settings.cl
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
      ; steps = 5
      ; learn_steps = 1
      ; exploit_steps = 1
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
  loop ~t:0 ~out ~state:(O.init (RNN.init ~r ~c')) []
