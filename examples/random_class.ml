(** Meta learning a Settings.classification with randomised labels across different sessions with a vanilla rnn *)
open Base

open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let batch_size = 256

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
  let cl = 5

  (* number of timesteps where an input sample is held fixed *)
  let t_per_sample = 10
  let t_per_sample_exploit = 10

  (* global mean and std *)
  let mu, sigma = 1., 0.1

  (* learn_t = exploit_t *)

  let learn_t = cl * t_per_sample
  let exploit_t = cl * t_per_sample_exploit
  let t_is_learn t = t < learn_t
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

module P = RNN_P.Make (Prms.P)

(* neural network *)
module RNN = struct
  module P = P

  let phi x = Maths.relu x

  (* input is the (input, labels, cue) and z is the internal state *)
  let forward ~(theta : P.M.t) ~input z =
    let bs = Tensor.shape input |> List.hd_exn in
    let input =
      Maths.concat
        (Maths.const input)
        (Maths.const (Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ]))
        ~dim:1
    in
    match z with
    | Some z ->
      let z_tmp =
        Maths.concat
          z
          (Maths.const (Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ]))
          ~dim:1
      in
      Maths.((z *@ theta.a) + (phi (z_tmp *@ theta.c) *@ theta.w) + (input *@ theta.b))
    | None ->
      (* this is the equivalent of initialising at z = 0 *)
      Maths.(input *@ theta.b)

  let prediction ~(theta : P.M.t) z = Maths.(relu (z *@ theta.o))

  let init ~r ~c' : P.tagged =
    let w =
      Convenience.gaussian_tensor_2d_normed
        ~kind:base.kind
        ~device:base.device
        ~a:c'
        ~b:r
        ~sigma:1.0
      |> Prms.free
    in
    let c =
      Convenience.gaussian_tensor_2d_normed
        ~kind:base.kind
        ~device:base.device
        ~a:(r + 1)
        ~b:c'
        ~sigma:1.0
      |> Prms.free
    in
    let b =
      Convenience.gaussian_tensor_2d_normed
        ~kind:base.kind
        ~device:base.device
        ~a:Int.((2 * Settings.cl) + 2)
        ~b:r
        ~sigma:1.0
      |> Prms.free
    in
    let o =
      Convenience.gaussian_tensor_2d_normed
        ~kind:base.kind
        ~device:base.device
        ~a:r
        ~b:Settings.cl
        ~sigma:1.0
      |> Prms.free
    in
    let a = Tensor.zeros ~kind:base.kind ~device:base.device [ r; r ] |> Prms.free in
    { w; c; b; a; o }

  type data = (Tensor.t * Tensor.t option) list
  type args = unit

  let f ~update ~data ~init ~args:() theta =
    let module L =
      Loss.CE (struct
        let scaling_factor = Float.(1. / of_int Settings.cl)
      end)
    in
    let result, _ =
      List.foldi data ~init:(init, None) ~f:(fun t (accu, z) (input, labels) ->
        if t % 1 = 0 then Stdlib.Gc.major ();
        let z = forward ~theta ~input z in
        let accu =
          match labels with
          | None -> accu
          | Some labels ->
            let pred = prediction ~theta z in
            let reduce_dim_list = Convenience.all_dims_but_first (Maths.primal pred) in
            let ell = L.f ~labels ~reduce_dim_list pred in
            (match update with
             | `loss_only u -> u accu (Some ell)
             | `loss_and_ggn u ->
               let delta_ggn =
                 let vtgt = Maths.tangent pred |> Option.value_exn in
                 L.vtgt_hessian_gv ~labels ~vtgt ~reduce_dim_list pred
               in
               u accu (Some (ell, Some delta_ggn)))
        in
        accu, Some z)
    in
    result

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
            let _, max_ypred = Tensor.max_dim ~keepdim:false ~dim:1 (Maths.primal pred) in
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

let _K = 256
(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)

type param_name =
  | W
  | C
  | B
  | A
  | O
[@@deriving compare]

let equal_param_name p1 p2 = compare_param_name p1 p2 = 0

let n_params_w, n_params_c, n_params_b, n_params_a, n_params_o =
  100, 100, 5, Int.(_K - 210), 5

let n_params_list = [ n_params_w; n_params_c; n_params_b; n_params_a; n_params_o ]
let cycle = true

module GGN : Wrapper.Auxiliary with module P = P = struct
  include struct
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

  module P = P
  module A = Make (Prms.P)

  type sampling_state = int

  let init_sampling_state () = 0

  let zero_params ~shape _K =
    Tensor.zeros ~device:base.device ~kind:base.kind (_K :: shape)

  let random_params ~shape _K =
    Tensor.randn ~device:base.device ~kind:base.kind (_K :: shape)

  let param_names_list = [ W; C; B; A; O ]

  let get_shapes (param_name : param_name) =
    match param_name with
    | W -> [ c'; r ]
    | C -> [ r + 1; c' ]
    | B -> [ Int.((2 * Settings.cl) + 2); r ]
    | A -> [ r; r ]
    | O -> [ r; Settings.cl ]

  let get_n_params (param_name : param_name) =
    match param_name with
    | W -> n_params_w
    | C -> n_params_c
    | B -> n_params_b
    | A -> n_params_a
    | O -> n_params_o

  let get_total_n_params (param_name : param_name) =
    let list_prod l = List.fold l ~init:1 ~f:(fun accu i -> accu * i) in
    list_prod (get_shapes param_name)

  let get_n_params_before_after (param_name : param_name) =
    let n_params_prefix_suffix_sums = Convenience.prefix_suffix_sums n_params_list in
    let param_idx =
      match param_name with
      | W -> 0
      | C -> 1
      | B -> 2
      | A -> 3
      | O -> 4
    in
    List.nth_exn n_params_prefix_suffix_sums param_idx

  (* approximation defined implicitly via Gv products *)
  let g12v ~(lambda : A.M.t) (v : P.M.t) : P.M.t =
    let open Maths in
    let tmp_einsum left right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    let w = tmp_einsum lambda.w_left lambda.w_right v.w in
    let c = tmp_einsum lambda.c_left lambda.c_right v.c in
    let b = tmp_einsum lambda.b_left lambda.b_right v.b in
    let a = tmp_einsum lambda.a_left lambda.a_right v.a in
    let o = tmp_einsum lambda.o_left lambda.o_right v.o in
    { w; c; b; a; o }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~param_name ~n_per_param v =
    let w = zero_params ~shape:(get_shapes W) n_per_param in
    let c = zero_params ~shape:(get_shapes C) n_per_param in
    let b = zero_params ~shape:(get_shapes B) n_per_param in
    let a = zero_params ~shape:(get_shapes A) n_per_param in
    let o = zero_params ~shape:(get_shapes O) n_per_param in
    let params_tmp = RNN_P.{ w; c; b; a; o } in
    match param_name with
    | W -> { params_tmp with w = v }
    | C -> { params_tmp with c = v }
    | B -> { params_tmp with b = v }
    | A -> { params_tmp with a = v }
    | O -> { params_tmp with o = v }

  let random_localised_vs _K : P.T.t =
    let random_localised_param_name param_name =
      let w_shape = get_shapes param_name in
      let before, after = get_n_params_before_after param_name in
      let w = random_params ~shape:w_shape (get_n_params param_name) in
      let zeros_before = zero_params ~shape:w_shape before in
      let zeros_after = zero_params ~shape:w_shape after in
      let final = Tensor.concat [ zeros_before; w; zeros_after ] ~dim:0 in
      final
    in
    { w = random_localised_param_name W
    ; c = random_localised_param_name C
    ; b = random_localised_param_name B
    ; a = random_localised_param_name A
    ; o = random_localised_param_name O
    }

  (* compute sorted eigenvalues, u_left and u_right. *)
  let eigenvectors_for_params ~lambda ~param_name =
    let left, right =
      match param_name with
      | W -> lambda.w_left, lambda.w_right
      | C -> lambda.c_left, lambda.c_right
      | B -> lambda.b_left, lambda.b_right
      | A -> lambda.a_left, lambda.a_right
      | O -> lambda.o_left, lambda.o_right
    in
    let u_left, s_left, _ = Tensor.svd ~some:true ~compute_uv:true Maths.(primal left) in
    let u_right, s_right, _ =
      Tensor.svd ~some:true ~compute_uv:true Maths.(primal right)
    in
    let s_left = Tensor.to_float1_exn s_left |> Array.to_list in
    let s_right = Tensor.to_float1_exn s_right |> Array.to_list in
    let s_all =
      List.mapi s_left ~f:(fun il sl ->
        List.mapi s_right ~f:(fun ir sr -> il, ir, Float.(sl * sr)))
      |> List.concat
      |> List.sort ~compare:(fun (_, _, a) (_, _, b) -> Float.compare b a)
      |> Array.of_list
    in
    s_all, u_left, u_right

  (* cache storage with a ref to memoize computed results *)
  let s_u_cache = ref []

  (* given param name, get eigenvalues and eigenvectors. *)
  let get_s_u ~lambda ~param_name =
    match List.Assoc.find !s_u_cache param_name ~equal:equal_param_name with
    | Some s -> s
    | None ->
      let s = eigenvectors_for_params ~lambda ~param_name in
      s_u_cache := (param_name, s) :: !s_u_cache;
      s

  let extract_local_vs ~s_all ~param_name ~u_left ~u_right ~selection =
    let n_per_param = get_n_params param_name in
    let local_vs =
      List.map selection ~f:(fun idx ->
        let il, ir, _ = s_all.(idx) in
        let slice_and_squeeze t dim idx =
          Tensor.squeeze_dim
            ~dim
            (Tensor.slice t ~dim ~start:(Some idx) ~end_:(Some (idx + 1)) ~step:1)
        in
        let u_l = slice_and_squeeze u_left 1 il in
        let u_r = slice_and_squeeze u_right 1 ir in
        let tmp = Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_l; u_r ] in
        Tensor.unsqueeze tmp ~dim:0)
      |> Tensor.concatenate ~dim:0
    in
    localise ~param_name ~n_per_param local_vs

  let eigenvectors_for_each_param ~lambda ~param_name ~sampling_state =
    let n_per_param = get_n_params param_name in
    let n_params = get_total_n_params param_name in
    let s_all, u_left, u_right = get_s_u ~lambda ~param_name in
    let selection =
      if cycle
      then
        List.init n_per_param ~f:(fun i ->
          ((sampling_state * n_per_param) + i) % n_params)
      else List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    extract_local_vs ~s_all ~param_name ~u_left ~u_right ~selection

  let eigenvectors ~(lambda : A.M.t) ~switch_to_learn t (_K : int) =
    let eigenvectors_each =
      List.map param_names_list ~f:(fun param_name ->
        eigenvectors_for_each_param ~lambda ~param_name ~sampling_state:t)
    in
    let vs =
      List.fold eigenvectors_each ~init:None ~f:(fun accu local_vs ->
        match accu with
        | None -> Some local_vs
        | Some a -> Some (P.map2 a local_vs ~f:(fun x y -> Tensor.concat ~dim:0 [ x; y ])))
    in
    (* reset s_u_cache and set sampling state to 0 if learn again *)
    if switch_to_learn then s_u_cache := [];
    let new_sampling_state = if switch_to_learn then 0 else t + 1 in
    Option.value_exn vs, new_sampling_state

  let init () =
    let init_eye size =
      Mat.(0.1 $* eye size) |> Tensor.of_bigarray ~device:base.device |> Prms.free
    in
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

(* --------------------------------
   -- Generic type of optimiser
   -------------------------------- *)

module type Do_with_T = sig
  module O :
    Optimizer.T
    with type 'a W.P.p = 'a RNN.P.p
     and type W.data = (Tensor.t * Tensor.t option) list
     and type W.args = RNN.args

  val name : string
  val config : iter:int -> (float, Bigarray.float32_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    Bos.Cmd.(v "rm" % "-f" % in_dir "aux") |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state ~time_elapsed running_avg =
      Stdlib.Gc.major ();
      let t = iter in
      let data = sample_data batch_size in
      let t0 = Unix.gettimeofday () in
      let loss, new_state = O.step ~config:(config ~iter) ~state ~data () in
      let t1 = Unix.gettimeofday () in
      let time_elapsed = Float.(time_elapsed + t1 - t0) in
      let running_avg =
        let loss_avg =
          match running_avg with
          | [] -> loss
          | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
        in
        (* save params *)
        if iter % 10 = 0
        then (
          (* save params *)
          O.W.P.T.save
            (RNN.P.value (O.params new_state))
            ~kind:base.ba_kind
            ~out:(in_dir name ^ "_params");
          (* avg error *)
          Convenience.print [%message (t : int) (loss_avg : float)];
          let train_acc =
            RNN.simulate
              ~data
              (RNN.P.map ~f:Maths.const (RNN.P.value (O.params new_state)))
          in
          let test_acc =
            let test_data = sample_data batch_size in
            RNN.simulate
              ~data:test_data
              (RNN.P.map ~f:Maths.const (RNN.P.value (O.params new_state)))
          in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array
                 [| Float.of_int t; time_elapsed; loss_avg; test_acc; train_acc |]
                 1
                 5)));
        []
      in
      if iter < max_iter
      then loop ~iter:(iter + 1) ~state:new_state ~time_elapsed (loss :: running_avg)
    in
    loop ~iter:0 ~state:init ~time_elapsed:0. []
end

(* --------------------------------
         -- SOFO
         -------------------------------- *)

module Do_with_SOFO : Do_with_T = struct
  module O = Optimizer.SOFO (RNN) (GGN)

  let name = "sofo"

  let config ~iter:_ =
    let aux =
      Optimizer.Config.SOFO.
        { (default_aux (in_dir "aux")) with
          config =
            Optimizer.Config.Adam.
              { default with base; learning_rate = Some 1e-3; eps = 1e-4 }
        ; steps = 5
        ; learn_steps = 100
        ; exploit_steps = 100
        }
    in
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 0.1
      ; n_tangents = _K
      ; rank_one = false
      ; damping = Some 1e-3
      ; aux = Some aux
      }

  let init = O.init (RNN.init ~r ~c')
end

(* --------------------------------
         -- Adam
         --------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (RNN)

  let config ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 1e-5 }

  let init = O.init (RNN.init ~r ~c')
end

let _ =
  let max_iter = 200000 in
  let optimise =
    match Cmdargs.get_string "-m" with
    | Some "sofo" ->
      let module X = Make (Do_with_SOFO) in
      X.optimise
    | Some "adam" ->
      let module X = Make (Do_with_Adam) in
      X.optimise
    | _ -> failwith "-m [sofo | fgd | adam]"
  in
  optimise max_iter
