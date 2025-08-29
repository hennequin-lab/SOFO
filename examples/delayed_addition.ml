(** Learning a delayed addition task to compare SOFO with adam. *)
open Utils

open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let batch_size = 256
let max_iter = 20000

let _ =
  Random.init 1999;
  (* Random.self_init (); *)
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default

module Settings = struct
  (* length of data *)
  let n_steps = 1000 (* 20 to 600 *)

  (* first signal upper bound *)
  let t1_bound = 10
  let t2_bound = Int.(n_steps / 2)
end

(* net parameters *)
let n = 128 (* number of neurons *)
let alpha = 0.25

module RNN_P = struct
  type 'a p =
    { c : 'a
    ; b : 'a
    ; o : 'a
    }
  [@@deriving prms]
end

module P = RNN_P.Make (Prms.Single)

(* neural network *)
module RNN = struct
  module P = P

  let phi x = Maths.relu x

  (* input is the (number, signal) pair and z is the internal state *)
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
      Sofo.gaussian_tensor_normed ~kind:base.kind ~device:base.device ~sigma:1.0 [ 3; n ]
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    (* initialise to repeat observation *)
    let o =
      Sofo.gaussian_tensor_normed ~kind:base.kind ~device:base.device ~sigma:1.0 [ n; 2 ]
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    { c; b; o }

  (* here data is a list of (x_t, optional labels). labels is x_t. *)
  let f ~data theta =
    let result, _ =
      let input_all, labels_all = data in
      let top_2, _ = List.split_n (Tensor.shape input_all) 2 in
      let time_list = List.range 0 Settings.n_steps in
      List.fold time_list ~init:(None, None) ~f:(fun (accu, z) t ->
        if t % 1 = 0 then Stdlib.Gc.major ();
        let input =
          let tmp =
            Tensor.slice ~dim:2 ~start:(Some t) ~end_:(Some (t + 1)) ~step:1 input_all
          in
          Tensor.reshape tmp ~shape:top_2
        in
        (* loss only calculated at the final timestep *)
        let labels = if t = Settings.n_steps - 1 then Some labels_all else None in
        let z = forward ~theta ~input z in
        let accu =
          match labels with
          | None -> accu
          | Some labels ->
            let pred = prediction ~theta z in
            let delta_ell = Loss.mse ~output_dims:[ 1 ] Maths.(of_tensor labels - pred) in
            let delta_ggn =
              Loss.mse_ggn
                ~output_dims:[ 1 ]
                (Maths.const pred)
                ~vtgt:(Maths.tangent_exn pred)
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
end

(* -----------------------------------------
   -- Generate addition data    ------
   ----------------------------------------- *)

let data_shape = [| 1; 2; Settings.n_steps |]

let sample () =
  let number_trace = Mat.uniform 1 Settings.n_steps in
  let signal_trace = Mat.zeros 1 Settings.n_steps in
  (* set indicator *)
  let t1 = Random.int_incl 0 (Settings.t1_bound - 1) in
  let t2 = Random.int_incl (t1 + 1) Settings.t2_bound in
  Mat.set signal_trace 0 t1 1.;
  Mat.set signal_trace 0 t2 1.;
  let target = Mat.(get number_trace 0 t1) +. Mat.(get number_trace 0 t2) in
  let target_mat = Mat.of_array [| target |] 1 1 in
  let input_mat = Mat.concat_horizontal number_trace signal_trace in
  let input_array = Arr.reshape input_mat data_shape in
  input_array, target_mat

let sample_data batch_size =
  let data_minibatch = Array.init batch_size ~f:(fun _ -> sample ()) in
  let input_array = Array.map data_minibatch ~f:fst in
  let target_array = Array.map data_minibatch ~f:snd in
  let input_tensor = Arr.concatenate ~axis:0 input_array in
  let target_mat = Mat.concatenate ~axis:0 target_array in
  let to_device = Tensor.of_bigarray ~device:base.device in
  to_device input_tensor, to_device target_mat

(* -----------------------------------------
   -- Optimization with SOFO    ------
   ----------------------------------------- *)

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
let _K = 128

type param_name =
  | C
  | B
  | O
[@@deriving compare]

let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
let param_names_list = [ C; B; O ]
let n_params_c, n_params_b, n_params_o = Int.(_K - 4), 2, 2
let n_params_list = [ n_params_c; n_params_b; n_params_o ]

module GGN : Auxiliary with module P = P = struct
  include struct
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

  module P = P

  (* module A = AA.Make (Prms.Single) *)
  module A = Make (Prms.Single)

  let init_sampling_state () = 0

  let zero_params ~shape _K =
    Tensor.zeros ~device:base.device ~kind:base.kind (_K :: shape)

  let random_params ~shape _K =
    Tensor.randn ~device:base.device ~kind:base.kind (_K :: shape)

  let get_shapes (param_name : param_name) =
    match param_name with
    | C -> [ n; n ]
    | B -> [ 3; n ]
    | O -> [ n; 2 ]

  let get_n_params (param_name : param_name) =
    match param_name with
    | C -> n_params_c
    | B -> n_params_b
    | O -> n_params_o

  let get_total_n_params (param_name : param_name) =
    let list_prod l = List.fold l ~init:1 ~f:(fun accu i -> accu * i) in
    list_prod (get_shapes param_name)

  let get_n_params_before_after (param_name : param_name) =
    let n_params_prefix_suffix_sums = prefix_suffix_sums n_params_list in
    let param_idx =
      match param_name with
      | C -> 0
      | B -> 1
      | O -> 2
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
  let localise ~param_name ~n_per_param v =
    let c = zero_params ~shape:(get_shapes C) n_per_param in
    let b = zero_params ~shape:(get_shapes B) n_per_param in
    let o = zero_params ~shape:(get_shapes O) n_per_param in
    let params_tmp = RNN_P.{ c; b; o } in
    match param_name with
    | C -> { params_tmp with c = v }
    | B -> { params_tmp with b = v }
    | O -> { params_tmp with o = v }

  let random_localised_vs () =
    let random_localised_param_name param_name =
      let w_shape = get_shapes param_name in
      let before, after = get_n_params_before_after param_name in
      let w = random_params ~shape:w_shape (get_n_params param_name) in
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
  let eigenvectors_for_params ~lambda ~param_name =
    let left, right =
      match param_name with
      | C -> lambda.c_left, lambda.c_right
      | B -> lambda.b_left, lambda.b_right
      | O -> lambda.o_left, lambda.o_right
    in
    get_svals_u_left_right left right

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
    let local_vs = get_local_vs ~selection ~s_all ~u_left ~u_right in
    localise ~param_name ~n_per_param local_vs

  let eigenvectors_for_each_param ~lambda ~param_name ~sampling_state:_ =
    let n_per_param = get_n_params param_name in
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
      List.map param_names_list ~f:(fun param_name ->
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
    { c_left = init_eye n
    ; c_right = init_eye n
    ; b_left = init_eye 3
    ; b_right = init_eye n
    ; o_left = init_eye n
    ; o_right = init_eye 2
    }
end

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
    ; damping = `relative_from_top 1e-5
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
      print [%message (t : int) (loss_avg : float)];
      Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss_avg |] 1 2)));
    []
  in
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state (loss :: running_avg)

(* Start the sofo loop. *)
let _ =
  Bos.Cmd.(v "rm" % "-f" % in_dir "loss") |> Bos.OS.Cmd.run |> ignore;
  Bos.Cmd.(v "rm" % "-f" % in_dir "aux") |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out:(in_dir "loss") ~state:(O.init RNN.init) []
