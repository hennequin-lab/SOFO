(** Learning a b-bit flip-flop task as in (Sussillo, 2013) to compare SOFO with FORCE. *)
open Utils

open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let batch_size = 32

let _ =
  Random.init 1999;
  Owl_stats_prng.init 2000;
  Torch_core.Wrapper.manual_seed 2000

let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default
let max_iter = 100000

module Settings = struct
  (* can we still train with more flips? *)
  let b = 10
  let n_steps = 200
  let pulse_prob = 0.02
  let pulse_duration = 2
  let pulse_refr = 10
end

let n = 128

module RNN_P = struct
  type 'a p =
    { j : 'a
    ; fb : Maths.const Prms.Single.t
    ; b : 'a
    ; w : 'a
    }
  [@@deriving prms]
end

module P = RNN_P.Make (Prms.Single)

(* net parameters *)
let g = 0.5

let fb =
  Mat.(uniform ~a:(-1.) ~b:1. Settings.b n)
  |> Tensor.of_bigarray ~device:base.device
  |> Maths.of_tensor

(* neural network *)
module RNN = struct
  module P = P

  let tau = 10.

  let init () : P.param =
    let w =
      Mat.(gaussian n Settings.b /$ Float.(sqrt (of_int n)))
      |> Tensor.of_bigarray ~device:base.device
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    let j =
      Mat.(gaussian Int.(n + 1) n *$ Float.(g / sqrt (of_int n)))
      |> Tensor.of_bigarray ~device:base.device
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    let fb = Prms.Single.const fb in
    let b =
      Mat.(uniform ~a:(-1.) ~b:1. Settings.b n)
      |> Tensor.of_bigarray ~device:base.device
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    { j; fb; b; w }

  let phi = Maths.relu

  let forward ~(theta : _ Maths.some P.t) ~input z =
    let bs = Tensor.shape input |> List.hd_exn in
    let input = Maths.(of_tensor input *@ theta.b) in
    let phi_z = phi z in
    let prev_outputs = Maths.(phi_z *@ theta.w) in
    let feedback = Maths.(prev_outputs *@ theta.fb) in
    let phi_z =
      Maths.concat
        [ phi_z
        ; Maths.any
            (Maths.of_tensor (Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ]))
        ]
        ~dim:1
    in
    let dz = Maths.((neg z + (phi_z *@ theta.j) + feedback + input) / f tau) in
    Maths.(z + dz)

  let f ~data:(inputs, targets) (theta : _ Maths.some P.t) =
    let[@warning "-8"] [ n_steps; bs; _ ] = Tensor.shape inputs in
    let z0 =
      Tensor.(f 0.1 * randn ~device:base.device [ bs; n ]) |> Maths.of_tensor |> Maths.any
    in
    let scaling = Float.(1. / of_int Settings.n_steps) in
    let result, _ =
      List.fold (List.range 0 n_steps) ~init:(None, z0) ~f:(fun (accu, z) t ->
        Stdlib.Gc.major ();
        let input =
          Tensor.slice inputs ~step:1 ~dim:0 ~start:(Some t) ~end_:(Some (t + 1))
          |> Tensor.squeeze
        in
        let target =
          Tensor.slice targets ~step:1 ~dim:0 ~start:(Some t) ~end_:(Some (t + 1))
          |> Tensor.squeeze
        in
        let z = forward ~theta ~input z in
        let pred = Maths.(phi z *@ theta.w) in
        let accu =
          let delta_ell =
            Maths.(scaling $* Loss.mse ~output_dims:[ 1 ] Maths.(of_tensor target - pred))
          in
          let delta_ggn =
            Maths.C.(
              scaling
              $* Loss.mse_ggn
                   ~output_dims:[ 1 ]
                   (Maths.const pred)
                   ~vtgt:(Maths.tangent_exn pred))
          in
          match accu with
          | None -> Some (delta_ell, delta_ggn)
          | Some accu ->
            let ell_accu, ggn_accu = accu in
            Some (Maths.(ell_accu + delta_ell), Maths.C.(ggn_accu + delta_ggn))
        in
        accu, z)
    in
    Option.value_exn result

  let simulate ~data:(inputs, _) (theta : _ Maths.some P.t) =
    let[@warning "-8"] [ n_steps; bs; n_bits ] = Tensor.shape inputs in
    let z0 =
      Tensor.(f 0.1 * randn ~device:base.device [ bs; n ]) |> Maths.of_tensor |> Maths.any
    in
    let result, _ =
      List.fold (List.range 0 n_steps) ~init:([], z0) ~f:(fun (accu, z) t ->
        Stdlib.Gc.major ();
        let input =
          Tensor.slice inputs ~step:1 ~dim:0 ~start:(Some t) ~end_:(Some (t + 1))
          |> Tensor.squeeze
        in
        let z = forward ~theta ~input z in
        let pred = Maths.(phi z *@ theta.w) |> Maths.to_tensor in
        let accu = pred :: accu in
        accu, z)
    in
    List.rev_map result ~f:(fun x -> Tensor.reshape x ~shape:[ 1; bs; n_bits ])
    |> Tensor.concatenate ~dim:0
    |> Tensor.to_bigarray ~kind:base.ba_kind
end

(* -----------------------------------------
   -- Generate Flipflop data            ------
   ----------------------------------------- *)

open Bit_flipflop_common

let sample_batch_train =
  sample_batch
    ~pulse_prob:Settings.pulse_prob
    ~pulse_duration:Settings.pulse_duration
    ~pulse_refr:Settings.pulse_refr
    ~n_steps:Settings.n_steps
    ~b:Settings.b
    ~device:base.device

let sim_traj theta_prev =
  let data = sample_batch_train batch_size in
  let network = RNN.simulate ~data theta_prev in
  let first_trial x = x |> Arr.get_slice [ []; [ 0 ] ] |> Arr.squeeze in
  let targets = snd data |> Tensor.to_bigarray ~kind:base.ba_kind in
  let inputs = fst data |> Tensor.to_bigarray ~kind:base.ba_kind in
  Mat.save_txt (first_trial network) ~out:(in_dir "network");
  Mat.save_txt (first_trial targets) ~out:(in_dir "targets");
  Mat.save_txt (first_trial inputs) ~out:(in_dir "inputs");
  let err = Mat.(mean' (sqr (network - targets))) in
  Mat.(save_txt ~append:true ~out:(in_dir "true_err") (create 1 1 err))

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
let _K = 128

module RNN_Spec = struct
  type param_name =
    | J
    | B
    | W
  [@@deriving compare, sexp]

  let all = [ J; B; W ]

  let shape = function
    | J -> [ Int.(n + 1); n ]
    | B -> [ Settings.b; n ]
    | W -> [ n; Settings.b ]

  let n_params = function
    | J -> 50
    | B -> 50
    | W -> Int.(_K - 100)

  let n_params_list = [ 50; 50; Int.(_K - 100) ]
  let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
end

module RNN_Aux = struct
  type 'a p =
    { j_left : 'a
    ; j_right : 'a
    ; b_left : 'a
    ; b_right : 'a
    ; w_left : 'a
    ; w_right : 'a
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
      | RNN_Spec.J -> 0
      | RNN_Spec.B -> 1
      | RNN_Spec.W -> 2
    in
    List.nth_exn n_params_prefix_suffix_sums param_idx

  let g12v ~(lambda : ([< `const | `dual ] as 'a) A.t) (v : 'a P.t) : Maths.any P.t =
    let open Maths in
    let tmp_einsum left right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    let j = tmp_einsum lambda.j_left lambda.j_right v.j in
    let b = tmp_einsum lambda.b_left lambda.b_right v.b in
    let w = tmp_einsum lambda.w_left lambda.w_right v.w in
    { j; b; w; fb }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~(param_name : RNN_Spec.param_name) ~n_per_param v =
    let zero name =
      Tensor.zeros ~device:base.device ~kind:base.kind (n_per_param :: RNN_Spec.shape name)
    in
    let params_tmp = RNN_P.{ j = zero J; b = zero B; w = zero W; fb } in
    match param_name with
    | J -> { params_tmp with j = v }
    | B -> { params_tmp with b = v }
    | W -> { params_tmp with w = v }

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
      { j = random_localised_param_name J
      ; b = random_localised_param_name B
      ; w = random_localised_param_name W
      ; fb
      }

  (* compute sorted eigenvalues, u_left and u_right. *)
  let eigenvectors_for_params ~(lambda : ([< `const | `dual ] as 'a) A.t) ~param_name =
    let left, right =
      match param_name with
      | RNN_Spec.J -> lambda.j_left, lambda.j_right
      | RNN_Spec.B -> lambda.b_left, lambda.b_right
      | RNN_Spec.W -> lambda.w_left, lambda.w_right
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
      { j_left = init_eye Int.(n + 1)
      ; j_right = init_eye n
      ; b_left = init_eye Settings.b
      ; b_right = init_eye n
      ; w_left = init_eye n
      ; w_right = init_eye Settings.b
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
    ; learning_rate = Some 1.
    ; n_tangents = _K
    ; damping = `relative_from_top 1e-5
    ; aux = Some aux
    }

let rec sofo_loop ~t ~out ~state running_avg =
  Stdlib.Gc.major ();
  let data = sample_batch_train batch_size in
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
      (* simulate trajectory *)
      sim_traj theta;
      print [%message (t : int) (loss_avg : float)];
      Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss_avg |] 1 2)));
    []
  in
  if t < max_iter then sofo_loop ~t:Int.(t + 1) ~out ~state:new_state (loss :: running_avg)

(* Start the sofo loop. *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  sofo_loop ~t:0 ~out ~state:(O.init (RNN.init ())) []
