(** Learning a b-bit flip-flop task as in (Sussillo, 2013) to compare SOFO with FORCE. Reservoir setting *)
open Utils

open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let batch_size = 32
let max_iter = 1000

let _ =
  Random.init 1999;
  (* Random.self_init (); *)
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default

(* for FORCE *)
let alpha = Cmdargs.(get_float "-alpha" |> force ~usage:"-alpha [float]")
let force = Option.value (Cmdargs.get_bool "-force") ~default:false
let _K = 64

module Settings = struct
  let b = 3
  let n_steps = 200
  let pulse_prob = 0.02
  let pulse_duration = 2
  let pulse_refr = 10
end

let n = 1000

module RNN_P = struct
  type 'a p =
    { j : Maths.const Prms.Single.t
    ; fb : Maths.const Prms.Single.t
    ; w : 'a
    ; b : Maths.const Prms.Single.t
    }
  [@@deriving prms]
end

module P = RNN_P.Make (Prms.Single)

(* net parameters *)
let g = 0.5

let j =
  Mat.(gaussian n n *$ Float.(g / sqrt (of_int n)))
  |> Tensor.of_bigarray ~device:base.device
  |> Maths.of_tensor

let fb =
  Mat.(uniform ~a:(-1.) ~b:1. Settings.b n)
  |> Tensor.of_bigarray ~device:base.device
  |> Maths.of_tensor

let b =
  Mat.(uniform ~a:(-1.) ~b:1. Settings.b n)
  |> Tensor.of_bigarray ~device:base.device
  |> Maths.of_tensor

(* neural network *)
module RNN = struct
  module P = P

  let tau = 10.

  let init () : P.param =
    let w =
      Mat.(zeros n Settings.b)
      |> Tensor.of_bigarray ~device:base.device
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    { j; fb; w; b }

  let phi = Maths.tanh

  let forward ~(theta : _ Maths.some P.t) ~input z =
    let input = Maths.(of_tensor input *@ theta.b) in
    let phi_z = phi z in
    let feedback = Maths.(phi_z *@ theta.w *@ theta.fb) in
    let dz = Maths.((neg z + (phi_z *@ theta.j) + feedback + input) / f tau) in
    Maths.(z + dz)

  let f ~data:(inputs, targets) (theta : _ Maths.some P.t) =
    let[@warning "-8"] [ n_steps; bs; _ ] = Tensor.shape inputs in
    let z0 =
      Tensor.(f 0.1 * randn ~device:base.device [ bs; n ]) |> Maths.of_tensor |> Maths.any
    in
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
          let delta_ell = Loss.mse ~output_dims:[ 1 ] Maths.(of_tensor target - pred) in
          let delta_ggn =
            Loss.mse_ggn
              ~output_dims:[ 1 ]
              (Maths.const pred)
              ~vtgt:(Maths.tangent_exn pred)
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
let n_per_param = _K

module RNN_Spec = struct
  type param_name = W [@@deriving compare, sexp]

  let all = [ W ]

  let shape = function
    | W -> [ n; Settings.b ]

  let n_params = function
    | W -> _K

  let n_params_list = [ _K ]
  let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
end

module RNN_Aux = struct
  type 'a p =
    { w_left : 'a
    ; w_right : 'a
    }
  [@@deriving prms]
end

module A = RNN_Aux.Make (Prms.Single)

module GGN : Auxiliary with module P = P = struct
  module P = P
  module A = A
  module C = Ggn_common.GGN_Common (RNN_Spec)

  let init_sampling_state () = 0

  let g12v ~(lambda : ([< `const | `dual ] as 'a) A.t) (v : 'a P.t) : Maths.any P.t =
    let open Maths in
    let tmp_einsum left right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    let w = tmp_einsum lambda.w_left lambda.w_right v.w in
    { j; fb; w; b }

  let get_sides (lambda : ([< `const | `dual ] as 'a) A.t) = function
    | RNN_Spec.W -> lambda.w_left, lambda.w_right

  let random_localised_vs () =
    RNN_P.
      { w =
          Maths.of_tensor
            (C.random_params
               ~shape:[ n; Settings.b ]
               ~device:base.device
               ~kind:base.kind
               _K)
      ; j
      ; fb
      ; b
      }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~param_name:_ ~n_per_param:_ v = RNN_P.{ j; fb; b; w = v }
  let combine x y = P.map2 x y ~f:(fun a b -> Tensor.concat ~dim:0 [ a; b ])
  let wrap = P.map ~f:Maths.of_tensor

  let eigenvectors
        ~(lambda : [< `const | `dual ] A.t)
        ~switch_to_learn
        (t : int)
        (_k : int)
    : Forward_torch.Maths.const P.elt P.p * int
    =
    C.eigenvectors
      ~lambda
      ~switch_to_learn
      ~sampling_state:t
      ~get_sides
      ~combine
      ~wrap
      ~localise

  let init () =
    let init_eye size =
      Maths.(0.1 $* eye ~device:base.device ~kind:base.kind size) |> Prms.Single.free
    in
    RNN_Aux.{ w_left = init_eye n; w_right = init_eye Settings.b }
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

let id_bs =
  Tensor.of_bigarray ~device:base.device Mat.(Float.(of_int batch_size) $* eye batch_size)

let force_one_run
      ?(update_weights = true)
      ~n_steps
      (inputs, targets)
      pmat
      z
      (theta : _ Maths.some RNN.P.t)
  =
  let theta, pmat, _, errors, dws, outputs =
    List.fold
      (List.range 0 n_steps)
      ~init:(theta, pmat, z, [], [], [])
      ~f:(fun (theta, pmat, z, accu_err, accu_dw, accu_o) t ->
        Stdlib.Gc.major ();
        let inputs =
          Tensor.slice inputs ~step:1 ~dim:0 ~start:(Some t) ~end_:(Some (t + 1))
          |> Tensor.squeeze
        in
        let targets =
          Tensor.slice targets ~step:1 ~dim:0 ~start:(Some t) ~end_:(Some (t + 1))
          |> Tensor.squeeze
        in
        let r = RNN.phi z in
        let o = Maths.(r *@ theta.w) in
        let e = Maths.(o - of_tensor targets) |> Maths.to_tensor in
        (* update P *)
        let pmat =
          if not update_weights
          then pmat
          else (
            let inner =
              Tensor.(
                id_bs
                + einsum
                    [ Maths.to_tensor r; pmat; Maths.to_tensor r ]
                    ~path:None
                    ~equation:"ki,ij,lj->kl")
            in
            let u, s, _ = Tensor.svd inner ~some:true ~compute_uv:true in
            let tmp =
              Tensor.(einsum [ u; reciprocal_ (sqrt_ s) ] ~path:None ~equation:"ij,j->ij")
            in
            let half =
              Tensor.einsum
                [ pmat; Maths.to_tensor r; tmp ]
                ~path:None
                ~equation:"ij,kj,kl->il"
            in
            Tensor.(pmat - einsum [ half; half ] ~path:None ~equation:"ij,kj->ik"))
        in
        (* update the readout weights *)
        let theta, dw_norm =
          if not update_weights
          then theta, 0.
          else (
            let dw =
              Tensor.(
                einsum [ e; pmat; Maths.to_tensor r ] ~path:None ~equation:"ki,ab,kb->ai"
                / f Float.(of_int batch_size))
            in
            let dw_norm = Tensor.(norm dw |> to_float0_exn) in
            let w = Maths.(theta.w - of_tensor dw) in
            { theta with w }, dw_norm)
        in
        let z = RNN.forward ~theta ~input:inputs z in
        let e = Tensor.(mean (square e) |> to_float0_exn) in
        theta, pmat, z, e :: accu_err, dw_norm :: accu_dw, Maths.to_tensor o :: accu_o)
  in
  theta, pmat, errors, dws, outputs

let rec force_loop ~k (theta : _ Maths.some RNN.P.t) pmat =
  if k < max_iter
  then (
    let z =
      Tensor.(f 0.1 * randn ~device:base.device [ batch_size; n ])
      |> Maths.of_tensor
      |> Maths.any
    in
    let theta_prev = theta in
    let theta, pmat, _, dws, _ =
      force_one_run
        ~update_weights:true
        ~n_steps:Settings.n_steps
        (sample_batch_train batch_size)
        pmat
        z
        theta
    in
    let _, _, errors, _, _ =
      let n_steps = Settings.n_steps in
      let inputs, targets = sample_batch_train batch_size in
      force_one_run ~update_weights:false ~n_steps (inputs, targets) pmat z theta_prev
    in
    let errors = Mat.(of_array (List.rev errors |> Array.of_list) (-1) 1) in
    let dws = Mat.(of_array (List.rev dws |> Array.of_list) (-1) 1) in
    (* this is the error corresponding to the sofo loss *)
    let error = Mat.sum' errors in
    let e = Mat.(mean' (sqr errors)) in
    let dw = Mat.(mean' (sqr dws)) in
    print [%message (k : int) (error : float) (e : float) (dw : float)];
    if k % 10 = 0 then sim_traj theta_prev;
    Mat.(save_txt ~append:true ~out:(in_dir "info") (of_array [| e; dw |] 1 2));
    force_loop ~k:(k + 1) theta pmat)

(* Start the training loop *)
let _ =
  if force
  then (
    let pmat = Tensor.of_bigarray ~device:base.device Mat.(Float.(1. / alpha) $* eye n) in
    force_loop ~k:0 (RNN.P.any (RNN.P.value (RNN.init ()))) pmat)
  else (
    let out = in_dir "loss" in
    Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
    sofo_loop ~t:0 ~out ~state:(O.init (RNN.init ())) [])
