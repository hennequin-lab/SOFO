(** model recovery of RNN *)

open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Mat = Owl.Dense.Matrix.S
module Linalg = Owl.Linalg.S

let batch_size = 256
let max_iter = 20000

let _ =
  Random.init 1999;
  (* Random.self_init (); *)
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default
let n_batches = 20
let tmax = 100
let n = 128
let m = 6
let o = 3
let alpha = 0.25

module RNN_P = struct
  type 'a p =
    { w : 'a
    ; b : 'a
    ; o : 'a
    }
  [@@deriving prms]
end

let true_theta =
  let w =
    Sofo.gaussian_tensor_normed ~kind:base.kind ~device:base.device ~sigma:1.5 [ n; n ]
    |> Maths.of_tensor
  in
  let b =
    Sofo.gaussian_tensor_normed
      ~kind:base.kind
      ~device:base.device
      ~sigma:1.0
      [ m + 1; n ]
    |> Maths.of_tensor
  in
  (* initialise to repeat observation *)
  let o =
    Sofo.gaussian_tensor_normed ~kind:base.kind ~device:base.device ~sigma:1.0 [ n; o ]
    |> Maths.of_tensor
  in
  RNN_P.{ w; b; o }

module P = RNN_P.Make (Prms.Single)

(* neural network *)
module RNN = struct
  module P = P

  let phi x = Maths.tanh x

  let forward ~(theta : _ Maths.some P.t) ~input z =
    let bs = Maths.shape input |> List.hd_exn in
    let input =
      Maths.C.concat
        [ input
        ; Maths.of_tensor (Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ])
        ]
        ~dim:1
    in
    match z with
    | Some z ->
      let leak = Maths.(Float.(1. - alpha) $* z) in
      Maths.(leak + (alpha $* phi (z *@ theta.w) + (input *@ theta.b)))
    | None ->
      (* this is the equivalent of initialising at z = 0 *)
      Maths.(input *@ theta.b)

  let prediction ~(theta : _ Maths.some P.t) z = Maths.(z *@ theta.o)

  let init : P.param =
    let w =
      Sofo.gaussian_tensor_normed ~kind:base.kind ~device:base.device ~sigma:1.0 [ n; n ]
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    let b =
      Sofo.gaussian_tensor_normed
        ~kind:base.kind
        ~device:base.device
        ~sigma:1.0
        [ m + 1; n ]
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    (* initialise to repeat observation *)
    let o =
      Sofo.gaussian_tensor_normed ~kind:base.kind ~device:base.device ~sigma:1.0 [ n; o ]
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    { w; b; o }

  (* here data is a list of (u_t, y_t). *)
  let f_sofo ~data theta =
    let result, _ =
      List.fold data ~init:(None, None) ~f:(fun (accu, z) data ->
        let input, label = data in
        let z = forward ~theta ~input z in
        let accu =
          let pred = prediction ~theta z in
          let delta_ell = Loss.mse ~output_dims:[ 1 ] Maths.(label - pred) in
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
        accu, Some z)
    in
    Option.value_exn result

  let f_adam ~data theta =
    let result, _ =
      List.fold data ~init:(None, None) ~f:(fun (accu, z) data ->
        let input, label = data in
        let z = forward ~theta ~input z in
        let accu =
          let pred = prediction ~theta z in
          let delta_ell = Loss.mse ~output_dims:[ 1 ] Maths.(label - pred) in
          match accu with
          | None -> Some delta_ell
          | Some accu -> Some Maths.(accu + delta_ell)
        in
        accu, Some z)
    in
    Option.value_exn result
end

(* -----------------------------------------
   -- Generate Data    ------
   ----------------------------------------- *)
(* data distribution generated using the teacher [true_theta] *)
let data_batches =
  let u, _, _ = Linalg.qr Mat.(gaussian m m) in
  let s = Mat.init 1 m (fun i -> Float.(exp (neg (of_int i / of_int 10)))) in
  let s = Mat.(s /$ mean' s) in
  let sigma12 = Mat.(transpose (u * sqrt s)) |> Maths.of_bigarray ~device:base.device in
  Array.init n_batches ~f:(fun _ ->
    let data_u =
      List.init tmax ~f:(fun _ ->
        Maths.randn ~device:base.device ~kind:base.kind [ batch_size; m ]
        |> fun x -> Maths.(const (x *@ const sigma12)))
    in
    let _, data_o =
      List.fold_map data_u ~init:None ~f:(fun z u ->
        let z_new = RNN.forward ~theta:true_theta ~input:u z in
        let o = RNN.prediction ~theta:true_theta z_new in
        Some z_new, Maths.const o)
    in
    List.map2_exn data_u data_o ~f:(fun u o -> u, o))

let sample_data () = data_batches.(Random.int n_batches)

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
let _K = 128

module RNN_Spec = struct
  type param_name =
    | W
    | B
    | O
  [@@deriving compare, sexp]

  let all = [ W; B; O ]

  let shape = function
    | W -> [ n; n ]
    | B -> [ m + 1; n ]
    | O -> [ n; o ]

  let n_params = function
    | W -> Int.(_K - 4)
    | B -> 2
    | O -> 2

  let n_params_list = [ Int.(_K - 4); 2; 2 ]
  let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
end

module RNN_Aux = struct
  type 'a p =
    { w_left : 'a
    ; w_right : 'a
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
  module C = Ggn_common.GGN_Common (RNN_Spec)

  let init_sampling_state () = 0

  let g12v ~(lambda : ([< `const | `dual ] as 'a) A.t) (v : 'a P.t) : Maths.any P.t =
    let open Maths in
    let tmp_einsum left right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    let w = tmp_einsum lambda.w_left lambda.w_right v.w in
    let b = tmp_einsum lambda.b_left lambda.b_right v.b in
    let o = tmp_einsum lambda.o_left lambda.o_right v.o in
    { w; b; o }

  let get_sides (lambda : ([< `const | `dual ] as 'a) A.t) = function
    | RNN_Spec.W -> lambda.w_left, lambda.w_right
    | RNN_Spec.B -> lambda.b_left, lambda.b_right
    | RNN_Spec.O -> lambda.o_left, lambda.o_right

  let random_localised_vs () =
    let random_localised_param_name param_name =
      let w_shape = RNN_Spec.shape param_name in
      let before, after = C.get_n_params_before_after param_name in
      let w =
        C.random_params
          ~shape:w_shape
          ~device:base.device
          ~kind:base.kind
          (RNN_Spec.n_params param_name)
      in
      let zeros_before =
        C.zero_params ~shape:w_shape ~device:base.device ~kind:base.kind before
      in
      let zeros_after =
        C.zero_params ~shape:w_shape ~device:base.device ~kind:base.kind after
      in
      let final = Tensor.concat [ zeros_before; w; zeros_after ] ~dim:0 in
      Maths.of_tensor final
    in
    RNN_P.
      { w = random_localised_param_name W
      ; b = random_localised_param_name B
      ; o = random_localised_param_name O
      }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~(param_name : RNN_Spec.param_name) ~n_per_param v =
    let zero name =
      Tensor.zeros ~device:base.device ~kind:base.kind (n_per_param :: RNN_Spec.shape name)
    in
    let params_tmp = RNN_P.{ w = zero W; b = zero B; o = zero O } in
    match param_name with
    | W -> { params_tmp with w = v }
    | B -> { params_tmp with b = v }
    | O -> { params_tmp with o = v }

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
    RNN_Aux.
      { w_left = init_eye n
      ; w_right = init_eye n
      ; b_left = init_eye (m + 1)
      ; b_right = init_eye n
      ; o_left = init_eye n
      ; o_right = init_eye o
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

let rec loop ~t ~out ~state running_avg =
  Stdlib.Gc.major ();
  let data = sample_data () in
  let theta, tangents, new_sampling_state = O.prepare ~config state in
  let loss, ggn = RNN.f_sofo ~data theta in
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
  let out = in_dir "loss_sofo_aux" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init RNN.init) []

(* -----------------------------------------
   -- Optimization with Adam    ------
   ----------------------------------------- *)

(* module O = Optimizer.Adam (RNN.P)

let config =
  Optimizer.Config.Adam.
    { base
    ; beta_1 = 0.9
    ; beta_2 = 0.99
    ; eps = 1e-4
    ; learning_rate = Some Float.(0.001)
    ; weight_decay = None
    }

let rec loop ~t ~out ~state running_avg =
  Stdlib.Gc.major ();
  let theta = O.params state in
  let theta_ = O.P.value theta in
  let theta_dual =
    O.P.map theta_ ~f:(fun x ->
      let x =
        x |> Maths.to_tensor |> Tensor.copy |> Tensor.to_device ~device:base.device
      in
      let x = Tensor.set_requires_grad x ~r:true in
      Tensor.zero_grad x;
      Maths.of_tensor x)
  in
  let data = sample_data () in
  let loss, true_g =
    let loss = RNN.f_adam ~data (P.map theta_dual ~f:Maths.any) in
    let loss = Maths.to_tensor loss in
    Tensor.backward loss;
    ( Tensor.to_float0_exn loss
    , O.P.map2 theta (O.P.to_tensor theta_dual) ~f:(fun tagged p ->
        match tagged with
        | Prms.Pinned _ -> Maths.(f 0.)
        | _ -> Maths.of_tensor (Tensor.grad p)) )
  in
  let new_state = O.step ~config ~info:true_g state in
  let running_avg =
    let loss_avg =
      match running_avg with
      | [] -> loss
      | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
    in
    if t % 10 = 0
    then (
      (* save params *)
      print [%message (t : int) (loss_avg : float)];
      Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss_avg |] 1 2)));
    []
  in
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state (loss :: running_avg)

(* Start the loop. *)
let _ =
  let out = in_dir "loss_adam" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init RNN.init) [] *)
