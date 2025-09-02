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

module RNN_Spec = struct
  type param_name =
    | C
    | B
    | O
  [@@deriving compare, sexp]

  let all = [ C; B; O ]

  let shape = function
    | C -> [ n; n ]
    | B -> [ 3; n ]
    | O -> [ n; 2 ]

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
  module C = Ggn_common.GGN_Common (RNN_Spec)

  let init_sampling_state () = 0

  let g12v ~(lambda : ([< `const | `dual ] as 'a) A.t) (v : 'a P.t) : Maths.any P.t =
    let open Maths in
    let tmp_einsum left right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    let c = tmp_einsum lambda.c_left lambda.c_right v.c in
    let b = tmp_einsum lambda.b_left lambda.b_right v.b in
    let o = tmp_einsum lambda.o_left lambda.o_right v.o in
    { c; b; o }

  let get_sides (lambda : ([< `const | `dual ] as 'a) A.t) = function
    | RNN_Spec.C -> lambda.c_left, lambda.c_right
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
      { c = random_localised_param_name C
      ; b = random_localised_param_name B
      ; o = random_localised_param_name O
      }

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
