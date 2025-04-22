(** Learning a delayed addition task to compare SOFO with adam. *)

open Printf
open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let batch_size = 256
let max_iter = 10000

let _ =
  Random.init 1999;
  (* Random.self_init (); *)
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default

module Settings = struct
  (* length of data *)
  let n_steps = 128 (* 20 to 600 *)

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

module P = RNN_P.Make (Prms.P)

(* neural network *)
module RNN = struct
  module P = P

  let phi x = Maths.relu x

  (* input is the (number, signal) pair and z is the internal state *)
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
      let leak = Maths.(Float.(1. - alpha) $* z) in
      Maths.(leak + (alpha $* phi ((z *@ theta.c) + (input *@ theta.b))))
    | None ->
      (* this is the equivalent of initialising at z = 0 *)
      Maths.(phi (input *@ theta.b))

  let prediction ~(theta : P.M.t) z = Maths.(z *@ theta.o)

  let init : P.tagged =
    let c =
      Convenience.gaussian_tensor_2d_normed
        ~kind:base.kind
        ~device:base.device
        ~a:n
        ~b:n
        ~sigma:1.0
      |> Prms.free
    in
    let b =
      Convenience.gaussian_tensor_2d_normed
        ~kind:base.kind
        ~device:base.device
        ~a:3
        ~b:n
        ~sigma:1.0
      |> Prms.free
    in
    (* initialise to repeat observation *)
    let o =
      Convenience.gaussian_tensor_2d_normed
        ~kind:base.kind
        ~device:base.device
        ~a:n
        ~b:2
        ~sigma:1.0
      |> Prms.free
    in
    { c; b; o }

  type data = Tensor.t * Tensor.t
  type args = unit

  (* here data is a list of (x_t, optional labels). labels is x_t. *)
  let f ~update ~data ~init ~args:() theta =
    let module L =
      Loss.MSE (struct
        let scaling_factor = 1.
      end)
    in
    let result, _ =
      let input_all, labels_all = data in
      let top_2, _ = List.split_n (Tensor.shape input_all) 2 in
      let time_list = List.range 0 Settings.n_steps in
      List.fold time_list ~init:(init, None) ~f:(fun (accu, z) t ->
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
        let pred = prediction ~theta z in
        let accu =
          match labels with
          | None -> accu
          | Some labels ->
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
end

let _K = 128
(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)

type param_name =
  | C
  | B
  | O

let n_params_c, n_params_b, n_params_e, n_params_o = Int.(_K - 6), 2, 2, 2

module GGN : Wrapper.Auxiliary with module P = P = struct
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
  module A = Make (Prms.P)

  type sampling_state = unit

  let init_sampling_state () = ()

  let zero_params ~shape _K =
    Tensor.zeros ~device:base.device ~kind:base.kind (_K :: shape)

  let random_params ~shape _K =
    Tensor.randn ~device:base.device ~kind:base.kind (_K :: shape)

  (* approximation defined implicitly via Gv products *)
  let g12v ~(lambda : A.M.t) (v : P.M.t) : P.M.t =
    let open Maths in
    let tmp_einsum left right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    let c = tmp_einsum lambda.c_left lambda.c_right v.c in
    let b = tmp_einsum lambda.b_left lambda.b_right v.b in
    let o = tmp_einsum lambda.o_left lambda.o_right v.o in
    { c; b;  o }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~local ~param_name ~n_per_param v =
    let sample = if local then zero_params else random_params in
    let c = sample ~shape:[ n; n ] n_per_param in
    let b = sample ~shape:[ 3; n ] n_per_param in
    let o = sample ~shape:[ n; 2 ] n_per_param in
    let params_tmp = RNN_P.{ c; b;  o } in
    match param_name with
    | C -> { params_tmp with c = v }
    | B -> { params_tmp with b = v }
    | O -> { params_tmp with o = v }

  let random_localised_vs _K : P.T.t =
    { c = random_params ~shape:[ n; n ] _K
    ; b = random_params ~shape:[ 3; n ] _K
    ; o = random_params ~shape:[ n; 2 ] _K
    }

  let eigenvectors_for_each_params ~local ~lambda ~param_name =
    let left, right, n_per_param =
      match param_name with
      | C -> lambda.c_left, lambda.c_right, n_params_c
      | B -> lambda.b_left, lambda.b_right, n_params_b
      | O -> lambda.o_left, lambda.o_right, n_params_o
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
    (* randomly select the indices *)
    let n_params =
      Convenience.first_dim (Maths.primal left)
      * Convenience.first_dim (Maths.primal right)
    in
    let selection =
      List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    let selection = List.map selection ~f:(fun j -> s_all.(j)) in
    let local_vs =
      List.map selection ~f:(fun (il, ir, _) ->
        let u_left =
          Tensor.(
            squeeze_dim
              ~dim:1
              (slice u_left ~dim:1 ~start:(Some il) ~end_:(Some Int.(il + 1)) ~step:1))
        in
        let u_right =
          Tensor.(
            squeeze_dim
              ~dim:1
              (slice u_right ~dim:1 ~start:(Some ir) ~end_:(Some Int.(ir + 1)) ~step:1))
        in
        let tmp = Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_left; u_right ] in
        Tensor.unsqueeze tmp ~dim:0)
      |> Tensor.concatenate ~dim:0
    in
    local_vs |> localise ~local ~param_name ~n_per_param

  let eigenvectors ~(lambda : A.M.t) () (_K : int) =
    let param_names_list = [ C; B;  O ] in
    let eigenvectors_each =
      List.map param_names_list ~f:(fun param_name ->
        eigenvectors_for_each_params ~local:true ~lambda ~param_name)
    in
    let vs =
      List.fold eigenvectors_each ~init:None ~f:(fun accu local_vs ->
        match accu with
        | None -> Some local_vs
        | Some a -> Some (P.map2 a local_vs ~f:(fun x y -> Tensor.concat ~dim:0 [ x; y ])))
    in
    Option.value_exn vs, ()

  let init () =
    let init_eye size =
      Mat.(0.1 $* eye size) |> Tensor.of_bigarray ~device:base.device |> Prms.free
    in
    { c_left = init_eye n
    ; c_right = init_eye n
    ; b_left = init_eye 3
    ; b_right = init_eye n
    ; o_left = init_eye n
    ; o_right = init_eye 2
    }
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

(* --------------------------------
   -- Generic type of optimiser
   -------------------------------- *)

module type Do_with_T = sig
  module O :
    Optimizer.T
    with type 'a W.P.p = 'a RNN.P.p
     and type W.data = Tensor.t * Tensor.t
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
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array [| Float.of_int t; time_elapsed; loss_avg |] 1 3)));
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
              { default with base; learning_rate = Some 1e-2; eps = 1e-8 }
        ; steps = 3
        }
    in
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 0.1
      ; n_tangents = _K
      ; rank_one = false
      ; damping = Some 1e-5
      ; aux = None
      }

  let init = O.init RNN.init
end

(* --------------------------------
           -- Adam
           --------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (RNN)

  let config ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 1e-4 }

  let init = O.init RNN.init
end

let _ =
  let max_iter = 2000 in
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
