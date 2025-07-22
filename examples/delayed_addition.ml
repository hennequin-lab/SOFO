(** Learning a delayed addition task to compare SOFO with adam. *)

open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let batch_size = 128
let n_tangents = batch_size

let _ =
  Random.init 1999;
  (* Random.self_init (); *)
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default
let tan_from_act = Option.value (Cmdargs.get_bool "-tan_from_act") ~default:false

module Settings = struct
  (* length of data *)
  let n_steps = 128 (* 20 to 600 *)

  (* first signal upper bound *)
  let t1_bound = 10
  let t2_bound = Int.(n_steps / 2)
end

(* for w of shape [n_in, n_out], dw_i = eps_i activation_i *)
let sample_rand_tensor_activation ~k ~param_shape ~activation ~weight =
  let activation_sliced =
    Tensor.slice ~dim:0 ~start:(Some 0) ~end_:(Some k) ~step:1 activation
  in
  let from_act =
    (* eps of shape [n_out_next x k], activation of shape [k x n_in], w of shape [n_out x n_out_next] *)
    let n_out_next = List.last_exn (Tensor.shape weight) in
    let eps = Tensor.randn ~device:base.device ~kind:base.kind [ n_out_next; k ] in
    Tensor.einsum ~path:None [ weight; eps; activation_sliced ] ~equation:"oa,ab,bi->bio"
  in
  if k < n_tangents
  then (
    let from_gauss =
      Tensor.randn ~device:base.device ~kind:base.kind ((n_tangents - k) :: param_shape)
    in
    Tensor.concat [ from_act; from_gauss ] ~dim:0)
  else from_act

(* net parameters *)
let n = 128 (* number of neurons *)
let alpha = 0.25

module RNN_P = struct
  type 'a p =
    { c : 'a
    ; b : 'a
    ; e : 'a
    ; o : 'a
    }
  [@@deriving prms]
end

(* neural network *)
module RNN = struct
  module P = RNN_P.Make (Prms.P)

  let phi x = Maths.relu x

  (* input is the (number, signal) pair and z is the internal state *)
  let forward ~(theta : P.M.t) ~input z =
    match z with
    | Some z ->
      let leak = Maths.(Float.(1. - alpha) $* z) in
      Maths.(leak + (alpha $* phi (theta.e + (z *@ theta.c) + (const input *@ theta.b))))
    | None ->
      (* this is the equivalent of initialising at z = 0 *)
      Maths.(phi (theta.e + (const input *@ theta.b)))

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
        ~a:2
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
    let e = Tensor.zeros ~device:base.device [ 1; n ] |> Prms.free in
    { c; b; e; o }

  type data = Tensor.t * Tensor.t
  type args = unit

  (* here data is a list of (x_t, optional labels). labels is x_t. *)
  let f ~update ~data ~init ~args:() theta =
    let module L =
      Loss.MSE (struct
        let scaling_factor = 1.
      end)
    in
    let input_all, labels_all = data in
    let result, z_final =
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
    if tan_from_act
    then (
      let z = Option.value_exn z_final in
      (* TODO: final RNN state for c tangents and average input for b tangents. *)
      let new_tangents_c =
        sample_rand_tensor_activation
          ~k:n_tangents
          ~param_shape:(Maths.shape theta.c)
          ~activation:(Maths.primal z)
          ~weight:(Maths.primal theta.c)
      in
      let new_tangents_b =
        let input =
          Tensor.mean_dim ~dim:(Some [ 2 ]) ~keepdim:false ~dtype:base.kind input_all
        in
        sample_rand_tensor_activation
          ~k:n_tangents
          ~param_shape:(Maths.shape theta.b)
          ~activation:input
          ~weight:(Maths.primal theta.c)
      in
      let new_tangents_e =
        Tensor.randn
          ~device:base.device
          ~kind:base.kind
          (n_tangents :: Maths.shape theta.e)
      in
      let new_tangents_o =
        Tensor.(
          f Float.(1. / of_int n)
          * randn ~device:base.device ~kind:base.kind (n_tangents :: Maths.shape theta.o))
      in
      (* set new tangents for c and b *)
      (match snd theta.c with
       | Some (Deferred dc) -> Maths.Deferred.set_exn dc new_tangents_c
       | _ -> ());
      (match snd theta.b with
       | Some (Deferred db) -> Maths.Deferred.set_exn db new_tangents_b
       | _ -> ());
      (match snd theta.e with
       | Some (Deferred de) -> Maths.Deferred.set_exn de new_tangents_e
       | _ -> ());
      match snd theta.o with
      | Some (Deferred d_o) -> Maths.Deferred.set_exn d_o new_tangents_o
      | _ -> ());
    result
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
      let loss, new_state = O.step ~config:(config ~iter) ~state ~data ~args:() in
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
  module O = Optimizer.SOFO (RNN)

  let name = "sofo"

  let config ~iter:_ =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 0.1
      ; n_tangents
      ; rank_one = false
      ; damping = Some 1e-5
      ; momentum = None
      ; tan_from_act
      }

  let init = O.init ~config:(config ~iter:0) RNN.init
end

(* --------------------------------
           -- Adam
           --------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (RNN)

  let config ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 1e-5 }

  let init = O.init RNN.init
end

let _ =
  let max_iter = 10000 in
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
