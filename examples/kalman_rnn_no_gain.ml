(** Meta learning a kalman filter with a vanilla rnn. no gain. *)
open Base

open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

(* state dimension *)
let s = 1

module Data = Kalman_data.Make (Kalman_data.Default)

let t_max = 5000
let batch_size = 256
let n_trials_simulation = 500
let n_trials_baseline = 1000

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
   let data = minibatch ~t_max n_trials_baseline in
   (* save [x; y; b; kalman_x]*)
   let _ =
   Mat.save_txt ~out:(in_dir "example_session") (to_save ~random:false data.(0))
   in
   mse ~filter_fun:(kalman_filter ~random:false) data)
   |> Stats.mean

   let kalman_random_baseline =
   let open Data in
   Array.init 2 ~f:(fun _ ->
   let data = minibatch ~t_max n_trials_baseline in
   (* save [x; y; b; kalman_random_x]*)
   let _ =
   Mat.save_txt ~out:(in_dir "example_session_random") (to_save ~random:true data.(0))
   in
   mse ~filter_fun:(kalman_filter ~random:true) data)
   |> Stats.mean *)

let kalman_baseline = 0.4896232
let kalman_random_baseline = 1.3612788

module RNN_P = struct
  type 'a p =
    { c : 'a
    ; b : 'a
    ; o : 'a
    }
  [@@deriving prms]
end

module P = RNN_P.Make (Prms.P)

let r_vanilla = 100
let alpha = 0.25

(* neural network *)
module RNN = struct
  module P = P

  let phi x = Maths.relu x

  (* input is the noisy observation and z is the internal state *)
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
        ~a:r_vanilla
        ~b:r_vanilla
        ~sigma:1.0
      |> Prms.free
    in
    let b =
      Convenience.gaussian_tensor_2d_normed
        ~kind:base.kind
        ~device:base.device
        ~a:(s + 1)
        ~b:r_vanilla
        ~sigma:1.0
      |> Prms.free
    in
    (* initialise to repeat observation *)
    let o =
      let b =
        Prms.value b |> Tensor.slice ~dim:0 ~start:(Some 0) ~end_:(Some s) ~step:1
      in
      if s = 1
      then (
        let b2 = Tensor.(square_ (norm b)) in
        Tensor.(div_ (Convenience.trans_2d b) b2) |> Prms.free)
      else Tensor.pinverse b ~rcond:0. |> Prms.free
    in
    { c; b; o }

  type data = (Tensor.t * Tensor.t option) list
  type args = unit

  (* here data is a list of (x_t, optional labels). labels is x_t. *)
  let f ~update ~data ~init ~args:() theta =
    let module L =
      Loss.MSE (struct
        let scaling_factor = Float.(1. / of_int t_max)
      end)
    in
    let result, _ =
      List.foldi data ~init:(init, None) ~f:(fun t (accu, z) (input, labels) ->
        if t % 1 = 0 then Stdlib.Gc.major ();
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

  (* TODO: make sure the batch size is 1 here, otherwise nonsensical results. Simulate one sample trajectory *)
  let simulate_1d (data : (float, Kalman_data.lds) Kalman_data.state array) theta =
    Array.foldi data ~init:([], None) ~f:(fun t (accu, z) datum ->
      if t % 1 = 0 then Stdlib.Gc.major ();
      (* Convenience.print [%message ( a : int)]; *)
      let input =
        Tensor.of_bigarray
          ~device:base.device
          Mat.(of_array [| datum.Kalman_data.y |] 1 1)
      in
      let z = forward ~theta ~input z in
      let pred =
        prediction ~theta z |> Maths.primal |> fun x -> Tensor.get_float2 x 0 0
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
  Data.minibatch ~t_max bs
  |> Data.minibatch_as_data
  |> List.map ~f:(fun datum ->
    let to_device = Tensor.of_bigarray ~device:base.device in
    to_device datum.Kalman_data.y, Some (to_device datum.Kalman_data.x))

let simulate_1d ~f_name n_trials =
  let module Data = Kalman_data.Make (Kalman_data.Default) in
  let n_list = List.range 0 n_trials in
  List.iter n_list ~f:(fun i ->
    let data = (Data.minibatch ~t_max 1).(0) in
    (* 1d *)
    let kf_prediction = Mat.of_array (Data.kalman_filter ~random:false data) (-1) 1 in
    let kf_random_prediction =
      Mat.of_array (Data.kalman_filter ~random:true data) (-1) 1
    in
    let model_params =
      let params_ba = RNN.P.T.load (in_dir f_name ^ "_params") in
      RNN.P.map params_ba ~f:(fun x -> x |> Maths.const)
    in
    let model_prediction = RNN.simulate_1d data model_params in
    (* the columns are tau, b, beta, sigma_eps, observed_y, x, model prediction, kalman prediction and kalman random prediction *)
    let to_save = Mat.((model_prediction @|| kf_prediction) @|| kf_random_prediction) in
    Mat.(
      save_txt to_save ~out:(in_dir f_name ^ "_" ^ Int.to_string i ^ "_model_prediction")))

let _K = 128

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)

type param_name =
  | C
  | B
  | O
[@@deriving compare]

let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
let param_names_list = [ C; B; O ]
let n_params_c, n_params_b, n_params_o = Int.(_K - 4), 2, 2
let n_params_list = [ n_params_c; n_params_b; n_params_o ]

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

  type sampling_state = int

  let init_sampling_state () = 0

  let zero_params ~shape _K =
    Tensor.zeros ~device:base.device ~kind:base.kind (_K :: shape)

  let random_params ~shape _K =
    Tensor.randn ~device:base.device ~kind:base.kind (_K :: shape)

  let get_shapes (param_name : param_name) =
    match param_name with
    | C -> [ r_vanilla; r_vanilla ]
    | B -> [ s + 1; r_vanilla ]
    | O -> [ r_vanilla; s ]

  let get_n_params (param_name : param_name) =
    match param_name with
    | C -> n_params_c
    | B -> n_params_b
    | O -> n_params_o

  let get_total_n_params (param_name : param_name) =
    let list_prod l = List.fold l ~init:1 ~f:(fun accu i -> accu * i) in
    list_prod (get_shapes param_name)

  let get_n_params_before_after (param_name : param_name) =
    let n_params_prefix_suffix_sums = Convenience.prefix_suffix_sums n_params_list in
    let param_idx =
      match param_name with
      | C -> 0
      | B -> 1
      | O -> 2
    in
    List.nth_exn n_params_prefix_suffix_sums param_idx

  (* approximation defined implicitly via Gv products *)
  let g12v ~(lambda : A.M.t) (v : P.M.t) : P.M.t =
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

  let eigenvectors_for_each_param ~lambda ~param_name ~sampling_state ~cycle =
    let n_per_param = get_n_params param_name in
    let n_params = get_total_n_params param_name in
    let s_all, u_left, u_right =
      if cycle
      then get_s_u ~lambda ~param_name
      else eigenvectors_for_params ~lambda ~param_name
    in
    let selection =
      if cycle
      then
        List.init n_per_param ~f:(fun i ->
          ((sampling_state * n_per_param) + i) % n_params)
      else List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    extract_local_vs ~s_all ~param_name ~u_left ~u_right ~selection

  let eigenvectors ~(lambda : A.M.t) t (_K : int) =
    let eigenvectors_each =
      List.map param_names_list ~f:(fun param_name ->
        eigenvectors_for_each_param ~lambda ~param_name ~sampling_state:t ~cycle:true)
    in
    let vs =
      List.fold eigenvectors_each ~init:None ~f:(fun accu local_vs ->
        match accu with
        | None -> Some local_vs
        | Some a -> Some (P.map2 a local_vs ~f:(fun x y -> Tensor.concat ~dim:0 [ x; y ])))
    in
    Option.value_exn vs, t + 1

  let init () =
    let init_eye size =
      Mat.(0.1 $* eye size) |> Tensor.of_bigarray ~device:base.device |> Prms.free
    in
    { c_left = init_eye r_vanilla
    ; c_right = init_eye r_vanilla
    ; b_left = init_eye (s + 1)
    ; b_right = init_eye r_vanilla
    ; o_left = init_eye r_vanilla
    ; o_right = init_eye s
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
let learn_first_steps = 1000

module Do_with_SOFO : Do_with_T = struct
  module O = Optimizer.SOFO (RNN) (GGN)

  let name = "sofo"

  let config ~iter:_ =
    let aux =
      Optimizer.Config.SOFO.
        { (default_aux (in_dir "aux")) with
          config =
            Optimizer.Config.Adam.
              { default with base; learning_rate = Some 1e-3; eps = 1e-8 }
        ; steps = 3
        ; learn_first_steps = Some learn_first_steps
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

  let init = O.init RNN.init
end

(* --------------------------------
         -- Adam
         --------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (RNN)

  let config ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.001 }

  let init = O.init RNN.init
end

let _ =
  let max_iter = 2000 + learn_first_steps in
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

(* let _ =
      let f_name = "sofo" in
      simulate_1d ~f_name n_trials_simulation *)
