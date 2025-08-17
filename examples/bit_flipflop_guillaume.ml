(** Learning a b-bit flip-flop task as in (Sussillo, 2013) to compare SOFO with FORCE. *)

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
    { j : Maths.t
    ; fb : Maths.t
    ; w : 'a
    ; b : Maths.t
    }
  [@@deriving prms]
end

module P = RNN_P.Make (Prms.P)

(* net parameters *)
let g = 0.5

let j =
  Mat.(gaussian n n *$ Float.(g / sqrt (of_int n)))
  |> Tensor.of_bigarray ~device:base.device
  |> Maths.const

let fb =
  Mat.(uniform ~a:(-1.) ~b:1. Settings.b n)
  |> Tensor.of_bigarray ~device:base.device
  |> Maths.const

let b =
  Mat.(uniform ~a:(-1.) ~b:1. Settings.b n)
  |> Tensor.of_bigarray ~device:base.device
  |> Maths.const

(* neural network *)
module RNN = struct
  module P = P

  let tau = 10.

  let init () : P.tagged =
    let w =
      Mat.(zeros n Settings.b) |> Tensor.of_bigarray ~device:base.device |> Prms.free
    in
    { j; fb; w; b }

  let phi = Maths.tanh

  let forward ~(theta : P.M.t) ~input z =
    let input = Maths.(const input *@ theta.b) in
    let phi_z = phi z in
    let feedback = Maths.(phi_z *@ theta.w *@ theta.fb) in
    let dz = Maths.((neg z + (phi_z *@ theta.j) + feedback + input) /$ tau) in
    Maths.(z + dz)

  type data = Tensor.t * Tensor.t
  type args = unit

  let f ~update ~data:(inputs, targets) ~init ~args:() (theta : P.M.t) =
    let[@warning "-8"] [ n_steps; bs; _ ] = Tensor.shape inputs in
    let module L =
      Loss.MSE (struct
        let scaling_factor = Float.(1. / of_int n_steps)
      end)
    in
    let z0 = Tensor.(f 0.1 * randn ~device:base.device [ bs; n ]) |> Maths.const in
    let result, _ =
      List.fold (List.range 0 n_steps) ~init:(init, z0) ~f:(fun (accu, z) t ->
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
          let reduce_dim_list = Convenience.all_dims_but_first (Maths.primal pred) in
          let ell = L.f ~labels:target ~reduce_dim_list pred in
          match update with
          | `loss_only u -> u accu (Some ell)
          | `loss_and_ggn u ->
            let delta_ggn =
              let vtgt = Maths.tangent pred |> Option.value_exn in
              L.vtgt_hessian_gv ~labels:targets ~vtgt ~reduce_dim_list pred
            in
            u accu (Some (ell, Some delta_ggn))
        in
        accu, z)
    in
    result

  let simulate ~data:(inputs, _) (theta : P.M.t) =
    let[@warning "-8"] [ n_steps; bs; n_bits ] = Tensor.shape inputs in
    let z0 = Tensor.(f 0.1 * randn ~device:base.device [ bs; n ]) |> Maths.const in
    let result, _ =
      List.fold (List.range 0 n_steps) ~init:([], z0) ~f:(fun (accu, z) t ->
        Stdlib.Gc.major ();
        let input =
          Tensor.slice inputs ~step:1 ~dim:0 ~start:(Some t) ~end_:(Some (t + 1))
          |> Tensor.squeeze
        in
        let z = forward ~theta ~input z in
        let pred = Maths.(phi z *@ theta.w) |> Maths.primal in
        let accu = pred :: accu in
        accu, z)
    in
    List.rev_map result ~f:(fun x -> Tensor.reshape x ~shape:[ 1; bs; n_bits ])
    |> Tensor.concatenate ~dim:0
    |> Tensor.to_bigarray ~kind:base.ba_kind
end

(* -----------------------------------------
   -- Generate fliplop data          ------
   ----------------------------------------- *)

type pulse_state =
  { input : [ `off | `on of float * int | `refr of int ]
  ; output : float
  }

let sample_one_bit n_steps =
  let rec iter k state accu =
    if k = n_steps
    then
      List.rev_map accu ~f:(fun s ->
        ( (match s.input with
           | `on (s, _) -> s
           | _ -> 0.)
        , s.output ))
    else (
      let state =
        match state.input with
        | `off ->
          if Float.(Random.float 1. < Settings.pulse_prob)
          then (
            let s = if Random.bool () then 1. else -1. in
            { state with input = `on (s, 0) })
          else state
        | `on (s, 0) ->
          (* target output is set for the NEXT time step so
             the RNN has a chance to recurrently integrate the input *)
          { input = `on (s, 1); output = s }
        | `on (_, d) when d = Settings.pulse_duration ->
          (* enter a refractory state at the end of the pulse *)
          { state with input = `refr Settings.pulse_refr }
        | `on (s, d) -> { state with input = `on (s, d + 1) }
        | `refr 0 -> { state with input = `off }
        | `refr r -> { state with input = `refr (r - 1) }
      in
      iter (k + 1) state (state :: accu))
  in
  iter 0 { input = `off; output = 0. } []

(* returns time x bs x bits *)
let sample_batch ~n_steps bs =
  let data =
    List.init bs ~f:(fun _ -> List.init Settings.b ~f:(fun _ -> sample_one_bit n_steps))
  in
  let massage extract =
    data
    |> List.map ~f:(fun a ->
      List.map a ~f:(fun b ->
        let b = List.map b ~f:extract in
        Mat.of_array (Array.of_list b) 1 (-1))
      |> Array.of_list
      |> Mat.concatenate ~axis:0
      |> fun x -> Arr.expand x 3)
    |> Array.of_list
    |> Mat.concatenate ~axis:0
    |> Arr.transpose ~axis:[| 2; 0; 1 |]
  in
  let inputs = massage fst
  and outputs = massage snd in
  ( Tensor.of_bigarray ~device:base.device inputs
  , Tensor.of_bigarray ~device:base.device outputs )

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)

let n_per_param = _K
let cycle = true

module GGN : Wrapper.Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { w_left : 'a
      ; w_right : 'a
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

  (* approximation defined implicitly via Gv products *)
  let g12v ~(lambda : A.M.t) (v : P.M.t) : P.M.t =
    let open Maths in
    let tmp_einsum left right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    let w = tmp_einsum lambda.w_left lambda.w_right v.w in
    { j; fb; w; b }

  let localise v = RNN_P.{ j; fb; b; w = v }

  let random_localised_vs () : P.T.t =
    { w = random_params ~shape:[ n; Settings.b ] _K; j; fb; b }

  (* compute sorted eigenvalues, u_left and u_right. *)
  let eigenvectors_for_params ~lambda =
    let left, right = lambda.w_left, lambda.w_right in
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
  let s_u_cache = ref None

  (* given param name, get eigenvalues and eigenvectors. *)
  let get_s_u ~lambda =
    (* match List.Assoc.find !s_u_cache param_name ~equal:equal_param_name with *)
    match !s_u_cache with
    | Some s -> s
    | None ->
      let s = eigenvectors_for_params ~lambda in
      s_u_cache := Some s;
      s

  let extract_local_vs ~s_all ~u_left ~u_right ~selection =
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
    localise local_vs

  let eigenvectors_for_each_param ~lambda ~sampling_state =
    let n_per_param = n_per_param in
    let n_params = n * Settings.b in
    let s_all, u_left, u_right = get_s_u ~lambda in
    let selection =
      if cycle
      then
        List.init n_per_param ~f:(fun i ->
          ((sampling_state * n_per_param) + i) % n_params)
      else List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    extract_local_vs ~s_all ~u_left ~u_right ~selection

  let eigenvectors ~(lambda : A.M.t) ~switch_to_learn sampling_state (_K : int) =
    let eigenvectors_w = eigenvectors_for_each_param ~lambda ~sampling_state in
    let vs = eigenvectors_w in
    (* reset s_u_cache and set sampling state to 0 if learn again *)
    if switch_to_learn then s_u_cache := None;
    let new_sampling_state = if switch_to_learn then 0 else sampling_state + 1 in
    vs, new_sampling_state

  let init () =
    let init_eye size =
      Mat.(0.1 $* eye size) |> Tensor.of_bigarray ~device:base.device |> Prms.free
    in
    { w_left = init_eye n; w_right = init_eye Settings.b }
end

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
      (* let n_params =
        let params = RNN.P.value (O.params state) in
        O.W.P.T.numel params
      in *)
      let t = iter in
      let data = sample_batch ~n_steps:Settings.n_steps batch_size in
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
              { default with base; learning_rate = Some 1e-3; eps = 1e-8 }
        ; steps = 50
        ; learn_steps = 100
        ; exploit_steps = 100
        ; local = true
        }
    in
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 0.5
      ; n_tangents = _K
      ; rank_one = false
      ; damping = Some 1e-5
      ; aux = Some aux
      ; orthogonalize = false
      }

  let init = O.init (RNN.init ())
end

(* --------------------------------
       -- Adam
       --------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (RNN)

  let config ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.0001 }

  let init = O.init (RNN.init ())
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

(* let _ =
    let f_name = "sofo" in
    simulate ~f_name n_trials_simulation *)
