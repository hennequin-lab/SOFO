open Printf
open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let _ =
  Random.init 1999;
  (* Random.self_init (); *)
  Owl_stats_prng.init 1985;
  Torch_core.Wrapper.manual_seed 1985

let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run
let base = Optimizer.Config.Base.default

module PP = struct
  type 'a p =
    { w : 'a
    ; c : 'a
    ; a : 'a
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

(* neural network *)
module RNN = struct
  module P = P

  type input = unit

  let f ~(theta : P.M.t) ~input:_ y =
    let bs = Maths.shape y |> List.hd_exn in
    let y_tmp =
      Maths.concat
        y
        (Maths.const (Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ]))
        ~dim:1
    in
    Maths.((y *@ theta.a) + (relu (y_tmp *@ theta.c) *@ theta.w))

  let init ~d ~dh : P.tagged =
    let w =
      Convenience.gaussian_tensor_2d_normed
        ~kind:base.kind
        ~device:base.device
        ~a:dh
        ~b:d
        ~sigma:0.1
      |> Prms.free
    and c =
      Convenience.gaussian_tensor_2d_normed
        ~kind:base.kind
        ~device:base.device
        ~a:(d + 1)
        ~b:dh
        ~sigma:1.
      |> Prms.free
    and a =
      Tensor.(mul_scalar (eye ~n:d ~options:(base.kind, base.device)) (Scalar.f 0.9))
    in
    { w; c; a = Prms.free a }

  let simulate ~(theta : P.M.t) ~horizon y0 =
    let rec iter t accu y =
      if t = 0 then List.rev accu else iter (t - 1) (y :: accu) (f ~theta ~input:() y)
    in
    iter horizon [] (Maths.const y0)
end

(* feedforward model with mse loss *)
module FF =
  Wrapper.Recurrent
    (RNN)
    (Loss.MSE (struct
         let scaling_factor = 1.
       end))

(* -----------------------------------------
   -- Generate Lorenz data            ------
   ----------------------------------------- *)

open Lorenz_common

let d = 3
let dh = 400
let batch_size = 256
let num_epochs_to_run = 2000
let n_trials_simulation = 10
let train_data = data 32
let full_batch_size = Arr.(shape train_data).(1)
let train_data_batch = get_batch train_data
let test_horizon = 10000
let full_batch_size = Arr.(shape train_data).(1)
let _ = Convenience.print [%message (full_batch_size : int)]
let train_data_batch = get_batch train_data
let max_iter = Convenience.num_train_loops ~full_batch_size ~batch_size num_epochs_to_run
let epoch_of t = Convenience.epoch_of ~full_batch_size ~batch_size t

(* simulate n trials from saved parameters; first 3 columns are predictions and last 3 columns are ground truth *)
let simulate ~f_name n_trials =
  let model_params =
    let params_ba = FF.P.T.load (in_dir f_name ^ "_params") in
    RNN.P.map params_ba ~f:(fun x -> x |> Maths.const)
  in
  let n_list = List.range 0 n_trials in
  List.iter n_list ~f:(fun j ->
    (* ground truth obtained from integration *)
    let y_true = data_test (test_horizon - 1) in
    (* use same initial condition to simulate with model *)
    let init_cond_sim = Mat.get_slice [ [ 0 ]; [] ] y_true in
    let simulated_arr =
      RNN.simulate
        ~theta:model_params
        ~horizon:test_horizon
        Tensor.(of_bigarray ~device:base.device init_cond_sim)
      |> List.map ~f:(fun yt ->
        let yt = Maths.primal yt in
        let yt = Tensor.to_bigarray ~kind:base.ba_kind yt in
        Arr.expand yt 3)
      |> Array.of_list
      |> Arr.concatenate ~axis:0
      |> Arr.transpose ~axis:[| 1; 0; 2 |]
    in
    simulated_arr
    |> Arr.iter_slice ~axis:0 (fun yi ->
      let yi = Arr.squeeze yi in
      let yi_tot = Mat.concat_horizontal yi y_true in
      Mat.save_txt ~out:(in_dir (sprintf "%s_autonomous%i" f_name j)) yi_tot))

let _K = 128

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
type param_name =
  | W
  | C
  | A

let n_params_w, n_params_c, n_params_a = 60, 60, Int.(_K - 120)

module GGN : Wrapper.Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { w_left : 'a
      ; w_right : 'a
      ; c_left : 'a
      ; c_right : 'a
      ; a_left : 'a
      ; a_right : 'a
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
    let w = tmp_einsum lambda.w_left lambda.w_right v.w in
    let c = tmp_einsum lambda.c_left lambda.c_right v.c in
    let a = tmp_einsum lambda.a_left lambda.a_right v.a in
    { w; c; a }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~local ~param_name ~n_per_param v =
    let sample = if local then zero_params else random_params in
    let w = sample ~shape:[ dh; d ] n_per_param in
    let c = sample ~shape:[ d + 1; dh ] n_per_param in
    let a = sample ~shape:[ d; d ] n_per_param in
    let params_tmp = PP.{ w; c; a } in
    match param_name with
    | W -> { params_tmp with w = v }
    | C -> { params_tmp with c = v }
    | A -> { params_tmp with a = v }

  let random_localised_vs _K : P.T.t =
    { w = random_params ~shape:[ dh; d ] _K
    ; c = random_params ~shape:[ d + 1; dh ] _K
    ; a = random_params ~shape:[ d; d ] _K
    }

  let eigenvectors_for_each_params ~local ~lambda ~param_name =
    let left, right, n_per_param =
      match param_name with
      | W -> lambda.w_left, lambda.w_right, n_params_w
      | C -> lambda.c_left, lambda.c_right, n_params_c
      | A -> lambda.a_left, lambda.a_right, n_params_a
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
    let param_names_list = [ W; C; A ] in
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
    { w_left = init_eye dh
    ; w_right = init_eye d
    ; c_left = init_eye (d + 1)
    ; c_right = init_eye dh
    ; a_left = init_eye d
    ; a_right = init_eye d
    }
end

(* --------------------------------
   -- Generic type of optimiser
   -------------------------------- *)

module type Do_with_T = sig
  module O :
    Optimizer.T
    with type 'a W.P.p = 'a FF.P.p
     and type W.data = (unit * FF.args option) list
     and type W.args = FF.args

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
      let init_cond, data =
        let trajectory = train_data_batch batch_size in
        let init_cond = List.hd_exn trajectory in
        ( Tensor.of_bigarray ~device:base.device init_cond
        , List.mapi trajectory ~f:(fun tt x ->
            (* only label provided is the end point *)
            if tt = 31
            then (
              let x = Tensor.of_bigarray ~device:base.device x in
              (), Some x)
            else (), None) )
      in
      let t0 = Unix.gettimeofday () in
      let loss, new_state = O.step ~config:(config ~iter) ~state ~data init_cond in
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
  module O = Optimizer.SOFO (FF) (GGN)

  let name = "sofo"

  let config ~iter:_ =
    let aux =
      Optimizer.Config.SOFO.
        { (default_aux (in_dir "aux")) with
          config =
            Optimizer.Config.Adam.
              { default with base; learning_rate = Some 1e-3; eps = 1e-8 }
        ; steps = 3
        }
    in
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 0.3
      ; n_tangents = _K
      ; rank_one = false
      ; damping = Some 1e-5
      ; aux = Some aux
      }

  let init = O.init (RNN.init ~d ~dh)
end

(* --------------------------------
     -- Adam
     --------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (FF)

  let config ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.0001 }

  let init = O.init (RNN.init ~d ~dh)
end

let _ =
  let max_iter = 100000 in
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
