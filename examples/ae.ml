(* variational autoencoder on mnist *)
open Printf
open Base
open Forward_torch
open Torch
open Sofo
module Arr = Owl.Dense.Ndarray.D
module Mat = Owl.Dense.Matrix.D

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f64)
    ; ba_kind = Bigarray.float64
    }

let bs = 64
let full_batch_size = 60_000
let num_epochs_to_run = 5

let num_train_loops =
  Convenience.num_train_loops ~full_batch_size ~batch_size:bs num_epochs_to_run

let epoch_of t = Convenience.epoch_of ~full_batch_size ~batch_size:bs t
let input_dim = 28 * 28
let h_dim1 = 256
let latent_dim = 128
let layer_sizes = [| h_dim1; latent_dim; h_dim1; input_dim |]

module VAE = struct
  module MLP_Layer = struct
    type 'a t =
      { w : 'a
      ; b : 'a
      }
    [@@deriving prms]
  end

  module P = Prms.Array (MLP_Layer.Make (Prms.P))

  type data = Tensor.t
  type args = unit

  let phi = Maths.relu

  let ggn ~x_hat =
    let vtgt = Maths.tangent x_hat |> Option.value_exn in
    Tensor.einsum ~equation:"kma,jma->kj" [ vtgt; vtgt ] ~path:None

  let f ~update ~data:x ~init ~args:() (theta : P.M.t) =
    let module L =
      Loss.MSE (struct
        let scaling_factor = 1.
      end)
    in
    let pred =
      Array.foldi theta ~init:(Maths.const x) ~f:(fun i accu p ->
        let act_fun = if i = Int.(Array.length layer_sizes - 1) then 
          Maths.sigmoid else phi in
        act_fun Maths.((accu *@ p.w) + p.b))
    in
    let reduce_dim_list = Convenience.all_dims_but_first (Maths.primal pred) in
    let ell = L.f ~labels:x ~reduce_dim_list pred in
    match update with
    | `loss_only u -> u init (Some ell)
    | `loss_and_ggn u ->
      let ggn =
        let vtgt = Maths.tangent pred |> Option.value_exn in
        L.vtgt_hessian_gv ~labels:x ~vtgt ~reduce_dim_list pred
      in
      let _, final_s, _ = Tensor.svd ~some:true ~compute_uv:false ggn in
      final_s
      |> Tensor.reshape ~shape:[ -1; 1 ]
      |> Tensor.to_bigarray ~kind:base.ba_kind
      |> Owl.Mat.save_txt ~out:(in_dir (sprintf "svals"));
      u init (Some (ell, Some ggn))

  let init =
    let open MLP_Layer in
    Array.mapi layer_sizes ~f:(fun i n_o ->
      let n_i = if i = 0 then input_dim else layer_sizes.(i - 1) in
      let w =
        Convenience.gaussian_tensor_2d_normed
          ~kind:base.kind
          ~device:base.device
          ~a:n_i
          ~b:n_o
          ~sigma:1.
      in
      let b = Tensor.zeros ~kind:base.kind ~device:base.device [ 1; n_o ] in
      { w; b })
    |> P.map ~f:Prms.free
end

let dataset typ =
  let suffix =
    match typ with
    | `train -> "train"
    | `test -> "test"
  in
  let x = Owl.Arr.load_npy ("_data/x_" ^ suffix ^ ".npy") in
  let mu = 0.13062754273414612
  and sigma = 0.30810779333114624 in
  Owl.Arr.(((x /$ 255.) -$ mu) /$ sigma)

let train_set = dataset `train

let sample_data set_x =
  let a = Mat.row_num set_x in
  fun batch_size ->
    let ids = List.init batch_size ~f:(fun _ -> Random.int a) in
    let x_tensor =
      Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_x)
    in
    x_tensor

(* --------------------------------
   -- Generic type of optimiser
   -------------------------------- *)

module type Do_with_T = sig
  module O :
    Optimizer.T
    with type 'a W.P.p = 'a VAE.P.p
     and type W.data = Tensor.t
     and type W.args = unit

  val name : string
  val config_f : iter:int -> (float, Bigarray.float64_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state ~time_elapsed running_avg =
      Stdlib.Gc.major ();
      let data = sample_data train_set bs in
      let t0 = Unix.gettimeofday () in
      let config = config_f ~iter in
      let loss, new_state = O.step ~config ~state ~data ~args:() in
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
          (* avg error *)
          let t = epoch_of iter in
          Convenience.print [%message (t : float) (loss_avg : float)];
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array [| t; time_elapsed; loss_avg |] 1 3));
          O.W.P.T.save
            (VAE.P.value (O.params new_state))
            ~kind:base.ba_kind
            ~out:(in_dir name ^ "_params"));
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
  module O = Optimizer.SOFO (VAE)

  let config_f ~iter =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some Float.(100. / (1. +. (0. * sqrt (of_int iter))))
      ; n_tangents = 256
      ; sqrt = false
      ; rank_one = false
      ; damping = Some 1e-5
      ; momentum = None
      ; lm = false
      ; perturb_thresh = None
      }

  let name =
    let init_config = config_f ~iter:0 in
    let gamma_name =
      Option.value_map init_config.damping ~default:"none" ~f:Float.to_string
    in
    sprintf
      "ggn_lr_%s_sqrt_%s_damp_%s"
      (Float.to_string (Option.value_exn init_config.learning_rate))
      (Bool.to_string init_config.sqrt)
      gamma_name

  let init = O.init ~config:(config_f ~iter:0) VAE.init
end

(* --------------------------------
       -- Adam
       -------------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (VAE)

  let config_f ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.001 }

  let init = O.init VAE.init
end

let _ =
  let max_iter = num_train_loops in
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
