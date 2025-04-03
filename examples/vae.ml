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

let bs = 256
let full_batch_size = 60_000
let num_epochs_to_run = 10

let num_train_loops =
  Convenience.num_train_loops ~full_batch_size ~batch_size:bs num_epochs_to_run

let epoch_of t = Convenience.epoch_of ~full_batch_size ~batch_size:bs t
let sampling = false
let n_fisher = 30
let input_dim = 28 * 28
let h_dim1 = 10
let latent_dim = 5

let encoder_hidden_layers =
  [| input_dim, h_dim1; h_dim1, latent_dim; h_dim1, latent_dim |]

let decoder_hidden_layers = [| latent_dim, h_dim1; h_dim1, input_dim |]
let n_encoder = Array.length encoder_hidden_layers
let n_decoder = Array.length decoder_hidden_layers
let beta = 1.

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

  let encoder x (theta : P.M.t) =
    let theta_encoder = Array.sub theta ~pos:0 ~len:(n_encoder - 2) in
    let h_ =
      Array.fold theta_encoder ~init:(Maths.const x) ~f:(fun accu p ->
        phi Maths.((accu *@ p.w) + p.b))
    in
    let mean =
      let p_mean = theta.(n_encoder - 2) in
      Maths.((h_ *@ p_mean.w) + p_mean.b)
    in
    let log_std =
      let p_std = theta.(n_encoder - 1) in
      Maths.((h_ *@ p_std.w) + p_std.b)
    in
    mean, log_std

  let reparam ~_mean ~log_std =
    let std = Maths.exp log_std in
    Maths.(_mean + (std * const (Tensor.rand_like (Maths.primal_tensor_detach _mean))))

  let decoder z (theta : P.M.t) =
    let theta_decoder = Array.sub theta ~pos:n_encoder ~len:n_decoder in
    Array.foldi theta_decoder ~init:z ~f:(fun i accu p ->
      let act_fun = if i = n_decoder - 1 then phi else Maths.sigmoid in
      act_fun Maths.((accu *@ p.w) + p.b))

  let gaussian_llh ?mu ?(std_batched = false) ~std x =
    let inv_std = Maths.(f 1. / std) in
    let std_eq = if std_batched then "ma" else "a" in
    let error_term =
      let error =
        match mu with
        | None -> Maths.(einsum [ x, "ma"; inv_std, std_eq ] "ma")
        | Some mu -> Maths.(einsum [ x - mu, "ma"; inv_std, std_eq ] "ma")
      in
      Maths.einsum [ error, "ma"; error, "ma" ] "m"
    in
    let cov_term =
      match std_batched with
      | false -> Maths.(sum (log (sqr std)))
      | true -> Maths.(sum_dim (log (sqr std)) ~dim:(Some [ 1 ]) ~keepdim:false)
    in
    let const_term =
      let o = x |> Maths.primal |> Tensor.shape |> List.last_exn in
      Float.(log (2. * pi) * of_int o)
    in
    Maths.(0.5 $* (const_term $+ error_term + cov_term)) |> Maths.neg

  let ggn ~x_hat =
    let vtgt = Maths.tangent x_hat |> Option.value_exn in
    Tensor.einsum ~equation:"kma,jma->kj" [ vtgt; vtgt ] ~path:None

  let true_fisher ~x_hat =
    let x_hat_pri = Maths.primal x_hat in
    let n_tangents =
      List.hd_exn (Tensor.shape (Option.value_exn (Maths.tangent x_hat)))
    in
    let fisher_sampled =
      List.fold
        (List.init n_fisher ~f:(fun i -> i))
        ~init:(Tensor.f 0.)
        ~f:(fun accu _ ->
          let eps = Tensor.randn_like x_hat_pri in
          let x_hat_sample = Tensor.(eps + x_hat_pri) in
          let llh_t =
            Maths.(
              mean_dim ~dim:(Some [ 1 ]) ~keepdim:false (sqr (const x_hat_sample - x_hat)))
            |> Maths.tangent
            |> Option.value_exn
          in
          let fisher_half = Tensor.reshape llh_t ~shape:[ n_tangents; -1 ] in
          let fisher =
            Tensor.einsum ~equation:"ka,ja->kj" [ fisher_half; fisher_half ] ~path:None
          in
          Stdlib.Gc.major ();
          Tensor.(accu + fisher))
    in
    Tensor.(fisher_sampled / f (Float.of_int n_fisher))

  let loss ~data:x theta =
    let _mean, log_std = encoder x theta in
    let z = reparam ~_mean ~log_std in
    let x_hat = decoder z theta in
    (* how close the reconstructed image is to the original *)
    let rec_error =
      Maths.(sqr (x_hat - const x)) |> Maths.mean_dim ~dim:(Some [ 1 ]) ~keepdim:false
    in
    let kl_term =
      if sampling
      then (
        let prior =
          gaussian_llh
            ~std:
              (Tensor.ones [ latent_dim ] ~device:base.device ~kind:base.kind
               |> Maths.const)
            z
        in
        let neg_entropy =
          gaussian_llh ~std_batched:true ~mu:_mean ~std:(Maths.exp log_std) z
        in
        Maths.(neg_entropy - prior))
      else (
        let det1 = Maths.(2. $* sum_dim ~dim:(Some [ 1 ]) ~keepdim:false log_std) in
        let _const = Float.(of_int latent_dim) in
        let tr =
          log_std
          |> Maths.exp
          |> Maths.sqr
          |> Maths.sum_dim ~dim:(Some [ 1 ]) ~keepdim:false
        in
        let quad = Maths.(einsum [ _mean, "mb"; _mean, "mb" ] "m") in
        Maths.(0.5 $* tr - det1 + quad - f _const))
    in
    Maths.(rec_error + (f beta * kl_term)), x_hat

  let f ~update ~data:x ~init ~args:() (theta : P.M.t) =
    let loss, x_hat = loss ~data:x theta in
    match update with
    | `loss_only u -> u init (Some loss)
    | `loss_and_ggn u ->
      (* let ggn = ggn ~x_hat in *)
      let true_fisher = true_fisher ~x_hat in
      let _, final_s, _ = Tensor.svd ~some:true ~compute_uv:false true_fisher in
      final_s
      |> Tensor.reshape ~shape:[ -1; 1 ]
      |> Tensor.to_bigarray ~kind:base.ba_kind
      |> Owl.Mat.save_txt ~out:(in_dir (sprintf "svals"));
      u init (Some (loss, Some true_fisher))

  let init =
    let open MLP_Layer in
    let encoder =
      Array.map encoder_hidden_layers ~f:(fun (i, o) ->
        let w =
          Convenience.gaussian_tensor_2d_normed
            ~kind:base.kind
            ~device:base.device
            ~a:i
            ~b:o
            ~sigma:1.
        in
        let b = Tensor.zeros ~kind:base.kind ~device:base.device [ 1; o ] in
        { w; b })
    in
    let decoder =
      Array.map decoder_hidden_layers ~f:(fun (i, o) ->
        let w =
          Convenience.gaussian_tensor_2d_normed
            ~kind:base.kind
            ~device:base.device
            ~a:i
            ~b:o
            ~sigma:1.
        in
        let b = Tensor.zeros ~kind:base.kind ~device:base.device [ 1; o ] in
        { w; b })
    in
    Array.concat [ encoder; decoder ] |> P.map ~f:Prms.free
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
let test_set = dataset `test

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
      let loss, new_state = O.step ~config ~state ~data () in
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
          let data_test = sample_data test_set bs in
          let test_loss =
            let loss_t, _ =
              VAE.loss
                ~data:data_test
                (VAE.P.map ~f:Maths.const (VAE.P.value (O.params new_state)))
            in
            loss_t |> Maths.primal |> Tensor.mean |> Tensor.to_float0_exn
          in
          (* avg error *)
          let t = epoch_of iter in
          Convenience.print [%message (t : float) (loss_avg : float)];
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array [| t; time_elapsed; loss_avg; test_loss |] 1 4));
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
  module O = Optimizer.SOFO (VAE) (GGN)

  let config_f ~iter =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some Float.(1e-6 / (1. +. (0. * sqrt (of_int iter))))
      ; n_tangents = 512
      ; damping = Some 1e-3
      ; rank_one = false
      ; aux = None
      }

  let name = "sofo"

  let init = O.init  VAE.init
end

(* --------------------------------
       -- Adam
       -------------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (VAE)

  let config_f ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 1e-3 }

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
