open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let cast = Dense.Matrix.Generic.cast_d2s
let base = Optimizer.Config.Base.default

let cifar =
  match Cmdargs.(get_string "-dataset") with
  | Some "mnist" -> false
  | Some "cifar" -> true
  | _ -> failwith "-dataset [mnist | cifar]"

let tan_from_act = Option.value (Cmdargs.get_bool "-tan_from_act") ~default:false
let output_dim = 10
let batch_size = 256
let n_tangents = batch_size
let input_size = if cifar then Int.(32 * 32 * 3) else Int.(28 * 28)
let full_batch_size = 60_000
let layer_sizes = [| 128; output_dim |]
let n_layers = Array.length layer_sizes
let num_epochs_to_run = 200

let num_train_loops =
  Convenience.num_train_loops ~full_batch_size ~batch_size num_epochs_to_run

let epoch_of t = Convenience.epoch_of ~full_batch_size ~batch_size t

module MLP_Layer = struct
  type 'a t =
    { id : int
    ; w : 'a
    ; b : 'a
    }
  [@@deriving prms]
end

module P = Prms.Array (MLP_Layer.Make (Prms.P))

(* neural network *)
module MLP = struct
  module P = P

  type input = Tensor.t

  (* for w of shape [n_in, n_out], dw_i = eps_i activation_i *)
  let sample_rand_tensor_activation ~param_shape ~activation =
    let n_out = List.last_exn param_shape in
    let bs = List.hd_exn (Tensor.shape activation) in
    (* eps of shape [n_out x bs], activation of shape [bs x n_in] *)
    let eps = Tensor.randn ~device:base.device ~kind:base.kind [ n_out; bs ] in
    Tensor.einsum ~path:None [ eps; activation ] ~equation:"ob,bi->bio"

  let f ~(theta : P.M.t) ~(input : input) =
    Array.foldi theta ~init:(Maths.const input) ~f:(fun i accu wb ->
      let open MLP_Layer in
      if tan_from_act
      then (
        let new_tangents_w =
          sample_rand_tensor_activation
            ~param_shape:(Maths.shape wb.w)
            ~activation:(Maths.primal accu)
        in
        let new_tangents_b =
          Tensor.randn ~device:base.device ~kind:base.kind (n_tangents :: Maths.shape wb.b)
        in
        (* set new tangents for w *)
        (match snd wb.w with
         | Some (Deferred dw) -> Maths.Deferred.set_exn dw new_tangents_w
         | _ -> ());
        match snd wb.b with
        | Some (Deferred db) -> Maths.Deferred.set_exn db new_tangents_b
        | _ -> ());
      (* call pre_activation again to perform the actual pass *)
      let pre_activation = Maths.((accu *@ wb.w) + wb.b) in
      if i = Array.length layer_sizes - 1
      then pre_activation
      else Maths.relu pre_activation)

  let init =
    let open MLP_Layer in
    Array.mapi layer_sizes ~f:(fun i n_o ->
      let n_i = if i = 0 then input_size else layer_sizes.(i - 1) in
      let w =
        Convenience.gaussian_tensor_2d_normed
          ~kind:base.kind
          ~device:base.device
          ~a:n_i
          ~b:n_o
          ~sigma:1.
      in
      let b =
        Convenience.gaussian_tensor_2d_normed
          ~kind:base.kind
          ~device:base.device
          ~a:1
          ~b:n_o
          ~sigma:1.
      in
      { id = i; w; b })
    |> P.map ~f:Prms.free
end

(* feedforward model with ce loss *)
module FF =
  Wrapper.Feedforward
    (MLP)
    (Loss.CE (struct
         let scaling_factor = 1.
       end))

(* -----------------------------------------
   -- Read in data. ------
   ----------------------------------------- *)

(* let _ = Owl.Dataset.download_all () *)

let train_set, test_set =
  if cifar
  then (
    (* train data has size [10000 x 32 x 32 x 3 ]*)
    let x_train, y_train =
      List.fold
        (List.init 5 ~f:(fun id -> id))
        ~init:None
        ~f:(fun accu i ->
          let x_train, _, y_train = Owl.Dataset.load_cifar_train_data Int.(i + 1) in
          match accu with
          | None -> Some (x_train, y_train)
          | Some (x_train_accu, y_train_accu) ->
            let new_x_train_accu = Arr.concatenate [| x_train_accu; x_train |] ~axis:0 in
            let new_y_train_accu = Arr.concatenate [| y_train_accu; y_train |] ~axis:0 in
            Some (new_x_train_accu, new_y_train_accu))
      |> Option.value_exn
    in
    let x_test, _, y_test = Owl.Dataset.load_cifar_test_data () in
    (* from torchvision *)
    let mu = cast (Owl.Arr.of_array [| 0.4914; 0.4822; 0.4465 |] [| 1; 1; 1; 3 |])
    and sigma = cast (Owl.Arr.of_array [| 0.2023; 0.1994; 0.2010 |] [| 1; 1; 1; 3 |]) in
    (* cifar10 from Owl dataset already scaled between 0 and 1! *)
    let x_train = Owl.Arr.((x_train - mu) / sigma) in
    let x_test = Owl.Arr.((x_test - mu) / sigma) in
    (x_train, y_train), (x_test, y_test))
  else (
    let dataset_mnist typ =
      let suffix =
        match typ with
        | `train -> "train"
        | `test -> "test"
      in
      let x = Owl.Arr.load_npy ("_data/x_" ^ suffix ^ ".npy") in
      let y =
        Owl.Arr.load_npy ("_data/t_" ^ suffix ^ ".npy") |> Owl.Arr.one_hot output_dim
      in
      let mu = 0.13062754273414612
      and sigma = 0.30810779333114624 in
      let x = Owl.Arr.(((x /$ 255.) -$ mu) /$ sigma) in
      cast x, cast y
    in
    dataset_mnist `train, dataset_mnist `test)

let sample_data (set_x, set_y) =
  let a = (Arr.shape set_x).(0) in
  fun batch_size ->
    if batch_size < 0
    then
      ( Tensor.of_bigarray ~device:base.device set_x
        |> Tensor.reshape ~shape:[ -1; input_size ]
      , Tensor.of_bigarray ~device:base.device set_y )
    else (
      let ids = List.init batch_size ~f:(fun _ -> Random.int a) in
      let xs = Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_x) in
      let ys = Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_y) in
      Tensor.reshape ~shape:[ batch_size; input_size ] xs, ys)

let test_eval ~train_data theta =
  let x, y =
    match train_data with
    | None -> sample_data test_set (-1)
    | Some train_data -> train_data
  in
  let logits = MLP.f ~theta ~input:x |> Maths.primal in
  let _, max_y = Tensor.max_dim ~keepdim:false ~dim:1 y in
  let _, max_ypred = Tensor.max_dim ~keepdim:false ~dim:1 logits in
  Tensor.eq_tensor max_y max_ypred
  |> Tensor.to_dtype ~dtype:base.kind ~non_blocking:false ~copy:false
  |> Tensor.mean
  |> Tensor.to_float0_exn

(* --------------------------------
   -- Generic type of optimiser
   -------------------------------- *)

module type Do_with_T = sig
  module O :
    Optimizer.T
    with type 'a W.P.p = 'a MLP.P.p
     and type W.data = Tensor.t * Tensor.t
     and type W.args = unit

  val name : string
  val config : (float, Bigarray.float32_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state running_avg =
      Stdlib.Gc.major ();
      let data = sample_data train_set batch_size in
      let loss, new_state = O.step ~config ~state ~data ~args:() in
      let running_avg =
        let loss_avg =
          match running_avg with
          | [] -> loss
          | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
        in
        (* save params *)
        if iter % 10 = 0
        then (
          let e = epoch_of iter in
          let test_acc =
            test_eval ~train_data:None MLP.P.(const (value (O.params new_state)))
          in
          let train_acc =
            test_eval ~train_data:(Some data) MLP.P.(const (value (O.params new_state)))
          in
          (* let params = O.params state in *)
          (* let n_params = O.W.P.T.numel (O.W.P.map params ~f:(fun p -> Prms.value p)) in  *)
          (* avg error *)
          Convenience.print [%message (e : float) (test_acc : float)];
          (* save params *)
          if iter % 100 = 0
          then
            O.W.P.T.save
              (MLP.P.value (O.params new_state))
              ~kind:base.ba_kind
              ~out:(in_dir name ^ "_params");
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array [| e; loss_avg; test_acc; train_acc |] 1 4)));
        []
      in
      if iter < max_iter then loop ~iter:(iter + 1) ~state:new_state (loss :: running_avg)
    in
    loop ~iter:0 ~state:init []
end

(* --------------------------------
       -- SOFO
       -------------------------------- *)

module Do_with_SOFO : Do_with_T = struct
  module O = Optimizer.SOFO (FF)

  let name = "sofo"

  let config =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 0.1
      ; n_tangents
      ; rank_one = false
      ; damping = Some 1e-3
      ; momentum = None
      ; tan_from_act
      }

  let init = O.init ~config MLP.init
end

(* --------------------------------
       -- Adam
       --------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (FF)

  let config = Optimizer.Config.Adam.{ default with base; learning_rate = Some 1e-3 }
  let init = O.init MLP.init
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
