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

let output_dim = 10
let batch_size = 256
let input_size = if cifar then Int.(32 * 32 * 3) else Int.(28 * 28)
let full_batch_size = 60_000
let layer_sizes = [ 128; output_dim ]
let num_epochs_to_run = 200
let max_iter = Int.(full_batch_size * num_epochs_to_run / batch_size)
let epoch_of t = Float.(of_int t * of_int batch_size / of_int full_batch_size)

module MLP_Layer = struct
  type 'a t =
    { id : int
    ; w : 'a
    }
  [@@deriving prms]
end

module P = Prms.List (MLP_Layer.Make (Prms.Single))

(* neural network *)
module MLP = struct
  module P = P

  type input = Tensor.t

  let forward ~(theta : _ Maths.some P.t) ~(input : input) =
    let bs = Tensor.shape input |> List.hd_exn in
    List.foldi
      theta
      ~init:Maths.(any (of_tensor input))
      ~f:(fun i accu wb ->
        let open MLP_Layer in
        let accu =
          Maths.concat
            [ accu
            ; Maths.(
                any
                  (of_tensor (Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ])))
            ]
            ~dim:1
        in
        let pre_activation = Maths.(accu *@ wb.w) in
        if i = List.length layer_sizes - 1
        then pre_activation
        else Maths.relu pre_activation)

  let f ~data:(input, labels) theta =
    let pred = forward ~theta ~input in
    let ell =
      Loss.cross_entropy
        ~average_over:[ 0; 1 ]
        ~logit_dim:1
        ~labels:(Maths.of_tensor labels)
        pred
    in
    let ggn =
      Loss.cross_entropy_ggn
        ~average_over:[ 0; 1 ]
        (Maths.const pred)
        ~vtgt:(Maths.tangent_exn pred)
    in
    ell, ggn

  let init : P.param =
    let open MLP_Layer in
    List.mapi layer_sizes ~f:(fun i n_o ->
      let n_i = if i = 0 then input_size else List.nth_exn layer_sizes (i - 1) in
      let w =
        Sofo.gaussian_tensor_normed
          ~kind:base.kind
          ~device:base.device
          ~sigma:1.
          [ n_i + 1; n_o ]
        |> Maths.of_tensor
        |> Prms.Single.free
      in
      { id = i; w })
end

(* -----------------------------------------
   -- Read in data. ------
   ----------------------------------------- *)

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
  let logits = MLP.forward ~theta ~input:x |> Maths.to_tensor in
  let _, max_y = Tensor.max_dim ~keepdim:false ~dim:1 y in
  let _, max_ypred = Tensor.max_dim ~keepdim:false ~dim:1 logits in
  Tensor.eq_tensor max_y max_ypred
  |> Tensor.to_dtype ~dtype:base.kind ~non_blocking:false ~copy:false
  |> Tensor.mean
  |> Tensor.to_float0_exn

(* -----------------------------------------
   -- Optimization with SOFO    ------
   ----------------------------------------- *)

module O = Optimizer.SOFO (MLP.P)

let config =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some 1e-3
    ; n_tangents = 128
    ; damping = `relative_from_top 1e-3
    }

let rec loop ~t ~out ~state =
  Stdlib.Gc.major ();
  let data = sample_data train_set batch_size in
  let theta, tangents = O.prepare ~config state in
  let loss, ggn = MLP.f ~data theta in
  let new_state = O.step ~config ~info:{ loss; ggn; tangents } state in
  if t % 10 = 0
  then (
    (* save params *)
    O.P.C.save
      (MLP.P.value (O.params new_state))
      ~kind:base.ba_kind
      ~out:(in_dir "sofo_params");
    let loss = Maths.to_float_exn (Maths.const loss) in
    let e = epoch_of t in
    let test_acc =
      test_eval ~train_data:None MLP.P.(const (value (O.params new_state)))
    in
    let train_acc =
      test_eval ~train_data:(Some data) MLP.P.(const (value (O.params new_state)))
    in
    print [%message (e : float) (loss : float)];
    Owl.Mat.(
      save_txt
        ~append:true
        ~out
        (of_array [| Float.of_int t; loss; test_acc; train_acc |] 1 4)));
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state

(* Start the sofo loop. *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init MLP.init)
