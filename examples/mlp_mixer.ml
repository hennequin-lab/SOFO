open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let _ =
  Random.init 1985;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let cast = Dense.Matrix.Generic.cast_d2s
let base = Optimizer.Config.Base.default

let cifar =
  match Cmdargs.(get_string "-dataset") with
  | Some "mnist" -> false
  | Some "cifar" -> true
  | _ -> failwith "-dataset [mnist | cifar ]"

(* Input/output dimensions *)
let input_dim = if cifar then 32 else 28
let output_dim = 10

(* Batch configuration *)
let batch_size = 256
let full_batch_size = 60_000
let num_epochs_to_run = 200
let max_iter = Int.(full_batch_size * num_epochs_to_run / batch_size)
let epoch_of t = Float.(of_int t * of_int batch_size / of_int full_batch_size)

(* Network hyperparameters *)
let n_blocks = 2
let blocks_list = List.range 0 n_blocks
let in_channels = if cifar then 3 else 1
let patch_size = if cifar then 4 else 7

(* number of patches *)
let s = (input_dim / patch_size) ** 2

(* hidden size *)
let c = 32

(* expanded hidden size *)
let h = c * 2

type layer =
  | Patchify
  | TokenHidden of int
  | TokenOutput of int
  | ChannelHidden of int
  | ChannelOutput of int
  | Classification
[@@deriving compare, sexp]

module MLP_Layer = struct
  type 'a t =
    { layer_name : layer
    ; w : 'a
    }
  [@@deriving prms]
end

module P = Prms.List (MLP_Layer.Make (Prms.Single))

(* -----------------------------------------
   ---- Build MLP-mixer       ------
   ----------------------------------------- *)

(* append 1s to the last dim of x *)
let expand_dim x =
  let x_shape = Maths.shape x in
  let x_shape_exp_last = List.drop_last_exn x_shape in
  let x_ = Maths.to_tensor x in
  Maths.concat
    [ x
    ; Maths.any
        (Maths.of_tensor
           (Tensor.ones
              ~device:(Tensor.device x_)
              ~kind:(Tensor.kind x_)
              (x_shape_exp_last @ [ 1 ])))
    ]
    ~dim:(List.length x_shape - 1)

module MLP_mixer = struct
  module P = P

  type input = Tensor.t

  let phi = Maths.relu

  (* use conv2d for patchification; https://github.com/rishikksh20/MLP-Mixer-pytorch/blob/master/mlp-mixer.py; 
  https://github.com/d-li14/mlp-mixer.pytorch/blob/main/mixer.py *)
  let patchify ~(theta : _ Maths.some P.t) ~input =
    (* w has shape [out_channels x in_channels x kerl_x x kerl_y] *)
    Maths.conv2d
      ~bias:None
      ~stride:(patch_size, patch_size)
      ~w:(List.hd_exn theta).w
      (Maths.any (Maths.of_tensor input))

  let token_mixer ~(theta : _ Maths.some P.t) ~layer_idx1 ~layer_idx2 x =
    let hidden =
      let x_appended =
        let tmp = Maths.transpose x ~dims:[ 0; 2; 1 ] in
        expand_dim tmp
      in
      phi
        (Maths.einsum
           [ x_appended, "mcs"; Maths.any (List.nth_exn theta layer_idx1).w, "sq" ]
           "mqc")
    in
    let output =
      let hidden_appended =
        let tmp = Maths.transpose hidden ~dims:[ 0; 2; 1 ] in
        expand_dim tmp
      in
      Maths.einsum
        [ hidden_appended, "mcq"; Maths.any (List.nth_exn theta layer_idx2).w, "qs" ]
        "msc"
    in
    output

  let channel_mixer ~(theta : _ Maths.some P.t) ~layer_idx1 ~layer_idx2 x =
    let hidden =
      let x_appended = expand_dim x in
      phi
        (Maths.einsum
           [ x_appended, "msc"; Maths.any (List.nth_exn theta layer_idx1).w, "cq" ]
           "mqs")
    in
    let output =
      let hidden_appended =
        let tmp = Maths.transpose hidden ~dims:[ 0; 2; 1 ] in
        expand_dim tmp
      in
      Maths.einsum
        [ hidden_appended, "msq"; Maths.any (List.nth_exn theta layer_idx2).w, "qc" ]
        "msc"
    in
    output

  let forward ~(theta : _ Maths.some P.t) ~(input : input) =
    (* patchify and map to hidden -> blocks -> classification *)
    let patchified_image = patchify ~theta ~input in
    let patchified_permute = Maths.permute patchified_image ~dims:[ 0; 2; 3; 1 ] in
    let batch_size = List.hd_exn (Tensor.shape input) in
    let patchified_final = Maths.view patchified_permute ~size:[ batch_size; s; c ] in
    let mixer_blocks =
      List.fold blocks_list ~init:patchified_final ~f:(fun acc i ->
        let token_mixed =
          token_mixer
            ~theta
            ~layer_idx1:Int.((i * 4) + 1)
            ~layer_idx2:Int.((i * 4) + 2)
            acc
        in
        let channel_mixed =
          channel_mixer
            ~theta
            ~layer_idx1:Int.((i * 4) + 3)
            ~layer_idx2:Int.((i + 1) * 4)
            token_mixed
        in
        channel_mixed)
    in
    let mixer_blocked_mean = Maths.mean mixer_blocks ~dim:[ 1 ] ~keepdim:false in
    let final_layer = List.last_exn theta in
    let output =
      Maths.einsum [ mixer_blocked_mean, "mc"; Maths.any final_layer.w, "cd" ] "md"
    in
    output

  let f ~data:(input, labels) theta =
    let pred = forward ~theta ~input in
    let ell =
      Loss.cross_entropy ~output_dims:[ 1 ] ~labels:(Maths.of_tensor labels) pred
    in
    let ggn =
      Loss.cross_entropy_ggn
        ~output_dims:[ 1 ]
        (Maths.const pred)
        ~vtgt:(Maths.tangent_exn pred)
    in
    ell, ggn

  let init : P.param =
    let to_param x = x |> Maths.of_tensor |> Prms.Single.free in
    let open MLP_Layer in
    (* patchify and map to hidden;  w has shape [out_channels x in_channels x kerl_x x kerl_y] *)
    let patchify =
      let w =
        let n_i = in_channels * patch_size * patch_size in
        let normaliser = Float.(1. / sqrt (of_int n_i)) in
        Tensor.mul_scalar_
          (Tensor.randn
             ~kind:base.kind
             ~device:base.device
             [ c; in_channels; patch_size; patch_size ])
          (Scalar.f normaliser)
      in
      { layer_name = Patchify; w = to_param w }
    in
    let mixer_layers =
      List.map blocks_list ~f:(fun i ->
        let token_hidden =
          let w =
            Sofo.gaussian_tensor_normed
              ~kind:base.kind
              ~device:base.device
              ~sigma:1.
              [ s + 1; h ]
          in
          { layer_name = TokenHidden i; w = to_param w }
        in
        let token_output =
          let w =
            Sofo.gaussian_tensor_normed
              ~kind:base.kind
              ~device:base.device
              [ h + 1; s ]
              ~sigma:1.
          in
          { layer_name = TokenOutput i; w = to_param w }
        in
        let channel_hidden =
          let w =
            Sofo.gaussian_tensor_normed
              ~kind:base.kind
              ~device:base.device
              ~sigma:1.
              [ c + 1; h ]
          in
          { layer_name = ChannelHidden i; w = to_param w }
        in
        let channel_output =
          let w =
            Sofo.gaussian_tensor_normed
              ~kind:base.kind
              ~device:base.device
              ~sigma:1.
              [ h + 1; c ]
          in
          { layer_name = ChannelOutput i; w = to_param w }
        in
        [ token_hidden; token_output; channel_hidden; channel_output ])
      |> List.concat
    in
    let classification_head =
      let w =
        Sofo.gaussian_tensor_normed
          ~kind:base.kind
          ~device:base.device
          ~sigma:1.
          [ c; output_dim ]
      in
      { layer_name = Classification; w = to_param w }
    in
    List.concat [ [ patchify ]; mixer_layers; [ classification_head ] ]
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
    (* transpose so dim is [bs x channels x height x width]*)
    let transpose_ = Owl.Dense.Ndarray.S.transpose ~axis:[| 0; 3; 1; 2 |] in
    (transpose_ x_train, y_train), (transpose_ x_test, y_test))
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
  if cifar
  then (
    let a = (Arr.shape set_x).(0) in
    (* let a = Mat.row_num set_x in *)
    fun batch_size ->
      if batch_size < 0
      then
        ( Tensor.of_bigarray ~device:base.device set_x
        , Tensor.of_bigarray ~device:base.device set_y )
      else (
        let ids = List.init batch_size ~f:(fun _ -> Random.int a) in
        let xs = Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_x) in
        let ys = Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_y) in
        xs, ys))
  else (
    let mnist_input_dim = 28 in
    let a = Mat.row_num set_x in
    fun batch_size ->
      if batch_size < 0
      then (
        (* reshape x to [batch size x 1 x input_dim x input_dim]. *)
        let total_bs = Mat.row_num set_x in
        let xs_tensor = Tensor.of_bigarray ~device:base.device set_x in
        let xs =
          Tensor.reshape
            xs_tensor
            ~shape:[ total_bs; 1; mnist_input_dim; mnist_input_dim ]
        in
        xs, Tensor.of_bigarray ~device:base.device set_y)
      else (
        let ids = List.init batch_size ~f:(fun _ -> Random.int a) in
        let x_tensor =
          Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_x)
        in
        let xs =
          Tensor.reshape
            x_tensor
            ~shape:[ batch_size; 1; mnist_input_dim; mnist_input_dim ]
        in
        let ys = Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_y) in
        xs, ys))

let test_eval ~train_data theta =
  let x, y =
    match train_data with
    | None -> sample_data test_set (-1)
    | Some train_data -> train_data
  in
  let logits = MLP_mixer.forward ~theta ~input:x |> Maths.to_tensor in
  let _, max_y = Tensor.max_dim ~keepdim:false ~dim:1 y in
  let _, max_ypred = Tensor.max_dim ~keepdim:false ~dim:1 logits in
  Tensor.eq_tensor max_y max_ypred
  |> Tensor.to_dtype ~dtype:base.kind ~non_blocking:false ~copy:false
  |> Tensor.mean
  |> Tensor.to_float0_exn

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
let _K_w = 32
let _K = Int.(List.length blocks_list * _K_w)
let _ = print [%message (_K : int)]

module MLP_Mixer_Aux_Layer = struct
  type 'a p =
    { layer_name : layer
    ; w_left : 'a
    ; w_right : 'a
    }
  [@@deriving prms]
end

module A = Prms.List (MLP_Mixer_Aux_Layer.Make (Prms.Single))

module MLP_Mixer_Spec = struct
  type param_name = layer [@@deriving compare, sexp]

  let all =
    [ Patchify
    ; TokenHidden 0
    ; TokenOutput 0
    ; ChannelHidden 0
    ; ChannelOutput 0
    ; TokenHidden 1
    ; TokenOutput 1
    ; ChannelHidden 1
    ; ChannelOutput 1
    ; Classification
    ]

  let shape = function
    | Patchify -> [ c; Int.(in_channels * patch_size * patch_size) ]
    | TokenHidden 0 -> [ s + 1; h ]
    | TokenOutput 0 -> [ h + 1; s ]
    | ChannelHidden 0 -> [ c + 1; h ]
    | ChannelOutput 0 -> [ h + 1; c ]
    | TokenHidden 1 -> [ s + 1; h ]
    | TokenOutput 1 -> [ h + 1; s ]
    | ChannelHidden 1 -> [ c + 1; h ]
    | ChannelOutput 1 -> [ h + 1; c ]
    | Classification -> [ c; output_dim ]
    | _ -> assert false

  let n_params = function
    | _ -> _K_w

  let n_params_list = List.map all ~f:(fun _ -> _K_w)
  let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
end

module GGN : Auxiliary with module P = P = struct
  module P = P
  module A = A
  module C = Ggn_common.GGN_Common (MLP_Mixer_Spec)

  let init_sampling_state () = 0

  (* approximation defined implicitly via Gv products *)
  let g12v ~(lambda : ([< `const | `dual ] as 'a) A.t) (v : 'a P.t) : Maths.any P.t =
    let open Maths in
    let einsum_w ~left ~right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    List.map2_exn lambda v ~f:(fun lambda v ->
      let w =
        match lambda.layer_name with
        | Patchify ->
          let n_tangents = List.hd_exn (shape v.w) in
          let v_w = reshape v.w ~shape:[ n_tangents; c; -1 ] in
          einsum_w ~left:lambda.w_left ~right:lambda.w_right v_w
          |> reshape ~shape:[ n_tangents; c; in_channels; patch_size; patch_size ]
        | _ -> einsum_w ~left:lambda.w_left ~right:lambda.w_right v.w
      in
      MLP_Layer.{ layer_name = v.layer_name; w })

  let get_sides (lambda : ([< `const | `dual ] as 'a) A.t) (x : MLP_Mixer_Spec.param_name)
    : 'a A.elt * 'a A.elt
    =
    let rec find = function
      | [] -> failwith "layer not found"
      | { MLP_Mixer_Aux_Layer.layer_name; w_left; w_right } :: rest ->
        if MLP_Mixer_Spec.equal_param_name layer_name x
        then w_left, w_right
        else find rest
    in
    find lambda

  let random_localised_vs () =
    let n_per_param = _K_w in
    List.mapi MLP_Mixer_Spec.all ~f:(fun i param_name ->
      let w_shape = MLP_Mixer_Spec.shape param_name in
      let w =
        C.random_params ~shape:w_shape ~device:base.device ~kind:base.kind n_per_param
      in
      let zeros_before =
        C.zero_params ~shape:w_shape ~device:base.device ~kind:base.kind (n_per_param * i)
      in
      let zeros_after =
        C.zero_params
          ~shape:w_shape
          ~device:base.device
          ~kind:base.kind
          (n_per_param * (n_blocks - 1 - i))
      in
      let final =
        if n_blocks = 1 then w else Tensor.concat [ zeros_before; w; zeros_after ] ~dim:0
      in
      MLP_Layer.{ layer_name = param_name; w = Maths.of_tensor final })

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~(param_name : layer) ~n_per_param v =
    List.map MLP_Mixer_Spec.all ~f:(fun layer_name ->
      let w_shape = MLP_Mixer_Spec.shape layer_name in
      let w =
        C.zero_params ~shape:w_shape ~device:base.device ~kind:base.kind n_per_param
      in
      let params_tmp = MLP_Layer.{ layer_name; w } in
      if MLP_Mixer_Spec.equal_param_name layer_name param_name
      then MLP_Layer.{ params_tmp with w = v }
      else params_tmp)

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

  let init () : A.param =
    let init_eye size =
      Maths.(0.1 $* eye ~device:base.device ~kind:base.kind size) |> Prms.Single.free
    in
    List.map MLP_Mixer_Spec.all ~f:(fun param_name ->
      let w_shape = MLP_Mixer_Spec.shape param_name in
      let w_left = init_eye (List.hd_exn w_shape) in
      let w_right = init_eye (List.last_exn w_shape) in
      MLP_Mixer_Aux_Layer.{ layer_name = param_name; w_left; w_right })
end

(* -----------------------------------------
   -- Optimization with SOFO    ------
   ----------------------------------------- *)

module O = Optimizer.SOFO (MLP_mixer.P) (GGN)

let config =
  let aux =
    Optimizer.Config.SOFO.
      { (default_aux (in_dir "aux")) with
        config =
          Optimizer.Config.Adam.
            { default with base; learning_rate = Some 1e-3; eps = 1e-4 }
      ; steps = 5
      ; learn_steps = 1
      ; exploit_steps = 1
      }
  in
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some 0.005
    ; n_tangents = _K
    ; damping = `relative_from_top 1e-3
    ; aux = Some aux
    }

let rec loop ~t ~out ~state running_avg =
  Stdlib.Gc.major ();
  let data = sample_data train_set batch_size in
  let theta, tangents, new_sampling_state = O.prepare ~config state in
  let loss, ggn = MLP_mixer.f ~data theta in
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
      (* save params *)
      O.P.C.save
        (MLP_mixer.P.value (O.params new_state))
        ~kind:base.ba_kind
        ~out:(in_dir "sofo_params");
      let e = epoch_of t in
      let test_acc =
        test_eval ~train_data:None MLP_mixer.P.(const (value (O.params new_state)))
      in
      let train_acc =
        test_eval ~train_data:(Some data) MLP_mixer.P.(const (value (O.params new_state)))
      in
      print [%message (e : float) (loss_avg : float)];
      Owl.Mat.(
        save_txt
          ~append:true
          ~out
          (of_array [| Float.of_int t; loss_avg; test_acc; train_acc |] 1 4)));
    []
  in
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state (loss :: running_avg)

(* Start the sofo loop. *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init MLP_mixer.init) []
