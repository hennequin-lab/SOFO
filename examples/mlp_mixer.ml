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
let full_batch_size = 60_000
let batch_size = 64
let n_epochs_to_run = 70
let max_iter = Int.(full_batch_size * n_epochs_to_run / batch_size)
let epoch_of t = Convenience.epoch_of ~full_batch_size ~batch_size t

(* Network hyperparameters *)
let n_blocks = 2
let blocks_list = List.range 0 n_blocks
let n_layers = (n_blocks * 4) + 2
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

let equal_layer a b =
  match a, b with
  | Patchify, Patchify -> true
  | Classification, Classification -> true
  | TokenHidden i, TokenHidden j -> i = j
  | TokenOutput i, TokenOutput j -> i = j
  | ChannelHidden i, ChannelHidden j -> i = j
  | ChannelOutput i, ChannelOutput j -> i = j
  | _, _ -> false

let layer_list =
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

module MLP_Layer = struct
  type 'a t =
    { layer_name : layer
    ; w : 'a
    }
  [@@deriving prms]
end

module P = Prms.Array (MLP_Layer.Make (Prms.P))

(* -----------------------------------------
   ---- Build MLP-mixer       ------
   ----------------------------------------- *)
module MLP_mixer = struct
  module P = P

  type input = Tensor.t

  let phi = Maths.relu

  (* use conv2d for patchification; https://github.com/rishikksh20/MLP-Mixer-pytorch/blob/master/mlp-mixer.py; 
  https://github.com/d-li14/mlp-mixer.pytorch/blob/main/mixer.py *)
  let patchify ~(theta : P.M.t) ~input =
    (* w has shape [out_channels x in_channels x kerl_x x kerl_y] *)
    Maths.conv2d
      (Maths.const input)
      theta.(0).w
      ~bias:None
      ~stride:(patch_size, patch_size)

  let token_mixer ~(theta : P.M.t) ~layer_idx1 ~layer_idx2 x =
    let hidden =
      let x_appended =
        let tmp = Maths.transpose x ~dim0:1 ~dim1:2 in
        Convenience.expand_dim tmp
      in
      phi (Maths.einsum [ x_appended, "mcs"; theta.(layer_idx1).w, "sq" ] "mqc")
    in
    let output =
      let hidden_appended =
        let tmp = Maths.transpose hidden ~dim0:1 ~dim1:2 in
        Convenience.expand_dim tmp
      in
      Maths.einsum [ hidden_appended, "mcq"; theta.(layer_idx2).w, "qs" ] "msc"
    in
    output

  let channel_mixer ~(theta : P.M.t) ~layer_idx1 ~layer_idx2 x =
    let hidden =
      let x_appended = Convenience.expand_dim x in
      phi (Maths.einsum [ x_appended, "msc"; theta.(layer_idx1).w, "cq" ] "mqs")
    in
    let output =
      let hidden_appended =
        let tmp = Maths.transpose hidden ~dim0:1 ~dim1:2 in
        Convenience.expand_dim tmp
      in
      Maths.einsum [ hidden_appended, "msq"; theta.(layer_idx2).w, "qc" ] "msc"
    in
    output

  let f ~(theta : P.M.t) ~(input : input) =
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
    let mixer_blocked_mean =
      Maths.mean_dim mixer_blocks ~dim:(Some [ 1 ]) ~keepdim:false
    in
    let final_layer = theta.(n_layers - 1) in
    let output = Maths.einsum [ mixer_blocked_mean, "mc"; final_layer.w, "cd" ] "md" in
    output

  let init =
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
      { layer_name = Patchify; w }
    in
    let mixer_layers =
      List.map blocks_list ~f:(fun i ->
        let token_hidden =
          let w =
            Convenience.gaussian_tensor_2d_normed
              ~kind:base.kind
              ~device:base.device
              ~a:(s + 1)
              ~b:h
              ~sigma:1.
          in
          { layer_name = TokenHidden i; w }
        in
        let token_output =
          let w =
            Convenience.gaussian_tensor_2d_normed
              ~kind:base.kind
              ~device:base.device
              ~a:(h + 1)
              ~b:s
              ~sigma:1.
          in
          { layer_name = TokenOutput i; w }
        in
        let channel_hidden =
          let w =
            Convenience.gaussian_tensor_2d_normed
              ~kind:base.kind
              ~device:base.device
              ~a:(c + 1)
              ~b:h
              ~sigma:1.
          in
          { layer_name = ChannelHidden i; w }
        in
        let channel_output =
          let w =
            Convenience.gaussian_tensor_2d_normed
              ~kind:base.kind
              ~device:base.device
              ~a:(h + 1)
              ~b:c
              ~sigma:1.
          in
          { layer_name = ChannelOutput i; w }
        in
        [| token_hidden; token_output; channel_hidden; channel_output |])
    in
    let classification_head =
      let w =
        Convenience.gaussian_tensor_2d_normed
          ~kind:base.kind
          ~device:base.device
          ~a:c
          ~b:output_dim
          ~sigma:1.
      in
      { layer_name = Classification; w }
    in
    Array.concat ([ [| patchify |] ] @ mixer_layers @ [ [| classification_head |] ])
    |> P.map ~f:Prms.free
end

(* feedforward model with ce loss *)
module FF =
  Wrapper.Feedforward
    (MLP_mixer)
    (Loss.CE (struct
         let scaling_factor = 1.
       end))

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
  let logits = MLP_mixer.f ~theta ~input:x |> Maths.primal in
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
let _K = Int.(List.length layer_list * _K_w)
let _ = Convenience.print [%message (_K : int)]
let cycle = false

module GGN : Wrapper.Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { patchify_w_left : 'a
      ; patchify_w_right : 'a
      ; token_hidden_0_w_left : 'a
      ; token_hidden_0_w_right : 'a
      ; token_output_0_w_left : 'a
      ; token_output_0_w_right : 'a
      ; channel_hidden_0_w_left : 'a
      ; channel_hidden_0_w_right : 'a
      ; channel_output_0_w_left : 'a
      ; channel_output_0_w_right : 'a
      ; token_hidden_1_w_left : 'a
      ; token_hidden_1_w_right : 'a
      ; token_output_1_w_left : 'a
      ; token_output_1_w_right : 'a
      ; channel_hidden_1_w_left : 'a
      ; channel_hidden_1_w_right : 'a
      ; channel_output_1_w_left : 'a
      ; channel_output_1_w_right : 'a
      ; classification_w_left : 'a
      ; classification_w_right : 'a
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
    let einsum_w ~left ~right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    Array.map v ~f:(fun v ->
      let w =
        match v.layer_name with
        | Patchify ->
          let n_tangents = List.hd_exn (shape v.w) in
          let v_w = reshape v.w ~shape:[ n_tangents; c; -1 ] in
          einsum_w ~left:lambda.patchify_w_left ~right:lambda.patchify_w_right v_w
          |> reshape ~shape:[ n_tangents; c; in_channels; patch_size; patch_size ]
        | TokenHidden 0 ->
          einsum_w
            ~left:lambda.token_hidden_0_w_left
            ~right:lambda.token_hidden_0_w_right
            v.w
        | TokenOutput 0 ->
          einsum_w
            ~left:lambda.token_output_0_w_left
            ~right:lambda.token_output_0_w_right
            v.w
        | ChannelHidden 0 ->
          einsum_w
            ~left:lambda.channel_hidden_0_w_left
            ~right:lambda.channel_hidden_0_w_right
            v.w
        | ChannelOutput 0 ->
          einsum_w
            ~left:lambda.channel_output_0_w_left
            ~right:lambda.channel_output_0_w_right
            v.w
        | TokenHidden 1 ->
          einsum_w
            ~left:lambda.token_hidden_1_w_left
            ~right:lambda.token_hidden_1_w_right
            v.w
        | TokenOutput 1 ->
          einsum_w
            ~left:lambda.token_output_1_w_left
            ~right:lambda.token_output_1_w_right
            v.w
        | ChannelHidden 1 ->
          einsum_w
            ~left:lambda.channel_hidden_1_w_left
            ~right:lambda.channel_hidden_1_w_right
            v.w
        | ChannelOutput 1 ->
          einsum_w
            ~left:lambda.channel_output_1_w_left
            ~right:lambda.channel_output_1_w_right
            v.w
        | Classification ->
          einsum_w
            ~left:lambda.classification_w_left
            ~right:lambda.classification_w_right
            v.w
        | _ -> assert false
      in
      MLP_Layer.{ layer_name = v.layer_name; w })

  let get_shapes (layer_name : layer) =
    let w_shape =
      match layer_name with
      | Patchify -> [ c; in_channels; patch_size; patch_size ]
      | TokenHidden _ -> [ s + 1; h ]
      | TokenOutput _ -> [ h + 1; s ]
      | ChannelHidden _ -> [ c + 1; h ]
      | ChannelOutput _ -> [ h + 1; c ]
      | Classification -> [ c; output_dim ]
    in
    w_shape

  let get_n_params _ = _K_w

  let get_total_n_params (layer_name : layer) =
    let list_prod l = List.fold l ~init:1 ~f:(fun accu i -> accu * i) in
    list_prod (get_shapes layer_name)

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~(param_name : layer) ~n_per_param v =
    List.map layer_list ~f:(fun layer_name ->
      let w_shape = get_shapes layer_name in
      let w = zero_params ~shape:w_shape n_per_param in
      let params_tmp = MLP_Layer.{ layer_name; w } in
      if equal_layer layer_name param_name
      then MLP_Layer.{ params_tmp with w = v }
      else params_tmp)
    |> List.to_array

  let random_localised_vs () : P.T.t =
    let n_per_param = _K_w in
    List.mapi layer_list ~f:(fun i layer_name ->
      let w_shape = get_shapes layer_name in
      let w = random_params ~shape:w_shape n_per_param in
      let zeros_before = zero_params ~shape:w_shape (n_per_param * i) in
      let zeros_after = zero_params ~shape:w_shape (n_per_param * (n_layers - 1 - i)) in
      let final =
        if n_layers = 1 then w else Tensor.concat [ zeros_before; w; zeros_after ] ~dim:0
      in
      MLP_Layer.{ layer_name; w = final })
    |> List.to_array

  (* compute sorted eigenvalues, u_left and u_right. *)
  let eigenvectors_for_params ~lambda ~param_name =
    let left, right =
      match param_name with
      | Patchify -> lambda.patchify_w_left, lambda.patchify_w_right
      | TokenHidden 0 -> lambda.token_hidden_0_w_left, lambda.token_hidden_0_w_right
      | TokenOutput 0 -> lambda.token_output_0_w_left, lambda.token_output_0_w_right
      | ChannelHidden 0 -> lambda.channel_hidden_0_w_left, lambda.channel_hidden_0_w_right
      | ChannelOutput 0 -> lambda.channel_output_0_w_left, lambda.channel_output_0_w_right
      | TokenHidden 1 -> lambda.token_hidden_1_w_left, lambda.token_hidden_1_w_right
      | TokenOutput 1 -> lambda.token_output_1_w_left, lambda.token_output_1_w_right
      | ChannelHidden 1 -> lambda.channel_hidden_1_w_left, lambda.channel_hidden_1_w_right
      | ChannelOutput 1 -> lambda.channel_output_1_w_left, lambda.channel_output_1_w_right
      | Classification -> lambda.classification_w_left, lambda.classification_w_right
      | _ -> assert false
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
    match List.Assoc.find !s_u_cache param_name ~equal:equal_layer with
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
        let tmp =
          match param_name with
          | Patchify ->
            let u_r_reshaped =
              Tensor.reshape u_r ~shape:[ in_channels; patch_size; patch_size ]
            in
            Tensor.einsum ~path:None ~equation:"i,jkl->ijkl" [ u_l; u_r_reshaped ]
          | _ -> Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_l; u_r ]
        in
        (* let tmp = Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_l; u_r ] in *)
        Tensor.unsqueeze tmp ~dim:0)
      |> Tensor.concatenate ~dim:0
    in
    localise ~param_name ~n_per_param local_vs

  let eigenvectors_for_each_param ~lambda ~param_name ~sampling_state =
    let n_per_param = get_n_params param_name in
    let n_params = get_total_n_params param_name in
    let s_all, u_left, u_right = get_s_u ~lambda ~param_name in
    let selection =
      if cycle
      then
        List.init n_per_param ~f:(fun i ->
          ((sampling_state * n_per_param) + i) % n_params)
      else List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    extract_local_vs ~s_all ~param_name ~u_left ~u_right ~selection

  let eigenvectors ~(lambda : A.M.t) ~switch_to_learn t (_K : int) =
    let eigenvectors_each =
      List.map layer_list ~f:(fun param_name ->
        eigenvectors_for_each_param ~lambda ~param_name ~sampling_state:t)
    in
    let vs =
      List.fold eigenvectors_each ~init:None ~f:(fun accu local_vs ->
        match accu with
        | None -> Some local_vs
        | Some a -> Some (P.map2 a local_vs ~f:(fun x y -> Tensor.concat ~dim:0 [ x; y ])))
    in
    (* reset s_u_cache and set sampling state to 0 if learn again *)
    if switch_to_learn then s_u_cache := [];
    let new_sampling_state = if switch_to_learn then 0 else t + 1 in
    Option.value_exn vs, new_sampling_state

  let init () =
    let init_eye size =
      Mat.(0.1 $* eye size) |> Tensor.of_bigarray ~device:base.device |> Prms.free
    in
    let eye_h = init_eye h in
    let eye_h_1 = init_eye (h + 1) in
    let eye_s = init_eye s in
    let eye_s_1 = init_eye (s + 1) in
    let eye_c = init_eye c in
    let eye_c_1 = init_eye (c + 1) in
    { patchify_w_left = init_eye c
    ; patchify_w_right = init_eye Int.(in_channels * patch_size * patch_size)
    ; token_hidden_0_w_left = eye_s_1
    ; token_hidden_0_w_right = eye_h
    ; token_output_0_w_left = eye_h_1
    ; token_output_0_w_right = eye_s
    ; channel_hidden_0_w_left = eye_c_1
    ; channel_hidden_0_w_right = eye_h
    ; channel_output_0_w_left = eye_h_1
    ; channel_output_0_w_right = eye_c
    ; token_hidden_1_w_left = eye_s_1
    ; token_hidden_1_w_right = eye_h
    ; token_output_1_w_left = eye_h_1
    ; token_output_1_w_right = eye_s
    ; channel_hidden_1_w_left = eye_c_1
    ; channel_hidden_1_w_right = eye_h
    ; channel_output_1_w_left = eye_h_1
    ; channel_output_1_w_right = eye_c
    ; classification_w_left = eye_c
    ; classification_w_right = init_eye output_dim
    }
end

(* --------------------------------
   -- Generic type of optimiser
   -------------------------------- *)

module type Do_with_T = sig
  module O :
    Optimizer.T
    with type 'a W.P.p = 'a MLP_mixer.P.p
     and type W.data = Tensor.t * Tensor.t
     and type W.args = unit

  val name : string
  val config : iter:int -> (float, Bigarray.float32_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    Bos.Cmd.(v "rm" % "-f" % in_dir "aux") |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state running_avg =
      Stdlib.Gc.major ();
      let data = sample_data train_set batch_size in
      let loss, new_state = O.step ~config:(config ~iter) ~state ~data () in
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
            test_eval ~train_data:None MLP_mixer.P.(const (value (O.params new_state)))
          in
          let train_acc =
            test_eval
              ~train_data:(Some data)
              MLP_mixer.P.(const (value (O.params new_state)))
          in
          (* let params = O.params state in 
          let n_params = O.W.P.T.numel (O.W.P.map params ~f:(fun p -> Prms.value p)) in *)
          (* avg error *)
          Convenience.print [%message (e : float) (loss_avg : float) (test_acc : float)];
          (* save params *)
          if iter % 100 = 0
          then
            O.W.P.T.save
              (MLP_mixer.P.value (O.params new_state))
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
  module O = Optimizer.SOFO (FF) (GGN)

  let name = "sofo"

  let config ~iter:_ =
    let aux =
      Optimizer.Config.SOFO.
        { (default_aux (in_dir "aux")) with
          config =
            Optimizer.Config.Adam.
              { default with base; learning_rate = Some 1e-3; eps = 1e-4 }
        ; steps = 50
        ; learn_steps = 100
        ; exploit_steps = 100
        ; local = true
        }
    in
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 0.005
      ; n_tangents = _K
      ; rank_one = false
      ; damping = Some 1e-3
      ; aux = Some aux
      ; orthogonalize = true
      }

  let init = O.init MLP_mixer.init
end

(* --------------------------------
     -- Adam
     --------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (FF)

  let config ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 1e-3 }

  let init = O.init MLP_mixer.init
end

let _ =
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
