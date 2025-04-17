open Printf
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

(* for MNIST *)
let input_dim = 28
let output_dim = 10
let full_batch_size = 60_000
let batch_size = 64
let n_epochs_to_run = 70
let max_iter = Int.(full_batch_size * n_epochs_to_run / batch_size)
let epoch_of t = Convenience.epoch_of ~full_batch_size ~batch_size t

(* network hyperparameters *)
let n_blocks = 2
let blocks_list = List.range 0 n_blocks
let n_layers = (n_blocks * 4) + 2
let in_channels = 1
let patch_size = 7

(* number of patches; s=16 *)
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

type param_type =
  | W
  | B

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
    ; b : 'a
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

  (* use conv2d for patchification; https://github.com/rishikksh20/MLP-Mixer-pytorch/blob/master/mlp-mixer.py *)
  let patchify ~(theta : P.M.t) ~input =
    (* w has shape [out_channels x in_channels x kerl_x x kerl_y] *)
    Maths.conv2d
      (Maths.const input)
      theta.(0).w
      ~bias:(Maths.squeeze ~dim:0 theta.(0).b)
      ~stride:(patch_size, patch_size)

  let token_mixer ~(theta : P.M.t) ~layer_idx1 ~layer_idx2 x =
    let hidden = Maths.einsum [ x, "msc"; theta.(layer_idx1).w, "sq" ] "mqc" in
    let hidden_bias = Maths.(phi (hidden + unsqueeze ~dim:0 theta.(layer_idx1).b)) in
    let output = Maths.einsum [ hidden_bias, "mqc"; theta.(layer_idx2).w, "qs" ] "msc" in
    Maths.(output + unsqueeze ~dim:0 theta.(layer_idx2).b)

  let channel_mixer ~(theta : P.M.t) ~layer_idx1 ~layer_idx2 x =
    let hidden = Maths.einsum [ x, "msc"; theta.(layer_idx1).w, "cq" ] "mqs" in
    let hidden_bias = Maths.(phi (hidden + unsqueeze ~dim:0 theta.(layer_idx1).b)) in
    let output = Maths.einsum [ hidden_bias, "mqs"; theta.(layer_idx2).w, "qc" ] "msc" in
    Maths.(output + unsqueeze ~dim:0 theta.(layer_idx2).b)

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
    Maths.(output + final_layer.b)

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
      let b = Tensor.zeros ~kind:base.kind ~device:base.device [ 1; c ] in
      { layer_name = Patchify; w; b }
    in
    let mixer_layers =
      List.map blocks_list ~f:(fun i ->
        let token_hidden =
          let w =
            Convenience.gaussian_tensor_2d_normed
              ~kind:base.kind
              ~device:base.device
              ~a:s
              ~b:h
              ~sigma:1.
          in
          let b = Tensor.zeros ~kind:base.kind ~device:base.device [ h; c ] in
          { layer_name = TokenHidden i; w; b }
        in
        let token_output =
          let w =
            Convenience.gaussian_tensor_2d_normed
              ~kind:base.kind
              ~device:base.device
              ~a:h
              ~b:s
              ~sigma:1.
          in
          let b = Tensor.zeros ~kind:base.kind ~device:base.device [ s; c ] in
          { layer_name = TokenOutput i; w; b }
        in
        let channel_hidden =
          let w =
            Convenience.gaussian_tensor_2d_normed
              ~kind:base.kind
              ~device:base.device
              ~a:c
              ~b:h
              ~sigma:1.
          in
          let b = Tensor.zeros ~kind:base.kind ~device:base.device [ h; s ] in
          { layer_name = ChannelHidden i; w; b }
        in
        let channel_output =
          let w =
            Convenience.gaussian_tensor_2d_normed
              ~kind:base.kind
              ~device:base.device
              ~a:h
              ~b:c
              ~sigma:1.
          in
          let b = Tensor.zeros ~kind:base.kind ~device:base.device [ s; c ] in
          { layer_name = ChannelOutput i; w; b }
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
      let b = Tensor.zeros ~kind:base.kind ~device:base.device [ 1; output_dim ] in
      { layer_name = Classification; w; b }
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
   ---- Read in MNIST data       ------
   ----------------------------------------- *)

let dataset typ =
  let suffix =
    match typ with
    | `train -> "train"
    | `test -> "test"
  in
  let x = Owl.Arr.load_npy ("_data/x_" ^ suffix ^ ".npy") in
  let y = Owl.Arr.load_npy ("_data/t_" ^ suffix ^ ".npy") |> Owl.Arr.one_hot output_dim in
  let mu = 0.13062754273414612
  and sigma = 0.30810779333114624 in
  let x = Owl.Arr.(((x /$ 255.) -$ mu) /$ sigma) in
  cast x, cast y

let train_set = dataset `train
let test_set = dataset `test

let sample_data (set_x, set_y) =
  let a = Mat.row_num set_x in
  fun batch_size ->
    if batch_size < 0
    then (
      (* reshape x to [batch size x 1 x input_dim x input_dim]. *)
      let total_bs = Mat.row_num set_x in
      let xs_tensor = Tensor.of_bigarray ~device:base.device set_x in
      let xs = Tensor.reshape xs_tensor ~shape:[ total_bs; 1; input_dim; input_dim ] in
      xs, Tensor.of_bigarray ~device:base.device set_y)
    else (
      let ids = List.init batch_size ~f:(fun _ -> Random.int a) in
      let x_tensor =
        Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_x)
      in
      let xs = Tensor.reshape x_tensor ~shape:[ batch_size; 1; input_dim; input_dim ] in
      let ys = Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_y) in
      xs, ys)

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
let _K_w = 30
let _K_b = 2
let _K = Int.(List.length layer_list * (_K_w + _K_b))
let _ = Convenience.print [%message (_K : int)]

module GGN : Wrapper.Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { patchify_w_left : 'a
      ; patchify_w_right : 'a
      ; patchify_b_left : 'a
      ; patchify_b_right : 'a
      ; token_hidden_0_w_left : 'a
      ; token_hidden_0_w_right : 'a
      ; token_hidden_0_b_left : 'a
      ; token_hidden_0_b_right : 'a
      ; token_output_0_w_left : 'a
      ; token_output_0_w_right : 'a
      ; token_output_0_b_left : 'a
      ; token_output_0_b_right : 'a
      ; channel_hidden_0_w_left : 'a
      ; channel_hidden_0_w_right : 'a
      ; channel_hidden_0_b_left : 'a
      ; channel_hidden_0_b_right : 'a
      ; channel_output_0_w_left : 'a
      ; channel_output_0_w_right : 'a
      ; channel_output_0_b_left : 'a
      ; channel_output_0_b_right : 'a
      ; token_hidden_1_w_left : 'a
      ; token_hidden_1_w_right : 'a
      ; token_hidden_1_b_left : 'a
      ; token_hidden_1_b_right : 'a
      ; token_output_1_w_left : 'a
      ; token_output_1_w_right : 'a
      ; token_output_1_b_left : 'a
      ; token_output_1_b_right : 'a
      ; channel_hidden_1_w_left : 'a
      ; channel_hidden_1_w_right : 'a
      ; channel_hidden_1_b_left : 'a
      ; channel_hidden_1_b_right : 'a
      ; channel_output_1_w_left : 'a
      ; channel_output_1_w_right : 'a
      ; channel_output_1_b_left : 'a
      ; channel_output_1_b_right : 'a
      ; classification_w_left : 'a
      ; classification_w_right : 'a
      ; classification_b_left : 'a
      ; classification_b_right : 'a
      }
    [@@deriving prms]
  end

  let param_names_list =
    List.map layer_list ~f:(fun name -> [ name, W; name, B ]) |> List.concat

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
    let einsum_w ~left ~right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    Array.map v ~f:(fun v ->
      let w, b =
        match v.layer_name with
        | Patchify ->
          let w =
            let n_tangents = List.hd_exn (shape v.w) in
            let v_w = reshape v.w ~shape:[ n_tangents; c; -1 ] in
            einsum_w ~left:lambda.patchify_w_left ~right:lambda.patchify_w_right v_w
            |> reshape ~shape:[ n_tangents; c; in_channels; patch_size; patch_size ]
          in
          let b =
            einsum_w ~left:lambda.patchify_b_left ~right:lambda.patchify_b_right v.b
          in
          w, b
        | TokenHidden 0 ->
          let w =
            einsum_w
              ~left:lambda.token_hidden_0_w_left
              ~right:lambda.token_hidden_0_w_right
              v.w
          in
          let b =
            einsum_w
              ~left:lambda.token_hidden_0_b_left
              ~right:lambda.token_hidden_0_b_right
              v.b
          in
          w, b
        | TokenOutput 0 ->
          let w =
            einsum_w
              ~left:lambda.token_output_0_w_left
              ~right:lambda.token_output_0_w_right
              v.w
          in
          let b =
            einsum_w
              ~left:lambda.token_output_0_b_left
              ~right:lambda.token_output_0_b_right
              v.b
          in
          w, b
        | ChannelHidden 0 ->
          let w =
            einsum_w
              ~left:lambda.channel_hidden_0_w_left
              ~right:lambda.channel_hidden_0_w_right
              v.w
          in
          let b =
            einsum_w
              ~left:lambda.channel_hidden_0_b_left
              ~right:lambda.channel_hidden_0_b_right
              v.b
          in
          w, b
        | ChannelOutput 0 ->
          let w =
            einsum_w
              ~left:lambda.channel_output_0_w_left
              ~right:lambda.channel_output_0_w_right
              v.w
          in
          let b =
            einsum_w
              ~left:lambda.channel_output_0_b_left
              ~right:lambda.channel_output_0_b_right
              v.b
          in
          w, b
        | TokenHidden 1 ->
          let w =
            einsum_w
              ~left:lambda.token_hidden_1_w_left
              ~right:lambda.token_hidden_1_w_right
              v.w
          in
          let b =
            einsum_w
              ~left:lambda.token_hidden_1_b_left
              ~right:lambda.token_hidden_1_b_right
              v.b
          in
          w, b
        | TokenOutput 1 ->
          let w =
            einsum_w
              ~left:lambda.token_output_1_w_left
              ~right:lambda.token_output_1_w_right
              v.w
          in
          let b =
            einsum_w
              ~left:lambda.token_output_1_b_left
              ~right:lambda.token_output_1_b_right
              v.b
          in
          w, b
        | ChannelHidden 1 ->
          let w =
            einsum_w
              ~left:lambda.channel_hidden_1_w_left
              ~right:lambda.channel_hidden_1_w_right
              v.w
          in
          let b =
            einsum_w
              ~left:lambda.channel_hidden_1_b_left
              ~right:lambda.channel_hidden_1_b_right
              v.b
          in
          w, b
        | ChannelOutput 1 ->
          let w =
            einsum_w
              ~left:lambda.channel_output_1_w_left
              ~right:lambda.channel_output_1_w_right
              v.w
          in
          let b =
            einsum_w
              ~left:lambda.channel_output_1_b_left
              ~right:lambda.channel_output_1_b_right
              v.b
          in
          w, b
        | Classification ->
          let w =
            einsum_w
              ~left:lambda.classification_w_left
              ~right:lambda.classification_w_right
              v.w
          in
          let b =
            einsum_w
              ~left:lambda.classification_b_left
              ~right:lambda.classification_b_right
              v.b
          in
          w, b
        | _ -> assert false
      in
      MLP_Layer.{ layer_name = v.layer_name; w; b })

  let get_shapes (layer_name : layer) =
    let w_shape, b_shape =
      match layer_name with
      | Patchify -> [ c; in_channels; patch_size; patch_size ], [ 1; c ]
      | TokenHidden _ -> [ s; h ], [ h; c ]
      | TokenOutput _ -> [ h; s ], [ s; c ]
      | ChannelHidden _ -> [ c; h ], [ h; s ]
      | ChannelOutput _ -> [ h; c ], [ s; c ]
      | Classification -> [ c; output_dim ], [ 1; output_dim ]
    in
    w_shape, b_shape

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~(param_name : layer * param_type) ~n_per_param v =
    let param_name_p, param_type_p = param_name in
    List.map layer_list ~f:(fun layer_name ->
      let w_shape, b_shape = get_shapes layer_name in
      let w = zero_params ~shape:w_shape n_per_param in
      let b = zero_params ~shape:b_shape n_per_param in
      let params_tmp = MLP_Layer.{ layer_name; w; b } in
      if equal_layer layer_name param_name_p
      then (
        match param_type_p with
        | W -> MLP_Layer.{ params_tmp with w = v }
        | B -> MLP_Layer.{ params_tmp with b = v })
      else params_tmp)
    |> List.to_array

  let random_localised_vs _K : P.T.t =
    List.map layer_list ~f:(fun layer_name ->
      let w_shape, b_shape = get_shapes layer_name in
      let w = random_params ~shape:w_shape _K in
      let b = random_params ~shape:b_shape _K in
      MLP_Layer.{ layer_name; w; b })
    |> List.to_array

  let eigenvectors_for_each_params ~lambda ~(param_name : layer * param_type) =
    let vs, n_per_param =
      let left, right, n_per_param =
        match param_name with
        | Patchify, W -> lambda.patchify_w_left, lambda.patchify_w_right, _K_w
        | Patchify, B -> lambda.patchify_b_left, lambda.patchify_b_right, _K_b
        | TokenHidden 0, W ->
          lambda.token_hidden_0_w_left, lambda.token_hidden_0_w_right, _K_w
        | TokenHidden 0, B ->
          lambda.token_hidden_0_b_left, lambda.token_hidden_0_b_right, _K_b
        | TokenOutput 0, W ->
          lambda.token_output_0_w_left, lambda.token_output_0_w_right, _K_w
        | TokenOutput 0, B ->
          lambda.token_output_0_b_left, lambda.token_output_0_b_right, _K_b
        | ChannelHidden 0, W ->
          lambda.channel_hidden_0_w_left, lambda.channel_hidden_0_w_right, _K_w
        | ChannelHidden 0, B ->
          lambda.channel_hidden_0_b_left, lambda.channel_hidden_0_b_right, _K_b
        | ChannelOutput 0, W ->
          lambda.channel_output_0_w_left, lambda.channel_output_0_w_right, _K_w
        | ChannelOutput 0, B ->
          lambda.channel_output_0_b_left, lambda.channel_output_0_b_right, _K_b
        | TokenHidden 1, W ->
          lambda.token_hidden_1_w_left, lambda.token_hidden_1_w_right, _K_w
        | TokenHidden 1, B ->
          lambda.token_hidden_1_b_left, lambda.token_hidden_1_b_right, _K_b
        | TokenOutput 1, W ->
          lambda.token_output_1_w_left, lambda.token_output_1_w_right, _K_w
        | TokenOutput 1, B ->
          lambda.token_output_1_b_left, lambda.token_output_1_b_right, _K_b
        | ChannelHidden 1, W ->
          lambda.channel_hidden_1_w_left, lambda.channel_hidden_1_w_right, _K_w
        | ChannelHidden 1, B ->
          lambda.channel_hidden_1_b_left, lambda.channel_hidden_1_b_right, _K_b
        | ChannelOutput 1, W ->
          lambda.channel_output_1_w_left, lambda.channel_output_1_w_right, _K_w
        | ChannelOutput 1, B ->
          lambda.channel_output_1_b_left, lambda.channel_output_1_b_right, _K_b
        | Classification, W ->
          lambda.classification_w_left, lambda.classification_w_right, _K_w
        | Classification, B ->
          lambda.classification_b_left, lambda.classification_b_right, _K_b
        | _ -> assert false
      in
      let u_left, s_left, _ =
        Tensor.svd ~some:true ~compute_uv:true Maths.(primal left)
      in
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
          let tmp =
            match param_name with
            | Patchify, W ->
              let u_right_reshaped =
                Tensor.reshape u_right ~shape:[ in_channels; patch_size; patch_size ]
              in
              Tensor.einsum
                ~path:None
                ~equation:"i,jkl->ijkl"
                [ u_left; u_right_reshaped ]
            | _ -> Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_left; u_right ]
          in
          Tensor.unsqueeze tmp ~dim:0)
        |> Tensor.concatenate ~dim:0
      in
      local_vs, n_per_param
    in
    vs |> localise ~param_name ~n_per_param

  let eigenvectors ~(lambda : A.M.t) () (_K : int) =
    let eigenvectors_each =
      List.map param_names_list ~f:(fun param_name ->
        eigenvectors_for_each_params ~lambda ~param_name)
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
    let eye_h = init_eye h in
    let eye_s = init_eye s in
    let eye_c = init_eye c in
    { patchify_w_left = init_eye c
    ; patchify_w_right = init_eye Int.(in_channels * patch_size * patch_size)
    ; patchify_b_left = init_eye 1
    ; patchify_b_right = eye_c
    ; token_hidden_0_w_left = eye_s
    ; token_hidden_0_w_right = eye_h
    ; token_hidden_0_b_left = eye_h
    ; token_hidden_0_b_right = eye_c
    ; token_output_0_w_left = eye_h
    ; token_output_0_w_right = eye_s
    ; token_output_0_b_left = eye_s
    ; token_output_0_b_right = eye_c
    ; channel_hidden_0_w_left = eye_c
    ; channel_hidden_0_w_right = eye_h
    ; channel_hidden_0_b_left = eye_h
    ; channel_hidden_0_b_right = eye_s
    ; channel_output_0_w_left = eye_h
    ; channel_output_0_w_right = eye_c
    ; channel_output_0_b_left = eye_s
    ; channel_output_0_b_right = eye_c
    ; token_hidden_1_w_left = eye_s
    ; token_hidden_1_w_right = eye_h
    ; token_hidden_1_b_left = eye_h
    ; token_hidden_1_b_right = eye_c
    ; token_output_1_w_left = eye_h
    ; token_output_1_w_right = eye_s
    ; token_output_1_b_left = eye_s
    ; token_output_1_b_right = eye_c
    ; channel_hidden_1_w_left = eye_c
    ; channel_hidden_1_w_right = eye_h
    ; channel_hidden_1_b_left = eye_h
    ; channel_hidden_1_b_right = eye_s
    ; channel_output_1_w_left = eye_h
    ; channel_output_1_w_right = eye_c
    ; channel_output_1_b_left = eye_s
    ; channel_output_1_b_right = eye_c
    ; classification_w_left = eye_c
    ; classification_w_right = init_eye output_dim
    ; classification_b_left = init_eye 1
    ; classification_b_right = init_eye output_dim
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
        ; steps = 5
        }
    in
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 0.01
      ; n_tangents = _K
      ; rank_one = false
      ; damping = Some 1e-3
      ; aux = Some aux
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
