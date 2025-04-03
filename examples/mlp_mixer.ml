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
let groups = 1

module MLP_Layer = struct
  type 'a t =
    { layer_name : string
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
             [ c; Int.(in_channels / groups); patch_size; patch_size ])
          (Scalar.f normaliser)
      in
      let b = Tensor.zeros ~kind:base.kind ~device:base.device [ 1; c ] in
      { layer_name = "patchify"; w; b }
    in
    let mixer_layers =
      List.map blocks_list ~f:(fun i ->
        let block_idx = Int.to_string i in
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
          let layer_name = sprintf "token_hidden_%s" block_idx in
          { layer_name; w; b }
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
          let layer_name = sprintf "token_output_%s" block_idx in
          { layer_name; w; b }
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
          let layer_name = sprintf "channel_hidden_%s" block_idx in
          { layer_name; w; b }
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
          let layer_name = sprintf "channel_output_%s" block_idx in
          { layer_name; w; b }
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
      { layer_name = "classification"; w; b }
    in
    Array.concat ([ [| patchify |] ] @ mixer_layers @ [ [| classification_head |] ])
    |> P.map ~f:Prms.free
end

(* feedforward model with mse loss *)
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
let layer_names_list =
  [ "patchify"
  ; "token_hidden_0"
  ; "token_output_0"
  ; "channel_hidden_0"
  ; "channel_output_0"
  ; "token_hidden_1"
  ; "token_output_1"
  ; "channel_hidden_1"
  ; "channel_output_1"
  ; "classification"
  ]

let _K_w = 8
let _K_b = 1
let _K = Int.(List.length layer_names_list * (_K_w + _K_b))
let _ = Convenience.print [%message (_K : int)]

module GGN : Wrapper.Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { patchify_w_0 : 'a
      ; patchify_w_1 : 'a
      ; patchify_w_2 : 'a
      ; patchify_w_3 : 'a
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
    List.map layer_names_list ~f:(fun name -> [ name ^ "_w"; name ^ "_b" ]) |> List.concat

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
        | "patchify" ->
          let w =
            let w_0_dim = List.last_exn (shape lambda.patchify_w_0) in
            let w_1_dim = List.last_exn (shape lambda.patchify_w_1) in
            let w_2_dim = List.last_exn (shape lambda.patchify_w_2) in
            let w_3_dim = List.last_exn (shape lambda.patchify_w_3) in
            let right_tmp =
              kron lambda.patchify_w_3 lambda.patchify_w_2
              |> reshape ~shape:[ w_2_dim; w_3_dim; -1 ]
            in
            let left_tmp =
              kron lambda.patchify_w_1 lambda.patchify_w_0
              |> reshape ~shape:[ w_0_dim; w_1_dim; -1 ]
            in
            einsum [ left_tmp, "abc"; v.w, "kabde"; right_tmp, "def" ] "kcf"
          in
          let b =
            einsum_w ~left:lambda.patchify_b_left ~right:lambda.patchify_b_right v.b
          in
          w, b
        | "token_hidden_0" ->
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
        | "token_output_0" ->
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
        | "channel_hidden_0" ->
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
        | "channel_output_0" ->
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
        | "token_hidden_1" ->
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
        | "token_output_1" ->
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
        | "channel_hidden_1" ->
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
        | "channel_output_1" ->
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
        | "classification" ->
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

  let get_shapes layer_name =
    let w_shape, b_shape =
      match layer_name with
      | "patchify" -> [ c; Int.(in_channels / groups); patch_size; patch_size ], [ 1; c ]
      | "token_hidden_0" | "token_hidden_1" -> [ s; h ], [ h; c ]
      | "token_output_0" | "token_output_1" -> [ h; s ], [ s; c ]
      | "channel_hidden_0" | "channel_hidden_1" -> [ c; h ], [ h; s ]
      | "channel_output_0" | "channel_output_1" -> [ h; c ], [ s; c ]
      | "classification" -> [ c; output_dim ], [ 1; output_dim ]
      | _ -> assert false
    in
    w_shape, b_shape

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~param_name ~n_per_param v =
    List.map layer_names_list ~f:(fun layer_name ->
      let w_shape, b_shape = get_shapes layer_name in
      let w = zero_params ~shape:w_shape n_per_param in
      let b = zero_params ~shape:b_shape n_per_param in
      let params_tmp = MLP_Layer.{ layer_name; w; b } in
      match param_name with
      | _ when String.equal param_name (layer_name ^ "_w") ->
        MLP_Layer.{ params_tmp with w = v }
      | _ when String.equal param_name (layer_name ^ "_b") ->
        MLP_Layer.{ params_tmp with b = v }
      | _ -> params_tmp)
    |> List.to_array

  let random_localised_vs _K : P.T.t =
    List.map layer_names_list ~f:(fun layer_name ->
      let w_shape, b_shape = get_shapes layer_name in
      let w = random_params ~shape:w_shape _K in
      let b = random_params ~shape:b_shape _K in
      MLP_Layer.{ layer_name; w; b })
    |> List.to_array

  let eigenvectors_for_each_params ~lambda ~param_name =
    let vs, n_per_param =
      match param_name with
      (* special case, 4 dims *)
      | "patchify_w" ->
        let n_per_param = _K_w in
        let w_0, w_1, w_2, w_3 =
          ( lambda.patchify_w_0
          , lambda.patchify_w_1
          , lambda.patchify_w_2
          , lambda.patchify_w_3 )
        in
        let u_0, s_0, _ = Tensor.svd ~some:true ~compute_uv:true Maths.(primal w_0) in
        let u_1, s_1, _ = Tensor.svd ~some:true ~compute_uv:true Maths.(primal w_1) in
        let u_2, s_2, _ = Tensor.svd ~some:true ~compute_uv:true Maths.(primal w_2) in
        let u_3, s_3, _ = Tensor.svd ~some:true ~compute_uv:true Maths.(primal w_3) in
        let s_0 = Tensor.to_float1_exn s_0 |> Array.to_list in
        let s_1 = Tensor.to_float1_exn s_1 |> Array.to_list in
        let s_2 = Tensor.to_float1_exn s_2 |> Array.to_list in
        let s_3 = Tensor.to_float1_exn s_3 |> Array.to_list in
        let s_all =
          List.mapi s_0 ~f:(fun i0 s0 ->
            List.concat
              (List.mapi s_1 ~f:(fun i1 s1 ->
                 List.concat
                   (List.mapi s_2 ~f:(fun i2 s2 ->
                      List.mapi s_3 ~f:(fun i3 s3 ->
                        i0, i1, i2, i3, Float.(s0 * s1 * s2 * s3)))))))
          |> List.concat
          |> List.sort ~compare:(fun (_, _, _, _, a) (_, _, _, _, b) -> Float.compare b a)
          |> Array.of_list
        in
        (* randomly select the indices *)
        let n_params =
          Convenience.first_dim (Maths.primal w_0)
          * Convenience.first_dim (Maths.primal w_1)
          * Convenience.first_dim (Maths.primal w_2)
          * Convenience.first_dim (Maths.primal w_3)
        in
        let selection =
          List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
        in
        let selection = List.map selection ~f:(fun j -> s_all.(j)) in
        let local_vs =
          List.map selection ~f:(fun (i_0, i1, i2, i3, _) ->
            let u_0 =
              Tensor.(
                squeeze_dim
                  ~dim:1
                  (slice u_0 ~dim:1 ~start:(Some i_0) ~end_:(Some Int.(i_0 + 1)) ~step:1))
            in
            let u_1 =
              Tensor.(
                squeeze_dim
                  ~dim:1
                  (slice u_1 ~dim:1 ~start:(Some i1) ~end_:(Some Int.(i1 + 1)) ~step:1))
            in
            let u_2 =
              Tensor.(
                squeeze_dim
                  ~dim:1
                  (slice u_2 ~dim:1 ~start:(Some i2) ~end_:(Some Int.(i2 + 1)) ~step:1))
            in
            let u_3 =
              Tensor.(
                squeeze_dim
                  ~dim:1
                  (slice u_3 ~dim:1 ~start:(Some i3) ~end_:(Some Int.(i3 + 1)) ~step:1))
            in
            let tmp =
              Tensor.einsum ~path:None ~equation:"i,j,k,l->ijkl" [ u_0; u_1; u_2; u_3 ]
            in
            Tensor.unsqueeze tmp ~dim:0)
          |> Tensor.concatenate ~dim:0
        in
        local_vs, n_per_param
      (* all other cases, 2 dim *)
      | _ ->
        let left, right, n_per_param =
          match param_name with
          | "patchify_b" -> lambda.patchify_b_left, lambda.patchify_b_right, _K_b
          | "token_hidden_0_w" ->
            lambda.token_hidden_0_w_left, lambda.token_hidden_0_w_right, _K_w
          | "token_hidden_0_b" ->
            lambda.token_hidden_0_b_left, lambda.token_hidden_0_b_right, _K_b
          | "token_output_0_w" ->
            lambda.token_output_0_w_left, lambda.token_output_0_w_right, _K_w
          | "token_output_0_b" ->
            lambda.token_output_0_b_left, lambda.token_output_0_b_right, _K_b
          | "channel_hidden_0_w" ->
            lambda.channel_hidden_0_w_left, lambda.channel_hidden_0_w_right, _K_w
          | "channel_hidden_0_b" ->
            lambda.channel_hidden_0_b_left, lambda.channel_hidden_0_b_right, _K_b
          | "channel_output_0_w" ->
            lambda.channel_output_0_w_left, lambda.channel_output_0_w_right, _K_w
          | "channel_output_0_b" ->
            lambda.channel_output_0_b_left, lambda.channel_output_0_b_right, _K_b
          | "token_hidden_1_w" ->
            lambda.token_hidden_1_w_left, lambda.token_hidden_1_w_right, _K_w
          | "token_hidden_1_b" ->
            lambda.token_hidden_1_b_left, lambda.token_hidden_1_b_right, _K_b
          | "token_output_1_w" ->
            lambda.token_output_1_w_left, lambda.token_output_1_w_right, _K_w
          | "token_output_1_b" ->
            lambda.token_output_1_b_left, lambda.token_output_1_b_right, _K_b
          | "channel_hidden_1_w" ->
            lambda.channel_hidden_1_w_left, lambda.channel_hidden_1_w_right, _K_w
          | "channel_hidden_1_b" ->
            lambda.channel_hidden_1_b_left, lambda.channel_hidden_1_b_right, _K_b
          | "channel_output_1_w" ->
            lambda.channel_output_1_w_left, lambda.channel_output_1_w_right, _K_w
          | "channel_output_1_b" ->
            lambda.channel_output_1_b_left, lambda.channel_output_1_b_right, _K_b
          | "classification_w" ->
            lambda.classification_w_left, lambda.classification_w_right, _K_w
          | "classification_b" ->
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
                  (slice
                     u_right
                     ~dim:1
                     ~start:(Some ir)
                     ~end_:(Some Int.(ir + 1))
                     ~step:1))
            in
            let tmp = Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_left; u_right ] in
            Tensor.unsqueeze tmp ~dim:0)
          |> Tensor.concatenate ~dim:0
        in
        local_vs, n_per_param
      | _ -> assert false
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
    { patchify_w_0 = init_eye c
    ; patchify_w_1 = init_eye Int.(in_channels / groups)
    ; patchify_w_2 = init_eye patch_size
    ; patchify_w_3 = init_eye patch_size
    ; patchify_b_left = init_eye 1
    ; patchify_b_right = init_eye c
    ; token_hidden_0_w_left = init_eye s
    ; token_hidden_0_w_right = init_eye h
    ; token_hidden_0_b_left = init_eye h
    ; token_hidden_0_b_right = init_eye c
    ; token_output_0_w_left = init_eye h
    ; token_output_0_w_right = init_eye s
    ; token_output_0_b_left = init_eye s
    ; token_output_0_b_right = init_eye c
    ; channel_hidden_0_w_left = init_eye c
    ; channel_hidden_0_w_right = init_eye h
    ; channel_hidden_0_b_left = init_eye h
    ; channel_hidden_0_b_right = init_eye s
    ; channel_output_0_w_left = init_eye h
    ; channel_output_0_w_right = init_eye c
    ; channel_output_0_b_left = init_eye s
    ; channel_output_0_b_right = init_eye c
    ; token_hidden_1_w_left = init_eye s
    ; token_hidden_1_w_right = init_eye h
    ; token_hidden_1_b_left = init_eye h
    ; token_hidden_1_b_right = init_eye c
    ; token_output_1_w_left = init_eye h
    ; token_output_1_w_right = init_eye s
    ; token_output_1_b_left = init_eye s
    ; token_output_1_b_right = init_eye c
    ; channel_hidden_1_w_left = init_eye c
    ; channel_hidden_1_w_right = init_eye h
    ; channel_hidden_1_b_left = init_eye h
    ; channel_hidden_1_b_right = init_eye s
    ; channel_output_1_w_left = init_eye h
    ; channel_output_1_w_right = init_eye c
    ; channel_output_1_b_left = init_eye s
    ; channel_output_1_b_right = init_eye c
    ; classification_w_left = init_eye c
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
          (* avg error *)
          Convenience.print [%message (iter : int) (loss_avg : float) (test_acc : float)];
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
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.02 }

  let init = O.init MLP_mixer.init
end

let _ =
  let max_iter = 1000 in
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
