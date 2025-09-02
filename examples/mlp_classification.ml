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
let n_layers = List.length layer_sizes
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

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
let n_params_w = 170
let _K = List.length layer_sizes * n_params_w

module MLP_Aux_Layer = struct
  type 'a p =
    { w_left : 'a
    ; w_right : 'a
    }
  [@@deriving prms]
end

module A = Prms.List (MLP_Aux_Layer.Make (Prms.Single))

module MLP_Spec = struct
  type param_name = int [@@deriving compare, sexp]

  let all = [ 0; 1 ]

  let shape = function
    | 0 -> [ input_size + 1; List.nth_exn layer_sizes 0 ]
    | 1 -> [ List.nth_exn layer_sizes 0 + 1; List.nth_exn layer_sizes 1 ]
    | _ -> assert false

  let n_params = function
    | _ -> n_params_w

  let n_params_list = [ n_params_w; n_params_w ]
  let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
end

module GGN : Auxiliary with module P = P = struct
  module P = P
  module A = A
  module C = Ggn_common.GGN_Common (MLP_Spec)

  let init_sampling_state () = 0

  (* approximation defined implicitly via Gv products *)
  let g12v ~(lambda : ([< `const | `dual ] as 'a) A.t) (v : 'a P.t) : Maths.any P.t =
    let open Maths in
    List.map2_exn lambda v ~f:(fun lambda v ->
      let w = einsum [ lambda.w_left, "in"; v.w, "aij"; lambda.w_right, "jm" ] "anm" in
      MLP_Layer.{ id = 0; w })

  let get_sides (lambda : ([< `const | `dual ] as 'a) A.t) (x : MLP_Spec.param_name)
    : 'a A.elt * 'a A.elt
    =
    let lambda_x = List.nth_exn lambda x in
    lambda_x.w_left, lambda_x.w_right

  let random_localised_vs () =
    let n_per_param = n_params_w in
    List.init n_layers ~f:(fun id ->
      let w_shape = MLP_Spec.shape id in
      let w =
        C.random_params ~shape:w_shape ~device:base.device ~kind:base.kind n_per_param
      in
      let zeros_before =
        C.zero_params ~shape:w_shape ~device:base.device ~kind:base.kind (n_per_param * id)
      in
      let zeros_after =
        C.zero_params
          ~shape:w_shape
          ~device:base.device
          ~kind:base.kind
          (n_per_param * (n_layers - 1 - id))
      in
      let final =
        if n_layers = 1 then w else Tensor.concat [ zeros_before; w; zeros_after ] ~dim:0
      in
      MLP_Layer.{ id; w = Maths.of_tensor final })

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~param_name ~n_per_param v =
    List.init n_layers ~f:(fun id ->
      let params_tmp =
        let w_shape = MLP_Spec.shape id in
        let w =
          C.zero_params ~shape:w_shape ~device:base.device ~kind:base.kind n_per_param
        in
        MLP_Layer.{ id; w }
      in
      if id = param_name then { params_tmp with w = v } else params_tmp)

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
    List.init n_layers ~f:(fun id ->
      let w_shape = MLP_Spec.shape id in
      let w_left = init_eye (List.hd_exn w_shape) in
      let w_right = init_eye (List.last_exn w_shape) in
      MLP_Aux_Layer.{ w_left; w_right })
end

(* -----------------------------------------
   -- Optimization with SOFO    ------
   ----------------------------------------- *)

module O = Optimizer.SOFO (MLP.P) (GGN)

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
    ; learning_rate = Some 0.05
    ; n_tangents = _K
    ; damping = `relative_from_top 1e-3
    ; aux = Some aux
    }

let rec loop ~t ~out ~state running_avg =
  Stdlib.Gc.major ();
  let data = sample_data train_set batch_size in
  let theta, tangents, new_sampling_state = O.prepare ~config state in
  let loss, ggn = MLP.f ~data theta in
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
        (MLP.P.value (O.params new_state))
        ~kind:base.ba_kind
        ~out:(in_dir "sofo_params");
      let e = epoch_of t in
      let test_acc =
        test_eval ~train_data:None MLP.P.(const (value (O.params new_state)))
      in
      let train_acc =
        test_eval ~train_data:(Some data) MLP.P.(const (value (O.params new_state)))
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
  loop ~t:0 ~out ~state:(O.init MLP.init) []
