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
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let cast = Dense.Matrix.Generic.cast_d2s
let base = Optimizer.Config.Base.default
let output_dim = 10
let batch_size = 256

(* for cifar *)
let input_size = Int.(32 * 32 * 3)

(* for mlp *)
(* let input_size = Int.(28 * 28) *)
let full_batch_size = 50_000
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
    }
  [@@deriving prms]
end

module P = Prms.Array (MLP_Layer.Make (Prms.P))

(* neural network *)
module MLP = struct
  module P = P

  type input = Tensor.t

  let f ~(theta : P.M.t) ~(input : input) =
    let bs = Tensor.shape input |> List.hd_exn in 
    Array.foldi theta ~init:(Maths.const input) ~f:(fun i accu wb ->
      let open MLP_Layer in
      let accu =
        Maths.concat
          accu
          (Maths.const
             (Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ]))
          ~dim:1
      in
      let pre_activation = Maths.(accu *@ wb.w) in
      if false && i = Array.length layer_sizes - 1
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
          ~a:(n_i + 1)
          ~b:n_o
          ~sigma:1.
      in
      { id = i; w })
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
   -- Generate mnist data. ------
   ----------------------------------------- *)
(* let dataset typ =
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
      let xs = Tensor.reshape xs_tensor ~shape:[ total_bs; input_size ] in
      xs, Tensor.of_bigarray ~device:base.device set_y)
    else (
      let ids = List.init batch_size ~f:(fun _ -> Random.int a) in
      let x_tensor =
        Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_x)
      in
      let xs = Tensor.reshape x_tensor ~shape:[ batch_size; input_size ] in
      let ys = Tensor.of_bigarray ~device:base.device (Mat.get_fancy [ L ids ] set_y) in
      xs, ys) *)

(* -----------------------------------------
  -- Generate CIFAR data. ------
  ----------------------------------------- *)
let dataset =
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
  (x_train, y_train), (x_test, y_test)

let train_set, test_set = dataset

let sample_data (set_x, set_y) =
  let a = (Arr.shape set_x).(0) in
  (* let a = Mat.row_num set_x in *)
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

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)

let n_params_w = 330
let _K = Array.length layer_sizes * n_params_w

module GGN : Wrapper.Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { w_left : 'a
      ; w_right : 'a
      }
    [@@deriving prms]
  end

  module P = P
  module A = Prms.Array (Make (Prms.P))

  type sampling_state = unit

  let init_sampling_state () = ()

  let zero_params ~shape _K =
    Tensor.zeros ~device:base.device ~kind:base.kind (_K :: shape)

  let random_params ~shape _K =
    Tensor.randn ~device:base.device ~kind:base.kind (_K :: shape)

  (* approximation defined implicitly via Gv products *)
  let g12v ~(lambda : A.M.t) (v : P.M.t) : P.M.t =
    let open Maths in
    Array.map2_exn lambda v ~f:(fun lambda v ->
      let w = einsum [ lambda.w_left, "in"; v.w, "aij"; lambda.w_right, "jm" ] "anm" in
      MLP_Layer.{ id = 0; w })

  let get_shapes id =
    match id with
    | 0 -> [ input_size + 1; layer_sizes.(0) ]
    | 1 -> [ layer_sizes.(0)+ 1; layer_sizes.(1) ]
    | 2 -> [ layer_sizes.(1)+ 1; layer_sizes.(2) ]
    | _ -> assert false

  (* set tangents = zero for other parameters but v for this parameter *)

  let localise ~local ~id:i ~n_per_param v =
    Array.init n_layers ~f:(fun id ->
      let sample = if local then zero_params else random_params in
      let params_tmp =
        let w_shape = get_shapes id in
        let w = sample ~shape:w_shape n_per_param in
        MLP_Layer.{ id; w }
      in
      if id = i then { params_tmp with w = v } else params_tmp)

  let random_localised_vs _K : P.T.t =
    Array.init n_layers ~f:(fun id ->
      let w_shape = get_shapes id in
      let w = random_params ~shape:w_shape _K in
      MLP_Layer.{ id; w })

  let eigenvectors_for_each_params ~local ~lambda ~id =
    let left, right, n_per_param = lambda.w_left, lambda.w_right, n_params_w in
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
        Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_left; u_right ]
        |> Tensor.unsqueeze ~dim:0)
      |> Tensor.concatenate ~dim:0
    in
    localise ~local ~id ~n_per_param local_vs

  let eigenvectors ~(lambda : A.M.t) () (_K : int) =
    let concat_vs a b = P.map2 a b ~f:(fun x y -> Tensor.concat ~dim:0 [ x; y ]) in
    let vs =
      Array.foldi lambda ~init:None ~f:(fun id accu lambda ->
        let local_vs = eigenvectors_for_each_params ~local:true ~lambda ~id in
        match accu with
        | None -> Some local_vs
        | Some a -> Some (concat_vs a local_vs))
    in
    Option.value_exn vs, ()

  let init () =
    let init_eye size =
      Mat.(0.1 $* eye size) |> Tensor.of_bigarray ~device:base.device |> Prms.free
    in
    Array.init n_layers ~f:(fun id ->
      let w_shape = get_shapes id in
      let w_left = init_eye (List.hd_exn w_shape) in
      let w_right = init_eye (List.last_exn w_shape) in
      { w_left; w_right })
end

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
            test_eval ~train_data:None MLP.P.(const (value (O.params new_state)))
          in
          let train_acc =
            test_eval ~train_data:(Some data) MLP.P.(const (value (O.params new_state)))
          in
          (* let params = O.params state in
          let n_params = O.W.P.T.numel (O.W.P.map params ~f:(fun p -> Prms.value p)) in  *)
          (* avg error *)
          Convenience.print [%message (e : float) (loss_avg : float) (test_acc : float)];
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
      ; learning_rate = Some 0.03
      ; n_tangents = _K
      ; rank_one = false
      ; damping = Some 1e-3
      ; aux = Some aux
      }

  let init = O.init MLP.init
end

(* --------------------------------
       -- Adam
       --------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (FF)

  let config ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 1e-3 }

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
