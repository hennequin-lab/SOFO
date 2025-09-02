(* MLP model recovery (student/teacher setting) *)
open Base
open Forward_torch
open Torch
open Sofo
module Mat = Owl.Dense.Matrix.S
module Linalg = Owl.Linalg.S

let _ =
  Random.init 1985;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let d = 5
let layer_sizes = [| 25; 3 |]
let n_layers = Array.length layer_sizes
let bs = 600
let n_batches = 1
let max_iter = 2000
let _K = n_layers * 5

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f32)
    ; ba_kind = Bigarray.float32
    }

let ones_bs = Maths.ones ~device:base.device ~kind:base.kind [ bs; 1 ]

module MLP_Layer = struct
  type 'a p =
    { id : int
    ; w : 'a
    }
  [@@deriving prms]
end

module P = Prms.List (MLP_Layer.Make (Prms.Single))

let true_theta =
  List.init n_layers ~f:(fun id ->
    let n_i = if id = 0 then d else layer_sizes.(id - 1) in
    let n_o = layer_sizes.(id) in
    let w =
      Sofo.gaussian_tensor_normed
        ~kind:base.kind
        ~device:base.device
        ~sigma:1.
        [ n_i + 1; n_o ]
      |> Maths.of_tensor
    in
    MLP_Layer.{ id; w })

(* neural network *)
module MLP = struct
  module P = P

  let forward ~(theta : _ Maths.some P.t) ~input =
    let open Maths in
    List.fold theta ~init:input ~f:(fun accu p ->
      let accu = concat ~dim:1 [ accu; Maths.any ones_bs ] in
      tanh (accu *@ p.w))

  let f ~data:(input, labels) theta =
    let pred = forward ~theta ~input in
    let ell = Loss.mse ~output_dims:[ 1 ] Maths.(labels - pred) in
    let ggn =
      Loss.mse_ggn ~output_dims:[ 1 ] (Maths.const pred) ~vtgt:(Maths.tangent_exn pred)
    in
    ell, ggn

  let init : P.param =
    List.init n_layers ~f:(fun id ->
      let n_i = if id = 0 then d else layer_sizes.(id - 1) in
      let n_o = layer_sizes.(id) in
      let w =
        Sofo.gaussian_tensor_normed
          ~kind:base.kind
          ~device:base.device
          ~sigma:1.
          [ n_i + 1; n_o ]
        |> Maths.of_tensor
        |> Prms.Single.free
      in
      MLP_Layer.{ id; w })
end

(* -----------------------------------------
   -- Generate in data. ------
   ----------------------------------------- *)
(* data distribution generated using the teacher [true_theta] *)
let data_batches =
  let u, _, _ = Linalg.qr Mat.(gaussian d d) in
  let s = Mat.init 1 d (fun i -> Float.(exp (neg (of_int i / of_int 10)))) in
  let s = Mat.(s /$ mean' s) in
  let sigma12 = Mat.(transpose (u * sqrt s)) |> Maths.of_bigarray ~device:base.device in
  Array.init n_batches ~f:(fun _ ->
    let data_x =
      Maths.randn ~device:base.device ~kind:base.kind [ bs; d ]
      |> fun x -> Maths.(x *@ const sigma12)
    in
    let data_y = MLP.forward ~theta:true_theta ~input:data_x in
    data_x, data_y)

(* let normalising_const =
  Array.map data_batches ~f:(fun (_, data_y) ->
    Maths.(mean (sqr data_y) |> to_tensor) |> Tensor.to_float0_exn)
  |> Owl.Stats.mean
  |> fun x -> Float.(0.5 / x) *)

let sample_data () = data_batches.(Random.int n_batches)

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
let n_params_w = 5
let _K = n_layers * n_params_w

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
    | 0 -> [ d + 1; layer_sizes.(0) ]
    | 1 -> [ layer_sizes.(0) + 1; layer_sizes.(1) ]
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
            { default with base; learning_rate = Some 1e-3; eps = 1e-8 }
      ; steps = 5
      ; learn_steps = 1
      ; exploit_steps = 1
      }
  in
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some 0.1
    ; n_tangents = _K
    ; damping = `relative_from_top 1e-5
    ; aux = Some aux
    }

let rec loop ~t ~out ~state running_avg =
  Stdlib.Gc.major ();
  let data = sample_data () in
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
      print [%message (t : int) (loss_avg : float)];
      Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss_avg |] 1 2)));
    []
  in
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state (loss :: running_avg)

(* Start the sofo loop. *)
let _ =
  Bos.Cmd.(v "rm" % "-f" % in_dir "loss") |> Bos.OS.Cmd.run |> ignore;
  Bos.Cmd.(v "rm" % "-f" % in_dir "aux") |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out:(in_dir "loss") ~state:(O.init MLP.init) []
