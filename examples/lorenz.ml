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
  (* Random.self_init (); *)
  Owl_stats_prng.init 1985;
  Torch_core.Wrapper.manual_seed 1985

let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run
let base = Optimizer.Config.Base.default

module PP = struct
  type 'a p =
    { w : 'a
    ; c : 'a
    ; a : 'a
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.Single)

(* neural network *)
module RNN = struct
  module P = P

  let forward ~(theta : _ Maths.some P.t) ~input:_ y =
    let bs = Maths.shape y |> List.hd_exn in
    let y_tmp =
      Maths.concat
        [ y; Maths.(any (ones ~device:base.device ~kind:base.kind [ bs; 1 ])) ]
        ~dim:1
    in
    Maths.((y *@ theta.a) + (relu (y_tmp *@ theta.c) *@ theta.w))

  (* here data is a list of (x, optional labels) *)
  let f ~data ~y0 theta =
    let scaling = Float.(1. / of_int (List.length data)) in
    let result, _ =
      List.foldi
        data
        ~init:(None, Maths.(any (of_tensor y0)))
        ~f:(fun t (accu, y) (x, labels) ->
          if t % 1 = 0 then Stdlib.Gc.major ();
          let y = forward ~theta ~input:x y in
          let accu =
            match labels with
            | None -> accu
            | Some labels ->
              let delta_ell =
                Maths.(
                  scaling $* Loss.mse ~output_dims:[ 1 ] Maths.(of_tensor labels - y))
              in
              let delta_ggn =
                Maths.C.(
                  scaling
                  $* Loss.mse_ggn
                       ~output_dims:[ 1 ]
                       (Maths.const y)
                       ~vtgt:(Maths.tangent_exn y))
              in
              (match accu with
               | None -> Some (delta_ell, delta_ggn)
               | Some accu ->
                 let ell_accu, ggn_accu = accu in
                 Some (Maths.(ell_accu + delta_ell), Maths.C.(ggn_accu + delta_ggn)))
          in
          accu, y)
    in
    Option.value_exn result

  let init ~d ~dh : P.param =
    let w =
      Sofo.gaussian_tensor_normed ~kind:base.kind ~device:base.device ~sigma:0.1 [ dh; d ]
      |> Maths.of_tensor
      |> Prms.Single.free
    and c =
      Sofo.gaussian_tensor_normed
        ~kind:base.kind
        ~device:base.device
        ~sigma:1.
        [ d + 1; dh ]
      |> Maths.of_tensor
      |> Prms.Single.free
    and a = Maths.(eye ~device:base.device ~kind:base.kind d) |> Prms.Single.free in
    PP.{ w; c; a }

  let simulate ~(theta : _ Maths.some P.t) ~horizon y0 =
    let rec iter t accu y =
      if t = 0
      then List.rev accu
      else iter (t - 1) (y :: accu) (forward ~theta ~input:() y)
    in
    iter horizon [] Maths.(any (of_tensor y0))
end

(* -----------------------------------------
   -- Generate Lorenz data            ------
   ----------------------------------------- *)

let d = 3
let dh = 400
let batch_size = 256
let num_epochs_to_run = 4000
let n_trials_simulation = 10
let train_data = Lorenz_common.data 32
let test_horizon = 10000
let full_batch_size = Arr.(shape train_data).(1)
let _ = Sofo.print [%message (full_batch_size : int)]
let max_iter = Int.(full_batch_size * num_epochs_to_run / batch_size)

(* simulate n trials from saved parameters; first 3 columns are predictions and last 3 columns are ground truth *)
let simulate ~f_name n_trials =
  let model_params =
    let params_ba = P.C.load (in_dir f_name ^ "_params") in
    RNN.P.map params_ba ~f:(fun x -> x |> Maths.const)
  in
  let n_list = List.range 0 n_trials in
  List.iter n_list ~f:(fun j ->
    (* ground truth obtained from integration *)
    let y_true = Lorenz_common.data_test (test_horizon - 1) in
    (* use same initial condition to simulate with model *)
    let init_cond_sim = Mat.get_slice [ [ 0 ]; [] ] y_true in
    let simulated_arr =
      RNN.simulate
        ~theta:model_params
        ~horizon:test_horizon
        Tensor.(of_bigarray ~device:base.device init_cond_sim)
      |> List.map ~f:(fun yt ->
        let yt = Maths.to_tensor yt in
        let yt = Tensor.to_bigarray ~kind:base.ba_kind yt in
        Arr.expand yt 3)
      |> Array.of_list
      |> Arr.concatenate ~axis:0
      |> Arr.transpose ~axis:[| 1; 0; 2 |]
    in
    simulated_arr
    |> Arr.iter_slice ~axis:0 (fun yi ->
      let yi = Arr.squeeze yi in
      let yi_tot = Mat.concat_horizontal yi y_true in
      Mat.save_txt ~out:(in_dir (sprintf "%s_autonomous%i" f_name j)) yi_tot))

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
let _K = 128

type param_name =
  | W
  | C
  | A
[@@deriving compare]

let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
let n_params_w, n_params_c, n_params_a = 60, 60, Int.(_K - 120)
let n_params_list = [ n_params_w; n_params_c; n_params_a ]
let param_names_list = [ W; C; A ]

module GGN : Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { w_left : 'a
      ; w_right : 'a
      ; c_left : 'a
      ; c_right : 'a
      ; a_left : 'a
      ; a_right : 'a
      }
    [@@deriving prms]
  end

  module P = P

  (* module A = AA.Make (Prms.Single) *)
  module A = Make (Prms.Single)

  let init_sampling_state () = 0

  let zero_params ~shape _K =
    Tensor.zeros ~device:base.device ~kind:base.kind (_K :: shape)

  let random_params ~shape _K =
    Tensor.randn ~device:base.device ~kind:base.kind (_K :: shape)

  let get_shapes (param_name : param_name) =
    match param_name with
    | W -> [ dh; d ]
    | C -> [ d + 1; dh ]
    | A -> [ d; d ]

  let get_n_params (param_name : param_name) =
    match param_name with
    | W -> n_params_w
    | C -> n_params_c
    | A -> n_params_a

  let get_total_n_params (param_name : param_name) =
    let list_prod l = List.fold l ~init:1 ~f:(fun accu i -> accu * i) in
    list_prod (get_shapes param_name)

  let get_n_params_before_after (param_name : param_name) =
    let n_params_prefix_suffix_sums = prefix_suffix_sums n_params_list in
    let param_idx =
      match param_name with
      | W -> 0
      | C -> 1
      | A -> 2
    in
    List.nth_exn n_params_prefix_suffix_sums param_idx

  let g12v ~(lambda : ([< `const | `dual ] as 'a) A.t) (v : 'a P.t) : Maths.any P.t =
    let open Maths in
    let tmp_einsum left right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    let w = tmp_einsum lambda.w_left lambda.w_right v.w in
    let c = tmp_einsum lambda.c_left lambda.c_right v.c in
    let a = tmp_einsum lambda.a_left lambda.a_right v.a in
    { w; c; a }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~param_name ~n_per_param v =
    let w = zero_params ~shape:(get_shapes W) n_per_param in
    let c = zero_params ~shape:(get_shapes C) n_per_param in
    let a = zero_params ~shape:(get_shapes A) n_per_param in
    let params_tmp = PP.{ w; c; a } in
    match param_name with
    | W -> { params_tmp with w = v }
    | C -> { params_tmp with c = v }
    | A -> { params_tmp with a = v }

  let random_localised_vs () =
    let random_localised_param_name param_name =
      let w_shape = get_shapes param_name in
      let before, after = get_n_params_before_after param_name in
      let w = random_params ~shape:w_shape (get_n_params param_name) in
      let zeros_before = zero_params ~shape:w_shape before in
      let zeros_after = zero_params ~shape:w_shape after in
      let final = Tensor.concat [ zeros_before; w; zeros_after ] ~dim:0 in
      Maths.of_tensor final
    in
    PP.
      { w = random_localised_param_name W
      ; c = random_localised_param_name C
      ; a = random_localised_param_name A
      }

  (* compute sorted eigenvalues, u_left and u_right. *)
  let eigenvectors_for_params ~lambda ~param_name =
    let left, right =
      match param_name with
      | W -> lambda.w_left, lambda.w_right
      | C -> lambda.c_left, lambda.c_right
      | A -> lambda.a_left, lambda.a_right
    in
    let u_left, s_left, _ =
      Tensor.svd ~some:true ~compute_uv:true Maths.(to_tensor left)
    in
    let u_right, s_right, _ =
      Tensor.svd ~some:true ~compute_uv:true Maths.(to_tensor right)
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
    match List.Assoc.find !s_u_cache param_name ~equal:equal_param_name with
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
        let tmp = Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_l; u_r ] in
        Tensor.unsqueeze tmp ~dim:0)
      |> Tensor.concatenate ~dim:0
    in
    localise ~param_name ~n_per_param local_vs

  let eigenvectors_for_each_param ~lambda ~param_name ~sampling_state:_ =
    let n_per_param = get_n_params param_name in
    let n_params = get_total_n_params param_name in
    let s_all, u_left, u_right = get_s_u ~lambda ~param_name in
    let selection =
      (* List.init n_per_param ~f:(fun i ->
          ((sampling_state * n_per_param) + i) % n_params) *)
      List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    extract_local_vs ~s_all ~param_name ~u_left ~u_right ~selection

  let eigenvectors ~(lambda : _ Maths.some A.t) ~switch_to_learn t (_K : int) =
    let eigenvectors_each =
      List.map param_names_list ~f:(fun param_name ->
        eigenvectors_for_each_param ~lambda ~param_name ~sampling_state:t)
    in
    let vs =
      List.fold eigenvectors_each ~init:None ~f:(fun accu local_vs ->
        match accu with
        | None -> Some local_vs
        | Some a -> Some (P.map2 a local_vs ~f:(fun x y -> Tensor.concat ~dim:0 [ x; y ])))
    in
    let vs = Option.map vs ~f:(fun v -> P.map v ~f:Maths.of_tensor) in
    (* reset s_u_cache and set sampling state to 0 if learn again *)
    if switch_to_learn then s_u_cache := [];
    let new_sampling_state = if switch_to_learn then 0 else t + 1 in
    Option.value_exn vs, new_sampling_state

  let init () =
    let init_eye size =
      Maths.(0.1 $* eye ~device:base.device ~kind:base.kind size) |> Prms.Single.free
    in
    { w_left = init_eye dh
    ; w_right = init_eye d
    ; c_left = init_eye (d + 1)
    ; c_right = init_eye dh
    ; a_left = init_eye d
    ; a_right = init_eye d
    }
end

module O = Optimizer.SOFO (RNN.P) (GGN)

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
      ; local = true
      }
  in
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some 1.
    ; n_tangents = _K
    ; damping = `relative_from_top 1e-5
    ; aux = Some aux
    }

let rec loop ~t ~out ~state running_avg =
  Stdlib.Gc.major ();
  let init_cond, data =
    let trajectory = Lorenz_common.get_batch train_data batch_size in
    let init_cond = List.hd_exn trajectory in
    ( Tensor.of_bigarray ~device:base.device init_cond
    , List.mapi trajectory ~f:(fun tt x ->
        (* only label provided is the end point *)
        if tt = 31
        then (
          let x = Tensor.of_bigarray ~device:base.device x in
          (), Some x)
        else (), None) )
  in
  let theta, tangents, new_sampling_state = O.prepare ~config state in
  let loss, ggn = RNN.f ~data ~y0:init_cond theta in
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
        (RNN.P.value (O.params new_state))
        ~kind:base.ba_kind
        ~out:(in_dir "sofo_params");
      print [%message (t : int) (loss_avg : float)];
      Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss_avg |] 1 2)));
    []
  in
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state (loss :: running_avg)

(* Start the sofo loop. *)
let _ =
  Bos.Cmd.(v "rm" % "-f" % in_dir "loss") |> Bos.OS.Cmd.run |> ignore;
  Bos.Cmd.(v "rm" % "-f" % in_dir "aux") |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out:(in_dir "loss") ~state:(O.init (RNN.init ~d ~dh)) []
