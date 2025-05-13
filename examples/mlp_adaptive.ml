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
let d = 64
let n_layers = 3
let bs = 128
let n_batches = 100
let _K = n_layers * 20

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f32)
    ; ba_kind = Bigarray.float32
    }

let ones_bs = Tensor.ones ~device:base.device ~kind:base.kind [ bs; 1 ] |> Maths.const

include struct
  type 'a p =
    { id : int
    ; w : 'a
    }
  [@@deriving prms]
end

module P = Prms.List (Make (Prms.P))

let init_theta () =
  List.init n_layers ~f:(fun id ->
    let z = Float.(1. / sqrt (of_int d)) in
    let w = Tensor.(f z * randn ~device:base.device ~kind:base.kind Int.[ d + 1; d ]) in
    { id; w = Maths.const w })

let true_theta = init_theta ()
let theta : P.tagged = init_theta () |> P.map ~f:(fun x -> Prms.free (Maths.primal x))
let n_params = P.M.numel true_theta
let ratio = Float.(100. * of_int _K / of_int n_params)
let _ = Convenience.print [%message (n_params : int) (ratio : float)]

let forward ~theta x =
  let open Maths in
  List.fold theta ~init:x ~f:(fun accu p ->
    let accu = concat ~dim:1 accu ones_bs in
    tanh (accu *@ p.w))

(* data distribution generated using the teacher [true_theta] *)
let data_batches =
  let u, _, _ = Linalg.qr Mat.(gaussian d d) in
  let s = Mat.init 1 d (fun i -> Float.(exp (neg (of_int i / of_int 10)))) in
  let s = Mat.(s /$ mean' s) in
  let sigma12 = Mat.(transpose (u * sqrt s)) |> Tensor.of_bigarray ~device:base.device in
  Array.init n_batches ~f:(fun _ ->
    let data_x =
      Tensor.randn ~device:base.device ~kind:base.kind [ bs; d ]
      |> Maths.const
      |> fun x -> Maths.(x *@ const sigma12)
    in
    let data_y = forward ~theta:true_theta data_x in
    data_x, data_y)

let normalising_const =
  Array.map data_batches ~f:(fun (_, data_y) ->
    Maths.(mean (sqr data_y) |> primal) |> Tensor.to_float0_exn)
  |> Owl.Stats.mean
  |> fun x -> Float.(0.5 / x)

let sample_data () = data_batches.(Random.int n_batches)

let loss_and_ggn ?(with_ggn = true) ~data:(data_x, data_y) (theta : P.M.t) =
  let open Maths in
  let y = forward ~theta data_x in
  let loss = mean_dim ~keepdim:false ~dim:(Some [ 1 ]) (sqr (data_y - y)) in
  let loss = f normalising_const * loss in
  let ggn =
    if with_ggn
    then (
      let y_t = Option.value_exn (tangent y) |> const in
      Some
        (einsum [ y_t, "kmi"; y_t, "lmi" ] "kl" / f Float.(of_int d)
         |> primal
         |> fun x -> Tensor.(f normalising_const * x)))
    else None
  in
  Option.iter ggn ~f:(fun ggn ->
    let s = Linalg.svdvals (Tensor.to_bigarray ~kind:base.ba_kind ggn) in
    Mat.(save_txt ~out:(in_dir "svals") (transpose s)));
  loss, ggn

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)

module GGN : Wrapper.Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { left : 'a
      ; right : 'a
      }
    [@@deriving prms]
  end

  module P = P
  module A = Prms.List (Make (Prms.P))

  type sampling_state = unit

  let init_sampling_state () = ()

  let zero_params ~shape _K =
    Tensor.zeros ~device:base.device ~kind:base.kind (_K :: shape)

  let random_params ~shape _K =
    Tensor.randn ~device:base.device ~kind:base.kind (_K :: shape)

  (* approximation defined implicitly via Gv products *)
  let g12v ~(lambda : A.M.t) (v : P.M.t) : P.M.t =
    let open Maths in
    List.map2_exn lambda v ~f:(fun lambda v ->
      let w = einsum [ lambda.left, "in"; v.w, "aij"; lambda.right, "jm" ] "anm" in
      { id = 0; w })

  let localise ~id:i ~_K v =
    List.init n_layers ~f:(fun id ->
      let w = if id = i then v else zero_params ~shape:[ d + 1; d ] _K in
      { id; w })

  let n_per_layer = _K / n_layers

  let random_localised_vs _K : P.T.t =
    List.init n_layers ~f:(fun id ->
      let w_shape = [ d + 1; d ] in
      let w = random_params ~shape:w_shape n_per_layer in
      let zeros_before = zero_params ~shape:w_shape (n_per_layer * id) in
      let zeros_after = zero_params ~shape:w_shape (n_per_layer * (n_layers - 1 - id)) in
      let final =
        if n_layers = 1 then w else Tensor.concat [ zeros_before; w; zeros_after ] ~dim:0
      in
      { id; w = final })

  let eigenvectors ~lambda () (_K : int) =
    let vs =
      (* for each layer, compute the eigenvectors of corresponding ggn and sample from it *)
      List.foldi lambda ~init:None ~f:(fun id accu lambda ->
        let u_left, s_left, _ =
          Tensor.svd ~some:true ~compute_uv:true Maths.(primal lambda.left)
        in
        let u_right, s_right, _ =
          Tensor.svd ~some:true ~compute_uv:true Maths.(primal lambda.right)
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
        let selection =
          List.permute (List.range 0 Int.(d * (d + 1)))
          |> List.sub ~pos:0 ~len:n_per_layer
        in
        let selection = List.map selection ~f:(fun j -> s_all.(j)) in
        let local_vs =
          List.map selection ~f:(fun (il, ir, _) ->
            let u_left =
              Tensor.(
                squeeze
                  (slice u_left ~dim:1 ~start:(Some il) ~end_:(Some Int.(il + 1)) ~step:1))
            in
            let u_right =
              Tensor.(
                squeeze
                  (slice
                     u_right
                     ~dim:1
                     ~start:(Some ir)
                     ~end_:(Some Int.(ir + 1))
                     ~step:1))
            in
            Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_left; u_right ]
            |> Tensor.unsqueeze ~dim:0)
          |> Tensor.concatenate ~dim:0
        in
        let local_vs = local_vs |> localise ~id ~_K:n_per_layer in
        match accu with
        | None -> Some local_vs
        | Some a -> Some (P.map2 a local_vs ~f:(fun x y -> Tensor.concat ~dim:0 [ x; y ])))
    in
    let vs = Option.value_exn vs in
    (* sanity checking: the sketch should be block-diagonal *)
    let _ =
      let open Maths in
      let g12v = g12v ~lambda (P.map vs ~f:const) in
      let sketch =
        P.fold g12v ~init:(f 0.) ~f:(fun accu (gv, _) ->
          let n_tangents = Convenience.first_dim (primal gv) in
          let gv = reshape gv ~shape:[ n_tangents; -1 ] in
          accu + einsum [ gv, "ki"; gv, "ki" ] "k")
      in
      sketch
      |> primal
      |> Tensor.reshape ~shape:[ -1; 1 ]
      |> Tensor.to_bigarray ~kind:Bigarray.float32
      |> Owl.Dense.Matrix.S.save_txt ~out:"sketch2"
    in
    vs, ()

  let init () =
    let left = Mat.(0.1 $* eye Int.(d + 1)) |> Tensor.of_bigarray ~device:base.device in
    let right = Mat.(0.1 $* eye d) |> Tensor.of_bigarray ~device:base.device in
    List.init n_layers ~f:(fun _ -> { left = Prms.free left; right = Prms.free right })
end

(* ------------------------------------------------
   --- Optimisation
   ------------------------------------------------ *)

module M = struct
  module P = P

  type data = Maths.t * Maths.t
  type args = bool

  let f ~update ~data ~init ~args:with_ggn (theta : P.M.t) =
    let loss, ggn = loss_and_ggn ~with_ggn ~data theta in
    match update with
    | `loss_only u -> u init (Some loss)
    | `loss_and_ggn u -> u init (Some (loss, ggn))
end

(* --------------------------------
   -- Generic type of optimiser
   -------------------------------- *)

module type Do_with_T = sig
  module O :
    Optimizer.T
    with type 'a W.P.p = 'a M.P.p
     and type W.data = Maths.t * Maths.t
     and type W.args = bool

  val name : string
  val config : iter:int -> (float, Bigarray.float32_elt) O.config
  val init : O.state
  val with_ggn : bool
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    Bos.Cmd.(v "rm" % "-f" % in_dir "aux") |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state running_avg data =
      Stdlib.Gc.major ();
      let data = if iter % 2 = 0 then sample_data () else data in
      let loss, new_state = O.step ~config:(config ~iter) ~state ~data with_ggn in
      let running_avg =
        let loss_avg =
          match running_avg with
          | [] -> loss
          | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
        in
        (* save params *)
        if iter % 10 = 0
        then (
          (* avg error *)
          Convenience.print [%message (iter : int) (loss_avg : float)];
          let t = iter in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array [| Float.of_int t; loss_avg |] 1 2)));
        []
      in
      if iter < max_iter
      then loop ~iter:(iter + 1) ~state:new_state (loss :: running_avg) data
    in
    loop ~iter:0 ~state:init [] (sample_data ())
end

(* --------------------------------
   -- SOFO
   -------------------------------- *)

module Do_with_SOFO : Do_with_T = struct
  module O = Optimizer.SOFO (M) (GGN)

  let name = "sofo_adapt"

  let config ~iter:k =
    let aux =
      Optimizer.Config.SOFO.
        { (default_aux (in_dir "aux")) with
          config =
            Optimizer.Config.Adam.{ default with learning_rate = Some 1e-4; eps = 1e-4 }
        ; steps = 5
        }
    in
    Optimizer.Config.SOFO.
      { base
      ; learning_rate =
          (let lr = Cmdargs.(get_float "-lr" |> default 0.1) in
           Some Float.(lr / sqrt (1. + (0. * (of_int k / 100.)))))
      ; n_tangents = _K
      ; rank_one = false
      ; damping = Some 1e-3
      ; aux = Some aux
      }

  let init = O.init theta
  let with_ggn = true
end

(* --------------------------------
   -- Adam
   -------------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (M)

  let config ~iter:k =
    Optimizer.Config.Adam.
      { default with
        base
      ; learning_rate =
          (let lr = Cmdargs.(get_float "-lr" |> default 0.02) in
           Some Float.(lr / sqrt (1. + (0. * (of_int k / 100.)))))
      }

  let init = O.init theta
  let with_ggn = false
end

let _ =
  let max_iter = 100000 in
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
