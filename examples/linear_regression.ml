(* Example a simple linear regression task to learn W where y = W x. *)
open Utils
open Base
open Torch
open Forward_torch
open Sofo

let print s = Stdio.print_endline (Sexp.to_string_hum s)
let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)
module P = Prms.Single

module Model = struct
  module P = P

  let f ~(theta : _ Maths.some P.t) input = Maths.(input *@ theta)

  let init ~d_in ~d_out : P.param =
    let sigma = Float.(1. / sqrt (of_int d_in)) in
    let theta =
      Maths.(sigma $* randn ~kind:base.kind ~device:base.device [ d_in; d_out ])
    in
    P.free theta
end

let _K = 10

(* -----------------------------------------
   -- Generate linear regression data.
   ----------------------------------------- *)

let d_in, d_out = 100, 3

let data_minibatch =
  let teacher = Model.init ~d_in ~d_out |> Model.P.value in
  let input_cov_sqrt =
    let u, _ = Maths.C.qr (Maths.randn ~device:base.device [ d_in; d_in ]) in
    let lambda =
      Array.init d_in ~f:(fun i -> Float.(1. / (1. + square (of_int Int.(i + 1)))))
      |> Maths.of_array ~device:base.device ~shape:[ d_in; 1 ]
      |> fun x -> Maths.C.(x / mean x)
    in
    Maths.C.(sqrt lambda * u)
  in
  fun bs ->
    let x = Maths.(randn ~device:base.device [ bs; d_in ] *@ input_cov_sqrt) in
    let y = Model.f ~theta:teacher x in
    x, y

(* -----------------------------------------
   -- Optimization loop       ------
   ----------------------------------------- *)

let batch_size = 512
let max_iter = 10_000
let _K = 10
let n_per_param = _K

module RNN_Spec = struct
  type param_name = W [@@deriving compare, sexp]

  let all = [ W ]

  let shape = function
    | W -> [ d_in; d_out ]

  let n_params = function
    | W -> _K

  let n_params_list = [ _K ]
  let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
end

module RNN_Aux = struct
  type 'a p =
    { w_left : 'a
    ; w_right : 'a
    }
  [@@deriving prms]
end

module A = RNN_Aux.Make (Prms.Single)

module GGN : Auxiliary with module P = P = struct
  module P = P
  module A = A

  let init_sampling_state () = 0

  (* cache storage with a ref to memoize computed results *)
  let s_u_cache = ref []

  let g12v ~(lambda : ([< `const | `dual ] as 'a) A.t) (v : 'a P.t) : Maths.any P.t =
    Maths.einsum [ lambda.w_left, "in"; v, "aij"; lambda.w_right, "jm" ] "anm"

  let random_localised_vs () =
    Tensor.randn ~device:base.device ~kind:base.kind [ _K; d_in; d_out ]
    |> Maths.of_tensor

  let eigenvectors ~(lambda : _ Maths.some A.t) ~switch_to_learn t (_K : int) =
    let left, right, n_per_param = lambda.w_left, lambda.w_right, _K in
    let s_all, u_left, u_right = get_svals_u_left_right left right in
    (* randomly select the indices *)
    let n_params =
      Int.((Maths.shape left |> List.hd_exn) * (Maths.shape right |> List.hd_exn))
    in
    let selection =
      List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    let vs = get_local_vs ~selection ~s_all ~u_left ~u_right |> Maths.of_tensor in
    (* reset s_u_cache and set sampling state to 0 if learn again *)
    if switch_to_learn then s_u_cache := [];
    let new_sampling_state = if switch_to_learn then 0 else Int.(t + 1) in
    vs, new_sampling_state

  let init () =
    let init_eye size =
      Owl.Dense.Matrix.S.(0.1 $* eye size)
      |> Tensor.of_bigarray ~device:base.device
      |> Maths.of_tensor
      |> Prms.Single.free
    in
    RNN_Aux.{ w_left = init_eye d_in; w_right = init_eye d_out }
end

module O = Optimizer.SOFO (Model.P) (GGN)

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
    { base; learning_rate = Some 0.1; n_tangents = _K; damping = `none; aux = Some aux }

let rec loop ~t ~out ~(state : O.state) =
  let open Maths in
  Stdlib.Gc.major ();
  let x, y = data_minibatch batch_size in
  let theta, tangents, new_sampling_state = O.prepare ~config state in
  let y_pred = Model.f ~theta x in
  let loss = Loss.mse ~output_dims:[ 1 ] (y - y_pred) in
  let ggn = Loss.mse_ggn ~output_dims:[ 1 ] (const y) ~vtgt:(tangent_exn y_pred) in
  let new_state =
    O.step
      ~config
      ~info:{ loss; ggn; tangents; sampling_state = new_sampling_state }
      state
  in
  if t % 100 = 0
  then (
    let loss = to_float_exn (const loss) in
    print [%message (t : int) (loss : float)];
    Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss |] 1 2)));
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state

(* Start the loop. *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % in_dir "loss") |> Bos.OS.Cmd.run |> ignore;
  Bos.Cmd.(v "rm" % "-f" % in_dir "aux") |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init (Model.init ~d_in ~d_out))
