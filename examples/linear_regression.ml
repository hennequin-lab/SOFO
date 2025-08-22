(* Example a simple linear regression task to learn W where y = W x. *)
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

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
module GGN : Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { theta_left : 'a
      ; theta_right : 'a
      }
    [@@deriving prms]
  end

  module P = P

  (* module A = AA.Make (Prms.Single) *)
  module A = Make (Prms.Single)

  type sampling_state = int

  let init_sampling_state () = 0

  (* cache storage with a ref to memoize computed results *)
  let s_u_cache = ref []

  let g12v ~(lambda : ([< `const | `dual ] as 'a) A.t) (v : 'a P.t) : Maths.any P.t =
    Maths.einsum [ lambda.theta_left, "in"; v, "aij"; lambda.theta_right, "jm" ] "anm"

  let random_localised_vs () : Tensor.t P.p =
    Tensor.randn ~device:base.device ~kind:base.kind [ _K; d_in; d_out ]

  let eigenvectors ~(lambda : _ Maths.some A.t) ~switch_to_learn t (_K : int) =
    let left, right, n_per_param = lambda.theta_left, lambda.theta_right, _K in
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
    (* randomly select the indices *)
    let n_params =
      Int.((Maths.shape left |> List.hd_exn) * (Maths.shape right |> List.hd_exn))
    in
    let selection =
      List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    let vs =
      List.map selection ~f:(fun idx ->
        let il, ir, _ = s_all.(idx) in
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
        let tmp = Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_left; u_right ] in
        Tensor.unsqueeze tmp ~dim:0)
      |> Tensor.concatenate ~dim:0
    in
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
    { theta_left = init_eye d_in; theta_right = init_eye d_out }
end

(* TODO: need to write the aux loop inside optimisation. *)
module O = Optimizer.SOFO (Model.P) (GGN)

let config =
  Optimizer.Config.SOFO.
    { base; learning_rate = Some 0.1; n_tangents = _K; damping = `none; aux = None }

let rec loop ~t ~out ~state =
  Stdlib.Gc.major ();
  let x, y = data_minibatch batch_size in
  (* extract aux learning flags. if switch to learn, switch to learning the ggn at the NEXT iteration *)
  let aux_learn, aux_exploit, switch_to_learn =
    match config.aux with
    | None -> false, false, false
    | Some { learn_steps; exploit_steps; _ } ->
      let rem = t % (learn_steps + exploit_steps) in
      let learn = rem < learn_steps in
      learn, not learn, rem = learn_steps + exploit_steps - 1
  in
  let tangents , new_sampling_state =
    match aux_exploit, aux_learn with
    | true, _ ->
      let lambda =
        state.aux |> O.params |> A.A.map ~f:(fun x -> Maths.const (Prms.value x))
      in
      let _vs, new_sampling_state =
        A.eigenvectors ~lambda ~switch_to_learn state.sampling_state config.n_tangents
      in
      _vs, new_sampling_state
    | false, true, ->
      let _vs = A.random_localised_vs () in
      _vs, sampling_state, None
    | false, false ->
      let _vs, _ =
        init_tangents
          ~base:config.base
          ~rank_one:config.rank_one
          ~n_tangents:config.n_tangents
          ~prev_seeds:state.prev_seeds
          ~ortho:config.orthogonalize
          theta
      in
      _vs, sampling_state

      in
      _vs, sampling_state
  in
  let theta, tangents = O.prepare ~config state in
  let y_pred = Model.f ~theta x in
  let loss = Loss.mse ~output_dims:[ 1 ] (y - y_pred) in
  let ggn = Loss.mse_ggn ~output_dims:[ 1 ] (const y) ~vtgt:(tangent_exn y_pred) in
  let new_state = O.step ~config ~info:{ loss; ggn; tangents } state in
  if t % 100 = 0
  then (
    let loss = to_float_exn (const loss) in
    print [%message (t : int) (loss : float)];
    Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss |] 1 2)));
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state

(* Start the loop. *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init (Model.init ~d_in ~d_out))
