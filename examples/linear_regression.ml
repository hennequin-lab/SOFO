(* example on a simple linear regression problem where y = W x, x of dim [n] and y of dim [d] *)
open Base
open Torch
open Forward_torch
open Sofo

let in_dir = Cmdargs.in_dir "-d"
let batch_size = 512
let max_iter = 10_000
let base = Optimizer.Config.Base.default

(* input dim *)
let n = 100

(* output dim *)
let d = 1

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)

(* Linear regression = simple one-layer neural network *)
module P = Prms.P

module One_layer = struct
  module P = P

  type input = Tensor.t

  (* forward pass where y = W *@ x *)
  let f ~(theta : P.M.t) ~(input : input) = Maths.(theta *@ const input)

  (* initialise parameter theta *)
  let init ~n ~d : P.tagged =
    let sigma = Float.(1. / sqrt (of_int n)) in
    Tensor.(f sigma * randn ~kind:base.kind ~device:base.device [ d; n ]) |> Prms.free
end

(* Feedforward model wrapper with MSE loss *)
module FF =
  Wrapper.Feedforward
    (One_layer)
    (Loss.MSE (struct
         let scaling_factor = 1.
       end))

(* SOFO Optimiser *)
(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)

module GGN : Wrapper.Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { theta_left : 'a
      ; theta_right : 'a
      }
    [@@deriving prms]
  end

  module P = P
  module A = Make (Prms.P)

  type sampling_state = unit

  let init_sampling_state () = ()

  (* approximation defined implicitly via Gv products *)
  let g12v ~(lambda : A.M.t) (v : P.M.t) : P.M.t =
    let theta =
      Maths.einsum [ lambda.theta_left, "in"; v, "aij"; lambda.theta_right, "jm" ] "anm"
    in
    theta

  let random_localised_vs _K : P.T.t =
    Tensor.randn ~device:base.device ~kind:base.kind [ _K; d; n ]

  let eigenvectors ~(lambda : A.M.t) () (_K : int) =
    let left, right, n_per_param = lambda.theta_left, lambda.theta_right, _K in
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
    let vs =
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
        let tmp = Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_left; u_right ] in
        Tensor.unsqueeze tmp ~dim:0)
      |> Tensor.concatenate ~dim:0
    in
    vs, ()

  let init () =
    let init_eye size =
      Owl.Dense.Matrix.S.(0.1 $* eye size)
      |> Tensor.of_bigarray ~device:base.device
      |> Prms.free
    in
    { theta_left = init_eye d; theta_right = init_eye n }
end

module O = Optimizer.SOFO (FF) (GGN)

let config =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some 100.
    ; n_tangents = 10
    ; rank_one = false
    ; damping = None
    ; aux = None
    }

(* Adam optimizer *)
(* module O = Optimizer.Adam (FF)
let config = Optimizer.Config.Adam.{ default with base; learning_rate = Some 1. } *)

(* -----------------------------------------
   -- Generate linear regression data.
   ----------------------------------------- *)

(* input covariance ( = Fisher information matrix in this case) *)
let input_cov12 =
  let lambda =
    Array.init n ~f:(fun i -> Float.(1. / (1. + square (of_int Int.(i + 1)))))
    |> Tensor.of_float1 ~device:base.device
  in
  let lambda = Tensor.(lambda / mean lambda) in
  let u, _ =
    Tensor.linalg_qr ~a:Tensor.(randn ~device:base.device [ n; n ]) ~mode:"complete"
  in
  Tensor.(u * sqrt lambda)

(* true W *)
let teacher : One_layer.P.M.t =
  One_layer.init ~n ~d |> One_layer.P.value |> One_layer.P.const

(* data for mini batch. *)
let minibatch bs =
  let x = Tensor.(matmul input_cov12 (randn ~device:base.device [ n; bs ])) in
  let y = One_layer.f ~theta:teacher ~input:x |> Maths.primal in
  x, y

(* optimization loop *)
let rec loop ~t ~out ~state =
  Stdlib.Gc.major ();
  let data = minibatch batch_size in
  let loss, new_state = O.step ~config ~state ~data () in
  if t % 100 = 0
  then (
    Convenience.print [%message (t : int) (loss : float)];
    Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss |] 1 2)));
  if t < max_iter then loop ~t:(t + 1) ~out ~state:new_state

(* start the loop *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init (One_layer.init ~n ~d))
