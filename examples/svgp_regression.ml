open Base
open Owl
open Torch
open Svgp
open Sofo
open Forward_torch
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run
let base = Optimizer.Config.Base.default
let max_iter = 2000

(* make up some data *)
let n = 100
let n_total = n * 10

(* number of inducing points *)
let n_ind = 40
let noise_scale = 0.2

(* y = w x, x of shape [n_total x 1], y of shape [n_total x 1] *)
let x = Mat.gaussian ~mu:Const.pi ~sigma:1.5 n_total 1
let y = Mat.((x * sin x /$ 2.) + (noise_scale $* gaussian n_total 1))
let _ = Mat.(save_txt ~out:(in_dir "data") (x @|| y))

let sample_batch =
  let ids = Array.init n_total ~f:Fn.id in
  fun () ->
    let ids = Stats.choose ids n |> Array.to_list in
    let slice = Mat.(get_fancy [ L ids ]) in
    ( Maths.of_bigarray ~device:base.device (slice x)
    , Maths.of_bigarray ~device:base.device (slice y) )

(* simple squared-exponential kernel *)
module Squared_exponential = struct
  module Kernel_parameters = struct
    type 'a p =
      { variance : 'a
      ; lengthscale : 'a
      }
    [@@deriving prms]
  end

  module P = Kernel_parameters.Make (Prms.Single)

  let kernel ~theta x1 x2 =
    let open Kernel_parameters in
    let final_shape = [ List.hd_exn (Maths.shape x1); List.hd_exn (Maths.shape x2) ] in
    let x2_T = Maths.transpose x2 ~dims:[ 1; 0 ] in
    let x1_broadcasted = Maths.broadcast_to x1 ~size:final_shape
    and x2_broadcasted = Maths.(broadcast_to x2_T ~size:final_shape) in
    let d = Maths.((x1_broadcasted - x2_broadcasted) / theta.lengthscale) in
    Maths.(theta.variance * exp (0.5 $* neg (sqr d)))

  let kernel_diag ~theta x =
    let open Kernel_parameters in
    let n = List.hd_exn (Maths.shape x) in
    Maths.(theta.variance * any (ones ~device:base.device ~kind:base.kind [ n; 1 ]))
end

module V = Variational_regression (Squared_exponential)

(* initialise the parameters *)
let theta =
  Variational_regression_parameters.
    { kernel =
        Squared_exponential.Kernel_parameters.
          { variance =
              Maths.ones [ 1 ] ~kind:base.kind ~device:base.device
              |> Prms.Single.bounded ~above:(Maths.f 0.01) ~below:(Maths.f 1.)
          ; lengthscale =
              Maths.ones [ 1 ] ~kind:base.kind ~device:base.device
              |> Prms.Single.bounded ~above:(Maths.f 0.01) ~below:(Maths.f 10.)
          }
    ; beta =
        Maths.(
          Float.(1. / square noise_scale)
          $* ones [ 1 ] ~kind:base.kind ~device:base.device)
        |> Prms.Single.bounded ~above:(Maths.f 1e-3)
    ; z =
        Mat.(transpose (linspace 0. Const.pi2 n_ind))
        |> Maths.of_bigarray ~device:base.device
        |> Prms.Single.free
    ; nu =
        Maths.zeros ~kind:base.kind ~device:base.device [ n_ind; 1 ] |> Prms.Single.free
    ; psi = Maths.eye ~kind:base.kind ~device:base.device n_ind |> Prms.Single.free
    }

(* training loop *)
module O = Optimizer.Adam (V.P)

let config _ =
  Optimizer.Config.Adam.
    { base
    ; beta_1 = 0.9
    ; beta_2 = 0.99
    ; eps = 1e-4
    ; learning_rate = Some 1e-3
    ; weight_decay = None
    }

let rec iter t state running_avg =
  let theta = O.params state in
  let theta_ = O.P.value theta in
  let theta_dual =
    O.P.map theta_ ~f:(fun x ->
      let x =
        x |> Maths.to_tensor |> Tensor.copy |> Tensor.to_device ~device:base.device
      in
      let x = Tensor.set_requires_grad x ~r:true in
      Tensor.zero_grad x;
      Maths.of_tensor x)
  in
  let data = sample_batch () in
  let loss, true_g =
    let loss = V.negative_bound ~theta:theta_dual ~n_total data |> Maths.to_tensor in
    Tensor.backward loss;
    ( Tensor.to_float0_exn loss
    , O.P.map2 theta (O.P.to_tensor theta_dual) ~f:(fun tagged p ->
        match tagged with
        | Prms.Pinned _ -> Maths.(f 0.)
        | _ -> Maths.of_tensor (Tensor.grad p)) )
  in
  let state' = O.step ~config:(config t) ~info:true_g state in
  let running_avg =
    let loss_avg =
      match running_avg with
      | [] -> loss
      | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
    in
    (* book-keeping *)
    if t % 100 = 0
    then (
      print [%message (t : int) (loss : float)];
      let to_float x = x |> Maths.to_tensor |> Tensor.to_float0_exn in
      Mat.(
        save_txt
          ~append:true
          ~out:(in_dir "info")
          (of_array
             [| Float.of_int t
              ; loss_avg
              ; to_float theta_.beta
              ; to_float theta_.kernel.lengthscale
              ; to_float theta_.kernel.variance
             |]
             1
             5));
      Mat.(save_txt ~out:(in_dir "z") (Maths.to_bigarray ~kind:base.ba_kind theta_.z)));
    []
  in
  if t < max_iter then iter (t + 1) state' (loss :: running_avg) else O.(params state')

let theta = iter 0 (O.init theta) []

(* test my posterior on a bunch of regularly spaced x_star *)
let x_star = Mat.linspace (-4.) 10. 200 |> Mat.transpose
let x_star_maths = x_star |> Maths.of_bigarray ~device:base.device
let y_star = Mat.(x_star * sin x_star /$ 2.)
let mu, sigma = V.infer ~theta x_star_maths

let _ =
  let ans =
    Maths.(
      concat
        ~dim:1
        [ any x_star_maths
        ; any (Maths.of_bigarray ~device:base.device y_star)
        ; mu
        ; unsqueeze ~dim:1 (diagonal ~offset:0 sigma)
        ])
    |> Maths.const
  in
  Mat.(save_txt ~out:(in_dir "posterior") (Maths.to_bigarray ~kind:base.ba_kind ans))

let _ =
  V.P.C.save
    ~out:(in_dir "prms")
    ~kind:base.ba_kind
    (V.P.map (V.P.value theta) ~f:Maths.const)
