(* example script testing automatic tuning of learning rate of SOFO. 
  following and simplifying https://arxiv.org/abs/1807.02811. *)
open Owl
open Base
open Torch
open Forward_torch
open Maths
open Sofo
open Svgp
module Mat = Dense.Matrix.S

let in_dir = Cmdargs.in_dir "-d"
let bayes_opt = Option.value (Cmdargs.get_bool "-bayes_opt") ~default:false
let base = Optimizer.Config.Base.default

let _ =
  Random.init 1985;
  (* Random.self_init (); *)
  Owl_stats_prng.init 1985;
  Torch_core.Wrapper.manual_seed 1985

(* -----------------------------------------
   -- Generate linear regression data.
   ----------------------------------------- *)

let d_in, d_out = 100, 3

let teacher =
  let sigma = Float.(1. / sqrt (of_int d_in)) in
  sigma $* randn ~kind:base.kind ~device:base.device [ d_in; d_out ]

let data_minibatch =
  let input_cov_sqrt =
    let u, _ = C.qr (randn ~device:base.device [ d_in; d_in ]) in
    let lambda =
      Array.init d_in ~f:(fun i -> Float.(1. / (1. + square (of_int Int.(i + 1)))))
      |> of_array ~device:base.device ~shape:[ d_in; 1 ]
      |> fun x -> C.(x / mean x)
    in
    C.(sqrt lambda * u)
  in
  fun bs ->
    let x = C.(randn ~device:base.device [ bs; d_in ] *@ input_cov_sqrt) in
    let y = Maths.(x *@ teacher) in
    x, y

(* -----------------------------------------
   -- SVGP for BayesOpt.
   ----------------------------------------- *)
let n_alpha = 10
let alpha_low = 1.1
let alpha_high = 2.5
let n_ind = 5
let max_iter_gp = 1000

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
        Maths.(Float.(1. / square 0.2) $* ones [ 1 ] ~kind:base.kind ~device:base.device)
        |> Prms.Single.bounded ~above:(Maths.f 1e-3)
    ; z =
        Mat.(transpose (exp10 (linspace alpha_low alpha_high n_ind)))
        |> Maths.of_bigarray ~device:base.device
        |> Prms.Single.free
    ; nu =
        Maths.zeros ~kind:base.kind ~device:base.device [ n_ind; 1 ] |> Prms.Single.free
    ; psi = Maths.eye ~kind:base.kind ~device:base.device n_ind |> Prms.Single.free
    }

(* training loop *)
module Bayes_O = Optimizer.Adam (V.P)

let config _ =
  Optimizer.Config.Adam.
    { base
    ; beta_1 = 0.9
    ; beta_2 = 0.99
    ; eps = 1e-4
    ; learning_rate = Some 1e-3
    ; weight_decay = None
    }

let gp_opt data =
  let rec iter t state running_avg =
    let theta = Bayes_O.params state in
    let theta_ = Bayes_O.P.value theta in
    let theta_dual =
      Bayes_O.P.map theta_ ~f:(fun x ->
        let x =
          x |> Maths.to_tensor |> Tensor.copy |> Tensor.to_device ~device:base.device
        in
        let x = Tensor.set_requires_grad x ~r:true in
        Tensor.zero_grad x;
        Maths.of_tensor x)
    in
    let loss, true_g =
      let loss =
        V.negative_bound ~theta:theta_dual ~n_total:n_alpha data |> Maths.to_tensor
      in
      Tensor.backward loss;
      ( Tensor.to_float0_exn loss
      , Bayes_O.P.map2 theta (Bayes_O.P.to_tensor theta_dual) ~f:(fun tagged p ->
          match tagged with
          | Prms.Pinned _ -> Maths.(f 0.)
          | _ -> Maths.of_tensor (Tensor.grad p)) )
    in
    let state' = Bayes_O.step ~config:(config t) ~info:true_g state in
    let running_avg =
      let loss_avg =
        match running_avg with
        | [] -> loss
        | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
      in
      (* book-keeping *)
      if t % 100 = 0
      then (
        let to_float x = x |> Maths.to_tensor |> Tensor.to_float0_exn in
        Mat.(
          save_txt
            ~append:true
            ~out:(in_dir "gp_info")
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
    if t < max_iter_gp
    then iter Int.(t + 1) state' (loss :: running_avg)
    else Bayes_O.(params state')
  in
  iter 0 (Bayes_O.init theta) []

(* -----------------------------------------
   -- Wolfe-unaware line search
   ----------------------------------------- *)
(* let c1 = 0.05
let c2 = 0.8 *)
let alpha_range = Mat.linspace alpha_low alpha_high n_alpha |> Mat.exp10 |> Mat.to_array
let cdf x = Maths.((f 1. + erf (x / f Float.(sqrt 2.))) / f 2.)

(* expected improvement, eqn 9 of [Mahsereci and Hennig, 2015] *)
(* let ei ~mu ~sigma f_best =
  let z = Maths.((f_best - mu) / (1e-8 $+ sigma)) in
  let tmp1 = Maths.((f_best - mu) / f 2. * (1. $+ erf (z / sqrt (f 2.)))) in
  let tmp2 = Maths.(sigma / f Float.(sqrt (2. * pi)) * exp (neg (sqr z / f 2.))) in
  Maths.(tmp1 + tmp2) 

(* expected improvement, eqn 8. *)
let ei ~mu ~sigma f_best =
  let delta = Maths.(mu - f_best) in
  let tmp1 = Maths.(relu delta) in
  let tmp2 = Maths.(sigma * pdf (delta / sigma)) in
  let tmp3 = Maths.(abs delta * cdf (delta / sigma)) in
  Maths.(tmp1 + tmp2 - tmp3)  *)

(* expected improvement, blog *)
let ei ~mu ~sigma f_best =
  let z = Maths.((mu - f_best) / sigma) in
  let tmp1 = Maths.((mu - f_best) * cdf z) in
  let tmp2 = Maths.(sigma * pdf z) in
  Maths.(tmp1 + tmp2)

let alpha_search model =
  let neg_loss_list =
    Array.map alpha_range ~f:(fun alpha ->
      (* the model accepts a learning rate and outputs a negative loss, which we wish to maximise *)
      let neg_loss = model alpha |> Maths.reshape ~shape:[ 1; 1 ] in
      neg_loss)
    |> List.of_array
  in
  (* alpha_tensor and neg_loss_tensor both of shape [n_alpha x 1] *)
  let alpha_tensor =
    Mat.of_array alpha_range n_alpha 1 |> Maths.of_bigarray ~device:base.device
  in
  let neg_loss_tensor = Maths.concat neg_loss_list ~dim:0 in
  (* fit gp *)
  let opt_gp_theta = gp_opt (alpha_tensor, neg_loss_tensor) in
  let f_best =
    let _, alpha_idx =
      Tensor.max_dim ~dim:0 ~keepdim:false (Maths.to_tensor neg_loss_tensor)
    in
    let alpha_idx = Tensor.to_int0_exn alpha_idx in
    List.nth_exn neg_loss_list alpha_idx
  in
  (* compute EI *)
  let ei_max_idx, _ =
    Array.foldi alpha_range ~init:None ~f:(fun i accu alpha ->
      let mu_f, sigma_f =
        V.infer
          ~theta:opt_gp_theta
          Maths.(f alpha * ones ~device:base.device ~kind:base.kind [ 1; 1 ])
      in
      let ei = ei ~mu:mu_f ~sigma:sigma_f f_best |> Maths.const |> Maths.to_float_exn in
      match accu with
      | None -> Some (i, ei)
      | Some (i_max, ei_max) ->
        if Float.(ei >= ei_max) then Some (i, ei) else Some (i_max, ei_max))
    |> Option.value_exn
  in
  alpha_range.(ei_max_idx)

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)

module Model = struct
  module P = Prms.Single

  let f ~(theta : _ some P.t) input = Maths.(input *@ theta)

  let init ~d_in ~d_out : P.param =
    let sigma = Float.(1. / sqrt (of_int d_in)) in
    let theta = sigma $* randn ~kind:base.kind ~device:base.device [ d_in; d_out ] in
    P.free theta
end

module O = Optimizer.SOFO (Model.P)

let init_config =
  Optimizer.Config.SOFO.
    { base; learning_rate = None; n_tangents = 10; damping = `relative_from_top 1e-5 }

(* -----------------------------------------
   -- Optimization loop       ------
   ----------------------------------------- *)

let batch_size = 256
let max_iter = 10_000
let max_iter_alpha_opt = 10

(* TODO: fix tangents to be used when line searching for the optimal learning rate *)
let model ~x ~y ~tangents ~state alpha =
  let config = { init_config with learning_rate = Some alpha } in
  let rec bayes_opt_loop t state =
    let theta, _ = O.prepare ~config state in
    let y_pred = Model.f ~theta x in
    let loss = Loss.mse ~average_over:[ 0; 1 ] (y - y_pred) in
    let ggn = Loss.mse_ggn ~average_over:[ 0; 1 ] (const y) ~vtgt:(tangent_exn y_pred) in
    let state = O.step ~config ~info:{ loss; ggn; tangents } state in
    (* return neg loss for BayesOpt *)
    if t < max_iter_alpha_opt then bayes_opt_loop Int.(t + 1) state else Maths.neg loss
  in
  bayes_opt_loop 0 state

let rec loop ~t ~out ~state ~alpha_opt =
  Stdlib.Gc.major ();
  let x, y = data_minibatch batch_size in
  let config = { init_config with learning_rate = Some alpha_opt } in
  let theta, tangents = O.prepare ~config state in
  let alpha_opt =
    if bayes_opt && t % 100 = 0 && t > 0
    then (
      let model_fn = model ~x ~y ~tangents ~state in
      alpha_search model_fn)
    else alpha_opt
  in
  let y_pred = Model.f ~theta x in
  let loss = Loss.mse ~average_over:[ 0; 1 ] (y - y_pred) in
  let ggn = Loss.mse_ggn ~average_over:[ 0; 1 ] (const y) ~vtgt:(tangent_exn y_pred) in
  let new_state =
    let config = { init_config with learning_rate = Some alpha_opt } in
    O.step ~config ~info:{ loss; ggn; tangents } state
  in
  if t % 100 = 0
  then (
    let loss_float = to_float_exn (const loss) in
    print [%message (t : int) (loss_float : float) (alpha_opt : float)];
    Owl.Mat.save_txt
      ~append:true
      ~out
      (Owl.Mat.of_array [| Float.of_int t; loss_float; alpha_opt |] 1 3));
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state ~alpha_opt

(* Start the loop. *)
let _ =
  let out = in_dir "loss_lr_bayes." in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  Bos.Cmd.(v "rm" % "-f" % in_dir "gp_info") |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init (Model.init ~d_in ~d_out)) ~alpha_opt:100.
