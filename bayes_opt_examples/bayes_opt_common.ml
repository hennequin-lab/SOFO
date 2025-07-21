open Owl
open Base
open Torch
open Forward_torch
open Sofo
open Svgp
module Mat = Dense.Matrix.S

let in_dir = Cmdargs.in_dir "-d"
let base = Optimizer.Config.Base.default

(* -----------------------------------------
   -- SVGP for BayesOpt.
   ----------------------------------------- *)

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
let theta ~alpha_low ~alpha_high =
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

let gp_opt ~alpha_low ~alpha_high ~n_alpha data =
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
  iter 0 (Bayes_O.init (theta ~alpha_low ~alpha_high)) []

(* -----------------------------------------
      -- Search for optimal lr
      ----------------------------------------- *)
let cdf x = Maths.((f 1. + erf (x / f Float.(sqrt 2.))) / f 2.)

(* expected improvement *)
let ei ~mu ~sigma f_best =
  let z = Maths.((mu - f_best) / sigma) in
  let tmp1 = Maths.((mu - f_best) * cdf z) in
  let tmp2 = Maths.(sigma * pdf z) in
  Maths.(tmp1 + tmp2)

let alpha_search ~alpha_low ~alpha_high ~n_alpha model =
  let alpha_range =
    Mat.linspace alpha_low alpha_high n_alpha |> Mat.exp10 |> Mat.to_array
  in
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
  let opt_gp_theta =
    gp_opt ~alpha_low ~alpha_high ~n_alpha (alpha_tensor, neg_loss_tensor)
  in
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
