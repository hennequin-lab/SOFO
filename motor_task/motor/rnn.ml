open Base
open Torch
open Forward_torch
open Sofo
include Rnn_typ
module P = PP.Make (Prms.P)

module Make (X : sig
    val dt : float
    val tau : float
    val n : int
    val internal_noise_std : float option
  end) =
struct
  open X
  module P = PP.Make (Prms.P)
  open PP

  let init ~(base : ('a, 'b) Optimizer.Config.Base.t) ~n_input_channels =
    let randn =
      Convenience.gaussian_tensor_2d_normed ~device:base.device ~kind:base.kind
    in
    let init_cond = Tensor.(zeros ~device:base.device [ 1; n ]) in
    let w = randn ~a:(n + 1) ~b:n ~sigma:Float.(0.5 / sqrt (of_int n)) in
    let b = randn ~a:n_input_channels ~b:n ~sigma:0.1 in
    let feedback_weights () =
      let sigma = 0.1 in
      randn ~a:1 ~b:n ~sigma
    in
    let output_weights () =
      let sigma = Float.(0.1 / sqrt (of_int n)) in
      randn ~a:n ~b:1 ~sigma
    in
    { init_cond = Prms.free init_cond
    ; w = Prms.free w
    ; b = Prms.free b
    ; f1 = Prms.free (feedback_weights ())
    ; f2 = Prms.free (feedback_weights ())
    ; f3 = Prms.free (feedback_weights ())
    ; f4 = Prms.free (feedback_weights ())
    ; c1 = Prms.free (output_weights ())
    ; c2 = Prms.free (output_weights ())
    }

  let phi x = Maths.relu x

  let step_forward ?noise ~prms input (z, a) =
    let bs = Tensor.shape input |> List.hd_exn in
    let r =
      match noise with
      | None -> phi z
      | Some eta -> Maths.(const eta * phi z)
    in
    let z =
      let z_leak = Maths.(Float.(1. - (dt / tau)) $* z) in
      let r =
        Maths.concat
          r
          (Maths.const
             (Tensor.ones
                ~device:(Tensor.device input)
                ~kind:(Tensor.kind input)
                [ bs; 1 ]))
          ~dim:1
      in
      Maths.(
        z_leak
        + (Float.(dt / tau)
           $* (r *@ prms.w)
              + (Maths.const input *@ prms.b)
              + (a.Arm.pos.x1 *@ prms.f1)
              + (a.Arm.pos.x2 *@ prms.f2)
              + (a.Arm.vel.x1 *@ prms.f3)
              + (a.Arm.vel.x2 *@ prms.f4)))
    in
    let torques = Maths.(r *@ prms.c1, r *@ prms.c2) in
    let a = Arm.integrate ~dt a torques in
    z, a, torques

  let draw_noise ~device:d ~t_max ~bs =
    Option.map internal_noise_std ~f:(fun sigma ->
      Tensor.(f 1. + (f Float.(2. * sigma) * (rand ~device:d [ t_max; bs; n ] - f 0.5))))

  let noise_slice noise t =
    Option.map noise ~f:(fun noise ->
      Tensor.(
        slice ~dim:0 ~start:(Some t) ~end_:(Some Int.(t + 1)) ~step:1 noise
        |> squeeze
        |> Tensor.(max (f 0.))))

  let forward ~prms ~t_max input =
    let bs = Convenience.first_dim (input 0) in
    let noise = draw_noise ~device:(Tensor.device (input 0)) ~t_max ~bs in
    let rec iter t (z, a) accu =
      Stdlib.Gc.major ();
      if t = t_max
      then List.rev accu
      else (
        let noise = noise_slice noise t in
        let z, a, torques = step_forward ?noise ~prms (input t) (z, a) in
        let result = { network = z; arm = a; torques } in
        iter (t + 1) (z, a) (result :: accu))
    in
    let z0 = Maths.concat_list ~dim:0 (List.init bs ~f:(fun _ -> prms.init_cond)) in
    let a0 = Arm.theta_init bs in
    iter 0 (z0, a0) []
end
