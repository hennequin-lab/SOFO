(* Linear Gaussian Dynamics, with same state/control/cost parameters constant across trials and across time *)
open Printf
open Base
open Forward_torch
open Torch
open Sofo

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f64)
    ; ba_kind = Bigarray.float64
    }

let m = 24
let o = 48
let bs = 256
let sigma_o = Maths.(f 0.1)

let c =
  Tensor.(
    f Float.(1. /. sqrt (of_int m)) * randn ~device:base.device ~kind:base.kind [ m; o ])
  |> Maths.const

module PP = struct
  type 'a p =
    { c : 'a
    ; d : 'a
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

let theta =
  let c = Prms.free (Tensor.randn_like (Maths.primal c)) in
  let d =
    Prms.create
      ~above:(Tensor.f 0.001)
      (Tensor.ones ~device:base.device ~kind:base.kind [ m ])
  in
  PP.{ c; d }

let sample_data () =
  let us = Tensor.(randn ~device:base.device ~kind:base.kind [ bs; m ]) |> Maths.const in
  let xs = Maths.(us *@ c) in
  let ys = Maths.(xs + (sigma_o * const (Tensor.randn_like (primal xs)))) in
  us, ys

let solver a y =
  let ell = Maths.cholesky a in
  let ell_t = Maths.transpose ~dim0:0 ~dim1:1 ell in
  let _x = Maths.linsolve_triangular ~left:false ~upper:true ell_t y in
  Maths.linsolve_triangular ~left:false ~upper:false ell _x

let u_opt ~(theta : P.M.t) y =
  let a = Maths.(einsum [ theta.c, "ji"; theta.c, "jk" ] "ik") in
  let a =
    Maths.(
      a
      + diag_embed
          ~offset:0
          ~dim1:0
          ~dim2:1
          (sqr sigma_o * const Tensor.(ones ~device:base.device ~kind:base.kind [ o ])))
  in
  let solution = solver a y in
  Maths.(einsum [ theta.c, "ij"; solution, "mj" ] "mi")

let gaussian_llh ?mu ~std x =
  let inv_std = Maths.(f 1. / std) in
  let error_term =
    let error =
      match mu with
      | None -> Maths.(einsum [ x, "ma"; inv_std, "a" ] "ma")
      | Some mu -> Maths.(einsum [ x - mu, "ma"; inv_std, "a" ] "ma")
    in
    Maths.einsum [ error, "ma"; error, "ma" ] "m" |> Maths.reshape ~shape:[ -1; 1 ]
  in
  let cov_term = Maths.(2. $* sum (log std)) |> Maths.reshape ~shape:[ 1; 1 ] in
  let const_term =
    let o = x |> Maths.primal |> Tensor.shape |> List.last_exn in
    Float.(log (2. * pi) * of_int o)
  in
  Maths.(0.5 $* (const_term $+ error_term + cov_term))
  |> Maths.(mean_dim ~keepdim:false ~dim:(Some [ 1 ]))
  |> Maths.neg

module M = struct
  module P = P

  type data = Maths.t
  type args = unit

  let cal_ggn ~y ~like_hess =
    let vtgt = Maths.tangent y |> Option.value_exn in
    let vtgt_hess = Tensor.einsum ~equation:"kma,a->kma" [ vtgt; like_hess ] ~path:None in
    Tensor.einsum ~equation:"kma,jma->kj" [ vtgt_hess; vtgt ] ~path:None

  let f ~update ~data:y ~init ~args:() (theta : P.M.t) =
    let u_opt = u_opt ~theta y in
    let u_sampled =
      Maths.(
        u_opt + (const (Tensor.randn_like (primal u_opt)) * Maths.unsqueeze ~dim:0 theta.d))
    in
    let y_pred = Maths.(u_opt *@ c) in
    let lik_term =
      gaussian_llh
        ~mu:y_pred
        ~std:
          Maths.(sigma_o * const (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
        y
    in
    let prior_term =
      gaussian_llh
        ~std:Maths.(const (Tensor.ones ~device:base.device ~kind:base.kind [ m ]))
        u_sampled
    in
    let q_term = gaussian_llh ~mu:u_opt ~std:theta.d u_sampled in
    let neg_elbo = Maths.(lik_term + prior_term - q_term) |> Maths.neg in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      (* TODO: use fisher *)
      let emp_fisher =
        let neg_elbo_t = Maths.tangent neg_elbo |> Option.value_exn in
        let fisher_half =
          Tensor.reshape neg_elbo_t ~shape:[ List.hd_exn (Tensor.shape neg_elbo_t); -1 ]
        in
        Tensor.(matmul fisher_half (transpose fisher_half ~dim0:0 ~dim1:1))
      in
      (* TODO: use ggn *)
      let ggn =
        let llh_ggn =
          cal_ggn
            ~y:Maths.(u_sampled *@ theta.c)
            ~like_hess:
              Tensor.(
                f 1.
                / square (Maths.primal sigma_o)
                * ones ~device:base.device ~kind:base.kind [ o ])
        in
        let prior_ggn =
          cal_ggn
            ~y:u_sampled
            ~like_hess:Tensor.(ones ~device:base.device ~kind:base.kind [ m ])
        in
        let entropy_ggn =
          cal_ggn ~y:u_sampled ~like_hess:Tensor.(f 1. / square (Maths.primal theta.d))
        in
        Tensor.(llh_ggn + prior_ggn)
      in
      u init (Some (neg_elbo, Some ggn))
end

let max_iter = 2000

let config ~base_lr ~gamma ~iter:_ =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some base_lr
    ; n_tangents = 128
    ; rank_one = false
    ; damping = gamma
    ; momentum = None
    ; lm = false
    ; perturb_thresh = None
    }

module O = Optimizer.SOFO (M)

(* let config ~base_lr ~gamma:_ ~iter:_ =
  Optimizer.Config.Adam.{ default with learning_rate = Some base_lr }

module O = Optimizer.Adam (M) *)

let optimise ~max_iter ~f_name ~init config_f =
  let rec loop ~iter ~state ~time_elapsed running_avg =
    Stdlib.Gc.major ();
    let config = config_f ~iter in
    let _, y = sample_data () in
    let t0 = Unix.gettimeofday () in
    let loss, new_state = O.step ~config ~state ~data:y ~args:() in
    let t1 = Unix.gettimeofday () in
    let time_elapsed = Float.(time_elapsed + t1 - t0) in
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
            ~out:(in_dir f_name)
            (of_array [| Float.of_int t; time_elapsed; loss_avg |] 1 3));
        O.W.P.T.save
          (M.P.value (O.params new_state))
          ~kind:base.ba_kind
          ~out:(in_dir f_name ^ "_params"));
      []
    in
    if iter < max_iter
    then loop ~iter:(iter + 1) ~state:new_state ~time_elapsed (loss :: running_avg)
  in
  (* ~config:(config_f ~iter:0) *)
  loop ~iter:0 ~state:(O.init ~config:(config_f ~iter:0) init) ~time_elapsed:0. []

let lr_rates = [ 5. ]
let damping_list = [ Some 1e-5 ]
let meth = "sofo"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    List.iter damping_list ~f:(fun gamma ->
      let config_f = config ~base_lr:eta ~gamma in
      let gamma_name = Option.value_map gamma ~default:"none" ~f:Float.to_string in
      let init, f_name =
        theta, sprintf "fa_elbo_%s_lr_%s_damp_%s" meth (Float.to_string eta) gamma_name
      in
      Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
      Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
      optimise ~max_iter ~f_name ~init config_f))

(* let lr_rates = [ 0.1 ]
let meth = "adam"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    let config_f = config ~base_lr:eta ~gamma:None in
    let init, f_name = theta, sprintf "lgs_elbo_%s_lr_%s" meth (Float.to_string eta) in
    Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
    Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
    optimise ~max_iter ~f_name ~init config_f) *)
