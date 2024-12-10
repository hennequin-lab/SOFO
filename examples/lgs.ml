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

(* -----------------------------------------
   -- Control Problem / Data Generation ---
   ----------------------------------------- *)
module Dims = struct
  let a = 6
  let b = 3
  let o = 4
  let tmax = 10
  let m = 128
  let k = 64
  let batch_const = true
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.cuda_if_available ()
end

module S = Lds_data.Sample_LDS (Dims)
module Data = Lds_data.Make_LDS_Tensor (Dims) (Lds_data.Sample_LDS)

(* control costs - time invariant *)
let _Cxx = Data.sample_q_xx ()
let _Cxu = None
let _Cuu = Data.sample_q_uu ()

(* TODO: for now we set x0=0? *)
let x0 = Tensor.zeros ~device:Dims.device ~kind:Dims.kind [ Dims.m; Dims.a ]

(* in the linear gaussian case, _Fx, _Fu, c, b and cov invariant across time *)
let f_list : Tensor.t Lds_data.f_params list =
  let _Fx = Data.sample_fx () in
  let _Fu = Data.sample_fu () in
  let c = Data.sample_c () in
  let b = Data.sample_b () in
  let cov = Data.sample_output_cov () in
  List.init (Dims.tmax + 1) ~f:(fun _ ->
    Lds_data.
      { _Fx_prod = _Fx
      ; _Fu_prod = _Fu
      ; _f = None
      ; _c = Some c
      ; _b = Some b
      ; _cov = Some cov
      })

let sample_data () =
  (* generate ground truth params and data *)
  let data_gen ~x0 =
    (* need to sample these first to get the trajectory *)
    let u_list = Data.sample_u_list () in
    Data.traj_rollout ~x0 ~f_list ~u_list
  in
  let x_list, o_list = data_gen ~x0 in
  let o_list = List.map o_list ~f:(fun o -> Option.value_exn o) in
  x_list, o_list

(* -----------------------------------------
   ----- Utilitiy Functions ------
   ----------------------------------------- *)

let tmp_einsum a b =
  if Dims.batch_const
  then Maths.einsum [ a, "ma"; b, "ab" ] "mb"
  else Maths.einsum [ a, "ma"; b, "mab" ] "mb"

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)
let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run

let base =
  Optimizer.Config.Base.
    { default with kind = Torch_core.Kind.(T f64); ba_kind = Bigarray.float64 }

let max_iter = 1000

let config ~base_lr ~gamma ~iter:_ =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some base_lr
    ; n_tangents = 128
    ; rank_one = false
    ; damping = gamma
    ; momentum = None
    }

(* let config ~base_lr ~gamma:_ ~iter:_ =
  Optimizer.Config.Adam.{ default with learning_rate = Some base_lr } *)

module LGS = struct
  module PP = struct
    type 'a p =
      { _Fx_prod : 'a (* generative model *)
      ; _Fu_prod : 'a
      ; _c : 'a
      ; _b : 'a
      ; _cov_noise : 'a (* sqrt of covariance of emission noise *)
      ; _cov_u : 'a (* sqrt of covariance of prior over u *)
      ; _cov_pos : 'a (* sqrt of recognition model; covariance of posterior *)
      }
    [@@deriving prms]
  end

  module P = PP.Make (Prms.P)

  type args = unit
  type data = Tensor.t * Tensor.t list

  (* create params for lds from f *)
  let params_from_f ~(theta : P.M.t) ~x0
    : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
    =
    Lqr.Params.
      { x0 = Some x0
      ; params =
          List.init (Dims.tmax + 1) ~f:(fun _ ->
            Lds_data.Temp.
              { _f = None
              ; _Fx_prod = theta._Fx_prod
              ; _Fu_prod = theta._Fu_prod
              ; _cx = None
              ; _cu = None
              ; _Cxx = Maths.const _Cxx
              ; _Cxu
              ; _Cuu = Maths.const _Cuu
              })
      }

  let optimal_u ~(theta : P.M.t) ~x0 =
    (* use lqr to obtain the optimal u*)
    let p = params_from_f ~x0 ~theta |> S.map_implicit in
    let sol = Lqr.solve ~batch_const:Dims.batch_const p in
    (* TODO: Matheron sampling of u *)
    let u_list = List.map sol ~f:(fun s -> s.u) in
    let sampled_u_list =
      List.map u_list ~f:(fun u ->
        let noise =
          let eps =
            Tensor.randn ~device:Dims.device ~kind:Dims.kind [ Dims.m; Dims.b ]
            |> Maths.const
          in
          Maths.einsum [ eps, "ma"; theta._cov_pos, "ab" ] "mb"
        in
        Maths.(noise + u))
    in
    u_list, sampled_u_list

  (* optimal u determined from lqr *)
  let pred_u ~data (theta : P.M.t) =
    let x0, _ = data in
    optimal_u ~theta ~x0:(Maths.const x0)

  (* rollout x list under sampled u *)
  let rollout_x ~u_list ~x0 (theta : P.M.t) =
    let x0_tan = Maths.const x0 in
    let _, x_list =
      List.fold u_list ~init:(x0_tan, [ x0_tan ]) ~f:(fun (x, x_list) u ->
        let new_x = Maths.(tmp_einsum x theta._Fx_prod + tmp_einsum u theta._Fu_prod) in
        new_x, new_x :: x_list)
    in
    List.rev x_list

  (* gaussian llh with diagonal covariance *)
  let gaussian_llh ~g_mean ~g_cov ~x =
    let g_cov = Maths.sqr g_cov in
    let error_term =
      let error = Maths.(x - g_mean) in
      let tmp = tmp_einsum error (Maths.inv_sqr g_cov) in
      Maths.einsum [ tmp, "ma"; error, "ma" ] "m" |> Maths.reshape ~shape:[ -1; 1 ]
    in
    let cov_term =
      if Dims.batch_const
      then Maths.(sum (log (diagonal g_cov ~offset:0))) |> Maths.reshape ~shape:[ 1; 1 ]
      else
        Maths.(sum_dim ~dim:(Some [ 0 ]) ~keepdim:true (log (diagonal g_cov ~offset:0)))
        |> Maths.reshape ~shape:[ -1; 1 ]
    in
    Maths.(0.5 $* error_term + cov_term) |> Maths.squeeze ~dim:(-1) |> Maths.neg

  let elbo ~u_list ~x_o_list ~optimal_u_list (theta : P.M.t) =
    (* calculate the likelihood term *)
    let llh =
      List.foldi x_o_list ~init:None ~f:(fun t accu (x, o) ->
        if t % 1 = 0 then Stdlib.Gc.major ();
        let increment =
          gaussian_llh
            ~g_mean:o
            ~g_cov:theta._cov_noise
            ~x:Maths.(tmp_einsum x theta._c + theta._b)
        in
        match accu with
        | None -> Some increment
        | Some accu -> Some Maths.(accu + increment))
      |> Option.value_exn
    in
    let prior =
      List.foldi u_list ~init:None ~f:(fun t accu u ->
        if t % 1 = 0 then Stdlib.Gc.major ();
        let u_zeros = Tensor.zeros_like (Maths.primal u) |> Maths.const in
        let increment = gaussian_llh ~g_mean:u_zeros ~g_cov:theta._cov_u ~x:u in
        match accu with
        | None -> Some increment
        | Some accu -> Some Maths.(accu + increment))
      |> Option.value_exn
    in
    let entropy =
      List.fold2_exn optimal_u_list u_list ~init:None ~f:(fun accu u_opt u ->
        let increment = gaussian_llh ~g_mean:u_opt ~g_cov:theta._cov_pos ~x:u in
        match accu with
        | None -> Some increment
        | Some accu -> Some Maths.(accu + increment))
      |> Option.value_exn
    in
    Maths.((llh + prior - entropy) /$ Float.of_int (List.length u_list))

  (* calculate optimal u *)
  let f ~update ~data ~init ~args:_ (theta : P.M.t) =
    let optimal_u_list, u_list = pred_u ~data theta in
    let module L = Loss.RL_loss in
    let x0, o_list = data in
    let rolled_out_x_list = rollout_x ~u_list ~x0 theta in
    (* These lists go from 1 to T *)
    let o_except_first = List.map o_list ~f:(fun o -> Maths.const o) in
    let x_except_first = List.tl_exn rolled_out_x_list in
    let x_o_list = List.map2_exn x_except_first o_except_first ~f:(fun x o -> x, o) in
    let neg_elbo = elbo ~u_list ~x_o_list ~optimal_u_list theta |> Maths.neg in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let delta_ggn =
        let vtgt = Maths.tangent neg_elbo |> Option.value_exn in
        L.vtgt_gv ~vtgt
      in
      u init (Some (neg_elbo, Some delta_ggn))

  let init : P.tagged =
    let _Fx_prod =
      Convenience.gaussian_tensor_2d_normed
        ~device:Dims.device
        ~kind:Dims.kind
        ~a:Dims.a
        ~b:Dims.a
        ~sigma:0.1
      |> Prms.free
    in
    let _Fu_prod =
      Convenience.gaussian_tensor_2d_normed
        ~device:Dims.device
        ~kind:Dims.kind
        ~a:Dims.b
        ~b:Dims.a
        ~sigma:0.1
      |> Prms.free
    in
    let _c =
      Convenience.gaussian_tensor_2d_normed
        ~device:Dims.device
        ~kind:Dims.kind
        ~a:Dims.a
        ~b:Dims.o
        ~sigma:0.1
      |> Prms.free
    in
    let _b =
      Convenience.gaussian_tensor_2d_normed
        ~device:Dims.device
        ~kind:Dims.kind
        ~a:1
        ~b:Dims.o
        ~sigma:0.1
      |> Prms.free
    in
    let _cov_noise =
      Tensor.(abs (Tensor.randn ~device:Dims.device ~kind:Dims.kind [ Dims.o ]))
      |> Tensor.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
      |> Prms.free
    in
    let _cov_u =
      Tensor.(abs (Tensor.randn ~device:Dims.device ~kind:Dims.kind [ Dims.b ]))
      |> Tensor.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
      |> Prms.free
    in
    let _cov_pos =
      Tensor.(abs (Tensor.randn ~device:Dims.device ~kind:Dims.kind [ Dims.b ]))
      |> Tensor.diag_embed ~offset:0 ~dim1:(-2) ~dim2:(-1)
      |> Prms.free
    in
    { _Fx_prod; _Fu_prod; _c; _b; _cov_noise; _cov_u; _cov_pos }
end

module O = Optimizer.SOFO (LGS)
(* module O = Optimizer.Adam (LGS) *)

let optimise ~max_iter ~f_name config_f =
  let rec loop ~iter ~state ~time_elapsed running_avg =
    Stdlib.Gc.major ();
    let config = config_f ~iter in
    let data =
      let _, o_list = sample_data () in
      x0, o_list
    in
    let t0 = Unix.gettimeofday () in
    let loss, new_state = O.step ~config ~state ~data ~args:() in
    let t1 = Unix.gettimeofday () in
    let time_elapsed = Float.(time_elapsed + t1 - t0) in
    let running_avg =
      if iter % 1 = 0
      then (
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
          O.W.P.T.save
            (LGS.P.value (O.params new_state))
            ~kind:base.ba_kind
            ~out:(in_dir f_name ^ "_params");
          let t = iter in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir f_name)
              (of_array [| Float.of_int t; time_elapsed; loss_avg |] 1 3)));
        [])
      else running_avg
    in
    if iter < max_iter
    then loop ~iter:(iter + 1) ~state:new_state ~time_elapsed (loss :: running_avg)
  in
  (* ~config:(config_f ~iter:0) *)
  loop ~iter:0 ~state:(O.init ~config:(config_f ~iter:0) LGS.(init)) ~time_elapsed:0. []

let lr_rates = [ 50. ]
let damping_list = [ Some 1e-3 ]
let meth = "sofo"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    List.iter damping_list ~f:(fun gamma ->
      let config_f = config ~base_lr:eta ~gamma in
      let gamma_name = Option.value_map gamma ~default:"none" ~f:Float.to_string in
      let f_name = sprintf "lgs_%s_lr_%s_damp_%s" meth (Float.to_string eta) gamma_name in
      Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
      Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
      optimise ~max_iter ~f_name config_f))

(* let lr_rates = [ 0.0001 ]
let meth = "adam"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    let config_f = config ~base_lr:eta ~gamma:None in
    let f_name = sprintf "lgs_%s_lr_%s" meth (Float.to_string eta) in
    Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
    Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
    optimise ~max_iter ~f_name config_f) *)
