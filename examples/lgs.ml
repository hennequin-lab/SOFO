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
  let a = 24
  let b = 10
  let o = 8
  let tmax = 10
  let m = 128
  let batch_const = true
  let kind = Torch_core.Kind.(T f64)
  let device = Torch.Device.cuda_if_available ()
end

module Data = Lds_data.Make_LDS_Tensor (Dims)

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
  (* need to sample these first to get the trajectory *)
  let u_list = Data.sample_u_list () in
  let x_list, o_list = Data.traj_rollout ~x0 ~f_list ~u_list in
  let o_list = List.map o_list ~f:(fun o -> Option.value_exn o) in
  u_list, x_list, o_list

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

  (* special care to be taken when dealing with elbo loss *)
  module ELBO_loss = struct
    type 'a with_args = 'a

    let vtgt_hessian_gv ~rolled_out_x_list ~u_list (theta : P.M.t) =
      (* fold ggn across time *)
      let ggn_final ~o_list ~like_hess =
        List.fold o_list ~init:(Tensor.f 0.) ~f:(fun accu o ->
          let vtgt = Maths.tangent o |> Option.value_exn in
          let vtgt_hess =
            Tensor.einsum ~equation:"kma,ab->kmb" [ vtgt; like_hess ] ~path:None
          in
          let increment =
            Tensor.einsum ~equation:"kma,jma->kj" [ vtgt_hess; vtgt ] ~path:None
          in
          Tensor.(accu + increment))
      in
      let llh_ggn =
        let like_hess =
          let cov_inv = theta._cov_noise |> Maths.sqr |> Maths.inv_sqr in
          let tmp = Maths.einsum [ theta._c, "ab"; cov_inv, "bc" ] "ac" in
          Maths.einsum [ tmp, "ac"; theta._c, "bc" ] "ab" |> Maths.primal
        in
        ggn_final ~o_list:rolled_out_x_list ~like_hess
      in
      let prior_ggn =
        let like_hess = theta._cov_u |> Maths.sqr |> Maths.inv_sqr |> Maths.primal in
        ggn_final ~o_list:u_list ~like_hess
      in
      let entropy_ggn =
        let like_hess = theta._cov_pos |> Maths.sqr |> Maths.inv_sqr |> Maths.primal in
        ggn_final ~o_list:u_list ~like_hess
      in
      let final = Tensor.(llh_ggn + prior_ggn - entropy_ggn) in
      final
  end

  (* create params for lds from f *)
  let params_from_f ~(theta : P.M.t) ~x0 ~o_list
    : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
    =
    (* set o at time 0 as 0 *)
    let o_list_tmp = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
    let _cov_noise_inv = theta._cov_noise |> Maths.sqr |> Maths.inv_sqr in
    let _cov_u_inv = theta._cov_u |> Maths.sqr |> Maths.inv_sqr in
    Lqr.Params.
      { x0 = Some x0
      ; params =
          List.map o_list_tmp ~f:(fun o ->
            let c_trans = Maths.transpose theta._c ~dim0:1 ~dim1:0 in
            let _Cxx = Maths.(theta._c *@ _cov_noise_inv *@ c_trans) in
            let _cx =
              Maths.((theta._b *@ _cov_noise_inv *@ c_trans) - (const o *@ c_trans))
            in
            Lds_data.Temp.
              { _f = None
              ; _Fx_prod = theta._Fx_prod
              ; _Fu_prod = theta._Fu_prod
              ; _cx = Some _cx
              ; _cu = None
              ; _Cxx
              ; _Cxu = None
              ; _Cuu = _cov_u_inv
              })
      }

  let optimal_u ~(theta : P.M.t) ~x0 ~o_list =
    (* use lqr to obtain the optimal u *)
    let p =
      params_from_f ~x0 ~theta ~o_list
      |> Lds_data.map_implicit ~batch_const:Dims.batch_const
    in
    let sol = Lqr.solve ~batch_const:Dims.batch_const p in
    let optimal_u_list = List.map sol ~f:(fun s -> s.u) in
    let u_list =
      List.map optimal_u_list ~f:(fun u ->
        let noise =
          let eps =
            Tensor.randn ~device:Dims.device ~kind:Dims.kind [ Dims.m; Dims.b ]
            |> Maths.const
          in
          Maths.einsum [ eps, "ma"; theta._cov_pos, "ab" ] "mb"
        in
        Maths.(noise + u))
    in
    optimal_u_list, u_list

  (* optimal u determined from lqr *)
  let pred_u ~data (theta : P.M.t) =
    let x0, o_list = data in
    optimal_u ~theta ~x0:(Maths.const x0) ~o_list

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
      Maths.(sum (log (diagonal g_cov ~offset:0))) |> Maths.reshape ~shape:[ 1; 1 ]
    in
    let const_term =
      let o = x |> Maths.primal |> Tensor.shape |> List.last_exn in
      Tensor.of_float0 ~device:Dims.device Float.(log (2. * pi) * of_int o)
      |> Tensor.reshape ~shape:[ 1; 1 ]
      |> Maths.const
    in
    Maths.(0.5 $* error_term + cov_term + const_term)
    |> Maths.(mean_dim ~keepdim:false ~dim:(Some [ 1 ]))
    |> Maths.neg

  let elbo ~u_list ~x_o_list ~optimal_u_list (theta : P.M.t) =
    (* calculate the likelihood term *)
    let llh =
      List.foldi x_o_list ~init:None ~f:(fun t accu (x, o) ->
        if t % 1 = 0 then Stdlib.Gc.major ();
        let increment =
          gaussian_llh
            ~g_mean:o
            ~g_cov:(Maths.sqr theta._cov_noise)
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
        let increment =
          gaussian_llh ~g_mean:u_zeros ~g_cov:(Maths.sqr theta._cov_u) ~x:u
        in
        match accu with
        | None -> Some increment
        | Some accu -> Some Maths.(accu + increment))
      |> Option.value_exn
    in
    let entropy =
      List.fold2_exn optimal_u_list u_list ~init:None ~f:(fun accu u_opt u ->
        let increment =
          gaussian_llh ~g_mean:u_opt ~g_cov:(Maths.sqr theta._cov_pos) ~x:u
        in
        match accu with
        | None -> Some increment
        | Some accu -> Some Maths.(accu + increment))
      |> Option.value_exn
    in
    Maths.(llh + prior - entropy)

  (* calculate optimal u *)
  let f ~update ~data ~init ~args:_ (theta : P.M.t) =
    let module L = ELBO_loss in
    let x0, o_list = data in
    let optimal_u_list, u_list = pred_u ~data theta in
    let rolled_out_x_list = rollout_x ~u_list ~x0 theta in
    (* These lists go from 1 to T *)
    let o_except_first = List.map o_list ~f:(fun o -> Maths.const o) in
    let x_except_first = List.tl_exn rolled_out_x_list in
    let x_o_list = List.map2_exn x_except_first o_except_first ~f:(fun x o -> x, o) in
    let neg_elbo = Maths.(neg (elbo ~u_list ~x_o_list ~optimal_u_list theta)) in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let ggn = L.vtgt_hessian_gv ~rolled_out_x_list:x_except_first ~u_list theta in
      u init (Some (neg_elbo, Some ggn))

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

  let simulate ~theta ~data =
    (* infer the optimal u *)
    let optimal_u_list, _ = pred_u ~data theta in
    (* rollout to obtain x *)
    let rolled_out_x_list = rollout_x ~u_list:optimal_u_list ~x0 theta in
    (* noiseless observation *)
    let noiseless_o_list =
      List.map rolled_out_x_list ~f:(fun x -> Maths.((x *@ theta._c) + theta._b))
      |> List.tl_exn
    in
    let o_error =
      List.fold2_exn noiseless_o_list (snd data) ~init:0. ~f:(fun accu o1 o2 ->
        let error = Tensor.(norm (Maths.primal o1 - o2)) |> Tensor.to_float0_exn in
        accu +. error)
    in
    o_error
end

module O = Optimizer.SOFO (LGS)
(* module O = Optimizer.Adam (LGS) *)

let optimise ~max_iter ~f_name config_f =
  let rec loop ~iter ~state ~time_elapsed running_avg =
    Stdlib.Gc.major ();
    let config = config_f ~iter in
    let data =
      let _, _, o_list = sample_data () in
      x0, o_list
    in
    let t0 = Unix.gettimeofday () in
    let loss, new_state = O.step ~config ~state ~data ~args:() in
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
        (* simulation error *)
        let o_error =
          let _, _, o_list = sample_data () in
          let data = x0, o_list in
          LGS.simulate ~theta:(LGS.P.const (LGS.P.value (O.params new_state))) ~data
        in
        Convenience.print [%message (iter : int) (o_error : float)];
        (* avg error *)
        Convenience.print [%message (iter : int) (loss_avg : float)];
        let t = iter in
        Owl.Mat.(
          save_txt
            ~append:true
            ~out:(in_dir f_name)
            (of_array [| Float.of_int t; time_elapsed; loss_avg; o_error |] 1 4));
        O.W.P.T.save
          (LGS.P.value (O.params new_state))
          ~kind:base.ba_kind
          ~out:(in_dir f_name ^ "_params"));
      []
    in
    if iter < max_iter
    then loop ~iter:(iter + 1) ~state:new_state ~time_elapsed (loss :: running_avg)
  in
  (* ~config:(config_f ~iter:0) *)
  loop ~iter:0 ~state:(O.init ~config:(config_f ~iter:0) LGS.(init)) ~time_elapsed:0. []

let lr_rates = [ 1e-8 ]
let damping_list = [ Some 1e-2 ]
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

(* let lr_rates = [ 0.01 ]
let meth = "adam"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    let config_f = config ~base_lr:eta ~gamma:None in
    let f_name = sprintf "lgs_%s_lr_%s" meth (Float.to_string eta) in
    Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
    Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
    optimise ~max_iter ~f_name config_f) *)
