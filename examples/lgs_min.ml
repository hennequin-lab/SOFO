(* minimal example of supervised learning on lgs *)
open Printf
open Base
open Forward_torch
open Torch
open Sofo
module Mat = Owl.Dense.Matrix.D

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

(* lds data generation *)
let a = 24
let b = 10
let o = 48
let tmax = 10
let batch_size = 512
let kind = Torch_core.Kind.(T f64)
let device = Torch.Device.cuda_if_available ()
let base = Optimizer.Config.Base.{ default with kind; ba_kind = Bigarray.float64 }
let x0 = Tensor.zeros ~device ~kind [ batch_size; a ]

let sample_stable ~a =
  let a = Mat.gaussian a a in
  let r =
    a
    |> Owl.Linalg.D.eigvals
    |> Owl.Dense.Matrix.Z.abs
    |> Owl.Dense.Matrix.Z.re
    |> Mat.max'
  in
  Mat.(Float.(0.8 / r) $* a)

let _Fx = Tensor.of_bigarray ~device (sample_stable ~a)
let _Fu = Tensor.randn ~device ~kind [ b; a ]
let c = Tensor.randn ~device ~kind [ a; o ]

(* x list goes from 0 to T but o list goes from 1 to T *)
let rollout ~u_list =
  let _, x_list, o_list =
    List.fold u_list ~init:(x0, [ x0 ], []) ~f:(fun (x_prev, x_list, o_list) u ->
      let x_curr = Tensor.(matmul x_prev _Fx + matmul u _Fu) in
      let o_curr = Tensor.matmul x_curr c in
      x_curr, x_curr :: x_list, o_curr :: o_list)
  in
  List.rev x_list, List.rev o_list

let sample_data () =
  let u_list =
    List.init tmax ~f:(fun _ -> Tensor.randn ~device ~kind [ batch_size; b ])
  in
  let x_list, o_list = rollout ~u_list in
  u_list, x_list, o_list

(* generative model to learn _Fx and _Fu *)
module LGS = struct
  module PP = struct
    type 'a p =
      { _Fx_prod : 'a (* generative model *)
      ; _Fu_prod : 'a
      ; _c : 'a
      }
    [@@deriving prms]
  end

  module P = PP.Make (Prms.P)

  type args = unit
  type data = Tensor.t list * Tensor.t list * Tensor.t list (* u list, x list, o list *)

  (* create params for lds from f *)
  let params_from_f ~(theta : P.M.t) ~x0 ~o_list
    : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
    =
    (* set o at time 0 as 0 *)
    let o_list_tmp = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
    let _Cxx = Maths.(einsum [ theta._c, "ab"; theta._c, "cb" ] "ac") in
    let _Cuu =
      Tensor.(mul_scalar (eye ~options:(base.kind, base.device) ~n:b) (Scalar.f 0.001))
      |> Maths.const
    in
    Lqr.Params.
      { x0 = Some x0
      ; params =
          List.map o_list_tmp ~f:(fun o ->
            let _cx = Maths.(neg (einsum [ const o, "mb"; theta._c, "ab" ] "ma")) in
            Lds_data.Temp.
              { _f = None
              ; _Fx_prod = theta._Fx_prod
              ; _Fu_prod = theta._Fu_prod
              ; _cx = Some _cx
              ; _cu = None
              ; _Cxx
              ; _Cxu = None
              ; _Cuu
              })
      }

  let pred_u ~data (theta : P.M.t) =
    let _, _, o_list = data in
    let p =
      params_from_f ~x0:(Maths.const x0) ~theta ~o_list
      |> Lds_data.map_naive ~batch_const:true
    in
    let sol, _ = Lqr._solve ~laplace:false ~batch_const:true p in
    List.map sol ~f:(fun s -> s.u)

  let rollout ~u_list (theta : P.M.t) =
    let _, _, o_list =
      List.fold
        u_list
        ~init:(Maths.const x0, [ Maths.const x0 ], [])
        ~f:(fun (x_prev, x_list, o_list) u ->
          let x_curr = Maths.((x_prev *@ theta._Fx_prod) + (u *@ theta._Fu_prod)) in
          let o_curr = Maths.(x_curr *@ theta._c) in
          x_curr, x_curr :: x_list, o_curr :: o_list)
    in
    List.rev o_list

  let f ~update ~data ~init ~args:() (theta : P.M.t) =
    let u_list, _, _ = data in
    let pred_u_list = pred_u ~data theta in
    let neg_llh =
      let llh =
        List.fold2_exn pred_u_list u_list ~init:None ~f:(fun accu u1 u2 ->
          let error = Maths.(u1 - const u2) in
          let reduce_dim_list = Convenience.all_dims_but_first u2 in
          let increment =
            Maths.(mean_dim ~keepdim:false ~dim:(Some reduce_dim_list) (sqr error))
          in
          match accu with
          | None -> Some increment
          | Some accu -> Some Maths.(accu + increment))
        |> Option.value_exn
      in
      Maths.(llh /$ Float.of_int tmax)
    in
    match update with
    | `loss_only u -> u init (Some neg_llh)
    | `loss_and_ggn u ->
      let ggn =
        let ggn =
          List.fold pred_u_list ~init:None ~f:(fun accu u ->
            let vtgt = Maths.tangent u |> Option.value_exn in
            let n_samples = Convenience.first_dim vtgt in
            let vtgt_mat = Tensor.reshape vtgt ~shape:[ n_samples; -1 ] in
            let increment = Convenience.a_b_trans vtgt_mat vtgt_mat in
            match accu with
            | None -> Some increment
            | Some accu -> Some Tensor.(accu + increment))
          |> Option.value_exn
        in
        Tensor.(div_scalar ggn (Scalar.f (Float.of_int tmax)))
      in
      u init (Some (neg_llh, Some ggn))

  let init : P.tagged =
    let _Fx_prod = Tensor.randn ~device ~kind [ a; a ] |> Prms.free in
    let _Fu_prod = Tensor.randn ~device ~kind [ b; a ] |> Prms.free in
    let _c = Tensor.randn ~device ~kind [ a; o ] |> Prms.free in
    (* let _Fx_prod = _Fx |> Prms.free in
    let _Fu_prod = _Fu |> Prms.free in
    let _c = c |> Prms.free in *)
    { _Fx_prod; _Fu_prod; _c }

  let error x y ~reduce_dim_list =
    Tensor.(
      mean_dim
        (square (Maths.primal x - y))
        ~dim:(Some reduce_dim_list)
        ~keepdim:false
        ~dtype:base.kind)
    |> Tensor.mean
    |> Tensor.to_float0_exn

  (* calculate the error between observations *)
  let simulate ~theta ~data =
    let u_list, _, o_list = data in
    (* rollout to obtain o *)
    let rolled_out_o_list = rollout ~u_list:(List.map u_list ~f:Maths.const) theta in
    let reduce_dim_list = Convenience.all_dims_but_first (List.hd_exn o_list) in
    let error =
      List.fold2_exn rolled_out_o_list o_list ~init:0. ~f:(fun accu x1 x2 ->
        let error = error x1 x2 ~reduce_dim_list in
        accu +. error)
    in
    Float.(error / of_int tmax)

  let simulate_u ~theta ~data =
    let pred_u = pred_u ~data theta in
    let u_list, _, _ = data in
    let reduce_dim_list = Convenience.all_dims_but_first (List.hd_exn u_list) in
    let error =
      List.fold2_exn pred_u u_list ~init:0. ~f:(fun accu x1 x2 ->
        let error = error x1 x2 ~reduce_dim_list in
        accu +. error)
    in
    Float.(error / of_int tmax)
end

let config ~base_lr ~gamma ~iter:_ =
  Optimizer.Config.SOFO.
    { base
    ; learning_rate = Some base_lr
    ; n_tangents = 256
    ; rank_one = false
    ; damping = gamma
    ; momentum = None
    ; lm = false
    ; perturb_thresh = None
    ; sqrt = false
    }

module O = Optimizer.SOFO (LGS)

(* let config ~base_lr ~gamma:_ ~iter:_ =
  Optimizer.Config.Adam.{ default with learning_rate = Some base_lr }

module O = Optimizer.Adam (LGS) *)

let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run
let max_iter = 2000

let optimise ~max_iter ~f_name ~init config_f =
  let rec loop ~iter ~state ~time_elapsed running_avg =
    Stdlib.Gc.major ();
    let config = config_f ~iter in
    let data = sample_data () in
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
      if iter % 1 = 0
      then (
        (* simulation error *)
        let sim_error, sim_u_error =
          (* let data = sample_data () in *)
          ( LGS.simulate ~theta:(LGS.P.const (LGS.P.value (O.params state))) ~data
          , LGS.simulate_u ~theta:(LGS.P.const (LGS.P.value (O.params state))) ~data )
        in
        (* avg error *)
        Convenience.print [%message (iter : int) (loss_avg : float)];
        let t = iter in
        Owl.Mat.(
          save_txt
            ~append:true
            ~out:(in_dir f_name)
            (of_array
               [| Float.of_int t; time_elapsed; loss_avg; sim_error; sim_u_error |]
               1
               5));
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
  loop ~iter:0 ~state:(O.init ~config:(config_f ~iter:0) init) ~time_elapsed:0. []

let lr_rates = [ 1. ]
let damping_list = [ Some 1e-5 ]
let meth = "sofo"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    List.iter damping_list ~f:(fun gamma ->
      let config_f = config ~base_lr:eta ~gamma in
      let gamma_name = Option.value_map gamma ~default:"none" ~f:Float.to_string in
      let init, f_name =
        ( LGS.(init)
        , sprintf "lgs_elbo_%s_lr_%s_damp_%s" meth (Float.to_string eta) gamma_name )
      in
      Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
      Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
      optimise ~max_iter ~f_name ~init config_f))

(* let lr_rates = [ 0.01 ]
let meth = "adam"

let _ =
  List.iter lr_rates ~f:(fun eta ->
    let config_f = config ~base_lr:eta ~gamma:None in
    let init, f_name =
      LGS.(init), sprintf "lgs_elbo_%s_lr_%s" meth (Float.to_string eta)
    in
    Bos.Cmd.(v "rm" % "-f" % in_dir f_name) |> Bos.OS.Cmd.run |> ignore;
    Bos.Cmd.(v "rm" % "-f" % in_dir (f_name ^ "_llh")) |> Bos.OS.Cmd.run |> ignore;
    optimise ~max_iter ~f_name ~init config_f)  *)
