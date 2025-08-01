(* Lorenz attractor with no controls, with same state/control/cost parameters constant across trials and across time; 
  use a mini-MGU2 (ilqr-vae paper Appendix C.2) as generative model. use kroneckered posterior cov. *)
open Base
open Forward_torch
open Torch
open Sofo
open Lds_data

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let in_dir = Cmdargs.in_dir "-d"
let _ = Bos.Cmd.(v "rm" % "-f" % in_dir "info") |> Bos.OS.Cmd.run

(* -----------------------------------------
   -- Generate Lorenz data            ------
   ----------------------------------------- *)

open Lorenz_common

(* state dim *)
let n = 20

(* control dim *)
let m = 5
let o = 3
let batch_size = 128

(* tmax needs to be divisible by 8 *)
let tmax = 80
let tmax_simulate = 1000
let train_data = Lorenz_common.data tmax
let train_data_batch = get_batch train_data
let max_iter = 100000
let base = Optimizer.Config.Base.default

(* list of clean data * list of noisy data *)
let sample_data () =
  let trajectory = train_data_batch batch_size in
  let traj_no_init = List.tl_exn trajectory in
  let both =
    List.map traj_no_init ~f:(fun x ->
      let x =
        Tensor.of_bigarray ~device:base.device x |> Tensor.to_type ~type_:base.kind
      in
      let noise = Tensor.(f 0.1 * rand_like x) in
      x, Tensor.(noise + x))
  in
  List.map both ~f:fst, List.map both ~f:snd

let x0 = Maths.zeros ~device:base.device ~kind:base.kind [ batch_size; n ] |> Maths.any
let eye_m = Maths.(eye m ~device:base.device ~kind:base.kind)
let ones_o = Maths.(ones ~device:base.device ~kind:base.kind [ o ])

(* -----------------------------------------
   ----- Utilitiy Functions ------
   ----------------------------------------- *)

let gaussian_llh ?mu ?(diagonal_inv_std = true) ~inv_std x =
  let d = x |> Maths.shape |> List.last_exn in
  let error_term =
    let error =
      match mu with
      | None ->
        if diagonal_inv_std
        then Maths.(einsum [ x, "ma"; inv_std, "a" ] "ma")
        else Maths.(einsum [ x, "ma"; inv_std, "ab" ] "mb")
      | Some mu ->
        if diagonal_inv_std
        then Maths.(einsum [ x - mu, "ma"; inv_std, "a" ] "ma")
        else Maths.(einsum [ x - mu, "ma"; inv_std, "ab" ] "mb")
    in
    Maths.einsum [ error, "ma"; error, "ma" ] "m"
  in
  let cov_term = Maths.(neg (sum (log (sqr inv_std))) |> reshape ~shape:[ 1 ]) in
  let const_term = Float.(log (2. * pi) * of_int d) in
  Maths.(0.5 $* (const_term $+ error_term + cov_term)) |> Maths.neg

let precision_of_log_var log_var = Maths.(exp (neg log_var))
let std_of_log_var log_var = Maths.(exp (f 0.5 * log_var))
let chol_of_log_var log_var = Maths.(cholesky (exp log_var))
let sqrt_precision_of_log_var log_var = Maths.(exp (f (-0.5) * log_var))

(* list of length T of [m x b] to matrix of [m x b x T] *)
let concat_time (u_list : Maths.any Maths.t list) =
  List.map u_list ~f:(fun u -> Maths.unsqueeze ~dim:(-1) u) |> Maths.concat ~dim:2

(* concat a list of [m x 3] tensors to [T x m x 3] mat *)
let t_list_to_mat data =
  Tensor.concat (List.map data ~f:(Tensor.unsqueeze ~dim:0)) ~dim:0
  |> Tensor.to_bigarray ~kind:base.ba_kind

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)

let conv_threshold = 1e-4

module PP = struct
  type 'a p =
    { _U_f : 'a (* generative model *)
    ; _U_h : 'a
    ; _b_f : 'a
    ; _b_h : 'a
    ; _W : 'a
    ; _c : 'a
    ; _b : 'a
    ; _log_prior_var : 'a (* log of the diagonal covariance of prior over u *)
    ; _log_obs_var : 'a (* log of the diagonal covariance of emission noise; *)
    ; _log_space_var : 'a (* log of the FULL covariance of space factor; *)
    ; _log_time_var : 'a (* log of the FULL covariance of time factor; *)
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.Single)

module MGU = struct
  module P = P

  let q_of u =
    let open Maths in
    let ell = einsum [ u, "ik"; u, "jk" ] "ij" |> cholesky in
    linsolve_triangular ~left:true ~upper:false ell u

  let _Fx_reparam (q, d) =
    let q = q_of q in
    let open Maths in
    let d = exp d in
    let left_factor = sqrt d in
    let right_factor = f 1. / sqrt (f 1. + d) in
    einsum [ left_factor, "qi"; q, "ij"; right_factor, "qj" ] "ji"

  let pre_sig ~x (theta : _ Maths.some P.t) = Maths.((x *@ theta._U_f) + theta._b_f)

  let pre_g ~f_t ~x (theta : _ Maths.some P.t) =
    Maths.((f_t * x *@ theta._U_h) + theta._b_h)

  let x_hat ~pre_g ~u (theta : _ Maths.some P.t) =
    Maths.(soft_relu pre_g + (u *@ theta._W))

  (* rollout x list under sampled u *)
  let rollout_one_step ~x ~u (theta : _ Maths.some P.t) =
    let pre_sig = pre_sig ~x theta in
    let f_t = Maths.sigmoid pre_sig in
    let pre_g = pre_g ~f_t ~x theta in
    let x_hat = x_hat ~pre_g ~u theta in
    let new_x = Maths.(((f 1. - f_t) * x) + (f_t * x_hat)) in
    new_x

  (* (1 + e^-x)^{-2} (e^-x) *)
  let d_sigmoid x = Maths.(sigmoid x * (f 1. - sigmoid x))

  (* 1/2 [1 + (x^2 + 4)^{-1/2} x] *)
  let d_soft_relu x =
    let tmp = Maths.(sqr x + f 4.) in
    let tmp2 = Maths.(f 1. / sqrt tmp) in
    Maths.(((tmp2 * x) + f 1.) / f 2.)

  (* _Fu and _Fx TESTED. *)
  let _Fu ~x (theta : _ Maths.some P.t) =
    let open Maths in
    match x with
    | Some x ->
      let pre_sig = pre_sig ~x theta in
      let f_t = sigmoid pre_sig in
      einsum [ f_t, "ma"; theta._W, "ba" ] "mba"
    | None -> any (zeros ~device:base.device ~kind:base.kind [ batch_size; m; n ])

  let _Fx ~x ~u (theta : _ Maths.some P.t) =
    let open Maths in
    match x, u with
    | Some x, Some u ->
      let pre_sig = pre_sig ~x theta in
      let f_t = sigmoid pre_sig in
      let pre_g = pre_g ~f_t ~x theta in
      let x_hat = x_hat ~pre_g ~u theta in
      let tmp_einsum2 a b = einsum [ a, "ba"; b, "ma" ] "mba" in
      let term1 = diag_embed (f 1. - f_t) ~offset:0 ~dim1:(-2) ~dim2:(-1) in
      let term2 =
        let tmp = d_sigmoid pre_sig * (x_hat - x) in
        tmp_einsum2 theta._U_f tmp
      in
      let term3 =
        let tmp1 = tmp_einsum2 theta._U_h (d_soft_relu pre_g) in
        let tmp2 = tmp_einsum2 theta._U_f (d_sigmoid pre_sig) in
        let tmp3 = diag_embed f_t ~offset:0 ~dim1:(-2) ~dim2:(-1) in
        let tmp4 = tmp2 + tmp3 in
        let tmp5 = einsum [ tmp4, "mab"; tmp1, "mbc" ] "mac" in
        f_t * tmp5
      in
      let final = term1 + term2 + term3 in
      final
    | _ -> any (zeros ~device:base.device ~kind:base.kind [ batch_size; n; n ])

  let rollout_y ~u_list (theta : _ Maths.some P.t) =
    let _, y_list =
      List.fold u_list ~init:(x0, []) ~f:(fun (x, accu) u ->
        let new_x = rollout_one_step ~x ~u theta in
        let new_y = Maths.(einsum [ new_x, "ma"; theta._c, "ab" ] "mb" + theta._b) in
        Stdlib.Gc.major ();
        new_x, new_y :: accu)
    in
    List.rev y_list

  let rollout_sol ~u_list (theta : _ Maths.some P.t) =
    let _, x_list =
      List.fold u_list ~init:(x0, []) ~f:(fun (x, accu) u ->
        let new_x = rollout_one_step ~x ~u theta in
        Stdlib.Gc.major ();
        new_x, Lqr.Solution.{ u = Some u; x = Some new_x } :: accu)
    in
    List.rev x_list

  (* artificially add one to tau so it goes from 0 to T *)
  let extend_tau_list (tau : Maths.any Maths.t option Lqr.Solution.p list) =
    let u_list = List.map tau ~f:(fun s -> s.u) in
    let x_list = List.map tau ~f:(fun s -> s.x) in
    let u_ext = u_list @ [ None ] in
    let x_ext = Some x0 :: x_list in
    List.map2_exn u_ext x_ext ~f:(fun u x -> Lqr.Solution.{ u; x })

  (* optimal u determined from lqr *)
  let pred_u ~data (theta : _ Maths.some P.t) =
    let open Maths in
    let c_trans = transpose theta._c ~dims:[ 1; 0 ] in
    let _obs_var_inv = exp (neg theta._log_obs_var) * ones_o in
    let _Cxx_batched =
      let _Cxx =
        Maths.(einsum [ theta._c, "ab"; _obs_var_inv, "b"; c_trans, "bc" ] "ac")
        |> Maths.unsqueeze ~dim:0
      in
      Maths.broadcast_to _Cxx ~size:[ batch_size; n; n ]
    in
    let _Cuu_batched =
      let _Cuu =
        Maths.(diag_embed (exp theta._log_prior_var) ~offset:0 ~dim1:(-2) ~dim2:(-1))
        |> Maths.unsqueeze ~dim:0
      in
      Maths.broadcast_to _Cuu ~size:[ batch_size; m; m ]
    in
    let _cx_common = einsum [ theta._b, "ab"; _obs_var_inv, "b"; c_trans, "bc" ] "ac" in
    let params_func (tau : any t option Lqr.Solution.p list)
      : (any t option, (any t, any t -> any t) Lqr.momentary_params list) Lqr.Params.p
      =
      (* set o at time 0 as 0 *)
      let o_list_extended = Tensor.zeros_like (List.hd_exn data) :: data in
      let tau_extended = extend_tau_list tau in
      let tmp_list =
        Lqr.Params.
          { x0 = Some x0
          ; params =
              List.map2_exn o_list_extended tau_extended ~f:(fun o s ->
                let _cx =
                  let tmp1 = einsum [ any (of_tensor o), "ab"; _obs_var_inv, "b" ] "ab" in
                  let tmp2 =
                    einsum
                      [ Option.value_exn s.x, "ab"; theta._c, "bc"; _obs_var_inv, "c" ]
                      "ac"
                  in
                  _cx_common + ((tmp2 - tmp1) *@ c_trans)
                in
                let _cu =
                  match s.u with
                  | None -> None
                  | Some u -> Some (einsum [ u, "ma"; _Cuu_batched, "mab" ] "mb")
                in
                Lds_data.Temp.
                  { _f = None
                  ; _Fx_prod = _Fx ~x:s.x ~u:s.u theta
                  ; _Fu_prod = _Fu ~x:s.x theta
                  ; _cx = Some _cx
                  ; _cu
                  ; _Cxx = _Cxx_batched
                  ; _Cxu = None
                  ; _Cuu = _Cuu_batched
                  })
          }
      in
      Lds_data.map_naive tmp_list ~batch_const:false
    in
    (* given a trajectory and parameters, calculate average cost across batch (summed over time) *)
    let cost_func (tau : any t option Lqr.Solution.p list) =
      let x_list = List.map tau ~f:(fun s -> s.x |> Option.value_exn) in
      let u_list = List.map tau ~f:(fun s -> s.u |> Option.value_exn) in
      let x_cost =
        let x_cost_list =
          List.map2_exn x_list data ~f:(fun x data ->
            let tmp1 =
              einsum
                [ x, "ma"; theta._c, "ab"; _obs_var_inv, "b"; c_trans, "bc"; x, "mc" ]
                "m"
            in
            let tmp2 =
              let diff =
                einsum
                  [ broadcast_to theta._b ~size:[ batch_size; o ] - of_tensor data, "mb"
                  ; _obs_var_inv, "b"
                  ; c_trans, "bc"
                  ]
                  "mc"
              in
              f 2. * einsum [ x, "ma"; diff, "ma" ] "m"
            in
            tmp1 + tmp2)
        in
        List.fold x_cost_list ~init:(any (f 0.)) ~f:(fun accu c -> accu + c)
      in
      let u_cost =
        List.fold
          u_list
          ~init:(any (f 0.))
          ~f:(fun accu u -> accu + einsum [ u, "ma"; _Cuu_batched, "mab"; u, "mb" ] "m")
      in
      x_cost + u_cost |> to_tensor |> Tensor.mean |> Tensor.to_float0_exn
    in
    let u_init =
      List.init tmax ~f:(fun _ ->
        any (zeros ~device:base.device ~kind:base.kind [ batch_size; m ]))
    in
    let tau_init = rollout_sol ~u_list:u_init theta in
    (* TODO: is there a more elegant way? Currently I need to set batch_const to false since _Fx and _Fu has batch dim. *)
    (* use lqr to obtain the optimal u *)
    let f_theta = rollout_one_step theta in
    let sol, backward_info =
      Ilqr._isolve
        ~linesearch:false
        ~batch_const:false
        ~gamma:0.5
        ~f_theta
        ~cost_func
        ~params_func
        ~conv_threshold
        ~tau_init
        2000
    in
    List.map sol ~f:(fun s -> s.u |> Option.value_exn), backward_info

  let kronecker_sample ~optimal_u_list (theta : _ Maths.some P.t) =
    let open Maths in
    (* sample u from the kronecker formation *)
    let u_list =
      let optimal_u = concat_time optimal_u_list in
      let xi = randn ~device:base.device ~kind:base.kind [ batch_size; m; tmax ] |> any in
      let xi_space =
        einsum [ xi, "mbt"; chol_of_log_var theta._log_space_var, "ba" ] "mat"
      in
      let xi_time =
        einsum [ xi_space, "mat"; chol_of_log_var theta._log_time_var, "tk" ] "mak"
      in
      let meaned = xi_time + optimal_u in
      List.init tmax ~f:(fun i ->
        slice ~dim:2 ~start:i ~end_:Int.(i + 1) ~step:1 meaned
        |> reshape ~shape:[ batch_size; m ])
    in
    u_list

  let neg_elbo ~data (theta : _ Maths.some P.t) =
    let open Maths in
    (* obtain u from lqr *)
    let optimal_u_list, _ = pred_u ~data theta in
    let u_sampled = kronecker_sample ~optimal_u_list theta in
    (* calculate the likelihood term *)
    let y_pred = rollout_y ~u_list:u_sampled theta in
    let lik_term =
      let inv_sigma_o_expanded = sqrt_precision_of_log_var theta._log_obs_var * ones_o in
      List.fold2_exn
        data
        y_pred
        ~init:(any (f 0.))
        ~f:(fun accu o y_pred ->
          accu + gaussian_llh ~mu:y_pred ~inv_std:inv_sigma_o_expanded (any (of_tensor o)))
    in
    (* calculate the kl term using samples *)
    let optimal_u = concat_time optimal_u_list in
    let kl =
      let prior =
        List.foldi u_sampled ~init:None ~f:(fun t accu u ->
          if t % 1 = 0 then Stdlib.Gc.major ();
          let increment =
            gaussian_llh ~inv_std:(any (ones [ m ] ~device:base.device ~kind:base.kind)) u
          in
          match accu with
          | None -> Some increment
          | Some accu -> Some (accu + increment))
        |> Option.value_exn
      in
      let neg_entropy =
        let u = concat_time u_sampled |> reshape ~shape:[ batch_size; -1 ] in
        let optimal_u = reshape optimal_u ~shape:[ batch_size; -1 ] in
        let inv_std =
          let _space_var_inv = inv_sqr (exp theta._log_space_var) in
          let _time_var_inv = inv_sqr (exp theta._log_time_var) in
          cholesky (kron _space_var_inv _time_var_inv)
        in
        gaussian_llh ~diagonal_inv_std:false ~mu:optimal_u ~inv_std u
      in
      neg_entropy - prior
    in
    mean ~dim:[ 0 ] (neg (lik_term - kl) / f Float.(of_int tmax * of_int o)), y_pred

  let ggn ~y_pred (theta : _ Maths.some P.t) =
    let open Maths in
    let obs_precision = precision_of_log_var theta._log_obs_var in
    let obs_precision_p = of_tensor (to_tensor obs_precision) in
    let sigma2_t = tangent (exp theta._log_obs_var) |> Option.value_exn in
    List.fold y_pred ~init:(f 0.) ~f:(fun accu y_pred ->
      let mu_t = tangent y_pred |> Option.value_exn |> const in
      let ggn_part1 = C.(einsum [ mu_t, "kmo"; mu_t, "lmo" ] "kl" * obs_precision_p) in
      let ggn_part2 =
        C.(
          einsum
            [ ( Float.(0.5 * of_int o * of_int batch_size)
                $* sigma2_t * sqr obs_precision_p
              , "ky" )
            ; sigma2_t, "ly"
            ]
            "kl")
      in
      C.(accu + (Float.(1. / (of_int o * of_int tmax)) $* ggn_part1 + ggn_part2)))

  let f ~data (theta : _ Maths.some P.t) =
    let neg_elbo, _ = neg_elbo ~data theta in
    (* let ggn = ggn ~y_pred theta in *)
    neg_elbo, None

  let init : P.param =
    let to_param a = a |> Maths.of_tensor |> Prms.Single.free in
    let _U_f =
      Sofo.gaussian_tensor_normed ~device:base.device ~kind:base.kind ~sigma:1. [ n; n ]
      |> to_param
    in
    let _U_h =
      Sofo.gaussian_tensor_normed ~device:base.device ~kind:base.kind ~sigma:1. [ n; n ]
      |> to_param
    in
    let _b_f = Tensor.zeros ~device:base.device [ 1; n ] |> to_param in
    let _b_h = Tensor.zeros ~device:base.device [ 1; n ] |> to_param in
    let _W =
      Sofo.gaussian_tensor_normed ~device:base.device ~kind:base.kind ~sigma:1. [ m; n ]
      |> to_param
    in
    let _c =
      Sofo.gaussian_tensor_normed ~device:base.device ~kind:base.kind ~sigma:1. [ n; o ]
      |> to_param
    in
    let _b = Tensor.zeros ~device:base.device [ 1; o ] |> to_param in
    let _log_prior_var =
      Tensor.(log (f Float.(square 1.) * ones ~device:base.device ~kind:base.kind [ m ]))
      |> to_param
    in
    let _log_obs_var =
      Tensor.(log (f Float.(square 1.) * ones ~device:base.device ~kind:base.kind [ 1 ]))
      |> to_param
    in
    let _log_space_var =
      Tensor.(log (ones ~device:base.device ~kind:base.kind [ m; m ])) |> to_param
    in
    let _log_time_var =
      Tensor.(log (ones ~device:base.device ~kind:base.kind [ tmax; tmax ])) |> to_param
    in
    { _U_f
    ; _U_h
    ; _b_f
    ; _b_h
    ; _W
    ; _c
    ; _b
    ; _log_prior_var
    ; _log_obs_var
    ; _log_space_var
    ; _log_time_var
    }

  let simulate ~theta ~data =
    (* infer the optimal u *)
    let optimal_u_list, _ = pred_u ~data theta in
    (* rollout x and y *)
    let _, y_list_rev =
      List.fold optimal_u_list ~init:(x0, []) ~f:(fun (x, y_list) u ->
        let x = rollout_one_step ~x ~u theta in
        let y = Maths.((x *@ theta._c) + theta._b) in
        x, y :: y_list)
    in
    let o_list_rev =
      List.map y_list_rev ~f:(fun y ->
        Maths.(
          y
          + einsum
              [ ( any (Maths.randn ~device:base.device ~kind:base.kind [ batch_size; o ])
                , "ma" )
              ; std_of_log_var theta._log_obs_var, "a"
              ]
              "ma"))
    in
    optimal_u_list, List.rev y_list_rev, List.rev o_list_rev

  (* simulate the autonomous dynamics of Lorenz given initial condition. *)
  let simulate_auto ~theta =
    let init_cond =
      Maths.randn ~device:base.device ~kind:base.kind [ 1; n ] |> Maths.any
    in
    (* rollout y *)
    let _, y_list_rev =
      List.fold
        (List.init tmax_simulate ~f:(fun i -> i))
        ~init:(init_cond, [])
        ~f:(fun (x, y_list) _ ->
          let x =
            rollout_one_step
              ~x
              ~u:Maths.(any (zeros ~device:base.device ~kind:base.kind [ 1; m ]))
              theta
          in
          let y = Maths.((x *@ theta._c) + theta._b) in
          x, y :: y_list)
    in
    List.rev y_list_rev
end

(* module O = Optimizer.SOFO (MGU.P)

let config =
  Optimizer.Config.SOFO.
    { base; learning_rate = Some 1.; n_tangents = 128; damping = `relative_from_top 1e-5 }

let rec loop ~t ~out ~state running_avg =
  Stdlib.Gc.major ();
  let data_unnoised, data = sample_data () in
  let theta, tangents = O.prepare ~config state in
  let loss, ggn = MGU.f ~data (P.map theta ~f:Maths.any) in
  let new_state = O.step ~config ~info:{ loss; ggn; tangents } state in
  let loss = Maths.to_float_exn (Maths.const loss) in
  let running_avg =
    let loss_avg =
      match running_avg with
      | [] -> loss
      | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
    in
    if t % 10 = 0
    then (
      let theta = MGU.P.value (O.params new_state) in
      (* simulate trajectory *)
      let y_list, o_list = MGU.simulate ~theta:(MGU.P.map theta ~f:Maths.any) ~data in
      let y_list_t = List.map y_list ~f:Maths.to_tensor
      and o_list_t = List.map o_list ~f:Maths.to_tensor in
      Arr.(save_npy ~out:(in_dir "o") (t_list_to_mat data));
      Arr.(save_npy ~out:(in_dir "y") (t_list_to_mat data_unnoised));
      Arr.(save_npy ~out:(in_dir "y_inferred") (t_list_to_mat y_list_t));
      Arr.(save_npy ~out:(in_dir "o_gen") (t_list_to_mat o_list_t));
      (* save params *)
      O.P.C.save theta ~kind:base.ba_kind ~out:(in_dir "sofo_params");
      print [%message (t : int) (loss_avg : float)];
      Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss_avg |] 1 2)));
    []
  in
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state (loss :: running_avg)

(* Start the sofo loop. *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init MGU.init) [] *)

module O = Optimizer.Adam (MGU.P)

let config =
  Optimizer.Config.Adam.
    { base
    ; beta_1 = 0.9
    ; beta_2 = 0.99
    ; eps = 1e-4
    ; learning_rate = Some 1e-3
    ; weight_decay = None
    }

let rec loop ~t ~out ~state running_avg =
  Stdlib.Gc.major ();
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
  let data_unnoised, data = sample_data () in
  let loss, true_g =
    let loss, _ = MGU.f ~data (P.map theta_dual ~f:Maths.any) in
    let loss = Maths.to_tensor loss in
    Tensor.backward loss;
    ( Tensor.to_float0_exn loss
    , O.P.map2 theta (O.P.to_tensor theta_dual) ~f:(fun tagged p ->
        match tagged with
        | Prms.Pinned _ -> Maths.(f 0.)
        | _ -> Maths.of_tensor (Tensor.grad p)) )
  in
  let new_state = O.step ~config ~info:true_g state in
  let running_avg =
    let loss_avg =
      match running_avg with
      | [] -> loss
      | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
    in
    if t % 10 = 0
    then (
      let theta = MGU.P.value (O.params new_state) in
      (* simulate trajectory *)
      let u_list, y_list, o_list =
        MGU.simulate ~theta:(MGU.P.map theta ~f:Maths.any) ~data
      in
      let y_list_auto = MGU.simulate_auto ~theta:(MGU.P.map theta ~f:Maths.any) in
      let u_list_t = List.map u_list ~f:Maths.to_tensor
      and y_list_t = List.map y_list ~f:Maths.to_tensor
      and o_list_t = List.map o_list ~f:Maths.to_tensor
      and y_list_auto_t = List.map y_list_auto ~f:Maths.to_tensor in
      Arr.(save_npy ~out:(in_dir "o") (t_list_to_mat data));
      Arr.(save_npy ~out:(in_dir "y") (t_list_to_mat data_unnoised));
      Arr.(save_npy ~out:(in_dir "u_inferred") (t_list_to_mat u_list_t));
      Arr.(save_npy ~out:(in_dir "y_inferred") (t_list_to_mat y_list_t));
      Arr.(save_npy ~out:(in_dir "y_auto") (t_list_to_mat y_list_auto_t));
      Arr.(save_npy ~out:(in_dir "o_gen") (t_list_to_mat o_list_t));
      (* save params *)
      O.P.C.save theta ~kind:base.ba_kind ~out:(in_dir "sofo_params");
      print [%message (t : int) (loss_avg : float)];
      Owl.Mat.(save_txt ~append:true ~out (of_array [| Float.of_int t; loss_avg |] 1 2)));
    []
  in
  if t < max_iter then loop ~t:Int.(t + 1) ~out ~state:new_state (loss :: running_avg)

(* Start the sofo loop. *)
let _ =
  let out = in_dir "loss" in
  Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore;
  loop ~t:0 ~out ~state:(O.init MGU.init) []
