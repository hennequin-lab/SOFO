(* Linear Gaussian Dynamics on HVS data *)
open Base
open Forward_torch
open Torch
open Sofo
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

let _ =
  Random.init 1998;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f32)
    ; ba_kind = Bigarray.float32
    }

(* batch size. *)
let bs = 32
let in_dir = Cmdargs.in_dir "-d"
let data_folder = Option.value (Cmdargs.get_string "-data") ~default:"DH1"

(* -----------------------------------------
   --- Utility Functions ---
   ----------------------------------------- *)

(* solves for xA = y, A = ell (ell)^T. NOTE: since linsolve_triangular only deals with rectangular matrix B 
but not vector, this function does not apply to ell with a batch dimension in front! *)
let solver_chol ell y =
  let ell_t = Maths.transpose ~dim0:(-1) ~dim1:(-2) ell in
  let _x = Maths.linsolve_triangular ~left:false ~upper:true ell_t y in
  Maths.linsolve_triangular ~left:false ~upper:false ell _x

let gaussian_llh ?mu ~inv_std x =
  let d = x |> Maths.primal |> Tensor.shape |> List.last_exn in
  let error_term =
    let error =
      match mu with
      | None -> Maths.(einsum [ x, "ma"; inv_std, "a" ] "ma")
      | Some mu -> Maths.(einsum [ x - mu, "ma"; inv_std, "a" ] "ma")
    in
    Maths.einsum [ error, "ma"; error, "ma" ] "m"
  in
  let cov_term = Maths.(neg (sum (log (sqr inv_std))) |> reshape ~shape:[ 1 ]) in
  let const_term = Float.(log (2. * pi) * of_int d) in
  Maths.(0.5 $* (const_term $+ error_term + cov_term)) |> Maths.neg

let gaussian_llh_chol ?mu ~precision_chol:ell x =
  let d = x |> Maths.primal |> Tensor.shape |> List.last_exn in
  let error =
    match mu with
    | None -> x
    | Some mu -> Maths.(x - mu)
  in
  let error_term = Maths.einsum [ error, "ma"; ell, "ai"; ell, "bi"; error, "mb" ] "m" in
  let cov_term =
    Maths.(neg (sum (log (sqr (diagonal ~offset:0 ell)))) |> reshape ~shape:[ 1 ])
  in
  let const_term = Float.(log (2. * pi) * of_int d) in
  Maths.(-0.5 $* (const_term $+ error_term + cov_term))

let make_a_prms ~n target_sa =
  let w =
    let w = Mat.(gaussian n n) in
    let sa =
      let eigvals = Owl.Linalg.S.eigvals w |> Owl.Dense.Matrix.Generic.cast_c2z in
      Owl.Dense.Matrix.Z.re eigvals |> Owl.Dense.Matrix.D.max'
    in
    Mat.(Float.(target_sa / sa) $* w)
  in
  Owl.Linalg.S.expm Mat.((w - eye n) *$ 0.1)

let precision_of_log_var log_var = Maths.(exp (neg log_var))
let sqrt_precision_of_log_var log_var = Maths.(exp (f (-0.5) * log_var))
let std_of_log_var log_var = Maths.(exp (f 0.5 * log_var))
let detach x = x |> Maths.primal_tensor_detach |> Maths.const

(* -----------------------------------------
   -- Data Read In ---
   ----------------------------------------- *)
let load_npy_data hvs_folder =
  let folder_path = "_data/hvs_npy/" ^ hvs_folder in
  (* get array of .npy files, each contains a mat of shape [T x n_channels] *)
  let files = Stdlib.Sys.readdir folder_path |> List.of_array in
  let npy_files = List.filter ~f:(fun f -> Stdlib.Filename.check_suffix f ".npy") files in
  let full_paths =
    List.map ~f:(fun filename -> Stdlib.Filename.concat folder_path filename) npy_files
  in
  (* load data *)
  List.map ~f:(fun path -> Arr.load_npy path) full_paths

(* array of [T x n_channels] files. *)
let data_raw = load_npy_data data_folder

let standardize (mat : Mat.mat) : Mat.mat =
  let mean_ = Mat.mean ~axis:0 mat in
  let std_ = Mat.std ~axis:0 mat in
  Mat.(div (mat - mean_) std_)

let chunking ~tmax mat =
  let shape = Arr.shape mat in
  let t = shape.(0) in
  let num_blocks = t / tmax in
  List.init num_blocks ~f:(fun i ->
    Arr.get_slice [ [ tmax * i; (tmax * (i + 1)) - 1 ]; [] ] mat)

let tmax = 500

(* array of [tmax x n_channels] files *)
let data =
  let chunk_lst =
    List.map data_raw ~f:(fun mat ->
      (* TODO: standardise per time series *)
      let mat = standardize mat in
      let tmax_mat = Mat.row_num mat in
      if tmax_mat < tmax then None else Some (chunking ~tmax mat))
  in
  List.concat
    (List.filter_map
       ~f:(function
         | Some lst -> Some lst
         | None -> None)
       chunk_lst)
  |> List.permute

let data_train, data_test =
  let full_batch_size = List.length data in
  List.split_n data Float.(to_int (of_int full_batch_size * 0.8))

let data_train_size = List.length data_train

(* -----------------------------------------
   -- Problem Definition ---
   ----------------------------------------- *)

(* state dim *)
let n = Option.value (Cmdargs.get_int "-n") ~default:8

(* control dim *)
let m = 3

(* observation dim *)
let o = 16 * 16

let _b_0_true =
  Tensor.(
    f Float.(1. /. sqrt (of_int n)) * randn ~device:base.device ~kind:base.kind [ n; n ])
  |> Maths.const

let _b_true =
  Tensor.(
    f Float.(1. /. sqrt (of_int m)) * randn ~device:base.device ~kind:base.kind [ m; n ])
  |> Maths.const

let to_device = Tensor.of_bigarray ~device:base.device

let sample_data ~sampling_state_data:_ bs =
  (* list (length bs) of tensor of shape [tmax x n_channels] to list (length tmax) of shape [bs x n_channels] *)
  let to_list bs_of_data =
    List.init tmax ~f:(fun i ->
      List.map bs_of_data ~f:(fun data ->
        Tensor.slice data ~dim:0 ~start:(Some i) ~end_:(Some (i + 1)) ~step:1)
      |> Tensor.concat ~dim:0)
  in
  if bs > 0
  then (
    (* cycle through the data points *)
    (* let indices =
    List.init bs ~f:(fun i -> ((sampling_state_data * bs) + i) % full_batch_size)
  in *)
    let indices =
      List.permute (List.range 0 data_train_size) |> List.sub ~pos:0 ~len:bs
    in
    let bs_of_data =
      List.map indices ~f:(fun idx -> List.nth_exn data idx |> to_device)
    in
    to_list bs_of_data)
  else (
    let bs_of_data = List.map data_test ~f:to_device in
    to_list bs_of_data)

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)
(* each trial has the same parameters. *)
let batch_const = true

module PP = struct
  type 'a p =
    { _a : 'a
    ; _b : 'a
    ; _b_0 : 'a (* b at time step 0 has same dimension as state *)
    ; _c : 'a
    ; _d : 'a (* o ~ N(cx + d, obs_var *)
    ; _log_prior_var_0 : 'a (* log of cov over input at step 0 *)
    ; _log_prior_var : 'a (* log of cov over input *)
    ; _log_obs_var : 'a (* log of covariance of emission noise *)
    ; _log_scaling_factor : 'a
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

module LGS = struct
  module P = P

  type args = unit
  type data = Tensor.t list

  (* create params for lds from f *)
  let params_from_f ~(theta : P.M.t) ~x0 ~o_list
    : (Maths.t option, (Maths.t, Maths.t option) Lds_data.Temp.p list) Lqr.Params.p
    =
    (* set o at time 0 as 0. o list goes from t=0 to T *)
    let o_list_tmp = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
    let _obs_precision = precision_of_log_var theta._log_obs_var in
    let _Cxx =
      Maths.(einsum [ theta._c, "ab"; _obs_precision, "b"; theta._c, "cb" ] "ac")
    in
    let _prior_precision =
      precision_of_log_var theta._log_prior_var
      |> Maths.diag_embed ~offset:0 ~dim1:0 ~dim2:1
    in
    let _prior_precision_0 =
      precision_of_log_var theta._log_prior_var_0
      |> Maths.diag_embed ~offset:0 ~dim1:0 ~dim2:1
    in
    Lqr.Params.
      { x0 = Some x0
      ; params =
          List.mapi o_list_tmp ~f:(fun i o ->
            let _cx =
              Maths.(
                einsum [ theta._d, "mb"; _obs_precision, "b"; theta._c, "cb" ] "mc"
                - einsum [ const o, "mb"; _obs_precision, "b"; theta._c, "cb" ] "mc")
            in
            Lds_data.Temp.
              { _f = None
              ; _Fx_prod = theta._a
              ; _Fu_prod = (if i = 0 then _b_0_true else _b_true)
              ; _cx = Some _cx
              ; _cu = None
              ; _Cxx
              ; _Cxu = None
              ; _Cuu = (if i = 0 then _prior_precision_0 else _prior_precision)
              })
      }

  (* rollout y (clean observation) list under sampled u (u_0, ..., u_{T-1}) *)
  let rollout ~z0 ~u_list (theta : P.M.t) =
    let tmp_einsum a b = Maths.einsum [ a, "ma"; b, "ab" ] "mb" in
    let _Fx_prod = theta._a in
    let _, y_list_rev =
      List.foldi
        u_list
        ~init:(Maths.const z0, [])
        ~f:(fun i (z, y_list) u ->
          let new_z =
            Maths.(
              tmp_einsum z _Fx_prod + tmp_einsum u (if i = 0 then _b_0_true else _b_true))
          in
          let new_y = Maths.(tmp_einsum new_z theta._c + theta._d) in
          new_z, new_y :: y_list)
    in
    List.rev y_list_rev

  (* approximate kalman filtered distribution of u *)
  (* TODO: check this function after going through the derivation again *)
  let sample_and_kl
        ~z0
        ~_Fx
        ~_Fu
        ~_Fu_0
        ~_c
        ~_d
        ~_log_prior_var_0
        ~_log_prior_var
        ~obs_precision
        ~scaling_factor
        ustars
        o_list
    =
    let open Maths in
    let btrinv_0 = einsum [ _Fu_0, "ij"; _c, "jo"; obs_precision, "o" ] "io" in
    let btrinv = einsum [ _Fu, "ij"; _c, "jo"; obs_precision, "o" ] "io" in
    (* posterior precision of filtered covariance of u *)
    let precision_chol_0 =
      let _prior_precision_0 =
        precision_of_log_var _log_prior_var_0
        |> Maths.diag_embed ~offset:0 ~dim1:0 ~dim2:1
      in
      let tmp =
        scaling_factor * const (Tensor.ones ~device:base.device ~kind:base.kind [ n ])
        |> unsqueeze ~dim:0
      in
      (_prior_precision_0 + einsum [ btrinv_0, "io"; _c, "jo"; _Fu_0, "kj" ] "ik"
       |> cholesky)
      * tmp
    in
    let precision_chol =
      let _prior_precision =
        precision_of_log_var _log_prior_var |> Maths.diag_embed ~offset:0 ~dim1:0 ~dim2:1
      in
      let tmp =
        scaling_factor * const (Tensor.ones ~device:base.device ~kind:base.kind [ m ])
        |> unsqueeze ~dim:0
      in
      (_prior_precision + einsum [ btrinv, "io"; _c, "jo"; _Fu, "kj" ] "ik" |> cholesky)
      * tmp
    in
    (* ustars go from 0 to T-1 and o goes from 1 to T *)
    let ustars_o_list = List.map2_exn ustars o_list ~f:(fun u o -> u, const o) in
    let _, kl, us =
      List.foldi
        ustars_o_list
        ~init:(const z0, f 0., [])
        ~f:(fun i (z, kl, us) (ustar, o) ->
          Stdlib.Gc.major ();
          let _Fu = if i = 0 then _Fu_0 else _Fu in
          let precision_chol = if i = 0 then precision_chol_0 else precision_chol in
          let btrinv = if i = 0 then btrinv_0 else btrinv in
          let zpred = (z *@ _Fx) + (ustar *@ _Fu) in
          let ypred = (zpred *@ _c) + _d in
          let delta = o - ypred in
          (* posterior mean of filtered u *)
          let mu =
            let tmp = einsum [ btrinv, "io"; delta, "mo" ] "mi" in
            solver_chol precision_chol tmp
          in
          (* sample from posterior filtered covariance of u. *)
          let u_diff_elbo =
            Maths.linsolve_triangular
              ~left:false
              ~upper:false
              precision_chol
              (const
                 (Tensor.randn
                    ~device:base.device
                    ~kind:base.kind
                    [ (List.hd_exn (Tensor.shape z0)); (if i = 0 then n else m) ]))
          in
          let u_sample = mu + u_diff_elbo in
          (* u_final used to propagate dynamics *)
          let u_final = ustar + u_sample in
          (* propagate that sample to update z *)
          let z = zpred + (u_sample *@ _Fu) in
          (* update the KL divergence *)
          let kl =
            let prior_term =
              let inv_std =
                sqrt_precision_of_log_var
                  (if i = 0 then _log_prior_var_0 else _log_prior_var)
              in
              gaussian_llh ~inv_std u_final
            in
            (* sticking the landing idea where gradients w.r.t variational parameters removed in entropy term. *)
            let q_term =
              gaussian_llh_chol
                ~mu:(detach mu)
                ~precision_chol:(detach precision_chol)
                u_sample
            in
            kl + q_term - prior_term
          in
          z, kl, u_final :: us)
    in
    kl, List.rev us

  let elbo ~data:(o_list : Tensor.t list) (theta : P.M.t) =
    let obs_precision =(precision_of_log_var theta._log_obs_var) in
    (* use lqr to obtain the optimal u *)
    let z0 =
      let bs = List.hd_exn (Tensor.shape (List.hd_exn o_list)) in
      Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n ]
    in
    let p =
      params_from_f ~x0:(Maths.const z0) ~theta ~o_list |> Lds_data.map_naive ~batch_const
    in
    let sol, _ = Lqr._solve ~batch_const p in
    let ustars = List.map sol ~f:(fun s -> s.u) in
    let scaling_factor = Maths.exp theta._log_scaling_factor in
    let kl, u_sampled =
      sample_and_kl
        ~z0
        ~_Fx:theta._a
        ~_Fu:_b_true
        ~_Fu_0:_b_0_true
        ~_c:theta._c
        ~_d:theta._d
        ~_log_prior_var_0:theta._log_prior_var_0
        ~_log_prior_var:theta._log_prior_var
        ~obs_precision
        ~scaling_factor
        ustars
        o_list
    in
    let y_pred = rollout ~z0 ~u_list:u_sampled theta in
    let lik_term =
      let inv_std_o = (sqrt_precision_of_log_var theta._log_obs_var) in
      List.fold2_exn
        o_list
        y_pred
        ~init:Maths.(f 0.)
        ~f:(fun accu o y_pred ->
          Stdlib.Gc.major ();
          Maths.(accu + gaussian_llh ~mu:y_pred ~inv_std:inv_std_o (Maths.const o)))
    in
    Maths.(neg (lik_term - kl) / f Float.(of_int tmax * of_int o)), y_pred

  let ggn ~y_pred (theta : P.M.t) =
    let obs_precision =
      precision_of_log_var theta._log_obs_var |> Maths.primal |> Maths.const
    in
    let obs_var_t =
      Maths.(tangent (exp theta._log_obs_var)) |> Option.value_exn |> Maths.const
    in
    List.fold y_pred ~init:(Maths.f 0.) ~f:(fun accu y_pred ->
      let mu_t = Maths.tangent y_pred |> Option.value_exn |> Maths.const in
      let ggn_part1 =
        Maths.(einsum [ mu_t, "kmo"; obs_precision, "o"; mu_t, "lmo" ] "kl")
      in
      (* CHECKED this agrees with mine *)
      let ggn_part2 =
        Maths.(
          einsum
            [ f Float.(0.5 * of_int bs) * obs_var_t, "ky"
            ; sqr obs_precision, "y"
            ; obs_var_t, "ly"
            ]
            "kl")
      in
      Maths.(
        accu + const (primal ((ggn_part1 + ggn_part2) / f Float.(of_int o * of_int tmax)))))
    |> Maths.primal

  let f ~update ~data ~init ~args:() (theta : P.M.t) =
    let neg_elbo, y_pred = elbo ~data theta in
    match update with
    | `loss_only u -> u init (Some neg_elbo)
    | `loss_and_ggn u ->
      let ggn = ggn ~y_pred theta in
      let _ =
        let _, s, _ = Owl.Linalg.S.svd (Tensor.to_bigarray ~kind:base.ba_kind ggn) in
        Mat.(save_txt ~out:(in_dir "svals") (transpose s))
      in
      u init (Some (neg_elbo, Some ggn))

  let init : P.tagged =
    let _b = Prms.const (Maths.primal _b_true) in
    let _b_0 = Prms.const (Maths.primal _b_0_true) in
    let _a = make_a_prms ~n 0.8 |> Tensor.of_bigarray ~device:base.device |> Prms.free in
    let _c =
      Prms.free
        Tensor.(
          f Float.(1. /. sqrt (of_int n))
          * randn ~device:base.device ~kind:base.kind [ n; o ])
    in
    let _d = Prms.free Tensor.(zeros ~device:base.device ~kind:base.kind [ 1; o ]) in
    let _log_prior_var_0 =
      Tensor.(log (f Float.(square 1.) * ones ~device:base.device ~kind:base.kind [ n ]))
      |> Prms.free
    in
    let _log_prior_var =
      Tensor.(log (f Float.(square 1.) * ones ~device:base.device ~kind:base.kind [ m ]))
      |> Prms.free
    in
    let _log_obs_var =
      Tensor.(log (f Float.(square 1.) * ones ~device:base.device ~kind:base.kind [ o ]))
      |> Prms.free
    in
    let _log_scaling_factor =
      Prms.create
        ~above:(Tensor.f 0.1)
        Tensor.(log (ones [ 1 ] ~device:base.device ~kind:base.kind))
    in
    { _a
    ; _b
    ; _b_0
    ; _c
    ; _d
    ; _log_prior_var_0
    ; _log_prior_var
    ; _log_obs_var
    ; _log_scaling_factor
    }

  let simulate ~bs_sim (theta : P.M.t) =
    let u_list =
      List.init tmax ~f:(fun i ->
        let u_shape, log_cov_u =
          if i = 0
          then [ bs_sim; n ], theta._log_prior_var_0
          else [ bs_sim; m ], theta._log_prior_var
        in
        let tmp =
          Tensor.randn ~device:base.device ~kind:base.kind u_shape |> Maths.const
        in
        Maths.(tmp * std_of_log_var log_cov_u))
    in
    let y = 
      let z0 = Tensor.zeros ~device:base.device ~kind:base.kind [ bs_sim; n ] in

      rollout ~z0 ~u_list theta in
    let o_list =
      List.map y ~f:(fun y ->
        let eps =
          Tensor.randn ~device:base.device ~kind:base.kind [ bs_sim; o ] |> Maths.const
        in
        Maths.(y + (eps * std_of_log_var theta._log_obs_var)))
    in
    o_list
end

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
type param_name =
  | A
  | C
  | D
  | Log_prior_var_0
  | Log_prior_var
  | Log_obs_var
  | Log_scaling_factor
[@@deriving compare]

let _K = 128

let param_names_list =
  [ A; C; D; Log_prior_var_0; Log_prior_var; Log_obs_var; Log_scaling_factor ]

let equal_param_name p1 p2 = compare_param_name p1 p2 = 0
let n_params_a = 60
let n_params_d = 10
let n_params_c = Int.(_K - 4 - n_params_a - n_params_d)
let n_params_log_prior_var_0 = 1
let n_params_log_prior_var = 1
let n_params_log_obs_var = 1
let n_params_log_scaling_factor = 1
let cycle = true

let n_params_list =
  [ n_params_a
  ; n_params_c
  ; n_params_d
  ; n_params_log_prior_var_0
  ; n_params_log_prior_var
  ; n_params_log_obs_var
  ; n_params_log_scaling_factor
  ]

module GGN : Wrapper.Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { a_left : 'a
      ; a_right : 'a
      ; c_left : 'a
      ; c_right : 'a
      ; d_left : 'a
      ; d_right : 'a
      ; log_prior_var_0_left : 'a
      ; log_prior_var_0_right : 'a
      ; log_prior_var_left : 'a
      ; log_prior_var_right : 'a
      ; log_obs_var_left : 'a
      ; log_obs_var_right : 'a
      ; log_scaling_factor_left : 'a
      ; log_scaling_factor_right : 'a
      }
    [@@deriving prms]
  end

  module P = P
  module A = Make (Prms.P)

  type sampling_state = int

  let init_sampling_state () = 0

  let zero_params ~shape _K =
    Tensor.zeros ~device:base.device ~kind:base.kind (_K :: shape)

  let random_params ~shape _K =
    Tensor.randn ~device:base.device ~kind:base.kind (_K :: shape)

  let get_shapes (param_name : param_name) =
    match param_name with
    | A -> [ n; n ]
    | C -> [ n; o ]
    | D -> [ 1; o ]
    | Log_prior_var_0 -> [ n ]
    | Log_prior_var -> [ m ]
    | Log_obs_var -> [ o ]
    | Log_scaling_factor -> [ 1 ]

  let get_n_params (param_name : param_name) =
    match param_name with
    | A -> n_params_a
    | C -> n_params_c
    | D -> n_params_d
    | Log_prior_var_0 -> n_params_log_prior_var_0
    | Log_prior_var -> n_params_log_prior_var
    | Log_obs_var -> n_params_log_obs_var
    | Log_scaling_factor -> n_params_log_scaling_factor

  let get_total_n_params (param_name : param_name) =
    let list_prod l = List.fold l ~init:1 ~f:(fun accu i -> accu * i) in
    list_prod (get_shapes param_name)

  let get_n_params_before_after (param_name : param_name) =
    let n_params_prefix_suffix_sums = Convenience.prefix_suffix_sums n_params_list in
    let param_idx =
      match param_name with
      | A -> 0
      | C -> 1
      | D -> 2
      | Log_prior_var_0 -> 3
      | Log_prior_var -> 4
      | Log_obs_var -> 5
      | Log_scaling_factor -> 6
    in
    List.nth_exn n_params_prefix_suffix_sums param_idx

  (* approximation defined implicitly via Gv products *)
  let g12v ~(lambda : A.M.t) (v : P.M.t) : P.M.t =
    let open Maths in
    let _a = einsum [ lambda.a_left, "in"; v._a, "aij"; lambda.a_right, "jm" ] "anm" in
    (* TODO: is there a more ergonomic way to deal with constant parameters? *)
    let _b =
      Tensor.zeros [ _K; m; n ] ~device:base.device ~kind:base.kind |> Maths.const
    in
    let _b_0 =
      Tensor.zeros [ _K; n; n ] ~device:base.device ~kind:base.kind |> Maths.const
    in
    let _c = einsum [ lambda.c_left, "in"; v._c, "aij"; lambda.c_right, "jm" ] "anm" in
    let _d = einsum [ lambda.d_left, "in"; v._d, "aij"; lambda.d_right, "jm" ] "anm" in
    let _log_prior_var_0 =
      einsum
        [ lambda.log_prior_var_0_left, "in"
        ; reshape v._log_prior_var_0 ~shape:[ -1; 1; n ], "aij"
        ; lambda.log_prior_var_0_right, "jm"
        ]
        "anm"
      |> reshape ~shape:[ -1; n ]
    in
    let _log_prior_var =
      einsum
        [ lambda.log_prior_var_left, "in"
        ; reshape v._log_prior_var ~shape:[ -1; 1; m ], "aij"
        ; lambda.log_prior_var_right, "jm"
        ]
        "anm"
      |> reshape ~shape:[ -1; m ]
    in
    let _log_obs_var =
      einsum
        [ lambda.log_obs_var_left, "in"
        ; reshape v._log_obs_var ~shape:[ -1; 1; o ], "aij"
        ; lambda.log_obs_var_right, "jm"
        ]
        "anm"
      |> reshape ~shape:[ -1; o ]
    in
    let _log_scaling_factor =
      einsum
        [ lambda.log_scaling_factor_left, "in"
        ; reshape v._log_scaling_factor ~shape:[ -1; 1; 1 ], "aij"
        ; lambda.log_scaling_factor_right, "jm"
        ]
        "anm"
      |> reshape ~shape:[ -1; 1 ]
    in
    { _a
    ; _b
    ; _b_0
    ; _c
    ; _d
    ; _log_prior_var_0
    ; _log_prior_var
    ; _log_obs_var
    ; _log_scaling_factor
    }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~param_name ~n_per_param v =
    let _a = zero_params ~shape:(get_shapes A) n_per_param in
    let _c = zero_params ~shape:(get_shapes C) n_per_param in
    let _d = zero_params ~shape:(get_shapes D) n_per_param in
    let _log_prior_var_0 = zero_params ~shape:(get_shapes Log_prior_var_0) n_per_param in
    let _log_prior_var = zero_params ~shape:(get_shapes Log_prior_var) n_per_param in
    let _log_obs_var = zero_params ~shape:(get_shapes Log_obs_var) n_per_param in
    let _log_scaling_factor =
      zero_params ~shape:(get_shapes Log_scaling_factor) n_per_param
    in
    let _b = zero_params ~shape:[ m; n ] n_per_param in
    let _b_0 = zero_params ~shape:[ n; n ] n_per_param in
    let params_tmp =
      PP.
        { _a
        ; _b
        ; _b_0
        ; _c
        ; _d
        ; _log_prior_var_0
        ; _log_prior_var
        ; _log_obs_var
        ; _log_scaling_factor
        }
    in
    match param_name with
    | A -> { params_tmp with _a = v }
    | C -> { params_tmp with _c = v }
    | D -> { params_tmp with _d = v }
    | Log_prior_var_0 -> { params_tmp with _log_prior_var_0 = v }
    | Log_prior_var -> { params_tmp with _log_prior_var = v }
    | Log_obs_var -> { params_tmp with _log_obs_var = v }
    | Log_scaling_factor -> { params_tmp with _log_scaling_factor = v }

  let random_localised_vs _K : P.T.t =
    let random_localised_param_name param_name =
      let w_shape = get_shapes param_name in
      let before, after = get_n_params_before_after param_name in
      let w = random_params ~shape:w_shape (get_n_params param_name) in
      let zeros_before = zero_params ~shape:w_shape before in
      let zeros_after = zero_params ~shape:w_shape after in
      let final = Tensor.concat [ zeros_before; w; zeros_after ] ~dim:0 in
      final
    in
    { _a = random_localised_param_name A
    ; _c = random_localised_param_name C
    ; _d = random_localised_param_name D
    ; _log_prior_var_0 = random_localised_param_name Log_prior_var_0
    ; _log_prior_var = random_localised_param_name Log_prior_var
    ; _log_obs_var = random_localised_param_name Log_obs_var
    ; _log_scaling_factor = random_localised_param_name Log_scaling_factor
    ; _b = zero_params ~shape:[ m; n ] _K
    ; _b_0 = zero_params ~shape:[ n; n ] _K
    }

  (* compute sorted eigenvalues, u_left and u_right. *)
  let eigenvectors_for_params ~lambda ~param_name =
    let left, right =
      match param_name with
      | A -> lambda.a_left, lambda.a_right
      | C -> lambda.c_left, lambda.c_right
      | D -> lambda.d_left, lambda.d_right
      | Log_prior_var_0 -> lambda.log_prior_var_0_left, lambda.log_prior_var_0_right
      | Log_prior_var -> lambda.log_prior_var_left, lambda.log_prior_var_right
      | Log_obs_var -> lambda.log_obs_var_left, lambda.log_obs_var_right
      | Log_scaling_factor ->
        lambda.log_scaling_factor_left, lambda.log_scaling_factor_right
    in
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
    s_all, u_left, u_right

  (* cache storage with a ref to memoize computed results *)
  let s_u_cache = ref []

  (* given param name, get eigenvalues and eigenvectors. *)
  let get_s_u ~lambda ~param_name =
    match List.Assoc.find !s_u_cache param_name ~equal:equal_param_name with
    | Some s -> s
    | None ->
      let s = eigenvectors_for_params ~lambda ~param_name in
      s_u_cache := (param_name, s) :: !s_u_cache;
      s

  let extract_local_vs ~s_all ~param_name ~u_left ~u_right ~selection =
    let n_per_param = get_n_params param_name in
    let local_vs =
      List.map selection ~f:(fun idx ->
        let il, ir, _ = s_all.(idx) in
        let slice_and_squeeze t dim idx =
          Tensor.squeeze_dim
            ~dim
            (Tensor.slice t ~dim ~start:(Some idx) ~end_:(Some (idx + 1)) ~step:1)
        in
        let u_l = slice_and_squeeze u_left 1 il in
        let u_r = slice_and_squeeze u_right 1 ir in
        let tmp =
          match param_name with
          | Log_scaling_factor | Log_obs_var | Log_prior_var_0 | Log_prior_var ->
            Tensor.(u_l * u_r)
          | _ -> Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_l; u_r ]
        in
        Tensor.unsqueeze tmp ~dim:0)
      |> Tensor.concatenate ~dim:0
    in
    localise ~param_name ~n_per_param local_vs

  let eigenvectors_for_each_param ~lambda ~param_name ~sampling_state =
    let n_per_param = get_n_params param_name in
    let n_params = get_total_n_params param_name in
    let s_all, u_left, u_right = get_s_u ~lambda ~param_name in
    let selection =
      if cycle
      then
        List.init n_per_param ~f:(fun i ->
          ((sampling_state * n_per_param) + i) % n_params)
      else List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    extract_local_vs ~s_all ~param_name ~u_left ~u_right ~selection

  let eigenvectors ~(lambda : A.M.t) ~switch_to_learn t (_K : int) =
    let eigenvectors_each =
      List.map param_names_list ~f:(fun param_name ->
        eigenvectors_for_each_param ~lambda ~param_name ~sampling_state:t)
    in
    let vs =
      List.fold eigenvectors_each ~init:None ~f:(fun accu local_vs ->
        match accu with
        | None -> Some local_vs
        | Some a -> Some (P.map2 a local_vs ~f:(fun x y -> Tensor.concat ~dim:0 [ x; y ])))
    in
    (* reset s_u_cache and set sampling state to 0 if learn again *)
    if switch_to_learn then s_u_cache := [];
    let new_sampling_state = if switch_to_learn then 0 else t + 1 in
    Option.value_exn vs, new_sampling_state

  let init () =
    let init_eye size =
      Mat.(0.1 $* eye size) |> Tensor.of_bigarray ~device:base.device |> Prms.free
    in
    { a_left = init_eye n
    ; a_right = init_eye n
    ; c_left = init_eye n
    ; c_right = init_eye o
    ; d_left = init_eye 1
    ; d_right = init_eye o
    ; log_prior_var_0_left = init_eye 1
    ; log_prior_var_0_right = init_eye n
    ; log_prior_var_left = init_eye 1
    ; log_prior_var_right = init_eye m
    ; log_obs_var_left = init_eye 1
    ; log_obs_var_right = init_eye o
    ; log_scaling_factor_left = init_eye 1
    ; log_scaling_factor_right = init_eye 1
    }
end

(* --------------------------------
   -- Generic type of optimiser
   -------------------------------- *)

module type Do_with_T = sig
  module O :
    Optimizer.T
    with type 'a W.P.p = 'a LGS.P.p
     and type W.data = Tensor.t list
     and type W.args = unit

  val name : string
  val config : iter:int -> (float, Bigarray.float32_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state ~time_elapsed running_avg =
      Stdlib.Gc.major ();
      let data =
        let o_list = sample_data ~sampling_state_data:iter bs in
        o_list
      in
      let t0 = Unix.gettimeofday () in
      let loss, new_state = O.step ~config:(config ~iter) ~state ~data () in
      let t1 = Unix.gettimeofday () in
      let time_elapsed = Float.(time_elapsed + t1 - t0) in
      (* let n_params =
        let params = LGS.P.value (O.params state) in
        O.W.P.T.numel params
      in *)
      let running_avg =
        let loss_avg =
          match running_avg with
          | [] -> loss
          | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
        in
        (* save params *)
        if iter % 10 = 0
        then (
          O.W.P.T.save
            (LGS.P.value (O.params new_state))
            ~kind:base.ba_kind
            ~out:(in_dir name ^ "_params");
          (* avg error *)
          Convenience.print [%message (iter : int) (loss_avg : float)];
          (* test elbo *)
          let test_loss =
            let data_test = sample_data ~sampling_state_data:iter (-1) in
            let neg_elbo, _ =
              LGS.elbo
                ~data:data_test
                (LGS.P.map ~f:Maths.const (LGS.P.value (O.params state)))
            in
            neg_elbo |> Maths.mean|> Maths.primal |> Tensor.to_float0_exn
          in
          let t = iter in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array [| Float.of_int t; time_elapsed; loss_avg; test_loss |] 1 4)));
        []
      in
      if iter < max_iter
      then loop ~iter:(iter + 1) ~state:new_state ~time_elapsed (loss :: running_avg)
    in
    loop ~iter:0 ~state:init ~time_elapsed:0. []
end

(* --------------------------------
     -- SOFO
     -------------------------------- *)

module Do_with_SOFO : Do_with_T = struct
  module O = Optimizer.SOFO (LGS) (GGN)

  let name = "sofo_n_" ^ Int.to_string n ^ "_m_" ^ Int.to_string m

  let config ~iter:_ =
    let aux =
      Optimizer.Config.SOFO.
        { (default_aux (in_dir "aux")) with
          config =
            Optimizer.Config.Adam.
              { default with base; learning_rate = Some 1e-2; eps = 1e-8 }
        ; steps = 50
        ; learn_steps = 100
        ; exploit_steps = 100
        }
    in
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some 0.05
      ; n_tangents = _K
      ; rank_one = false
      ; damping = Some 1e-3
      ; aux = None
      ; orthogonalize = false
      }

  let init = O.init LGS.init
end

(* --------------------------------
     -- Adam
     --------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam_n_" ^ Int.to_string n ^ "_m_" ^ Int.to_string m

  module O = Optimizer.Adam (LGS)

  let config ~iter:t =
    Optimizer.Config.Adam.
      { default with
        base
      ; learning_rate = Some Float.(0.001 / sqrt ((of_int t + 1.) / 50.))
      }

  let init = O.init LGS.init
end

let _ =
  let max_iter = 100000 in
  let optimise =
    match Cmdargs.get_string "-m" with
    | Some "sofo" ->
      let module X = Make (Do_with_SOFO) in
      X.optimise
    | Some "adam" ->
      let module X = Make (Do_with_Adam) in
      X.optimise
    | _ -> failwith "-m [sofo | fgd | adam]"
  in
  optimise max_iter

(* let _ =
  let adam_params = LGS.P.T.load ~device:base.device (in_dir "adam_params") in
  LGS.P.T.save_npy adam_params ~kind:base.ba_kind ~out:(in_dir "adam_params") *)
