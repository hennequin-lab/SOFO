(* Linear Gaussian Dynamics, with same state/control/cost parameters constant across trials and across time *)
open Printf
open Base
open Forward_torch
open Torch
open Sofo
module Mat = Owl.Dense.Matrix.D
module Arr = Owl.Dense.Ndarray.D
module Linalg = Owl.Linalg.D
module Z = Owl.Dense.Matrix.Z

let _ =
  Random.init 1999;
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f64)
    ; ba_kind = Bigarray.float64
    }

(* -----------------------------------------
   ----- Utilitiy Functions ------
   ----------------------------------------- *)
let cleanup () = Stdlib.Gc.major ()
let in_dir = Cmdargs.in_dir "-d"
let tmp_einsum a b = Maths.einsum [ a, "ma"; b, "ab" ] "mb"

(* make sure fx is stable *)
let sample_stable ~a =
  let w =
    let tmp = Mat.gaussian a a in
    let r = tmp |> Linalg.eigvals |> Z.re |> Mat.max' in
    Mat.(Float.(0.8 / r) $* tmp)
  in
  let w_i = Mat.((w - eye a) *$ 0.1) in
  Owl.Linalg.Generic.expm w_i

let save_svals m =
  let _, s, _ = Tensor.svd ~some:true ~compute_uv:false m in
  s
  |> Tensor.reshape ~shape:[ -1; 1 ]
  |> Tensor.to_bigarray ~kind:base.ba_kind
  |> Owl.Mat.save_txt ~out:(in_dir (sprintf "svals"))

(* -----------------------------------------
   ----- LQR Functions ------
   ----------------------------------------- *)

(* everything has to be optional, because
   perhaps none of those input parameters will have tangents *)
type 'a momentary_params_common =
  { _A : 'a option
  ; _B : 'a option
  ; _Cxx : 'a option
  ; _Cxu : 'a option
  ; _Cuu : 'a option
  }

type 'a momentary_params =
  { common : 'a momentary_params_common
  ; _f : 'a option
  ; _cx : 'a option
  ; _cu : 'a option
  }

(* params starts at time idx 0 and ends at time index T. Note that at time index T only _Cxx and _cx is used *)
module Params = struct
  type ('a, 'p) p =
    { x0 : 'a
    ; params : 'p
    }
  [@@deriving prms]
end

module Solution = struct
  type 'a p =
    { u : 'a
    ; x : 'a
    }
  [@@deriving prms]
end

type backward_common_info =
  { _Quu_chol : Maths.t option
  ; _Quu_chol_T : Maths.t option
  ; _V : Maths.t option
  ; _K : Maths.t option
  }

type backward_info =
  { _K : Maths.t option
  ; _k : Maths.t option
  }

let maybe_btr = Option.map ~f:Maths.btr

let ( +? ) a b =
  match a, b with
  | None, None -> None
  | Some a, None -> Some a
  | None, Some b -> Some b
  | Some a, Some b -> Some Maths.(a + b)

let ( @? ) a b =
  match a, b with
  | Some a, Some b ->
    (match List.length (Maths.shape b) with
     | 3 -> Some (Maths.einsum [ a, "ab"; b, "mbc" ] "mac")
     | 2 -> Some (Maths.einsum [ a, "ab"; b, "mb" ] "ma")
     | _ -> failwith "not batch multipliable")
  | _ -> None

let maybe_unsqueeze ~dim a =
  match a with
  | None -> None
  | Some a -> Some (Maths.unsqueeze a ~dim)

let maybe_squeeze ~dim a =
  match a with
  | None -> None
  | Some a -> Some (Maths.squeeze a ~dim)

let maybe_einsum (a, opsA) (b, opsB) opsC =
  match a, b with
  | Some a, Some b -> Some (Maths.einsum [ a, opsA; b, opsB ] opsC)
  | _ -> None

let neg_inv_symm ~is_vector _b (ell, ell_T) =
  match ell, ell_T with
  | Some ell, Some ell_T ->
    let _b =
      if not is_vector
      then _b
      else Option.map _b ~f:(fun x -> Maths.reshape x ~shape:(Maths.shape x @ [ 1 ]))
    in
    let _y = Option.map _b ~f:(Maths.linsolve_triangular ~left:true ~upper:false ell) in
    Option.map _y ~f:(fun x ->
      Maths.linsolve_triangular ~left:true ~upper:true ell_T x |> Maths.neg)
  | _ -> None

(* backward recursion: all the (most expensive) common bits. common goes from 0 to T *)
let backward_common (common : Maths.t momentary_params_common list) =
  let _V =
    match List.last common with
    | None -> failwith "LQR needs a time horizon >= 1"
    | Some z -> z._Cxx
  in
  (* info goes from 0 to t-1 *)
  let _, info =
    let common_except_last = List.drop_last_exn common in
    List.fold_right common_except_last ~init:(_V, []) ~f:(fun z (_V, info_list) ->
      cleanup ();
      let _Quu, _Qxx, _Qux =
        let _V_unsqueezed = maybe_unsqueeze _V ~dim:0 in
        let tmp = z._A @? _V_unsqueezed in
        let _Quu =
          let tmp2 = z._B @? maybe_btr (z._B @? _V_unsqueezed) |> maybe_squeeze ~dim:0 in
          z._Cuu +? tmp2
        in
        let _Qxx =
          let tmp2 = z._A @? maybe_btr tmp |> maybe_squeeze ~dim:0 in
          z._Cxx +? tmp2
        in
        let _Qux =
          let tmp2 = z._B @? maybe_btr tmp |> maybe_squeeze ~dim:0 in
          maybe_btr z._Cxu +? tmp2
        in
        _Quu, _Qxx, _Qux
      in
      (* compute LQR gain parameters to be used in the subsequent fwd pass *)
      let _Quu_inv = Option.map _Quu ~f:Maths.inv_sqr in
      let _Quu_chol = Option.map _Quu ~f:Maths.cholesky in
      let _Quu_chol_T = maybe_btr _Quu_chol in
      let _K = neg_inv_symm ~is_vector:false _Qux (_Quu_chol, _Quu_chol_T) in
      (* update the value function *)
      let _V_new = _Qxx +? maybe_einsum (_Qux, "ab") (_K, "ac") "bc" in
      _V_new, { _Quu_chol; _Quu_chol_T; _V; _K } :: info_list)
  in
  info

let _k ~z ~_f _qu =
  match _qu with
  | None -> None
  | Some _qu ->
    let _qu_swapped = Some (Maths.permute ~dims:[ 1; 0 ] _qu) in
    let _k =
      neg_inv_symm ~is_vector:false _qu_swapped (z._Quu_chol, z._Quu_chol_T)
      |> Option.value_exn
      |> Maths.permute ~dims:[ 1; 0 ]
    in
    Some _k

let _qu_qx ~z ~_cu ~_cx ~_f ~_B ~_A _v =
  let tmp = _v +? maybe_einsum (z._V, "ab") (_f, "mb") "ma" in
  _cu +? (_B @? tmp), _cx +? (_A @? tmp)

let v ~_K ~_qx ~_qu = _qx +? maybe_einsum (_K, "ba") (_qu, "mb") "ma"

(* backward recursion; assumes all the common (expensive) stuff has already been
   computed *)
let backward
      common_info
      (params : (Maths.t option, Maths.t momentary_params list) Params.p)
  =
  let _v =
    match List.last params.params with
    | None -> failwith "LQR needs a time horizon >= 1"
    | Some z -> z._cx
  in
  (* info (k/K) goes from 0 to T-1 *)
  let _, info =
    let params_except_last = List.drop_last_exn params.params in
    List.fold2_exn
      (List.rev common_info)
      (List.rev params_except_last)
      ~init:(_v, [])
      ~f:(fun (_v, info_list) z p ->
        cleanup ();
        let _qu, _qx =
          _qu_qx ~z ~_cu:p._cu ~_cx:p._cx ~_f:p._f ~_B:p.common._B ~_A:p.common._A _v
        in
        (* compute LQR gain parameters to be used in the subsequent fwd pass *)
        let _k = _k ~z ~_f:p._f _qu in
        (* update the value function *)
        let _v = v ~_K:z._K ~_qx ~_qu in
        _v, { _k; _K = z._K } :: info_list)
  in
  info

let _u ~_k ~_K ~x = _k +? maybe_einsum (x, "ma") (_K, "ba") "mb"

let _x ~_f ~_A ~_B ~x ~u =
  _f +? maybe_einsum (x, "ma") (_A, "ab") "mb" +? maybe_einsum (u, "ma") (_B, "ab") "mb"

let forward params (backward_info : backward_info list) =
  let open Params in
  (* u goes from 0 to T-1, x goes from 1 to T *)
  let _, solution =
    let params_except_last = List.drop_last_exn params.params in
    List.fold2_exn
      params_except_last
      backward_info
      ~init:(params.x0, [])
      ~f:(fun (x, accu) p b ->
        cleanup ();
        (* compute the optimal control inputs *)
        let u = _u ~_k:b._k ~_K:b._K ~x in
        (* fold them into the state update *)
        let x = _x ~_f:p._f ~_A:p.common._A ~_B:p.common._B ~x ~u in
        x, Solution.{ u; x } :: accu)
  in
  List.rev solution

let _solve p =
  let common_info = backward_common (List.map p.Params.params ~f:(fun x -> x.common)) in
  cleanup ();
  let bck = backward common_info p in
  cleanup ();
  let sol =
    bck
    |> forward p
    |> List.map ~f:(fun s ->
      Solution.{ x = Option.value_exn s.x; u = Option.value_exn s.u })
  in
  sol

module Temp = struct
  type ('a, 'o) p =
    { _f : 'o
    ; _A : 'a
    ; _B : 'a
    ; _cx : 'o
    ; _cu : 'o
    ; _Cxx : 'a
    ; _Cxu : 'o
    ; _Cuu : 'a
    }
  [@@deriving prms]
end

module O = Prms.Option (Prms.P)
module Input = Params.Make (O) (Prms.List (Temp.Make (Prms.P) (O)))

let map_naive (x : Input.M.t) =
  let params =
    List.map x.params ~f:(fun p ->
      { common =
          { _A = Some p._A
          ; _B = Some p._B
          ; _Cxx = Some p._Cxx
          ; _Cxu = p._Cxu
          ; _Cuu = Some p._Cuu
          }
      ; _f = p._f
      ; _cx = p._cx
      ; _cu = p._cu
      })
  in
  Params.{ x with params }

(* -----------------------------------------
   -- Control Problem / Data Generation ---
   ----------------------------------------- *)
let n = 24
let m = 10
let o = 40
let tmax = 10
let bs = 128

(* x_list goes from 1 to T and o list goes from 1 to T. *)
let traj_rollout ~x0 ~_A ~_B ~_c ~_std_o ~u_list =
  let tmp_einsum a b = Tensor.einsum ~equation:"ma,ab->mb" [ a; b ] ~path:None in
  let _size = List.hd_exn (Tensor.shape x0) in
  let _, x_list, o_list =
    List.fold u_list ~init:(x0, [], []) ~f:(fun (x, x_list, o_list) u ->
      let new_x = Tensor.(tmp_einsum x _A + tmp_einsum u _B) in
      let new_o =
        let noise =
          let eps = Tensor.randn ~device:base.device ~kind:base.kind [ _size; o ] in
          Tensor.einsum ~equation:"ma,a->ma" [ eps; _std_o ] ~path:None
        in
        Tensor.(noise + tmp_einsum new_x _c)
      in
      new_x, new_x :: x_list, new_o :: o_list)
  in
  List.rev x_list, List.rev o_list

(* in the linear gaussian case, _A, _B, c and cov invariant across time *)
let _std_o = Tensor.(f 1. * ones ~device:base.device ~kind:base.kind [ o ])
let ones_o = Tensor.(ones ~device:base.device ~kind:base.kind [ o ])
let _std_o_log = Tensor.(log _std_o)
let _A = sample_stable ~a:n |> Tensor.of_bigarray ~device:base.device
let _B = Tensor.randn ~device:base.device ~kind:base.kind [ m; n ]
let _c = Tensor.randn ~device:base.device ~kind:base.kind [ n; o ]

(** generalisation performance *)
let x0 = Tensor.zeros ~device:base.device ~kind:base.kind [ bs; n ]

(** data generation *)

(* test from an unseen set *)
let sample_test_data _size =
  let x0 = Tensor.zeros ~device:base.device ~kind:base.kind [ _size; n ] in
  let u_list =
    List.init tmax ~f:(fun _ ->
      Tensor.(randn ~device:base.device ~kind:base.kind [ _size; m ]))
  in
  let x_list, o_list = traj_rollout ~x0 ~_A ~_B ~_c ~_std_o ~u_list in
  u_list, x_list, o_list

(* -----------------------------------------
   -- Model setup and optimizer
   ----------------------------------------- *)

let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f64)
    ; ba_kind = Bigarray.float64
    }

let n_fisher = 30
let sample = false
let step = 0.1
let std_o_weight = 1.

module PP = struct
  (* note that all std live in log space *)
  type 'a p =
    { _S_params : 'a
    ; _L_params : 'a
      (* generative model; Fx = (1-step) I + step * W, W = I + (-I + S - S^T) LL^T *)
      (* _A_params : 'a *)
    ; _B_params : 'a
    ; _c_params : 'a
    ; _std_o_params : 'a (* sqrt of the diagonal of covariance of emission noise *)
    ; _std_space_params : 'a
      (* recognition model; sqrt of the diagonal of covariance of space factor *)
    ; _std_time_params : 'a (* sqrt of the diagonal of covariance of the time factor *)
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.P)

module LGS = struct
  module P = P

  type args = unit
  type data = Tensor.t list

  let _A (theta : P.M.t) =
    (* theta._A_params *)
    let eye_tmp = Maths.const (Tensor.eye ~n ~options:(base.kind, base.device)) in
    let _W =
      Maths.(
        eye_tmp
        + ((theta._S_params - transpose theta._S_params ~dim0:1 ~dim1:0 - eye_tmp)
           *@ (theta._L_params *@ transpose theta._L_params ~dim0:1 ~dim1:0)))
    in
    let tmp1 = Maths.(f (1. -. step) * eye_tmp) in
    let tmp2 = Maths.(f step * _W) in
    Maths.(tmp1 + tmp2)

  let step ~_A ~_B ~u ~prev_x = Maths.(tmp_einsum prev_x _A + tmp_einsum u _B)

  (* 1/ (x^2) *)
  let sqr_inv x = Maths.(1. $/ sqr x)

  (* list of length T of [m x b] to matrix of [m x b x T]*)
  let concat_time u_list =
    List.map u_list ~f:(fun u -> Maths.unsqueeze ~dim:(-1) u) |> Maths.concat_list ~dim:2

  let gaussian_llh ?mu ?(fisher_batched = false) ~std x =
    let inv_std = Maths.(f 1. / std) in
    let error_term =
      if fisher_batched
      then (
        (* dimension l is number of fisher samples *)
        let error =
          match mu with
          | None -> Maths.(einsum [ x, "lma"; inv_std, "a" ] "lma")
          | Some mu -> Maths.(einsum [ x - mu, "lma"; inv_std, "a" ] "lma")
        in
        Maths.einsum [ error, "lma"; error, "lma" ] "lm")
      else (
        let error =
          match mu with
          | None -> Maths.(einsum [ x, "ma"; inv_std, "a" ] "ma")
          | Some mu -> Maths.(einsum [ x - mu, "ma"; inv_std, "a" ] "ma")
        in
        Maths.einsum [ error, "ma"; error, "ma" ] "m")
    in
    let cov_term =
      let cov_term_shape = if fisher_batched then [ 1; 1 ] else [ 1 ] in
      Maths.(sum (log (sqr std))) |> Maths.reshape ~shape:cov_term_shape
    in
    let const_term =
      let o = x |> Maths.primal |> Tensor.shape |> List.last_exn in
      Float.(log (2. * pi) * of_int o)
    in
    Maths.(0.5 $* (const_term $+ error_term + cov_term)) |> Maths.neg

  (* special care to be taken when dealing with elbo loss *)
  module Elbo_loss = struct
    let fisher ?(fisher_batched = false) ~n:_ lik_term =
      let neg_lik_t = Maths.(tangent lik_term) |> Option.value_exn in
      let n_tangents = List.hd_exn (Tensor.shape neg_lik_t) in
      let fisher =
        if fisher_batched
        then (
          let fisher_half =
            Tensor.reshape neg_lik_t ~shape:[ n_tangents; n_fisher; -1 ]
          in
          Tensor.einsum ~equation:"kla,jla->lkj" [ fisher_half; fisher_half ] ~path:None)
        else (
          let fisher_half = Tensor.reshape neg_lik_t ~shape:[ n_tangents; -1 ] in
          Tensor.(matmul fisher_half (transpose fisher_half ~dim0:0 ~dim1:1)))
      in
      fisher

    let llh_increment ~_std_o_vec ~_std_o_extended ~new_o =
      let noise =
        Tensor.(
          _std_o_extended
          * randn (n_fisher :: Maths.shape new_o) ~device:base.device ~kind:base.kind)
      in
      let new_o_unsqueezed =
        List.init n_fisher ~f:(fun _ -> Maths.unsqueeze new_o ~dim:0)
        |> Maths.concat_list ~dim:0
      in
      let o_samples_batched = Maths.(const Tensor.(Maths.primal new_o + noise)) in
      gaussian_llh
        ~mu:new_o_unsqueezed
        ~std:_std_o_vec
        ~fisher_batched:true
        o_samples_batched

    let true_fisher ~u_list (theta : P.M.t) =
      let _std_o_vec = Maths.(exp theta._std_o_params * const ones_o) in
      let _std_o_extended =
        _std_o_vec |> Maths.primal |> Tensor.unsqueeze ~dim:0 |> Tensor.unsqueeze ~dim:0
      in
      let _A = _A theta in
      let _, llh_rollout =
        List.fold
          u_list
          ~init:(Maths.const x0, Maths.f 0.)
          ~f:(fun accu u ->
            let prev_x, llh_accu = accu in
            let new_x = step ~_A ~_B:theta._B_params ~u ~prev_x in
            let new_o = tmp_einsum new_x theta._c_params in
            let increment = llh_increment ~_std_o_vec ~_std_o_extended ~new_o in
            cleanup ();
            new_x, Maths.(increment + llh_accu))
      in
      let fisher_rollout =
        let fisher_batched = fisher ~n:o llh_rollout ~fisher_batched:true in
        Tensor.mean_dim fisher_batched ~dim:(Some [ 0 ]) ~keepdim:false ~dtype:base.kind
      in
      let final = Tensor.(fisher_rollout / f (Float.of_int tmax)) in
      save_svals final;
      final

    let ggn_increment ~new_o ~hess_y =
      let ggn_y_increment =
        let vtgt = Maths.tangent new_o |> Option.value_exn in
        let vtgt_hess =
          Tensor.einsum ~equation:"kma,a->kma" [ vtgt; hess_y ] ~path:None
        in
        Tensor.einsum ~equation:"kma,jma->kj" [ vtgt_hess; vtgt ] ~path:None
      in
      ggn_y_increment

    let ggn ~u_list (theta : P.M.t) =
      let _A = _A theta in
      let _std_o = Maths.exp theta._std_o_params in
      let _std_o_vec = Maths.(_std_o * const ones_o) in
      let precision = _std_o |> sqr_inv in
      let hess_y = _std_o_vec |> sqr_inv |> Maths.primal in
      let hess_sigma_o =
        Tensor.(
          f Float.(of_int Int.(bs * o) / 2.) * square (square (Maths.primal _std_o)))
      in
      let _std_o_extended =
        _std_o_vec |> Maths.primal |> Tensor.unsqueeze ~dim:0 |> Tensor.unsqueeze ~dim:0
      in
      let _, ggn_rollout =
        List.fold
          u_list
          ~init:(Maths.const x0, Tensor.f 0.)
          ~f:(fun accu u ->
            let prev_x, ggn_accu = accu in
            let new_x = Maths.(tmp_einsum prev_x _A + tmp_einsum u theta._B_params) in
            let new_o = tmp_einsum new_x theta._c_params in
            let increment = ggn_increment ~new_o ~hess_y in
            cleanup ();
            new_x, Tensor.(increment + ggn_accu))
      in
      let ggn_sigma =
        let vtgt = precision |> Maths.tangent |> Option.value_exn in
        let vtgt_h =
          Tensor.einsum ~equation:"ka,a->ka" [ vtgt; hess_sigma_o ] ~path:None
        in
        Tensor.einsum ~equation:"ka,ja->kj" [ vtgt_h; vtgt ] ~path:None
      in
      let final =
        Tensor.((ggn_rollout / f (Float.of_int tmax)) + (f std_o_weight * ggn_sigma))
      in
      save_svals final;
      final
  end

  (* create params for lds from f *)
  let params_from_f ~(theta : P.M.t) ~x0 ~o_list
    : (Maths.t option, (Maths.t, Maths.t option) Temp.p list) Params.p
    =
    (* set o at time 0 as 0 *)
    let o_list_tmp = Tensor.zeros_like (List.hd_exn o_list) :: o_list in
    let _cov_u_inv = Tensor.eye ~n:m ~options:(base.kind, base.device) |> Maths.const in
    let c_trans = Maths.transpose theta._c_params ~dim0:1 ~dim1:0 in
    let _std_o = Maths.exp theta._std_o_params in
    let _std_o_vec = Maths.(_std_o * const ones_o) in
    let _cov_o_inv = _std_o_vec |> sqr_inv in
    let _Cxx =
      let tmp = Maths.(einsum [ theta._c_params, "ab"; _cov_o_inv, "b" ] "ab") in
      Maths.(tmp *@ c_trans)
    in
    let _A = _A theta in
    Params.
      { x0 = Some x0
      ; params =
          List.map o_list_tmp ~f:(fun o ->
            let _cx =
              let tmp = Maths.(einsum [ const o, "ab"; _cov_o_inv, "b" ] "ab") in
              Maths.(neg (tmp *@ c_trans))
            in
            Temp.
              { _f = None
              ; _A
              ; _B = theta._B_params
              ; _cx = Some _cx
              ; _cu = None
              ; _Cxx
              ; _Cxu = None
              ; _Cuu = _cov_u_inv
              })
      }

  (* rollout x list under sampled u *)
  let rollout_x ~u_list ~x0 (theta : P.M.t) =
    let x0_tan = Maths.const x0 in
    let _A = _A theta in
    let _, x_list =
      List.fold u_list ~init:(x0_tan, [ x0_tan ]) ~f:(fun (x, x_list) u ->
        let new_x = step ~_A ~_B:theta._B_params ~u ~prev_x:x in
        new_x, new_x :: x_list)
    in
    List.rev x_list

  (* optimal u determined from lqr *)
  let pred_u ~data:o_list (theta : P.M.t) =
    (* use lqr to obtain the optimal u *)
    let p = params_from_f ~x0:(Maths.const x0) ~theta ~o_list |> map_naive  in
    let sol = _solve p in
    let optimal_u_list = List.map sol ~f:(fun s -> s.u) in
    (* sample u from the kronecker formation *)
    let u_list =
      let optimal_u = concat_time optimal_u_list in
      let xi =
        Tensor.randn ~device:base.device ~kind:base.kind [ bs; m; tmax ] |> Maths.const
      in
      let xi_space =
        Maths.einsum [ xi, "mbt"; Maths.exp theta._std_space_params, "b" ] "mbt"
      in
      let xi_time =
        Maths.einsum [ xi_space, "mat"; Maths.exp theta._std_time_params, "t" ] "mat"
      in
      let meaned = Maths.(xi_time + optimal_u) in
      List.init tmax ~f:(fun i ->
        Maths.slice ~dim:2 ~start:(Some i) ~end_:(Some (i + 1)) ~step:1 meaned
        |> Maths.reshape ~shape:[ bs; m ])
    in
    optimal_u_list, u_list

  (* u sampled from true posterior via Matheron sampling *)
  let pred_u_matheron ~data:o_list (theta : P.M.t) =
    (* use lqr to obtain the optimal u *)
    let optimal_u_list =
      let p = params_from_f ~x0:(Maths.const x0) ~theta ~o_list |> map_naive in
      let sol = _solve p in
      List.map sol ~f:(fun s -> s.u)
    in
    cleanup ();
    let sample_randn ~_mean ~_std ~dim =
      let eps =
        Tensor.randn ~device:base.device ~kind:base.kind [ bs; dim ] |> Maths.const
      in
      Maths.(einsum [ eps, "ma"; _std, "a" ] "ma" + _mean)
    in
    (* sample u from their priors N(0, I) *)
    let sampled_u =
      List.init tmax ~f:(fun _ ->
        sample_randn
          ~_mean:(Maths.const (Tensor.f 0.))
          ~_std:(Maths.const (Tensor.ones [ m ] ~device:base.device ~kind:base.kind))
          ~dim:m)
    in
    (* sample o with obsrevation noise *)
    let o_sampled =
      (* propagate prior sampled u through dynamics *)
      let x_rolled_out = rollout_x ~u_list:sampled_u ~x0 theta |> List.tl_exn in
      let std_o = theta._std_o_params |> Maths.exp in
      List.map x_rolled_out ~f:(fun x ->
        let _mean = Maths.(x *@ theta._c_params) in
        sample_randn ~_mean ~_std:std_o ~dim:o)
    in
    (* lqr on (o - o_sampled) *)
    let sol_delta_o =
      let delta_o_list =
        List.map2_exn o_list o_sampled ~f:(fun a b -> Tensor.(a - Maths.primal b))
      in
      let p =
        params_from_f ~x0:(Maths.const x0) ~theta ~o_list:delta_o_list |> map_naive
      in
      _solve p
    in
    cleanup ();
    (* final u samples *)
    let u_list =
      let optimal_u_list_delta_o = List.map sol_delta_o ~f:(fun s -> s.u) in
      List.map2_exn sampled_u optimal_u_list_delta_o ~f:(fun u delta_u ->
        Maths.(const (primal_tensor_detach (u + delta_u))))
    in
    optimal_u_list, u_list

  let llh ~o_list ~u_list ~optimal_u_list:_ (theta : P.M.t) =
    let u_o_list = List.map2_exn u_list o_list ~f:(fun u o -> u, o) in
    let _A = _A theta in
    let _std_o = Maths.exp theta._std_o_params in
    let _std_o_vec = Maths.(_std_o * const ones_o) in
    let _, llh =
      List.foldi
        u_o_list
        ~init:(Maths.const x0, None)
        ~f:(fun t accu (u, o) ->
          if t % 1 = 0 then cleanup ();
          let prev_x, llh_summed = accu in
          let new_x = step ~_A ~_B:theta._B_params ~u ~prev_x in
          let increment =
            gaussian_llh ~mu:o ~std:_std_o_vec (tmp_einsum new_x theta._c_params)
          in
          let new_llh_summed =
            match llh_summed with
            | None -> Some increment
            | Some accu -> Some Maths.(accu + increment)
          in
          cleanup ();
          new_x, new_llh_summed)
    in
    Option.value_exn llh

  let kl ~u_list ~optimal_u_list (theta : P.M.t) =
    let optimal_u = concat_time optimal_u_list in
    if sample
    then (
      let prior =
        List.foldi u_list ~init:None ~f:(fun t accu u ->
          if t % 1 = 0 then cleanup ();
          let increment =
            gaussian_llh
              ~std:(Tensor.ones [ m ] ~device:base.device ~kind:base.kind |> Maths.const)
              u
          in
          match accu with
          | None -> Some increment
          | Some accu -> Some Maths.(accu + increment))
        |> Option.value_exn
      in
      let neg_entropy =
        let u = concat_time u_list |> Maths.reshape ~shape:[ bs; -1 ] in
        let optimal_u = Maths.reshape optimal_u ~shape:[ bs; -1 ] in
        let std =
          Maths.(kron (exp theta._std_space_params) (exp theta._std_time_params))
        in
        gaussian_llh ~mu:optimal_u ~std u
      in
      Maths.(neg_entropy - prior))
    else (
      (* calculate the kl term analytically *)
      let std2 =
        Maths.(kron (exp theta._std_space_params) (exp theta._std_time_params))
      in
      let det1 = Maths.(2. $* sum (log std2)) in
      let _const = Float.of_int (m * tmax) in
      let tr =
        let tmp2 = Maths.(sqr (exp theta._std_space_params)) in
        let tmp3 = Maths.(kron tmp2 (sqr (exp theta._std_time_params))) in
        Maths.sum tmp3
      in
      let quad =
        Maths.einsum [ optimal_u, "mbt"; optimal_u, "mbt" ] "m" |> Maths.unsqueeze ~dim:1
      in
      let tmp = Maths.(tr - det1 -$ _const) |> Maths.reshape ~shape:[ 1; 1 ] in
      Maths.(0.5 $* tmp + quad) |> Maths.squeeze ~dim:1)

  let elbo ~o_list ~u_list ~optimal_u_list (theta : P.M.t) =
    let llh = llh ~o_list ~optimal_u_list ~u_list theta in
    let kl = kl ~u_list ~optimal_u_list theta in
    Maths.(llh - kl)

  let f ~update ~data ~init ~args:() (theta : P.M.t) =
    let module L = Elbo_loss in
    (* TODO: Matheron formulation *)
    let loss, u_list =
      let optimal_u_list, u_list = pred_u ~data theta in
      let loss_total =
        Maths.neg
          (elbo ~o_list:(List.map data ~f:Maths.const) ~u_list ~optimal_u_list theta)
      in
      Maths.(loss_total / f (Float.of_int tmax)), u_list
    in
    match update with
    | `loss_only u -> u init (Some loss)
    | `loss_and_ggn u ->
      let ggn = L.ggn ~u_list theta in
      u init (Some (loss, Some ggn))

  let init : P.tagged =
    let _S_params =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:n
        ~b:n
        ~sigma:0.1
      |> Prms.free
    in
    let _L_params =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:n
        ~b:n
        ~sigma:0.1
      |> Prms.free
    in
    let _B_params =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:m
        ~b:n
        ~sigma:0.1
      |> Prms.free
    in
    let _c_params =
      Convenience.gaussian_tensor_2d_normed
        ~device:base.device
        ~kind:base.kind
        ~a:n
        ~b:o
        ~sigma:1.
      |> Prms.free
    in
    let _std_o_params =
      Prms.create
        ~above:(Tensor.f (-5.))
        Tensor.(zeros ~device:base.device ~kind:base.kind [ 1 ])
      (* |> Prms.const *)
    in
    let _std_space_params =
      Prms.create
        Tensor.(zeros ~device:base.device ~kind:base.kind [ m ])
        ~above:(Tensor.f (-5.))
    in
    let _std_time_params =
      Prms.create
        Tensor.(zeros ~device:base.device ~kind:base.kind [ tmax ])
        ~above:(Tensor.f (-5.))
    in
    { _S_params
    ; _L_params (* _A_params *)
    ; _B_params
    ; _c_params
    ; _std_o_params
    ; _std_space_params
    ; _std_time_params
    }

  (* calculate the error between observations *)
  let simulate ~data ~(theta : P.M.t) =
    (* rollout under the given u *)
    let u_list, o_list = data in
    (* rollout to obtain x *)
    let rolled_out_x_list =
      rollout_x ~u_list:(List.map u_list ~f:Maths.const) ~x0 theta |> List.tl_exn
    in
    let rolled_out_o_list =
      List.map rolled_out_x_list ~f:(fun x -> tmp_einsum x theta._c_params)
    in
    let error =
      List.fold2_exn rolled_out_o_list o_list ~init:0. ~f:(fun accu x1 x2 ->
        let error = Tensor.(norm (Maths.primal x1 - x2)) |> Tensor.to_float0_exn in
        accu +. error)
    in
    Float.(error / of_int tmax)
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
  val config_f : iter:int -> (float, Bigarray.float64_elt) O.config
  val init : O.state
end

module Make (D : Do_with_T) = struct
  open D

  let optimise max_iter =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    let rec loop ~iter ~state ~time_elapsed running_avg =
      cleanup ();
      let _, _, o_list = sample_test_data bs in
      let t0 = Unix.gettimeofday () in
      let config = config_f ~iter in
      let loss, new_state = O.step ~config ~state ~data:o_list ~args:() in
      let std_o_mean =
        let a = LGS.P.value (O.params state) in
        a._std_o_params |> Tensor.mean |> Tensor.to_float0_exn
      in
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
          (* elbo on test set *)
          let _, _, test_o_list = sample_test_data bs in
          let elbo_test =
            let params = LGS.P.map ~f:Maths.const (LGS.P.value (O.params new_state)) in
            (* if matheron *)
            let optimal_u_list, u_list_test =
              LGS.pred_u_matheron ~data:test_o_list params
            in
            LGS.llh
              ~o_list:(List.map test_o_list ~f:Maths.const)
              ~u_list:u_list_test
              ~optimal_u_list
              params
            |> Maths.primal
            |> fun x ->
            Tensor.(x / f (Float.of_int tmax))
            |> Tensor.neg
            |> Tensor.mean
            |> Tensor.to_float0_exn
            (* if elbo *)
            (* let optimal_u_list_test, u_list_test = LGS.pred_u ~data:test_o_list params in
            LGS.elbo
              ~o_list:(List.map test_o_list ~f:Maths.const)
              ~u_list:u_list_test
              ~optimal_u_list:optimal_u_list_test
              params
            |> Maths.primal
            |> Tensor.neg
            |> Tensor.mean
            |> Tensor.to_float0_exn *)
          in
          (* simulation error *)
          let o_error =
            let u_list, _, o_list = sample_test_data bs in
            LGS.simulate
              ~theta:(LGS.P.const (LGS.P.value (O.params new_state)))
              ~data:(u_list, o_list)
          in
          (* avg error *)
          Convenience.print [%message (iter : int) (loss_avg : float) (elbo_test : float)];
          let t = iter in
          Owl.Mat.(
            save_txt
              ~append:true
              ~out:(in_dir name)
              (of_array
                 [| Float.of_int t
                  ; time_elapsed
                  ; loss_avg
                  ; o_error
                  ; std_o_mean
                  ; elbo_test
                 |]
                 1
                 6));
          O.W.P.T.save
            (LGS.P.value (O.params new_state))
            ~kind:base.ba_kind
            ~out:(in_dir name ^ "_params"));
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
  module O = Optimizer.SOFO (LGS)

  let config_f ~iter =
    Optimizer.Config.SOFO.
      { base
      ; learning_rate = Some Float.(0.1 / (1. +. (0. * sqrt (of_int iter))))
      ; n_tangents = 128
      ; sqrt = false
      ; rank_one = false
      ; damping = None
      ; momentum = None
      ; lm = false
      ; perturb_thresh = None
      }

  let name =
    let init_config = config_f ~iter:0 in
    let gamma_name =
      Option.value_map init_config.damping ~default:"none" ~f:Float.to_string
    in
    sprintf
      "ggn_lr_%s_sqrt_%s_damp_%s_std_o_weight_%s"
      (Float.to_string (Option.value_exn init_config.learning_rate))
      (Bool.to_string init_config.sqrt)
      gamma_name
      (Float.to_string std_o_weight)

  let init = O.init ~config:(config_f ~iter:0) LGS.init
end

(* --------------------------------
     -- Adam
     -------------------------------- *)

module Do_with_Adam : Do_with_T = struct
  let name = "adam"

  module O = Optimizer.Adam (LGS)

  let config_f ~iter:_ =
    Optimizer.Config.Adam.{ default with base; learning_rate = Some 0.001 }

  let init = O.init LGS.init
end

let _ =
  let max_iter = 10000 in
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
