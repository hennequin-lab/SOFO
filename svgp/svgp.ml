open Base
open Sofo
open Torch
open Forward_torch

module Variational_regression_parameters = struct
  type ('a, 'b) p =
    { kernel : 'a
    ; beta : 'b
    ; z : 'b
    ; nu : 'b (* parameter used to contruct m *)
    ; psi : 'b (* parameter used to construct S *)
    }
  [@@deriving prms]
end

module Variational_regression (K : sig
    module P : Prms.T

    val kernel
      :  theta:_ Maths.some P.t
      -> _ Maths.some P.elt
      -> _ Maths.some P.elt
      -> Maths.any P.elt

    val kernel_diag : theta:_ Maths.some P.t -> _ Maths.some P.elt -> Maths.any P.elt
  end) =
struct
  open Variational_regression_parameters
  module P = Variational_regression_parameters.Make (K.P) (Prms.Single)

  let fudge = 1E-6

  (* ---------------------------------- 
     ---  LEARNING
     ---------------------------------- *)

  (* bound on the marginal likelihood
     data assumed to be [n x something]
     this also returns the quantities needed to update the
     variational parameters *)
  let negative_bound ~(theta : _ Maths.some P.t) ~n_total (data_x, data_y) =
    let n = List.hd_exn (Maths.shape data_x) in
    assert (
      let[@warning "-8"] [ n'; d' ] = Maths.shape data_y in
      d' = 1 && n = n');
    (* likelihood term needs to be rescaled by this factor *)
    let lik_rescaling = Float.(of_int n_total / of_int n) in
    let k_nm = K.kernel ~theta:theta.kernel data_x theta.z in
    let kmm =
      let kmm = K.kernel ~theta:theta.kernel theta.z theta.z in
      let kmm_n = Maths.shape kmm |> List.hd_exn in
      (* regularize a tiny bit *)
      Maths.(
        kmm
        + (f fudge
           * eye
               ~device:(Tensor.device (Maths.to_tensor data_x))
               ~kind:(Tensor.kind (Maths.to_tensor data_x))
               kmm_n))
    in
    (* perform the various Cholesky decompositions required downstream *)
    let kmm_chol = Maths.cholesky kmm in
    let psi = Maths.tril ~_diagonal:0 theta.psi in
    (* the KL is remarkably simple and cheap to compute
       under this whitened variational parameterisation *)
    let kl_term =
      let term1 = Maths.(sum (sqr psi)) in
      let term2 = Maths.(sum (log (sqr (diagonal ~offset:0 psi)))) in
      let term3 = Maths.(sum (sqr theta.nu)) in
      Maths.(0.5 $* term1 - term2 + term3)
    in
    let nll_term =
      let term1 = Maths.(Float.(-0.5 * of_int n) $* log theta.beta) in
      let term2 =
        let knn_diag = K.kernel_diag ~theta:theta.kernel data_x in
        let zT =
          Maths.linsolve ~left:true kmm_chol (Maths.transpose k_nm)
        in
        let z = Maths.transpose zT ~dims:[ 1; 0 ] in
        let mu = Maths.(z *@ theta.nu) in
        Maths.(
          sum (sqr (mu - data_y)) + sum knn_diag + sum (sqr (z *@ psi)) - sum (sqr z))
      in
      Maths.(term1 + (0.5 $* theta.beta * term2))
    in
    Maths.(kl_term + (lik_rescaling $* nll_term))

  (* ---------------------------------- 
     ---  INFERENCE
     ---------------------------------- *)

  let infer ~theta =
    let theta = P.value theta in
    let psi = Maths.tril ~_diagonal:0 theta.psi in
    let nu = theta.nu in
    (* compute the necessary quantifies once and for all *)
    let kmm =
      let kmm = K.kernel ~theta:theta.kernel theta.z theta.z in
      let kmm_n = Maths.shape kmm |> List.hd_exn in
      (* regularize a tiny bit *)
      Maths.(
        kmm
        + (f fudge
           * eye
               ~device:(Tensor.device (Maths.to_tensor kmm))
               ~kind:(Tensor.kind (Maths.to_tensor kmm))
               kmm_n))
    in
    let kmm_chol = Maths.cholesky kmm in
    let am = Maths.linsolve ~left:true (Maths.transpose kmm_chol) nu in
    let tmp =
      let z =
        let psi_n = Maths.shape psi |> List.hd_exn in
        Maths.(
          (psi *@ transpose psi)
          - eye
              ~device:(Tensor.device (Maths.to_tensor psi))
              ~kind:(Tensor.kind (Maths.to_tensor psi))
              psi_n)
      in
      let z' = Maths.linsolve ~left:true (Maths.transpose kmm_chol) z in
      Maths.linsolve
        ~left:true
        (Maths.transpose kmm_chol)
        (Maths.transpose z')
      |> Maths.transpose
    in
    fun x_star ->
      let k = K.kernel ~theta:theta.kernel x_star theta.z in
      let k' = K.kernel ~theta:theta.kernel x_star x_star in
      let mu = Maths.(k *@ am) in
      (* k** + k L^{-T} (MM^T - I ) L^{-1} k *)
      let sigma = Maths.(k' + (k *@ tmp *@ transpose k)) in
      mu, sigma
end
