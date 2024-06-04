open Forward_torch
open Torch

module type T = sig
  type 'a with_args
  type labels

  (* loss function that computes the losses average over batch for each tangent given labels, model outputs, and a list of output dimensions. *)
  val f : (labels:labels -> reduce_dim_list:int list -> Maths.t -> Maths.t) with_args

  (* compute vtgtHgv from labels, model outputs and vg. *)
  val vtgt_hessian_gv
    : (labels:labels -> vtgt:Tensor.t -> reduce_dim_list:int list -> Maths.t -> Tensor.t)
        with_args
end

module type Reward = sig
  type 'a with_args

  val vtgt_gv : (vtgt:Tensor.t -> Tensor.t) with_args
end
