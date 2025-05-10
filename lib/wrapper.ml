open Base
open Forward_torch
open Torch
include Wrapper_typ

(* feedforward network *)
module Feedforward (F : Function) (L : Loss.T with type 'a with_args = 'a) = struct
  module P = F.P

  type data = F.input * L.labels
  type args = unit

  let f ~update ~data:(x, labels) ~init ~args:_ theta =
    let y = F.f ~theta ~input:x in
    let reduce_dim_list = Convenience.all_dims_but_first (Maths.primal y) in
    let ell = L.f ~labels ~reduce_dim_list y in
    match update with
    | `loss_only u -> u init (Some ell)
    | `loss_and_ggn u ->
      let delta_ggn =
        let vtgt = Maths.tangent y |> Option.value_exn in
        L.vtgt_hessian_gv ~labels ~vtgt ~reduce_dim_list y
      in
      u init (Some (ell, Some delta_ggn))
end

(* recurrent network *)
module Recurrent (F : Recurrent_function) (L : Loss.T with type 'a with_args = 'a) =
struct
  module P = F.P

  type data = (F.input * L.labels option) list
  type args = Tensor.t

  (* here data is a list of (x, optional labels) *)
  let f ~update ~data ~init ~args:y0 theta =
    let result, _ =
      List.foldi
        data
        ~init:(init, Maths.const y0)
        ~f:(fun t (accu, y) (x, labels) ->
          if t % 1 = 0 then Stdlib.Gc.major ();
          let y = F.f ~theta ~input:x y in
          let accu =
            match labels with
            | None -> accu
            | Some labels ->
              let reduce_dim_list = Convenience.all_dims_but_first (Maths.primal y) in
              let ell = L.f ~labels ~reduce_dim_list y in
              (match update with
               | `loss_only u -> u accu (Some ell)
               | `loss_and_ggn u ->
                 let delta_ggn =
                   let vtgt = Maths.tangent y |> Option.value_exn in
                   L.vtgt_hessian_gv ~labels ~vtgt ~reduce_dim_list y
                 in
                 u accu (Some (ell, Some delta_ggn)))
          in
          accu, y)
    in
    result
end
