open Base
open Torch
open Forward_torch
open Sofo
module Mat = Owl.Dense.Matrix.S

let base = Arm.base
let dt = 2e-3
let bins t = Float.(to_int (t / dt))
let t_tot = bins 2.0
let bs = 128 (* batch size *)

type item =
  { weights : Tensor.t
  ; target : Tensor.t Arm.pair
  ; output : Maths.t Arm.pair
  }

let squared_loss ~scaling_factor =
  let module L =
    Loss.Weighted_MSE (struct
      let scaling_factor = scaling_factor
    end)
  in
  let dummy_t = Tensor.f 0. in
  let dummy = Maths.const dummy_t in
  let g ~weights x =
    match Maths.tangent x with
    | Some vtgt ->
      L.vtgt_hessian_gv ~weights ~labels:dummy_t ~vtgt ~reduce_dim_list:[] dummy
    | None -> Tensor.(f 0.)
  in
  fun loss_items ->
    let ell =
      List.fold loss_items ~init:None ~f:(fun accu item ->
        let ell =
          let ell_ labels output =
            L.f ~weights:item.weights ~labels ~reduce_dim_list:[ 1 ] output
          in
          Maths.(ell_ item.target.x1 item.output.x1 + ell_ item.target.x2 item.output.x2)
        in
        match accu with
        | Some a -> Some Maths.(a + ell)
        | None -> Some ell)
      |> Option.value_exn
    in
    (* lazy dggn *)
    let dggn () =
      List.fold loss_items ~init:None ~f:(fun accu item ->
        let dggn =
          Tensor.(
            g ~weights:item.weights item.output.x1
            + g ~weights:item.weights item.output.x2)
        in
        match accu with
        | Some a -> Some Tensor.(a + dggn)
        | None -> Some dggn)
      |> Option.value_exn
    in
    ell, dggn
