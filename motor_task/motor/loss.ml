open Base
open Torch
open Forward_torch
module Mat = Owl.Dense.Matrix.S

let base = Arm.base
let dt = 2e-3
let bins t = Float.(to_int (t / dt))
let t_tot = bins 2.0
let bs = 128 (* batch size *)

type item =
  { weights : Tensor.t
  ; target : Tensor.t Arm.pair
  ; output : [ `const | `dual ] Maths.t Arm.pair
  }

let squared_loss ~scaling_factor =
  let g ~weights x =
    match Maths.tangent x with
    | Some vtgt ->
      let weights = Maths.of_tensor (Tensor.squeeze weights) in
      let n_samples = Maths.shape vtgt |> List.hd_exn in
      let vtgt_mat = Maths.reshape vtgt ~shape:[ n_samples; -1 ] in
      Maths.C.(
        Float.(2. * scaling_factor)
        $* einsum [ vtgt_mat, "ik"; weights, "k"; vtgt_mat, "jk" ] "ij")
    | None -> Maths.(f 0.)
  in
  fun loss_items ->
    let ell =
      List.fold loss_items ~init:None ~f:(fun accu item ->
        let ell =
          let ell_ labels output =
            Maths.(
              scaling_factor
              $* mean
                   ~keepdim:false
                   ~dim:[ 0; 1 ]
                   (Maths.of_tensor item.weights * sqr (output - of_tensor labels)))
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
          Maths.C.(
            g ~weights:item.weights item.output.x1
            + g ~weights:item.weights item.output.x2)
        in
        match accu with
        | Some a -> Some Maths.C.(a + dggn)
        | None -> Some dggn)
      |> Option.value_exn
    in
    ell, dggn
