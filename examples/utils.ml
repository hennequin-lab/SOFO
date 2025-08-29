open Base
open Torch
open Forward_torch

let get_svals_u_left_right left right =
  let u_left, s_left, _ = Tensor.svd ~some:true ~compute_uv:true Maths.(to_tensor left) in
  let u_right, s_right, _ =
    Tensor.svd ~some:true ~compute_uv:true Maths.(to_tensor right)
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

let get_local_vs ~selection ~s_all ~u_left ~u_right =
  List.map selection ~f:(fun idx ->
    let il, ir, _ = s_all.(idx) in
    let slice_and_squeeze t dim idx =
      Tensor.squeeze_dim
        ~dim
        (Tensor.slice t ~dim ~start:(Some idx) ~end_:(Some (idx + 1)) ~step:1)
    in
    let u_l = slice_and_squeeze u_left 1 il in
    let u_r = slice_and_squeeze u_right 1 ir in
    let tmp = Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_l; u_r ] in
    Tensor.unsqueeze tmp ~dim:0)
  |> Tensor.concatenate ~dim:0
