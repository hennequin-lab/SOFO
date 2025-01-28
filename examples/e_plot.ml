open Base
open Forward_torch
module Mat = Owl.Dense.Matrix.D
module Arr = Owl.Dense.Ndarray.D
module Linalg = Owl.Linalg.D
module Z = Owl.Dense.Matrix.Z

let in_dir = Cmdargs.in_dir "-d"
let a = 10

let sample_stable ~a =
  let a = Mat.gaussian a a in
  let r =
    a |> Linalg.eigvals |> Owl.Dense.Matrix.Z.abs |> Owl.Dense.Matrix.Z.re |> Mat.max'
  in
  Mat.(Float.(0.8 / r) $* a)

let sample_stable_new ~a =
  let w =
    let tmp = Mat.gaussian a a in
    let r = tmp |> Linalg.eigvals |> Z.re |> Mat.max' in
    Mat.(Float.(0.8 / r) $* tmp)
  in
  let w_i = Mat.(0.1 $* w - eye a) in
  Owl.Linalg.Generic.expm w_i

let eigvals ~a mat =
  let final_eigvals = Linalg.eigvals mat in
  Array.init a ~f:(fun i ->
    let eigval = Z.get final_eigvals 0 i in
    eigval.re, eigval.im)

(* Generate 100 samples and concatenate them into a 100x4 matrix *)

let _ =
  let num_samples = 100 in
  let data =
    List.init num_samples ~f:(fun _ ->
      let mat = sample_stable_new ~a in
      eigvals ~a mat |> Array.to_list |> List.concat_map ~f:(fun (re, im) -> [ re; im ]))
  in
  let matrix = Mat.of_arrays (Array.of_list_map data ~f:Array.of_list) in
  Mat.save_txt ~out:(in_dir "eigvals_new") matrix
