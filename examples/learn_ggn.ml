open Base
open Owl

let print s = Stdio.print_endline Sexp.(to_string_hum s)
let in_dir = Cmdargs.in_dir "-d"
let n = 200
let _K = 2

let u, s, q, u_screwed, u_noisy =
  let u, _, _ = Linalg.D.qr Mat.(gaussian n n) in
  let s = Mat.init 1 n (fun i -> Float.(1. / (1. + (of_int i / 2.)))) in
  let u_screwed =
    let rot, _, _ = Linalg.D.qr Mat.(gaussian n n) in
    Mat.(u / sqrt s *@ rot)
  in
  let u_noisy =
    let z =
      Mat.(u * sqrt s *@ Mat.gaussian n Int.(2 * n) /$ Float.(sqrt (of_int Int.(2 * n))))
    in
    let u, _, _ = Linalg.D.svd z in
    u
  in
  u, s, Mat.(u * s *@ transpose u), u_screwed, u_noisy

let _ = Mat.(save_txt ~out:(in_dir "s") (transpose s))
let xstar = Mat.gaussian n 1

let loss x =
  let e = Mat.(x - xstar) in
  Float.(0.5 * Mat.(mean' (e * (q *@ e))))

let grad x = Mat.(q *@ (x - xstar))
let sample_vs_naive _ _ = Mat.gaussian n _K

(* randomly sample from the eigenvectors of ggn *)
let sample_vs_smart u _ =
  let ids = List.init _K ~f:(fun _ -> Random.int n) in
  Mat.get_fancy [ R []; L ids ] u

(* systematically sample from the eigenvectors of ggn *)
let sample_vs_smarter u t =
  let ids = List.init _K ~f:(fun i -> ((t * _K) + i) % n) in
  Mat.get_fancy [ R []; L ids ] u

(* vanilla SOFO *)
let _ =
  let lr = 1. in
  let file = in_dir "clean_smarter" in
  Bos.Cmd.(v "rm" % "-f" % file) |> Bos.OS.Cmd.run |> ignore;
  let rec iter ~k x =
    let v = sample_vs_smarter u k in
    let v, _, _ = Linalg.D.qr v in
    let ggn_sketch = Mat.(transpose v *@ q *@ v) in
    let grad_sketch = Mat.(transpose v *@ q *@ (x - xstar)) in
    let u, s, _ = Linalg.D.svd ggn_sketch in
    let ng = Mat.(v *@ (u / s *@ transpose u *@ grad_sketch)) in
    let x = Mat.(x - (lr $* ng)) in
    let loss = loss x in
    Mat.(save_txt ~append:true ~out:file (of_array [| loss |] 1 1));
    if k < 2000 then iter ~k:(k + 1) x
  in
  iter ~k:0 Mat.(gaussian n 1)
