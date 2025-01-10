open Base
open Torch
open Forward_torch

let print s = Stdio.print_endline (Sexp.to_string_hum s)

module Make (P : Prms.T) (O : Prms.T) = struct
  let run (p : P.T.t) ~(f : P.M.t -> O.M.t) =
    (* batched JVP tests *)
    let k = 7 in
    let dot_prod_1, v, w =
      (* samples v where it has one extra dimension in front of size k *)
      let v = P.T.gaussian_like_k ~k p in
      let p =
        P.make_dual (P.map p ~f:Prms.free) ~t:(P.map v ~f:(fun v -> Maths.Direct v))
      in
      let o = f p in
      let w = O.T.gaussian_like (O.primal o) in
      let o = O.tangent o in
      let dp1 =
        O.fold2 w o ~init:(Tensor.f 0.) ~f:(fun accu (w, o, _) ->
          let wo = Tensor.(reshape (w * o) ~shape:[ k; -1 ]) in
          Tensor.(accu + sum_to_size ~size:[ k; 1 ] wo))
      in
      Tensor.squeeze dp1, v, w
    in
    (* compare with backward-pass torch *)
    let dot_prod_2 =
      let p =
        P.map p ~f:(fun x ->
          let x = Tensor.set_requires_grad (Tensor.copy x) ~r:true in
          Tensor.zero_grad x;
          Maths.const x)
      in
      (* make sure that every single input parameter has a zero gradient *)
      let _ =
        let dummy = P.M.dot_prod p p in
        Tensor.backward (Maths.primal dummy);
        P.iter p ~f:(fun x -> Tensor.zero_grad (Maths.primal x))
      in
      let o = f p in
      let surrogate = O.T.dot_prod w (O.primal o) in
      Tensor.backward surrogate;
      let g = P.map p ~f:(fun x -> Tensor.grad (Maths.primal x)) in
      let dp2 =
        P.fold2 v g ~init:(Tensor.f 0.) ~f:(fun accu (v, g, _) ->
          let vg = Tensor.(reshape (v * g) ~shape:[ k; -1 ]) in
          Tensor.(accu + sum_to_size ~size:[ k; 1 ] vg))
      in
      Tensor.squeeze dp2
    in
    let err =
      Tensor.(
        mean (square (dot_prod_1 - dot_prod_2)) / mean (square (dot_prod_1 + dot_prod_2)))
      |> Tensor.to_float0_exn
    in
    dot_prod_1, dot_prod_2, err
end
