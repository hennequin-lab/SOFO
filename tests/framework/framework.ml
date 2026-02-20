open Base
open Forward_torch
open Maths

let print s = Stdio.print_endline (Sexp.to_string_hum s)

module Make (P : Prms.T) (O : Prms.T) = struct
  let run (p : P.t) ~(f : P.t -> O.t) =
    let open Torch in
    (* batched JVP tests *)
    let k = 7 in
    let dot_prod_1, v, w =
      (* samples v where it has one extra dimension in front of size k *)
      let v = P.randn_like_k ~k p in
      let p = P.dual p ~tangent:(P.primal v) in
      let o = f p in
      let w = O.randn_like o in
      let o = O.tangent_exn o in
      let dp1 =
        O.fold2 w o ~init:(Maths.f 0.) ~f:(fun accu (w, o, _) ->
          let wo = reshape (w * const o) ~shape:[ k; -1 ] in
          accu + sum ~dim:[ 1 ] ~keepdim:false wo)
      in
      dp1, v, w
    in
    (* compare with backward-pass torch *)
    let dot_prod_2 =
      let p =
        P.map p ~f:(fun x ->
          let x = primal x in
          let x = Tensor.set_requires_grad (Tensor.copy x) ~r:true in
          Tensor.zero_grad x;
          const x)
      in
      (* make sure that every single input parameter has a zero gradient *)
      let _ =
        let dummy = P.dot_prod p p in
        Tensor.backward (primal dummy);
        P.iter p ~f:(fun x -> Tensor.zero_grad (primal x))
      in
      let o = f p in
      let surrogate = O.dot_prod w o in
      Tensor.backward (primal surrogate);
      let g = P.map p ~f:(fun x -> const (Tensor.grad (primal x))) in
      let dp2 =
        P.fold2 v g ~init:(Maths.f 0.) ~f:(fun accu (v, g, _) ->
          let vg = reshape (v * g) ~shape:[ k; -1 ] in
          accu + sum ~dim:[ 1 ] ~keepdim:false vg)
      in
      dp2
    in
    let err =
      mean (sqr (dot_prod_1 - dot_prod_2) / mean (sqr (dot_prod_1 + dot_prod_2)))
      |> to_float_exn
    in
    dot_prod_1, dot_prod_2, err
end
