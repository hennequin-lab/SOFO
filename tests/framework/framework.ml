open Base
open Torch
open Forward_torch

let print s = Stdio.print_endline (Sexp.to_string_hum s)

module Make (P : Prms.T) (O : Prms.T) = struct
  let run (p : P.T.t) ~(f : P.M.t -> O.M.t) =
    (* batched JVP tests *)
    let k = 2 in
    let dot_prod_1, v, w =
      (* samples v where it has one extra dimension in front of size k *)
      let v = P.T.gaussian_like_k ~k p in
      let p = P.make_dual p ~t:(P.map v ~f:(fun v -> Maths.Direct v)) in
      let o = f p |> O.tangent in
      let w = O.T.gaussian_like_k ~k o in
      O.T.dot_prod w o |> Tensor.to_float0_exn, v, w
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
      P.T.dot_prod v g |> Tensor.to_float0_exn
    in
    ( dot_prod_1
    , dot_prod_2
    , Float.(abs (dot_prod_1 - dot_prod_2) / (abs dot_prod_1 + abs dot_prod_2)) )
end
