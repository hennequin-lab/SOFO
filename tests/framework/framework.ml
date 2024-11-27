open Base
open Torch
open Forward_torch

let print s = Stdio.print_endline (Sexp.to_string_hum s)

module Make (P : Prms.T) (O : Prms.T) = struct
  let run (p : P.T.t) ~(f : P.M.t -> O.M.t) =
    let dot_prod_1, v, w =
      let v = P.T.gaussian_like p in
      let p =
        P.make_dual
          p
          ~t:
            (P.map v ~f:(fun v ->
               Maths.Direct (Tensor.reshape v ~shape:(1 :: Tensor.shape v))))
      in
      let o = f p |> O.tangent in
      let w = O.T.gaussian_like o in
      O.T.dot_prod w o |> Tensor.to_float0_exn, v, w
    in
    print [%message "dot product with sofo completed"];
    (* compare with backward-pass torch *)
    let dot_prod_2 =
      let p =
        P.map p ~f:(fun x ->
          let x = Tensor.set_requires_grad x ~r:true in
          Tensor.zero_grad x;
          Maths.const x)
      in
      let o = f p in
      let surrogate = O.T.dot_prod w (O.primal o) in
      Tensor.backward surrogate;
      let g = P.map (P.primal p) ~f:Tensor.grad in
      P.T.dot_prod v g |> Tensor.to_float0_exn
    in
    Float.(abs (dot_prod_1 - dot_prod_2) / (abs dot_prod_1 + abs dot_prod_2))
end
