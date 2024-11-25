open Base
open Torch
open Forward_torch

module Make (P : Prms.T) (O : Prms.T) = struct
  let run (p : Tensor.t P.p) ~(f : P.t' -> O.t') =
    let v = P.gaussian_like p |> P.map ~f:(fun v -> Maths.Direct v) in
    let p = P.make_dual p ~t:v in
    let o = f p |> O.tangent  in
    let w = O.gaussian_like o |> O.const in
let dot_prod_1 = O.dot_prod w (O.const o) in 
(* compare with backward-pass torch *)



end
