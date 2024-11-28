open Base
open Torch
open Forward_torch
module Arr = Owl.Arr
module Mat = Owl.Mat
module Linalg = Owl.Linalg.D

let print s = Stdio.print_endline (Sexp.to_string_hum s)
let device = Torch.Device.Cpu
let kind = Torch_core.Kind.(T f64)

module Temp = struct
  type 'a p = { a : 'a } [@@deriving prms]
end

module Input = Lqr.Params.Make (Prms.P) (Prms.Array (Temp.Make (Prms.P)))

(* module Output = Prms.List (Lqr.Solution.Make (Prms.P)) *)
module Output = Prms.P

let tmax = 2
let f (x : Input.M.t) : Output.M.t = Input.M.dot_prod x x
let r () = Tensor.randn ~device [ 1 + Random.int 2; 1 + Random.int 2; 1 + Random.int 2 ]

let check_grad (x : Input.T.t) =
  let module F = Framework.Make (Input) (Output) in
  F.run x ~f

let _ =
  let x =
    Lqr.Params.
      { x0 = r ()
      ; params =
          (let z1 = Temp.{ a = r () } in
           [| z1; z1 |])
      }
  in
  print [%message (check_grad x : float)]
