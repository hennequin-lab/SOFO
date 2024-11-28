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
  type ('a, 'o) p =
    { o : 'o
    ; a : 'a
    }
  [@@deriving prms]
end

module Input =
  Lqr.Params.Make (Prms.P) (Prms.List (Temp.Make (Prms.P) (Prms.Option (Prms.P))))

module Output = Prms.P

let f (x : Input.M.t) : Output.M.t = Input.M.dot_prod x x

let check_grad (x : Input.T.t) =
  let module F = Framework.Make (Input) (Output) in
  F.run x ~f

let _ =
  let x0 = Tensor.randn [ 5; 7 ] in
  let (x : Input.T.t) =
    Lqr.Params.
      { x0
      ; params =
          List.init 3 ~f:(fun _ ->
            Temp.{ o = Some (Tensor.randn [ 5; 5 ]); a = Tensor.randn [ 2; 3 ] })
      }
  in
  print [%message (check_grad x : float)]
