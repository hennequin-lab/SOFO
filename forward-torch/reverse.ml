open! Base
open Torch
open Maths

module ReverseMode = struct
  type nonrec dual =
    { p : const t
    ; mutable a : const t option
    }

  type _ Stdlib.Effect.t +=
    | Mul : (dual * dual) -> dual Stdlib.Effect.t
    | Add : (dual * dual) -> dual Stdlib.Effect.t

  let ( + ) a b = Stdlib.Effect.perform (Add (a, b))
  let ( * ) a b = Stdlib.Effect.perform (Mul (a, b))

  module Grad (P : Prms.T) = struct
    let eval f (x : const Maths.t P.p) =
      let x = P.map x ~f:(fun p -> { p; a = None }) in
      try f x with
      | effect Add (a, b), k ->
        Stdlib.Effect.Deep.continue k { p = C.(a.p + b.p); a = None }
      | effect Mul (a, b), k ->
        Stdlib.Effect.Deep.continue k { p = C.(a.p * b.p); a = None }

    (* f is prms -> scalar function *)
    let grad f (x : const Maths.t P.p) =
      let x = P.map x ~f:(fun p -> { p; a = Some (zeros_like p) }) in
      let fx =
        match f x with
        | result ->
          result.a <- Some (C.f 1.);
          result
        | effect Add (a, b), k ->
          let p = C.(a.p + b.p) in
          let result = { p; a = Some (zeros_like p) } in
          ignore (Stdlib.Effect.Deep.continue k result);
          (* propagate adjoints *)
          Option.iter result.a ~f:(fun c_bar ->
            a.a
            <- Some
                 (match a.a with
                  | None -> c_bar
                  | Some tmp -> C.(tmp + c_bar));
            b.a
            <- Some
                 (match b.a with
                  | None -> c_bar
                  | Some tmp -> C.(tmp + c_bar)));
          result
        | effect Mul (a, b), k ->
          (* initialise the torch Tensor gradient of a.p b.p to zero;
            i.e. prepare them for a reverse pass *)
            let p = C.(a.p * b.p) in
        let result = { p; a = Some (zeros_like p) } in
        ignore (Stdlib.Effect.Deep.continue k result);
        Option.iter result.a ~f:(fun r_bar ->
          let aux = Tensor.(sum (to_tensor r_bar * to_tensor p)) in
          Tensor.backward  aux;


          torch_effect_handler_2_1 op (a, b) k
      in
      fx.p, P.map x ~f:(fun x -> x.a)
  end
end
