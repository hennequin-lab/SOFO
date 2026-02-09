open! Base
open Torch
open Maths

type nonrec dual =
  { p : const t
  ; mutable a : const t option
  }

type _ Stdlib.Effect.t +=
  | Gen1 : ((const t -> const t) * dual) -> dual Stdlib.Effect.t
  | Gen2 : ((const t -> const t -> const t) * dual * dual) -> dual Stdlib.Effect.t
  | Einsum : ((dual * string) list * string) -> dual Stdlib.Effect.t

let of_const p = { p; a = None }
let get_primal { p; _ } = p

let lift1 f a = Stdlib.Effect.perform (Gen1 (f, a))
let lift2 f a b = Stdlib.Effect.perform (Gen2 (f, a, b))
let einsum ops rhs = Stdlib.Effect.perform (Einsum (ops, rhs))
let ( + ) a b = lift2 C.( + ) a b
let ( - ) a b = lift2 C.( - ) a b
let ( * ) a b = lift2 C.( * ) a b
let ( / ) a b = lift2 C.( / ) a b
(* These take a float - use dual instead *)
(* Better to define an effect for float and dual? *)
let ( $+ ) a b = 
  let a_dual = of_const Maths.(a $* ones_like (get_primal b)) in 
  a_dual + b
let ( $- ) a b = 
  let a_dual = of_const Maths.(a $* ones_like (get_primal b)) in 
  a_dual - b
(* let ( $* ) a b = lift2 C.( $* ) a b *)
let ( *@ ) a b = lift2 C.( *@ ) a b
let sigmoid a = lift1 C.sigmoid a
let tanh a = lift1 C.tanh a
let mean a = lift1 C.mean a
let sqr a = lift1 C.sqr a


module Make (P : Prms.T) = struct
  type _ Stdlib.Effect.t +=
    | Gen : ((const t P.p -> const t) * dual P.p) -> dual Stdlib.Effect.t
    | GenDual: ((dual P.p -> dual) * dual P.p) -> dual Stdlib.Effect.t

  let lift f x = Stdlib.Effect.perform (Gen (f, x))
  let lift_dual f x = 
    (* let x = P.map x ~f:(fun p -> { p; a = Some (zeros_like p) }) in *)
    Stdlib.Effect.perform (GenDual (f, x))

  let eval f (x : const Maths.t P.p) : 'a =
    let x = P.map x ~f:(fun p -> { p; a = None }) in
    match f x with
    | result -> result
    | effect Gen1 (f, a), k -> Stdlib.Effect.Deep.continue k { p = f a.p; a = None }
    | effect Gen2 (f, a, b), k ->
      Stdlib.Effect.Deep.continue k { p = f a.p b.p; a = None }

  let __prepare a =
    let a = to_tensor a.p in
    let device = Tensor.device a in
    let a = Tensor.copy a |> Tensor.to_device ~device in
    let a = Tensor.set_requires_grad a ~r:true in
    Tensor.zero_grad a;
    a

  let zero_adj p = { p; a = Some (zeros_like p) }
  let zero_adj_prms p = P.map p ~f:(fun p -> { p; a = Some (zeros_like p) })

  (* f is prms -> scalar function *)
  let grad f (x : dual P.p) =
    (* let x = P.map x ~f:(fun p -> { p; a = Some (zeros_like p) }) in *)
    let fx =
      match f x with
      | result ->
        result.a <- Some (C.f 1.);
        result
      (* Gen is not used *)
      | effect Gen (f, a), k ->
        let a_ = P.map ~f:__prepare a in
        let p = f (P.map a_ ~f:of_tensor) in
        let o = zero_adj p in
        let result = Stdlib.Effect.Deep.continue k o in
        (* use Torch's autodiff to propagate adjoints *)
        Option.iter o.a ~f:(fun r_bar ->
          let y = Tensor.(sum (to_tensor r_bar * to_tensor o.p)) in
          Tensor.backward y;
          P.iter2 a a_ ~f:(fun a a_ -> a.a <- Some (of_tensor (Tensor.grad a_))));
        result
      | effect GenDual (f, a), k ->
        let a_ = P.map ~f:__prepare a in
        let o = f (P.map a_ ~f:(fun x -> of_tensor x |> zero_adj)) in
        let result = Stdlib.Effect.Deep.continue k o in
        (* use Torch's autodiff to propagate adjoints *)
        Option.iter o.a ~f:(fun r_bar ->
          let y = Tensor.(sum (to_tensor r_bar * to_tensor o.p)) in
          Tensor.backward y;
          P.iter2 a a_ ~f:(fun a a_ -> a.a <- Some (of_tensor (Tensor.grad a_))));
        result
      | effect Gen1 (f, a), k ->
        (* prepare a for reverse pass after the continuation *)
        let a_ = __prepare a in
        let p = f (of_tensor a_) in
        let o = zero_adj p in
        let result = Stdlib.Effect.Deep.continue k o in
        (* use Torch's autodiff to propagate adjoints *)
        Option.iter o.a ~f:(fun r_bar ->
          let y = Tensor.(sum (to_tensor r_bar * to_tensor o.p)) in
          Tensor.backward y;
          a.a <- Some (of_tensor (Tensor.grad a_)));
        result
      | effect Gen2 (f, a, b), k ->
        (* prepare a for reverse pass after the continuation *)
        let a_ = __prepare a in
        let b_ = __prepare b in
        let p = f (of_tensor a_) (of_tensor b_) in
        let o = zero_adj p in
        let result = Stdlib.Effect.Deep.continue k o in
        (* use Torch's autodiff to propagate adjoints *)
        Option.iter o.a ~f:(fun r_bar ->
          let y = Tensor.(sum (to_tensor r_bar * to_tensor o.p)) in
          Tensor.backward y;
          a.a <- Some (of_tensor (Tensor.grad a_));
          b.a <- Some (of_tensor (Tensor.grad b_)));
        result
      | effect Einsum (ops, rhs), k ->
        let ops_ = ops |> Array.of_list |> Array.map ~f:(fun (x, _) -> __prepare x) in
        let ops__ = List.mapi ops ~f:(fun i (_, idx) -> of_tensor ops_.(i), idx) in
        let p = C.einsum ops__ rhs in
        let o = zero_adj p in
        let result = Stdlib.Effect.Deep.continue k o in
        Option.iter o.a ~f:(fun r_bar ->
          let y = Tensor.(sum (to_tensor r_bar * to_tensor o.p)) in
          Tensor.backward y;
          List.iteri ops ~f:(fun i (x, _) ->
            x.a <- Some (of_tensor (Tensor.grad ops_.(i)))));
        result
    in
    ( fx.p
    , P.map x ~f:(fun x ->
        match x.a with
        | Some g -> g
        | None -> zeros_like x.p) )
end
