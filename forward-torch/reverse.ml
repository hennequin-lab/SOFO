open! Base
open Torch
open Maths

type nonrec dual =
  { p : any t
  ; mutable a : const t option
  }

type _ Stdlib.Effect.t +=
  | Gen1 : ((any t -> any t) * dual) -> dual Stdlib.Effect.t
  | Gen2 : ((any t -> any t -> any t) * dual * dual) -> dual Stdlib.Effect.t

let const p = { p; a = None }
let primal { p; _ } = p
let lift1 f a = Stdlib.Effect.perform (Gen1 (f, a))
let lift2 f a b = Stdlib.Effect.perform (Gen2 (f, a, b))
let ( + ) a b = lift2 ( + ) a b
let ( - ) a b = lift2 ( - ) a b
let ( * ) a b = lift2 ( * ) a b
let ( / ) a b = lift2 ( / ) a b

(* These take a float - use dual instead *)
(* Better to define an effect for float and dual? *)
let ( $+ ) a b =
  let a_dual = const (any Maths.(a $* ones_like (primal b))) in
  a_dual + b

let ( $- ) a b =
  let a_dual = const (any Maths.(a $* ones_like (primal b))) in
  a_dual - b

(* let ( $* ) a b = lift2 C.( $* ) a b *)
let ( *@ ) a b = lift2 ( *@ ) a b
let sigmoid a = lift1 sigmoid a
let tanh a = lift1 tanh a
let mean a = lift1 mean a
let sqr a = lift1 sqr a

let eval f x =
  match f x with
  | result -> result
  | effect Gen1 (f, a), k -> Stdlib.Effect.Deep.continue k { p = f a.p; a = None }
  | effect Gen2 (f, a, b), k -> Stdlib.Effect.Deep.continue k { p = f a.p b.p; a = None }

let __prepare a =
  let a = to_tensor a.p in
  let device = Tensor.device a in
  let a_ = Tensor.copy a |> Tensor.to_device ~device in
  let a_ = Tensor.set_requires_grad a_ ~r:true in
  Tensor.zero_grad a_;
  a_

let zero_adj p = { p; a = Some (zeros_like p) }

module Make (P : Prms.T) = struct
  let const p = P.map p ~f:(fun p -> { p; a = Some (zeros_like p) })

  let update_adj adj delta =
    match adj with
    | None -> Some delta
    | Some a -> Some Maths.C.(a + delta)

  (* f is prms -> scalar function *)
  let grad f (x : dual P.p) =
    let fx =
      match f x with
      | result ->
        result.a <- Some (C.f 1.);
        result
      | effect Gen1 (f, a), k ->
        let p = f a.p in
        let o = zero_adj p in
        let result = Stdlib.Effect.Deep.continue k o in
        (* use Torch's autodiff to propagate adjoints *)
        Option.iter o.a ~f:(fun r_bar ->
          (* prepare a for reverse pass after the continuation *)
          let a_ = __prepare a in
          let p = f (any (of_tensor a_)) in
          let y = Tensor.(sum (to_tensor r_bar * to_tensor p)) in
          Tensor.backward y;
          a.a <- update_adj a.a (of_tensor (Tensor.grad a_)));
        result
      | effect Gen2 (f, a, b), k ->
        (* prepare a for reverse pass after the continuation *)
        let p = f a.p b.p in
        let o = zero_adj p in
        let result = Stdlib.Effect.Deep.continue k o in
        (* use Torch's autodiff to propagate adjoints *)
        Option.iter o.a ~f:(fun r_bar ->
          let a_ = __prepare a in
          let b_ = __prepare b in
          let p = f (any (of_tensor a_)) (any (of_tensor b_)) in
          let y = Tensor.(sum (to_tensor r_bar * to_tensor p)) in
          Tensor.backward y;
          a.a <- update_adj a.a (of_tensor (Tensor.grad a_));
          b.a <- update_adj b.a (of_tensor (Tensor.grad b_)));
        result
    in
    ( fx.p
    , P.map x ~f:(fun x ->
        match x.a with
        | Some g -> g
        | None -> zeros_like x.p) )
end
