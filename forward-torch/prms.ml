open Base
open Torch
include Prms_typ

let print s = Stdio.print_endline (Sexp.to_string_hum s)

let value = function
  | Const x -> x
  | Free x -> x
  | Bounded (x, _, _) -> x

let bind x ~f = f (value x)

(* map enforces Const and Bounds constraints *)
let map x ~f =
  match x with
  | Const x -> Const x
  | Free x -> Free (f x)
  | Bounded (x, lb, ub) ->
    let y = f x in
    let y = Option.value_map lb ~default:y ~f:(Tensor.max y) in
    let y = Option.value_map ub ~default:y ~f:(Tensor.min y) in
    Bounded (y, lb, ub)

(* can only pair values of the same tag; if they are both Bounded,
   bounds should be the same *)
let pair x1 x2 = Const (value x1, value x2)

let pair' (x1 : ('a, Tensor.t, Tensor.t) tag) (x2 : ('b, Tensor.t, Tensor.t) tag) =
  match x1 with
  | Const z1 -> Const z1
  | Free z1 -> Free (z1, value x2)
  | Bounded (z1, lb, ub) -> Bounded ((z1, value x2), lb, ub)

let pair'' (x1 : ('a, Tensor.t, Tensor.t) tag) (x2 : Tensor.t) =
  match x1 with
  | Const z1 -> Const z1
  | Free z1 -> Free (z1, x2)
  | Bounded (z1, lb, ub) -> Bounded ((z1, x2), lb, ub)

module Let_syntax = struct
  let ( let* ) x f = bind x ~f
  let ( and* ) = pair
  let ( let+ ) x f = map x ~f
  let ( and+ ) = pair'
  let ( and++ ) = pair''
end

open Let_syntax

let const x = Const x
let free x = Free x

let create ?above ?below x =
  match above, below with
  | None, None -> Free x
  | _ -> Bounded (x, above, below)

let pin ?to_ x =
  let* x = x in
  Const (Option.value ~default:x to_)

let bound ?above ?below = function
  | Free z -> Bounded (z, above, below)
  | _ -> failwith "Can only bound a Free parameter"

let numel t = List.fold ~init:1 ~f:( * ) (Tensor.shape t)

let cat label =
  if String.(label = "")
  then Fn.id
  else
    function
    | None -> Some [ label ]
    | Some p -> Some (label :: p)

module Make (B : Basic) : T with type 'a p = 'a B.p = struct
  let map_ = map

  include B

  type nonrec tagged = tagged p

  let iter x ~f = fold ?path:None x ~init:() ~f:(fun () (x, _) -> f x)
  let iter2 x y ~f = fold2 ?path:None x y ~init:() ~f:(fun () (x, y, _) -> f x y)

  module T = struct
    type t = Tensor.t p
    type elt = Tensor.t

    let zeros_like x = map x ~f:Tensor.zeros_like
    let ones_like x = map x ~f:Tensor.ones_like

    let gaussian_like ?mu ?sigma x =
      map x ~f:(fun x ->
        let z = Tensor.randn_like x in
        let z =
          match sigma with
          | None -> z
          | Some s -> Tensor.mul_scalar_ z (Scalar.f s)
        in
        match mu with
        | None -> z
        | Some m -> Tensor.add_scalar_ z (Scalar.f m))

    let numel x = fold x ~init:0 ~f:(fun accu (x, _) -> accu + numel x)

    let dot_prod x y =
      fold2 x y ~init:None ~f:(fun accu (x, y, _) ->
        let (z : Tensor.t) = Tensor.(sum (x * y)) in
        match accu with
        | None -> Some z
        | Some a -> Some Tensor.(a + z))
      |> Option.value_exn

    let sqr = map ~f:Tensor.square
    let sqrt = map ~f:Tensor.sqrt
    let ( + ) = map2 ~f:Tensor.( + )
    let ( - ) = map2 ~f:Tensor.( - )
    let ( * ) = map2 ~f:Tensor.( * )
    let ( / ) = map2 ~f:Tensor.( / )
    let ( $+ ) x = map ~f:(fun a -> Tensor.(add_scalar a (Scalar.f x)))
    let ( $* ) x = map ~f:(fun a -> Tensor.(mul_scalar a (Scalar.f x)))

    let save m ~kind ~out:filename =
      let m = map m ~f:(fun x -> Tensor.to_bigarray x ~kind) in
      let output = Stdio.Out_channel.create filename in
      Stdlib.Marshal.to_channel output m [ Stdlib.Marshal.No_sharing ];
      Stdio.Out_channel.close output

    let load ?device filename =
      let input = Stdio.In_channel.create filename in
      let m = Stdlib.Marshal.from_channel input in
      Stdio.In_channel.close input;
      map m ~f:(fun x -> Tensor.of_bigarray ?device x)

    let save_npy ?prefix ~kind ~out prms =
      let prms = map prms ~f:(fun x -> Tensor.to_bigarray x ~kind) in
      let path = Option.map prefix ~f:(fun s -> [ s ]) in
      let file = Npy.Npz.open_out out in
      fold ?path prms ~init:() ~f:(fun () (prm, path) ->
        let descr =
          match path with
          | None -> assert false
          | Some p -> String.concat ~sep:"/" (List.rev p)
        in
        Npy.Npz.write file descr prm);
      Npy.Npz.close_out file
  end

  let const = map ~f:Maths.const
  let value = map ~f:value
  let primal = map ~f:Maths.primal

  let tangent x =
    try map x ~f:(fun x -> Option.value_exn (Maths.tangent x)) with
    | _ -> raise Maths.Not_a_dual_number

  let make_dual x ~t = map2 x t ~f:(fun x t -> Maths.make_dual x ~t)

  module M = struct
    type t = Maths.t p
    type elt = Maths.t

    let zeros_like x = const (T.zeros_like (primal x))
    let ones_like x = const (T.ones_like (primal x))
    let gaussian_like ?mu ?sigma x = const (T.gaussian_like ?mu ?sigma (primal x))
    let numel x = T.numel (primal x)

    let dot_prod x y =
      fold2 x y ~init:None ~f:(fun accu (x, y, _) ->
        let (z : Maths.t) = Maths.(sum (x * y)) in
        match accu with
        | None -> Some z
        | Some a -> Some Maths.(a + z))
      |> Option.value_exn

    let sqr = map ~f:Maths.sqr
    let sqrt = map ~f:Maths.sqrt
    let ( + ) = map2 ~f:Maths.( + )
    let ( - ) = map2 ~f:Maths.( - )
    let ( * ) = map2 ~f:Maths.( * )
    let ( / ) = map2 ~f:Maths.( / )
    let ( $+ ) x = map ~f:(fun a -> Maths.(x $+ a))
    let ( $* ) x = map ~f:(fun a -> Maths.(x $* a))
  end

  module Let_syntax = struct
    let ( let* ) x f =
      let x = value x in
      f x

    let ( and* ) x y = map2 x y ~f:pair
    let ( let+ ) x f = map x ~f:(map_ ~f)
    let ( and+ ) x y = map2 x y ~f:pair'
    let ( and++ ) x y = map2 x y ~f:pair''
  end
end

module None : T with type 'a p = unit = Make (struct
    type 'a p = unit

    let map () ~f:_ = ()
    let map2 () () ~f:_ = ()
    let fold ?path:_ () ~init ~f:_ = init
    let fold2 ?path:_ () () ~init ~f:_ = init
  end)

module P = Make (struct
    type 'a p = 'a

    let map x ~f = f x
    let map2 x y ~f = f x y
    let fold ?path x ~init ~f = f init (x, path)
    let fold2 ?path x y ~init ~f = f init (x, y, path)
  end)

module Pair (P1 : T) (P2 : T) = Make (struct
    type 'a p = 'a P1.p * 'a P2.p

    let map (x1, x2) ~f = P1.map x1 ~f, P2.map x2 ~f
    let map2 (x1, x2) (y1, y2) ~f = P1.map2 x1 y1 ~f, P2.map2 x2 y2 ~f

    let fold ?path (x1, x2) ~init ~f =
      let init = P1.fold ?path x1 ~init ~f in
      P2.fold ?path x2 ~init ~f

    let fold2 ?path (x1, x2) (y1, y2) ~init ~f =
      let init = P1.fold2 ?path x1 y1 ~init ~f in
      P2.fold2 ?path x2 y2 ~init ~f
  end)

module Option (P : T) = Make (struct
    type 'a p = 'a P.p Option.t

    let map x ~f =
      match x with
      | Some x -> Some (P.map ~f x)
      | None -> None

    let map2 x y ~f =
      match x, y with
      | Some x, Some y -> Some (P.map2 ~f x y)
      | _ -> None

    let fold ?path x ~init ~f =
      match x with
      | Some x -> P.fold ?path x ~f ~init
      | None -> init

    let fold2 ?path x y ~init ~f =
      match x, y with
      | Some x, Some y -> P.fold2 ?path x y ~init ~f
      | _ -> init
  end)

module List (P : T) = Make (struct
    type 'a p = 'a P.p List.t

    let map x ~f = List.map x ~f:(P.map ~f)
    let map2 x y ~f = List.map2_exn x y ~f:(P.map2 ~f)

    let fold ?path x ~init ~f =
      List.foldi x ~init ~f:(fun i init w ->
        P.fold ?path:(cat (Int.to_string i) path) ~init ~f w)

    let fold2 ?path x y ~init ~f =
      let _, result =
        List.fold2_exn x y ~init:(0, init) ~f:(fun (i, init) w1 w2 ->
          i + 1, P.fold2 ?path:(cat (Int.to_string i) path) ~init ~f w1 w2)
      in
      result
  end)

module Array (P : T) = Make (struct
    type 'a p = 'a P.p array

    let map x ~f = Array.map x ~f:(P.map ~f)
    let map2 x y ~f = Array.map2_exn x y ~f:(P.map2 ~f)

    let fold ?path x ~init ~f =
      Array.foldi x ~init ~f:(fun i init w ->
        P.fold ?path:(cat (Int.to_string i) path) ~init ~f w)

    let fold2 ?path x y ~init ~f =
      let _, result =
        Array.fold2_exn x y ~init:(0, init) ~f:(fun (i, init) w1 w2 ->
          i + 1, P.fold2 ?path:(cat (Int.to_string i) path) ~init ~f w1 w2)
      in
      result
  end)
