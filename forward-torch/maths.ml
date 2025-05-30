open Base
open Torch

(*
   We will systematically work with the assumption that for a primal value of shape [n1; n2; ... ],
   the associated tangents have one more dimension, corresponding to tangent batch: [K; n1; n2; ... ]
*)

type tangent =
  | Explicit of const t
  | On_demand of (Device.t -> const t)

and const = [ `Const ]
and dual = [ `Dual ]
and 'a any = [< `Const | `Dual ] as 'a

and _ t =
  | Const of Tensor.t
  | Dual of Tensor.t * tangent

exception Not_dual
exception Wrong_shape of int list * int list
exception Wrong_device of Device.t * Device.t
exception Check_grad_failed

let assert_right_shape x dx =
  let sx = Tensor.shape x
  and sdx = Tensor.shape dx in
  let n = List.length sx in
  let n' = List.length sdx in
  if n' <> n + 1 || Poly.(List.tl_exn sdx <> sx) then raise (Wrong_shape (sx, sdx))

let assert_right_device x dx =
  let dx = Tensor.device x
  and ddx = Tensor.device dx in
  if Poly.(dx <> ddx) then raise (Wrong_device (dx, ddx))

let as_const = function
  | Const x -> Const x
  | Dual (x, _) -> Const x

let as_dual_exn = function
  | Dual _ as y -> y
  | _ -> raise Not_dual

let primal = function
  | Const x -> x
  | Dual (x, _) -> x

let tangent_tensor_of x v =
  match v with
  | Explicit dx -> primal dx
  | On_demand dx -> primal (dx (Tensor.device x))

let tangent = function
  | Dual (x, dx) -> tangent_tensor_of x dx
  | _ -> assert false (* will never happen *)

let const x = Const x

let dual ~dx = function
  | Const x ->
    assert_right_device x (primal dx);
    assert_right_shape x (primal dx);
    Dual (x, Explicit dx)
  | _ -> assert false (* will never happen *)

let dual_lazy ~dx = function
  | Const x ->
    let dx device =
      let dx = dx device in
      assert_right_shape x (primal dx);
      assert_right_device x (primal dx);
      dx
    in
    Dual (x, On_demand dx)
  | _ -> assert false (* will never happen *)

let _print s = Stdio.print_endline (Sexp.to_string_hum s)
let shape x = x |> primal |> Tensor.shape
let device x = x |> primal |> Tensor.device
let kind x = x |> primal |> Tensor.type_

(* int list starting from 1 and ending at the last dim of a *)
let _all_dims_but_first a = List.range 1 (List.length (Tensor.shape a))
let batch_dim dx = List.hd_exn (Tensor.shape dx)

(* constant scalar tensor *)
let f x = Const (Tensor.f x)

type 'a with_tensor_params = ?device:Device.t -> ?kind:Torch_core.Kind.packed -> 'a

let zeros ?device ?kind shape = Const (Tensor.zeros ?device ?kind shape)
let ones ?device ?kind ?scale shape = Const (Tensor.zeros ?device ?kind ?scale shape)
let rand ?device ?kind ?scale shape = Const (Tensor.rand ?device ?kind ?scale shape)
let randn ?device ?kind ?scale shape = Const (Tensor.randn ?device ?kind ?scale shape)
let zeros_like x = Const (Tensor.zeros_like (primal x))
let ones_like x = Const (Tensor.ones_like (primal x))
let rand_like x = Const (Tensor.rand_like (primal x))
let randn_like x = Const (Tensor.randn_like (primal x))

module Builder = struct
  type a = Tensor.t

  module Const = struct
    type unary_op = const t -> const t
    type binary_op = const t -> const t -> const t
    type unary_builder = a -> a
    type binary_builder = a -> a -> a

    let make_unary (f : unary_builder) : unary_op = function
      | Const x -> Const (f x)
      | _ -> assert false

    let make_binary (f : binary_builder) : binary_op =
      fun x y ->
      match x, y with
      | Const x, Const y -> Const (f x y)
      | _ -> assert false
  end

  module Any = struct
    type 'a unary_op = 'a any t -> 'a any t
    type ('a, 'b, 'c) binary_op = 'a any t -> 'b any t -> 'c any t

    type unary_builder =
      { f : a -> a
      ; df : f:a -> x:a -> a -> a
      }

    type binary_builder =
      { f : a -> a -> a
      ; dfx : f:a -> x:a -> y:a -> a -> a
      ; dfy : f:a -> x:a -> y:a -> a -> a
      ; dfxy : f:a -> x:a -> y:a -> a -> a -> a
      }

    let make_unary (b : unary_builder) : 'a any t -> 'a any t = function
      | Const x -> Const (b.f x)
      | Dual (x, dx) ->
        let f = b.f x in
        Dual (f, Explicit (const (b.df ~f ~x (tangent_tensor_of x dx))))

    let make_binary (b : binary_builder) : (_, _, _) binary_op =
      fun x y ->
      match x, y with
      | Const x, Const y -> Const (b.f x y)
      | Dual (x, dx), Const y ->
        let f = b.f x y in
        Dual (f, Explicit (const (b.dfx ~f ~x ~y (tangent_tensor_of x dx))))
      | Const x, Dual (y, dy) ->
        let f = b.f x y in
        Dual (f, Explicit (const (b.dfy ~f ~x ~y (tangent_tensor_of y dy))))
      | Dual (x, dx), Dual (y, dy) ->
        let f = b.f x y in
        Dual
          ( f
          , Explicit
              (const (b.dfxy ~f ~x ~y (tangent_tensor_of x dx) (tangent_tensor_of y dy)))
          )
  end
end

module Ops = struct
  (* reshape the size of x to size, and of each batch in dx to size. *)
  let view ~size =
    let f = Tensor.view ~size in
    let df ~f:_ ~x:_ dx =
      let size = batch_dim dx :: size in
      Tensor.view ~size dx
    in
    Builder.Any.{ f; df }

  (* reshape the size of x to size, and of each batch in dx to size. *)
  let reshape ~shape =
    let f = Tensor.reshape ~shape in
    let df ~f:_ ~x:_ dx =
      let shape = batch_dim dx :: shape in
      Tensor.reshape ~shape dx
    in
    Builder.Any.{ f; df }

  let permute ~dims =
    let f = Tensor.permute ~dims in
    let df ~f:_ ~x:_ dx =
      let tensor_dims = 0 :: List.map dims ~f:(fun i -> Int.(i + 1)) in
      Tensor.permute dx ~dims:tensor_dims
    in
    Builder.Any.{ f; df }

  (* reshape x with a dimension of size one inserted at dim *)
  let unsqueeze ~dim =
    let f = Tensor.unsqueeze ~dim in
    let df ~f:_ ~x:_ dx =
      let new_dim = if dim < 0 then dim else Int.(dim + 1) in
      Tensor.unsqueeze dx ~dim:new_dim
    in
    Builder.Any.{ f; df }

  (* reshape x with a dimension of size one removed at dim *)
  let squeeze ~dim =
    let f = Tensor.squeeze_dim ~dim in
    let df ~f:_ ~x:_ dx =
      let new_dim = if dim < 0 then dim else Int.(dim + 1) in
      Tensor.squeeze_dim dx ~dim:new_dim
    in
    Builder.Any.{ f; df }

  (* y = -x, dy = -dx *)
  let neg =
    let f = Tensor.neg in
    let df ~f:_ ~x:_ dx = Tensor.neg dx in
    Builder.Any.{ f; df }

  let trace =
    let f x =
      assert (
        match Tensor.shape x with
        | [ a; b ] when a = b -> true
        | _ -> false);
      Tensor.(reshape (trace x) ~shape:[ 1 ])
    in
    let df ~f:_ ~x:_ dx =
      Tensor.einsum [ dx ] ~equation:"qii->q" ~path:None
      |> Tensor.reshape ~shape:[ -1; 1 ]
    in
    Builder.Any.{ f; df }

  (* y = sin(x), dy = cos(x) dx *)
  let sin =
    let f = Tensor.sin in
    let df ~f:_ ~x dx = Tensor.(cos x * dx) in
    Builder.Any.{ f; df }

  (* y = cos(x), dy = -sin(x) dx *)
  let cos =
    let f = Tensor.cos in
    let df ~f:_ ~x dx = Tensor.(neg (sin x) * dx) in
    Builder.Any.{ f; df }

  (* y = x^2, dy = 2 x dx *)
  let sqr =
    let f = Tensor.square in
    let df ~f:_ ~x dx =
      let tmp = Tensor.(x * dx) in
      Tensor.mul_scalar_ tmp (Scalar.f 2.)
    in
    Builder.Any.{ f; df }

  (* y = x^{1/2}, dy = 1/2 x^{-1/2} dx *)
  let sqrt =
    let f = Tensor.sqrt in
    let df ~f:y ~x:_ dx =
      let tmp = Tensor.(dx / y) in
      Tensor.mul_scalar_ tmp (Scalar.f 0.5)
    in
    Builder.Any.{ f; df }

  (* y = log x, dy = dx/x *)
  let log =
    let f = Tensor.log in
    let df ~f:_ ~x dx = Tensor.(div dx x) in
    Builder.Any.{ f; df }

  (* y = exp(x), dy = exp(x) dx *)
  let exp =
    let f = Tensor.exp in
    let df ~f:y ~x:_ dx = Tensor.(dx * y) in
    Builder.Any.{ f; df }

  (* y = tanh(x), dy = (1 - tanh(x)^2) dx *)
  let tanh =
    let f = Tensor.tanh in
    let df ~f:y ~x:_ dx =
      let tmp = Tensor.(square y) in
      let tmp = Tensor.(neg_ tmp) in
      let tmp = Tensor.(add_scalar_ tmp Scalar.(f 1.)) in
      Tensor.mul_ tmp dx
    in
    Builder.Any.{ f; df }

  let relu =
    let f = Tensor.relu in
    let df ~f:_ ~x dx =
      let values = Tensor.(ones ~device:(device x) ~kind:(type_ x) []) in
      Tensor.mul dx (Tensor.heaviside ~values x)
    in
    Builder.Any.{ f; df }

  let sigmoid =
    let f = Tensor.sigmoid in
    let df ~f:y ~x:_ dx = Tensor.(mul dx (mul y (ones_like y - y))) in
    Builder.Any.{ f; df }

  let softplus =
    let f = Tensor.softplus in
    let df ~f:_ ~x dx = Tensor.(mul (sigmoid x) dx) in
    Builder.Any.{ f; df }

  (* like Torch's slice but not sharing data *)
  let[@warning "-16"] slice ?(step = 1) ?start ?end_ ~dim =
    let f = Tensor.slice_copy ~dim ~start ~end_ ~step in
    let df ~f:_ ~x:_ dx = Tensor.slice_copy ~dim:Int.(dim + 1) ~start ~end_ ~step dx in
    Builder.Any.{ f; df }

  (* y = sum of x_i, dy = sum of dx_i *)
  let sum =
    let f = Tensor.sum in
    let df ~f:_ ~x dx =
      (* make sure to preserve the batch dimension in dx *)
      let dim = List.mapi (Tensor.shape x) ~f:(fun i _ -> Int.(i + 1)) in
      Tensor.(sum_dim_intlist ~keepdim:false ~dtype:(type_ dx) ~dim:(Some dim) dx)
    in
    Builder.Any.{ f; df }

  let[@warning "-16"] sum_dim ?(keepdim = false) ~dim =
    let f x = Tensor.(sum_dim_intlist x ~dim:(Some dim) ~keepdim ~dtype:(type_ x)) in
    let df ~f:_ ~x:_ dx =
      (* make sure to preserve the batch dimension in dx *)
      let dim = List.map dim ~f:Int.succ in
      Tensor.(sum_dim_intlist ~dim:(Some dim) ~keepdim ~dtype:(type_ dx) dx)
    in
    Builder.Any.{ f; df }

  let mean =
    let f = Tensor.mean in
    let df ~f:_ ~x dx =
      let dim = List.mapi (Tensor.shape x) ~f:(fun i _ -> Int.(i + 1)) in
      Tensor.(mean_dim ~keepdim:false ~dtype:(type_ dx) ~dim:(Some dim) dx)
    in
    Builder.Any.{ f; df }

  let[@warning "-16"] mean_dim ?(keepdim = false) ~dim =
    let f x = Tensor.(mean_dim x ~dim:(Some dim) ~keepdim ~dtype:(type_ x)) in
    let df ~f:_ ~x:_ dx =
      (* make sure to preserve the batch dimension in dx *)
      let dim = List.map dim ~f:Int.succ in
      Tensor.(mean_dim ~dim:(Some dim) ~keepdim ~dtype:(type_ dx) dx)
    in
    Builder.Any.{ f; df }

  let ( + ) =
    let f = Tensor.( + ) in
    let dfx ~f:_ ~x:_ ~y:_ dx = dx in
    let dfy ~f:_ ~x:_ ~y:_ dy = dy in
    let dfxy ~f:_ ~x:_ ~y:_ dx dy = Tensor.(dx + dy) in
    Builder.Any.{ f; dfx; dfy; dfxy }

  let ( - ) =
    let f = Tensor.( - ) in
    let dfx ~f:_ ~x:_ ~y:_ dx = dx in
    let dfy ~f:_ ~x:_ ~y:_ dy = Tensor.neg dy in
    let dfxy ~f:_ ~x:_ ~y:_ dx dy = Tensor.(dx - dy) in
    Builder.Any.{ f; dfx; dfy; dfxy }

  (* z = x * y, dz = y dx + x dy *)
  let ( * ) =
    let f = Tensor.( * ) in
    let dfx ~f:_ ~x:_ ~y dx = Tensor.(mul dx y) in
    let dfy ~f:_ ~x ~y:_ dy = Tensor.(mul x dy) in
    let dfxy ~f:_ ~x ~y dx dy = Tensor.(add_ (mul x dy) (mul dx y)) in
    Builder.Any.{ f; dfx; dfy; dfxy }

  (* z = x / y, dz = 1/y dx - x/(y^2) dy *)
  let ( / ) =
    let f = Tensor.( / ) in
    let dfx ~f:_ ~x:_ ~y dx = Tensor.(div dx y) in
    let dfy ~f:z ~x:_ ~y dy = Tensor.(neg_ (div_ (dy * z) y)) in
    let dfxy ~f:_ ~x ~y dx dy =
      let y2 = Tensor.square y in
      Tensor.(div_ (sub_ (dx * y) (dy * x)) y2)
    in
    Builder.Any.{ f; dfx; dfy; dfxy }

  (* x = x + z *)
  let ( $+ ) z =
    let f x = Tensor.add_scalar x (Scalar.f z) in
    let df ~f:_ ~x:_ dx = dx in
    Builder.Any.{ f; df }

  (* x = x *z *)
  let ( $* ) z =
    let f x = Tensor.mul_scalar x (Scalar.f z) in
    let df ~f:_ ~x:_ dx = Tensor.(mul_scalar dx Scalar.(f z)) in
    Builder.Any.{ f; df }

  let ( $/ ) z =
    let f x = Tensor.(mul_scalar (reciprocal x) (Scalar.f z)) in
    let df ~f:_ ~x dx =
      let x2 = Tensor.square x in
      Tensor.(neg_ (div_ (mul_scalar dx (Scalar.f z)) x2))
    in
    Builder.Any.{ f; df }

  (* z = xy, dz = dx y + x dy *)
  let ( *@ ) =
    let f x y =
      if List.length (Tensor.shape x) < 2 && List.length (Tensor.shape y) < 2
      then failwith "( *@ ) does not operate on two vectors";
      Tensor.matmul x y
    in
    let dfx ~f:_ ~x:_ ~y dx = Tensor.(matmul dx y) in
    let dfy ~f:_ ~x ~y:_ dy = Tensor.(matmul x dy) in
    let dfxy ~f:_ ~x ~y dx dy = Tensor.(add_ (matmul dx y) (matmul x dy)) in
    Builder.Any.{ f; dfx; dfy; dfxy }

  let einsum operands return =
    let equation = String.concat ~sep:"," (List.map ~f:snd operands) ^ "->" ^ return in
    Tensor.einsum ~equation (List.map ~f:fst operands) ~path:None
end

(* Type-safe operations on primals *)
module Primal = struct
  let unary_of_any (b : Builder.Any.unary_builder) = b.f
  let binary_of_any (b : Builder.Any.binary_builder) = b.f
  let view ~size = Builder.Const.make_unary (unary_of_any (Ops.view ~size))
  let reshape ~shape = Builder.Const.make_unary (unary_of_any (Ops.reshape ~shape))
  let permute ~dims = Builder.Const.make_unary (unary_of_any (Ops.permute ~dims))
  let squeeze ~dim = Builder.Const.make_unary (unary_of_any (Ops.squeeze ~dim))
  let unsqueeze ~dim = Builder.Const.make_unary (unary_of_any (Ops.unsqueeze ~dim))
  let neg = Builder.Const.make_unary (unary_of_any Ops.neg)
  let trace = Builder.Const.make_unary (unary_of_any Ops.trace)
  let sin = Builder.Const.make_unary (unary_of_any Ops.sin)
  let cos = Builder.Const.make_unary (unary_of_any Ops.cos)
  let sqr = Builder.Const.make_unary (unary_of_any Ops.sqr)
  let sqrt = Builder.Const.make_unary (unary_of_any Ops.sqrt)
  let log = Builder.Const.make_unary (unary_of_any Ops.log)
  let exp = Builder.Const.make_unary (unary_of_any Ops.exp)
  let tanh = Builder.Const.make_unary (unary_of_any Ops.tanh)
  let relu = Builder.Const.make_unary (unary_of_any Ops.relu)
  let sigmoid = Builder.Const.make_unary (unary_of_any Ops.sigmoid)
  let softplus = Builder.Const.make_unary (unary_of_any Ops.softplus)

  let slice ?start ?end_ ?step ~dim =
    Builder.Const.make_unary (unary_of_any (Ops.slice ?start ?end_ ?step ~dim))

  let sum = Builder.Const.make_unary (unary_of_any Ops.sum)

  let sum_dim ?keepdim ~dim =
    Builder.Const.make_unary (unary_of_any (Ops.sum_dim ?keepdim ~dim))

  let mean = Builder.Const.make_unary (unary_of_any Ops.mean)

  let mean_dim ?keepdim ~dim =
    Builder.Const.make_unary (unary_of_any (Ops.mean_dim ?keepdim ~dim))

  let ( + ) = Builder.Const.make_binary (binary_of_any Ops.(( + )))
  let ( - ) = Builder.Const.make_binary (binary_of_any Ops.(( - )))
  let ( * ) = Builder.Const.make_binary (binary_of_any Ops.(( * )))
  let ( / ) = Builder.Const.make_binary (binary_of_any Ops.(( / )))
  let ( $+ ) z = Builder.Const.make_unary (unary_of_any Ops.(( $+ ) z))
  let ( $* ) z = Builder.Const.make_unary (unary_of_any Ops.(( $* ) z))
  let ( $/ ) z = Builder.Const.make_unary (unary_of_any Ops.(( $/ ) z))
  let ( *@ ) = Builder.Const.make_binary (binary_of_any Ops.(( *@ )))

  let einsum (operands : (_ t * string) list) return =
    let tangent_id = 'x' in
    assert (not (String.contains return tangent_id));
    assert (
      List.fold operands ~init:true ~f:(fun accu (_, eq) ->
        accu && not (String.contains eq tangent_id)));
    let y = Ops.einsum (List.map operands ~f:(fun (x, eq) -> primal x, eq)) return in
    Const y
end

let numel x = x |> primal |> Tensor.shape |> List.fold ~init:0 ~f:Int.( + ) |> Int.max 1
let view ~size = Builder.Any.make_unary (Ops.view ~size)
let reshape ~shape = Builder.Any.make_unary (Ops.reshape ~shape)
let permute ~dims = Builder.Any.make_unary (Ops.permute ~dims)
let squeeze ~dim = Builder.Any.make_unary (Ops.squeeze ~dim)
let unsqueeze ~dim = Builder.Any.make_unary (Ops.unsqueeze ~dim)
let neg x = Builder.Any.make_unary Ops.neg x
let trace x = Builder.Any.make_unary Ops.trace x
let sin x = Builder.Any.make_unary Ops.sin x
let cos x = Builder.Any.make_unary Ops.cos x
let sqr x = Builder.Any.make_unary Ops.sqr x
let sqrt x = Builder.Any.make_unary Ops.sqrt x
let log x = Builder.Any.make_unary Ops.log x
let exp x = Builder.Any.make_unary Ops.exp x
let tanh x = Builder.Any.make_unary Ops.tanh x
let relu x = Builder.Any.make_unary Ops.relu x
let sigmoid x = Builder.Any.make_unary Ops.sigmoid x
let softplus x = Builder.Any.make_unary Ops.softplus x

let slice ?start ?end_ ?step ~dim x =
  Builder.Any.make_unary (Ops.slice ?start ?end_ ?step ~dim) x

let sum x = Builder.Any.make_unary Ops.sum x
let sum_dim ?keepdim ~dim x = Builder.Any.make_unary (Ops.sum_dim ?keepdim ~dim) x
let mean x = Builder.Any.make_unary Ops.mean x
let mean_dim ?keepdim ~dim x = Builder.Any.make_unary (Ops.mean_dim ?keepdim ~dim) x
let ( + ) x = Builder.Any.make_binary Ops.(( + )) x
let ( - ) x = Builder.Any.make_binary Ops.(( - )) x
let ( * ) x = Builder.Any.make_binary Ops.(( * )) x
let ( / ) x = Builder.Any.make_binary Ops.(( / )) x
let ( $+ ) z = Builder.Any.make_unary Ops.(( $+ ) z)
let ( $* ) z = Builder.Any.make_unary Ops.(( $* ) z)
let ( $/ ) z = Builder.Any.make_unary Ops.(( $/ ) z)
let ( *@ ) x = Builder.Any.make_binary Ops.(( *@ )) x

(* einsum [ a, "ij"; b, "jk"; c, "ki" ] "ii" *)

let einsum (operands : (_ t * string) list) return =
  let tangent_id = 'x' in
  assert (not (String.contains return tangent_id));
  assert (
    List.fold operands ~init:true ~f:(fun accu (_, eq) ->
      accu && not (String.contains eq tangent_id)));
  let y = Ops.einsum (List.map operands ~f:(fun (x, eq) -> primal x, eq)) return in
  let dy =
    List.foldi operands ~init:None ~f:(fun i accu (op, eq) ->
      match op with
      | Const _ -> accu
      | Dual (x, dx) ->
        let ops =
          List.mapi operands ~f:(fun j (op', eq') ->
            if i = j
            then tangent_tensor_of x dx, String.of_char tangent_id ^ eq
            else primal op', eq')
        in
        let return = String.of_char tangent_id ^ return in
        let tangent_contrib = Ops.einsum ops return in
        (match accu with
         | None -> Some tangent_contrib
         | Some a -> Some Tensor.(a + tangent_contrib)))
  in
  match dy with
  | None -> Const y
  | Some dy -> Dual (y, Explicit (const dy))
