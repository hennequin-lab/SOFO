open Base
open Torch

(*
   We will systematically work with the assumption that for a primal value of shape [n1; n2; ... ],
   the associated tangents have one more dimension, corresponding to tangent batch: [K; n1; n2; ... ]
*)

type _ tangent_kind =
  | Explicit : Tensor.t -> Tensor.t tangent_kind
  | On_demand : (Device.t -> Tensor.t) -> (Device.t -> Tensor.t) tangent_kind

type tangent = Tangent : 'a tangent_kind -> tangent

type _ t =
  | Const : Tensor.t -> [> `Const ] t
  | Dual : Tensor.t * tangent -> [> `Dual ] t

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

let to_dual_exn : [ `Const | `Dual ] t -> [> `Dual ] t = function
  | Const _ -> raise Not_dual
  | Dual (_, _) as y -> y

let const x : [> `Const ] t = Const x

let dual ~dx (Const x : [ `Const ] t) : [> `Dual ] t =
  assert_right_device x dx;
  assert_right_shape x dx;
  Dual (x, Tangent (Explicit dx))

let dual_lazy ~dx (Const x : [ `Const ] t) : [> `Dual ] t =
  let dx device =
    let dx = dx device in
    assert_right_shape x dx;
    assert_right_device x dx;
    dx
  in
  Dual (x, Tangent (On_demand dx))

let tangent_tensor_of (Dual (x, Tangent dx) : [ `Dual ] t) : Tensor.t =
  match dx with
  | Explicit dx -> dx
  | On_demand dx -> dx (Tensor.device x)

let primal : [< `Const | `Dual ] t -> Tensor.t = function
  | Const x -> x
  | Dual (x, _) -> x

let tangent : [< `Dual ] t -> Tensor.t = function
  | Dual (_, _) as z -> tangent_tensor_of z

let force_lazy_tangent : [< `Dual ] t -> [ `Dual ] t = function
  | Dual (x, _) as z -> Dual (x, Tangent (Explicit (tangent_tensor_of z)))

let print s = Stdio.print_endline (Sexp.to_string_hum s)
let shape x = x |> primal |> Tensor.shape
let device x = x |> primal |> Tensor.device
let kind x = x |> primal |> Tensor.type_

(* int list starting from 1 and ending at the last dim of a *)
let all_dims_but_first a = List.range 1 (List.length (Tensor.shape a))

let append_batch ~tangent shape =
  let b = List.hd_exn (Tensor.shape tangent) in
  b :: shape

(* constant scalar tensor *)
let f x = Const (Tensor.f x)

module Builder = struct
  type a = Tensor.t

  module Const = struct
    type 'a unary_op = [ `Const ] t -> ([> `Const ] as 'a) t
    type 'a binary_op = [ `Const ] t -> [ `Const ] t -> ([> `Const ] as 'a) t
    type unary_builder = a -> a
    type binary_builder = a -> a -> a

    let make_unary (f : unary_builder) : 'a unary_op = function
      | Const x -> Const (f x)

    let make_binary (f : binary_builder) : 'a binary_op =
      fun x y ->
      match x, y with
      | Const x, Const y -> Const (f x y)
  end

  module Any = struct
    type 'a unary_op = [ `Const | `Dual ] t -> ([> `Const | `Dual ] as 'a) t

    type 'a binary_op =
      [ `Const | `Dual ] t -> [ `Const | `Dual ] t -> ([> `Const | `Dual ] as 'a) t

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

    let make_unary (b : unary_builder) : 'a unary_op = function
      | Const x -> Const (b.f x)
      | Dual (x, _) as xx ->
        let f = b.f x in
        Dual (f, Tangent (Explicit (b.df ~f ~x (tangent_tensor_of xx))))

    let make_binary (b : binary_builder) : 'a binary_op =
      fun x y ->
      match x, y with
      | Const x, Const y -> Const (b.f x y)
      | (Dual (x, _) as xx), Const y ->
        let f = b.f x y in
        Dual (f, Tangent (Explicit (b.dfx ~f ~x ~y (tangent_tensor_of xx))))
      | Const x, (Dual (y, _) as yy) ->
        let f = b.f x y in
        Dual (f, Tangent (Explicit (b.dfy ~f ~x ~y (tangent_tensor_of yy))))
      | (Dual (x, _) as xx), (Dual (y, _) as yy) ->
        let f = b.f x y in
        Dual
          ( f
          , Tangent
              (Explicit (b.dfxy ~f ~x ~y (tangent_tensor_of xx) (tangent_tensor_of yy)))
          )
  end
end

module Ops = struct
  (* reshape the size of x to size, and of each batch in dx to size. *)
  let view ~size =
    let f = Tensor.view ~size in
    let df ~f:_ ~x:_ dx =
      let size = append_batch ~tangent:dx size in
      Tensor.view ~size dx
    in
    Builder.Any.{ f; df }

  (* reshape the size of x to size, and of each batch in dx to size. *)
  let reshape ~shape =
    let f = Tensor.reshape ~shape in
    let df ~f:_ ~x:_ dx =
      let shape = append_batch ~tangent:dx shape in
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
  let slice ~dim ~start ~end_ ~step =
    let f = Tensor.slice_copy ~dim ~start ~end_ ~step in
    let df ~f:_ ~x:_ dx = Tensor.slice_copy ~dim:Int.(dim + 1) ~start ~end_ ~step dx in
    Builder.Any.{ f; df }

  (* y = sum of x_i, dy = sum of dx_i *)
  let sum =
    let f = Tensor.sum in
    let df ~f:_ ~x dx =
      (* if x has shape [m, n, k], dim is the int list [1, 2, 3]
         whereas dx has shape [B, m, n, k] *)
      let dim = List.mapi (Tensor.shape x) ~f:(fun i _ -> Int.(i + 1)) in
      (* retain the batch dimension, sum over the rest *)
      Tensor.(sum_dim_intlist ~keepdim:false ~dtype:(type_ dx) ~dim:(Some dim) dx)
    in
    Builder.Any.{ f; df }

  let sum_dim ~dim ~keepdim =
    let f x = Tensor.(sum_dim_intlist x ~dim ~keepdim ~dtype:(type_ x)) in
    let df ~f:_ ~x dx =
      (* make sure to preserve the batch dimension in dx *)
      let dim =
        match dim with
        | Some d -> List.map d ~f:Int.succ
        | None -> List.mapi (Tensor.shape x) ~f:(fun i _ -> Int.(i + 1))
      in
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

  let mean_dim ~dim ~keepdim =
    let f x = Tensor.(mean_dim x ~dim ~keepdim ~dtype:(type_ x)) in
    let df ~f:_ ~x dx =
      (* make sure to preserve the batch dimension in dx *)
      let dim =
        match dim with
        | Some d -> List.map d ~f:Int.succ
        | None -> List.mapi (Tensor.shape x) ~f:(fun i _ -> Int.(i + 1))
      in
      Tensor.(mean_dim ~dim:(Some dim) ~keepdim ~dtype:(type_ dx) dx)
    in
    Builder.Any.{ f; df }

  let ( + ) =
    let f = Tensor.( + ) in
    let dfx ~f:_ ~x:_ ~y:_ dx = dx in
    let dfy ~f:_ ~x:_ ~y:_ dy = dy in
    let dfxy ~f:_ ~x:_ ~y:_ dx dy = Tensor.(dx + dy) in
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

  let slice ~dim ~start ~end_ ~step =
    Builder.Const.make_unary (unary_of_any (Ops.slice ~dim ~start ~end_ ~step))

  let sum = Builder.Const.make_unary (unary_of_any Ops.sum)

  let sum_dim ~dim ~keepdim =
    Builder.Const.make_unary (unary_of_any (Ops.sum_dim ~dim ~keepdim))

  let mean = Builder.Const.make_unary (unary_of_any Ops.mean)

  let mean_dim ~dim ~keepdim =
    Builder.Const.make_unary (unary_of_any (Ops.mean_dim ~dim ~keepdim))

  let ( + ) = Builder.Const.make_binary (binary_of_any Ops.(( + )))

  let einsum (operands : ([ `Const ] t * string) list) return =
    let tangent_id = 'x' in
    assert (not (String.contains return tangent_id));
    assert (
      List.fold operands ~init:true ~f:(fun accu (_, eq) ->
        accu && not (String.contains eq tangent_id)));
    let result_primal =
      Ops.einsum (List.map operands ~f:(fun (Const x, eq) -> x, eq)) return
    in
    Const result_primal
end

let view ~size = Builder.Any.make_unary (Ops.view ~size)
let reshape ~shape = Builder.Any.make_unary (Ops.reshape ~shape)
let permute ~dims = Builder.Any.make_unary (Ops.permute ~dims)
let squeeze ~dim = Builder.Any.make_unary (Ops.squeeze ~dim)
let unsqueeze ~dim = Builder.Any.make_unary (Ops.unsqueeze ~dim)
let neg = Builder.Any.make_unary Ops.neg
let trace = Builder.Any.make_unary Ops.trace
let sin = Builder.Any.make_unary Ops.sin
let cos = Builder.Any.make_unary Ops.cos
let sqr = Builder.Any.make_unary Ops.sqr
let sqrt = Builder.Any.make_unary Ops.sqrt
let log = Builder.Any.make_unary Ops.log
let exp = Builder.Any.make_unary Ops.exp
let tanh = Builder.Any.make_unary Ops.tanh
let relu = Builder.Any.make_unary Ops.relu
let sigmoid = Builder.Any.make_unary Ops.sigmoid
let softplus = Builder.Any.make_unary Ops.softplus

let slice ~dim ~start ~end_ ~step =
  Builder.Any.make_unary (Ops.slice ~dim ~start ~end_ ~step)

let sum = Builder.Any.make_unary Ops.sum
let sum_dim ~dim ~keepdim = Builder.Any.make_unary (Ops.sum_dim ~dim ~keepdim)
let mean = Builder.Any.make_unary Ops.mean
let mean_dim ~dim ~keepdim = Builder.Any.make_unary (Ops.mean_dim ~dim ~keepdim)
let ( + ) = Builder.Any.make_binary Ops.(( + ))

(* einsum [ a, "ij"; b, "jk"; c, "ki" ] "ii" *)

let einsum (operands : ([ `Const | `Dual ] t * string) list) return =
  let tangent_id = 'x' in
  assert (not (String.contains return tangent_id));
  assert (
    List.fold operands ~init:true ~f:(fun accu (_, eq) ->
      accu && not (String.contains eq tangent_id)));
  let result_primal =
    Ops.einsum (List.map operands ~f:(fun (x, eq) -> primal x, eq)) return
  in
  let result_tangent =
    List.foldi operands ~init:None ~f:(fun i accu (op, eq) ->
      match op with
      | Const _ -> accu
      | Dual (_, _) as opp ->
        let ops =
          List.mapi operands ~f:(fun j (op', eq') ->
            if i = j
            then tangent_tensor_of opp, String.of_char tangent_id ^ eq
            else primal op', eq')
        in
        let return = String.of_char tangent_id ^ return in
        let tangent_contrib = Ops.einsum ops return in
        (match accu with
         | None -> Some tangent_contrib
         | Some a -> Some Tensor.(a + tangent_contrib)))
  in
  match result_tangent with
  | Some dr -> Dual (result_primal, Tangent (Explicit dr))
  | None -> Const result_primal
