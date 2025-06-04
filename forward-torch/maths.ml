open Base
open Torch

(*
   We will systematically work with the assumption that for a primal value of shape [n1; n2; ... ],
   the associated tangents have one more dimension, corresponding to tangent batch: [K; n1; n2; ... ]
*)

type explicit
type on_demand

type 'a num =
  | C : Tensor.t -> [ `const ] num
  | D : Tensor.t * _ tangent -> [ `dual ] num

and _ tangent =
  | Explicit : Tensor.t -> explicit tangent
  | On_demand : (Device.t -> Tensor.t) -> on_demand tangent

(* existential wrapper *)
type +_ t = E : 'a num -> _ t [@@unboxed]

exception Wrong_shape of int list * int list
exception Wrong_device of Device.t * Device.t
exception Check_grad_failed
exception Not_const
exception Not_dual

let _print s = Stdio.print_endline (Sexp.to_string_hum s)

let tangent_tensor_of : type a. Tensor.t -> a tangent -> Tensor.t =
  fun x v ->
  match v with
  | Explicit v -> v
  | On_demand v -> v (Tensor.device x)

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

let any x = (x :> [ `const | `dual ] t)
let of_tensor x : [ `const ] t = E (C x)

let to_tensor : [< `const | `dual ] t -> Tensor.t = function
  | E (C x) -> x
  | E (D (x, _)) -> x

let of_array ?device ~shape x =
  x |> Tensor.of_float1 ?device |> Tensor.reshape ~shape |> of_tensor

let to_float_exn x = x |> to_tensor |> Tensor.to_float0_exn

let const : [< `const | `dual ] t -> [ `const ] t = function
  | E (C x) -> E (C x)
  | E (D (x, _)) -> E (C x)

let shape x = x |> to_tensor |> Tensor.shape
let device x = x |> to_tensor |> Tensor.device
let kind x = x |> to_tensor |> Tensor.type_

let numel x =
  x |> to_tensor |> Tensor.shape |> List.fold ~init:0 ~f:Int.( + ) |> Int.max 1

let tangent_exn (E x : [< `const | `dual ] t) : [ `const ] t =
  match x with
  | D (x, dx) -> E (C (tangent_tensor_of x dx))
  | _ -> raise Not_dual

let dual : tangent:[ `const ] t -> [ `const ] t -> [ `dual ] t =
  fun ~tangent (E x) ->
  match x with
  | C x ->
    let dxp = to_tensor tangent in
    assert_right_device x dxp;
    assert_right_shape x dxp;
    E (D (x, Explicit dxp))
  | _ -> raise Not_const

let dual_on_demand : tangent:(Device.t -> [ `const ] t) -> [ `const ] t -> [ `dual ] t =
  fun ~tangent (E x) ->
  match x with
  | C x ->
    let dx_wrap device =
      let dx = tangent device in
      let dxp = to_tensor dx in
      assert_right_shape x dxp;
      assert_right_device x dxp;
      dxp
    in
    E (D (x, On_demand dx_wrap))
  | _ -> raise Not_const

let _all_dims_but_first a = List.range 1 (List.length (Tensor.shape a))
let batch_dim dx = List.hd_exn (Tensor.shape dx)
let first_dim x = List.hd_exn (Tensor.shape (to_tensor x))

(* constant scalar tensor *)
let f x : [ `const ] t = E (C (Tensor.f x))

type 'a with_tensor_params = ?device:Device.t -> ?kind:Torch_core.Kind.packed -> 'a

let zeros ?device ?kind shape : [ `const ] t = E (C (Tensor.zeros ?device ?kind shape))

let ones ?device ?kind ?scale shape : [ `const ] t =
  E (C (Tensor.ones ?device ?kind ?scale shape))

let rand ?device ?kind ?scale shape : [ `const ] t =
  E (C (Tensor.rand ?device ?kind ?scale shape))

let randn ?device ?kind ?scale shape : [ `const ] t =
  E (C (Tensor.randn ?device ?kind ?scale shape))

let zeros_like x : [ `const ] t = E (C (Tensor.zeros_like (to_tensor x)))

let zeros_like_k ~k x =
  let x = to_tensor x in
  let x = Tensor.(broadcast_to x ~size:(k :: shape x)) in
  E (C (Tensor.zeros_like x))

let ones_like x : [ `const ] t = E (C (Tensor.ones_like (to_tensor x)))
let rand_like x : [ `const ] t = E (C (Tensor.rand_like (to_tensor x)))
let randn_like x : [ `const ] t = E (C (Tensor.randn_like (to_tensor x)))

let randn_like_k ~k x =
  let x = to_tensor x in
  let x = Tensor.(broadcast_to x ~size:(k :: shape x)) in
  E (C (Tensor.randn_like x))

type unary_info =
  { f : Tensor.t -> Tensor.t
  ; df : f:Tensor.t -> x:Tensor.t -> dx:Tensor.t -> Tensor.t
  }

type binary_info =
  { f : Tensor.t -> Tensor.t -> Tensor.t
  ; dfx : f:Tensor.t -> x:Tensor.t -> y:Tensor.t -> dx:Tensor.t -> Tensor.t
  ; dfy : f:Tensor.t -> x:Tensor.t -> y:Tensor.t -> dy:Tensor.t -> Tensor.t
  ; dfxy :
      f:Tensor.t -> x:Tensor.t -> y:Tensor.t -> dx:Tensor.t -> dy:Tensor.t -> Tensor.t
  }

module Ops = struct
  (* reshape the size of x to size, and of each batch in dx to size. *)
  let view ~size =
    let f = Tensor.view ~size in
    let df ~f:_ ~x:_ ~dx =
      let size = batch_dim dx :: size in
      Tensor.view ~size dx
    in
    { f; df }

  (* reshape the size of x to size, and of each batch in dx to size. *)
  let reshape ~shape =
    let f = Tensor.reshape ~shape in
    let df ~f:_ ~x:_ ~dx =
      let shape = batch_dim dx :: shape in
      Tensor.reshape ~shape dx
    in
    { f; df }

  let permute ~dims =
    let f = Tensor.permute ~dims in
    let df ~f:_ ~x:_ ~dx =
      let tensor_dims = 0 :: List.map dims ~f:(fun i -> Int.(i + 1)) in
      Tensor.permute dx ~dims:tensor_dims
    in
    { f; df }

  (* reshape x with a dimension of size one inserted at dim *)
  let unsqueeze ~dim =
    let f = Tensor.unsqueeze ~dim in
    let df ~f:_ ~x:_ ~dx =
      let new_dim = if dim < 0 then dim else Int.(dim + 1) in
      Tensor.unsqueeze dx ~dim:new_dim
    in
    { f; df }

  (* reshape x with a dimension of size one removed at dim *)
  let squeeze ~dim =
    let f = Tensor.squeeze_dim ~dim in
    let df ~f:_ ~x:_ ~dx =
      let new_dim = if dim < 0 then dim else Int.(dim + 1) in
      Tensor.squeeze_dim dx ~dim:new_dim
    in
    { f; df }

  let min_binary_swaps permutation =
    try
      let perm = Array.of_list permutation in
      let n = Array.length perm in
      let swaps = ref [] in
      for i = 0 to n - 1 do
        if perm.(i) <> i
        then (
          let j = ref (i + 1) in
          while !j < n && perm.(!j) <> i do
            Int.incr j
          done;
          let temp = perm.(i) in
          perm.(i) <- perm.(!j);
          perm.(!j) <- temp;
          swaps := (i, !j) :: !swaps)
      done;
      List.rev !swaps
    with
    | _ -> failwith "Improper specification of axis permutation"

  let[@warning "-16"] transpose ?dims =
    let f x =
      match dims with
      | None ->
        assert (Int.(List.length (Tensor.shape x) = 2));
        Tensor.transpose ~dim0:0 ~dim1:1 x
      | Some dims ->
        assert (Int.(List.length (Tensor.shape x) = List.length dims));
        List.fold (min_binary_swaps dims) ~init:x ~f:(fun accu (dim0, dim1) ->
          Tensor.transpose accu ~dim0 ~dim1)
    in
    let df ~f:_ ~x:_ ~dx =
      match dims with
      | None ->
        assert (Int.(List.length (Tensor.shape dx) = 3));
        Tensor.transpose ~dim0:1 ~dim1:2 dx
      | Some dims ->
        assert (Int.(List.length (Tensor.shape dx) = 1 + List.length dims));
        List.fold (min_binary_swaps dims) ~init:dx ~f:(fun accu (dim0, dim1) ->
          Tensor.transpose accu ~dim0:Int.(dim0 + 1) ~dim1:Int.(dim1 + 1))
    in
    { f; df }

  (* y = -x, dy = -dx *)
  let neg =
    let f = Tensor.neg in
    let df ~f:_ ~x:_ ~dx = Tensor.neg dx in
    { f; df }

  let trace =
    let f x =
      assert (
        match Tensor.shape x with
        | [ a; b ] when a = b -> true
        | _ -> false);
      Tensor.(reshape (trace x) ~shape:[ 1 ])
    in
    let df ~f:_ ~x:_ ~dx =
      Tensor.einsum [ dx ] ~equation:"qii->q" ~path:None
      |> Tensor.reshape ~shape:[ -1; 1 ]
    in
    { f; df }

  (* y = sin(x), dy = cos(x) dx *)
  let sin =
    let f = Tensor.sin in
    let df ~f:_ ~x ~dx = Tensor.(cos x * dx) in
    { f; df }

  (* y = cos(x), dy = -sin(x) dx *)
  let cos =
    let f = Tensor.cos in
    let df ~f:_ ~x ~dx = Tensor.(neg (sin x) * dx) in
    { f; df }

  (* y = x^2, dy = 2 x dx *)
  let sqr =
    let f = Tensor.square in
    let df ~f:_ ~x ~dx =
      let tmp = Tensor.(x * dx) in
      Tensor.mul_scalar_ tmp (Scalar.f 2.)
    in
    { f; df }

  (* y = x^{1/2}, dy = 1/2 x^{-1/2} dx *)
  let sqrt =
    let f = Tensor.sqrt in
    let df ~f:y ~x:_ ~dx =
      let tmp = Tensor.(dx / y) in
      Tensor.mul_scalar_ tmp (Scalar.f 0.5)
    in
    { f; df }

  (* y = log x, dy = dx/x *)
  let log =
    let f = Tensor.log in
    let df ~f:_ ~x ~dx = Tensor.(div dx x) in
    { f; df }

  (* y = exp(x), dy = exp(x) dx *)
  let exp =
    let f = Tensor.exp in
    let df ~f:y ~x:_ ~dx = Tensor.(dx * y) in
    { f; df }

  (* y = tanh(x), dy = (1 - tanh(x)^2) dx *)
  let tanh =
    let f = Tensor.tanh in
    let df ~f:y ~x:_ ~dx =
      let tmp = Tensor.(square y) in
      let tmp = Tensor.(neg_ tmp) in
      let tmp = Tensor.(add_scalar_ tmp Scalar.(f 1.)) in
      Tensor.mul_ tmp dx
    in
    { f; df }

  let relu =
    let f = Tensor.relu in
    let df ~f:_ ~x ~dx =
      let values = Tensor.(ones ~device:(device x) ~kind:(type_ x) []) in
      Tensor.mul dx (Tensor.heaviside ~values x)
    in
    { f; df }

  let sigmoid =
    let f = Tensor.sigmoid in
    let df ~f:y ~x:_ ~dx = Tensor.(mul dx (mul y (ones_like y - y))) in
    { f; df }

  let softplus =
    let f = Tensor.softplus in
    let df ~f:_ ~x ~dx = Tensor.(mul (sigmoid x) dx) in
    { f; df }

  (* like Torch's slice but not sharing data *)
  let[@warning "-16"] slice ?start ?end_ ?(step = 1) ~dim =
    let f = Tensor.slice_copy ~dim ~start ~end_ ~step in
    let df ~f:_ ~x:_ ~dx = Tensor.slice_copy ~dim:Int.(dim + 1) ~start ~end_ ~step dx in
    { f; df }

  (* y = sum of x_i, dy = sum of dx_i *)
  let sum =
    let f = Tensor.sum in
    let df ~f:_ ~x ~dx =
      (* make sure to preserve the batch dimension in dx *)
      let dim = List.mapi (Tensor.shape x) ~f:(fun i _ -> Int.(i + 1)) in
      Tensor.(sum_dim_intlist ~keepdim:false ~dtype:(type_ dx) ~dim:(Some dim) dx)
    in
    { f; df }

  let[@warning "-16"] sum_dim ?(keepdim = false) ~dim =
    let f x = Tensor.(sum_dim_intlist x ~dim:(Some dim) ~keepdim ~dtype:(type_ x)) in
    let df ~f:_ ~x:_ ~dx =
      (* make sure to preserve the batch dimension in dx *)
      let dim = List.map dim ~f:Int.succ in
      Tensor.(sum_dim_intlist ~dim:(Some dim) ~keepdim ~dtype:(type_ dx) dx)
    in
    { f; df }

  let mean =
    let f = Tensor.mean in
    let df ~f:_ ~x ~dx =
      let dim = List.mapi (Tensor.shape x) ~f:(fun i _ -> Int.(i + 1)) in
      Tensor.(mean_dim ~keepdim:false ~dtype:(type_ dx) ~dim:(Some dim) dx)
    in
    { f; df }

  let[@warning "-16"] mean_dim ?(keepdim = false) ~dim =
    let f x = Tensor.(mean_dim x ~dim:(Some dim) ~keepdim ~dtype:(type_ x)) in
    let df ~f:_ ~x:_ ~dx =
      (* make sure to preserve the batch dimension in dx *)
      let dim = List.map dim ~f:Int.succ in
      Tensor.(mean_dim ~dim:(Some dim) ~keepdim ~dtype:(type_ dx) dx)
    in
    { f; df }

  let[@warning "-16"] logsumexp ?(keepdim = false) ~dim =
    let f x = Tensor.logsumexp x ~dim ~keepdim in
    let df ~f:y ~x ~dx =
      let s =
        if keepdim
        then Tensor.shape y
        else
          List.mapi (Tensor.shape x) ~f:(fun i di ->
            if List.mem ~equal:Int.( = ) dim i then 1 else di)
      in
      let tmp = Tensor.(x - view ~size:s y) in
      let tmp = Tensor.exp_ tmp in
      let tmp = Tensor.mul dx tmp in
      (* make sure to preserve the batch dimension in dx *)
      let dim = List.map dim ~f:Int.succ in
      Tensor.(sum_dim_intlist ~dim:(Some dim) ~keepdim ~dtype:(type_ x) tmp)
    in
    { f; df }

  let ( + ) =
    let f = Tensor.( + ) in
    let dfx ~f:_ ~x:_ ~y:_ ~dx = dx in
    let dfy ~f:_ ~x:_ ~y:_ ~dy = dy in
    let dfxy ~f:_ ~x:_ ~y:_ ~dx ~dy = Tensor.(dx + dy) in
    { f; dfx; dfy; dfxy }

  let ( - ) =
    let f = Tensor.( - ) in
    let dfx ~f:_ ~x:_ ~y:_ ~dx = dx in
    let dfy ~f:_ ~x:_ ~y:_ ~dy = Tensor.neg dy in
    let dfxy ~f:_ ~x:_ ~y:_ ~dx ~dy = Tensor.(dx - dy) in
    { f; dfx; dfy; dfxy }

  (* z = x * y, dz = y dx + x dy *)
  let ( * ) =
    let f = Tensor.( * ) in
    let dfx ~f:_ ~x:_ ~y ~dx = Tensor.(mul dx y) in
    let dfy ~f:_ ~x ~y:_ ~dy = Tensor.(mul x dy) in
    let dfxy ~f:_ ~x ~y ~dx ~dy = Tensor.(add_ (mul x dy) (mul dx y)) in
    { f; dfx; dfy; dfxy }

  (* z = x / y, dz = 1/y dx - x/(y^2) dy *)
  let ( / ) =
    let f = Tensor.( / ) in
    let dfx ~f:_ ~x:_ ~y ~dx = Tensor.(div dx y) in
    let dfy ~f:z ~x:_ ~y ~dy = Tensor.(neg_ (div_ (dy * z) y)) in
    let dfxy ~f:_ ~x ~y ~dx ~dy =
      let y2 = Tensor.square y in
      Tensor.(div_ (sub_ (dx * y) (dy * x)) y2)
    in
    { f; dfx; dfy; dfxy }

  (* x = x + z *)
  let ( $+ ) z =
    let f x = Tensor.add_scalar x (Scalar.f z) in
    let df ~f:_ ~x:_ ~dx = dx in
    { f; df }

  (* x = x *z *)
  let ( $* ) z =
    let f x = Tensor.mul_scalar x (Scalar.f z) in
    let df ~f:_ ~x:_ ~dx = Tensor.(mul_scalar dx Scalar.(f z)) in
    { f; df }

  let ( $/ ) z =
    let f x = Tensor.(mul_scalar (reciprocal x) (Scalar.f z)) in
    let df ~f:_ ~x ~dx =
      let x2 = Tensor.square x in
      Tensor.(neg_ (div_ (mul_scalar dx (Scalar.f z)) x2))
    in
    { f; df }

  (* z = xy, dz = dx y + x dy *)
  let ( *@ ) =
    let f x y =
      if List.length (Tensor.shape x) < 2 && List.length (Tensor.shape y) < 2
      then failwith "( *@ ) does not operate on two vectors";
      Tensor.matmul x y
    in
    let dfx ~f:_ ~x:_ ~y ~dx = Tensor.(matmul dx y) in
    let dfy ~f:_ ~x ~y:_ ~dy = Tensor.(matmul x dy) in
    let dfxy ~f:_ ~x ~y ~dx ~dy = Tensor.(add_ (matmul dx y) (matmul x dy)) in
    { f; dfx; dfy; dfxy }

  let einsum operands return =
    let equation = String.concat ~sep:"," (List.map ~f:snd operands) ^ "->" ^ return in
    Tensor.einsum ~equation (List.map ~f:fst operands) ~path:None
end

(* ----------------------------------------------------
   -- Generic operations on [< `const | `dual] t 
   ---------------------------------------------------- *)

let make_unary (z : unary_info) =
  let f (E x : ([< `const | `dual ] as 'a) t) : 'a t =
    match x with
    | C x -> E (C (z.f x))
    | D (x, dx) ->
      let f = z.f x in
      let dx = tangent_tensor_of x dx in
      E (D (f, Explicit (z.df ~f ~x ~dx)))
  in
  f

let make_binary (z : binary_info) =
  let f (E x : [< `const | `dual ] t) (E y : [< `const | `dual ] t) : [ `const | `dual ] t
    =
    match x, y with
    | C x, C y -> E (C (z.f x y))
    | D (x, dx), C y ->
      let f = z.f x y in
      E (D (f, Explicit (z.dfx ~f ~x ~y ~dx:(tangent_tensor_of x dx))))
    | C x, D (y, dy) ->
      let f = z.f x y in
      E (D (f, Explicit (z.dfy ~f ~x ~y ~dy:(tangent_tensor_of y dy))))
    | D (x, dx), D (y, dy) ->
      let f = z.f x y in
      let dx = tangent_tensor_of x dx in
      let dy = tangent_tensor_of y dy in
      E (D (f, Explicit (z.dfxy ~f ~x ~y ~dx ~dy)))
  in
  f

let view ~size = make_unary (Ops.view ~size)
let reshape ~shape = make_unary (Ops.reshape ~shape)
let permute ~dims = make_unary (Ops.permute ~dims)
let squeeze ~dim = make_unary (Ops.squeeze ~dim)
let unsqueeze ~dim = make_unary (Ops.unsqueeze ~dim)
let transpose ?dims x = make_unary Ops.(transpose ?dims) x
let neg x = make_unary Ops.neg x
let trace x = make_unary Ops.trace x
let sin x = make_unary Ops.sin x
let cos x = make_unary Ops.cos x
let sqr x = make_unary Ops.sqr x
let sqrt x = make_unary Ops.sqrt x
let log x = make_unary Ops.log x
let exp x = make_unary Ops.exp x
let tanh x = make_unary Ops.tanh x
let relu x = make_unary Ops.relu x
let sigmoid x = make_unary Ops.sigmoid x
let softplus x = make_unary Ops.softplus x
let slice ?start ?end_ ?step ~dim = make_unary Ops.(slice ?start ?end_ ?step ~dim)
let mean x = make_unary Ops.mean x
let sum x = make_unary Ops.sum x
let sum_dim ?keepdim ~dim = make_unary (Ops.sum_dim ?keepdim ~dim)
let mean_dim ?keepdim ~dim = make_unary (Ops.mean_dim ?keepdim ~dim)
let logsumexp ?keepdim ~dim = make_unary (Ops.logsumexp ?keepdim ~dim)
let ( + ) x = make_binary Ops.( + ) x
let ( - ) x = make_binary Ops.( - ) x
let ( * ) x = make_binary Ops.( * ) x
let ( / ) x = make_binary Ops.( / ) x
let ( $+ ) z = make_unary Ops.(( $+ ) z)
let ( $* ) z = make_unary Ops.(( $* ) z)
let ( $/ ) z = make_unary Ops.(( $/ ) z)
let ( *@ ) x = make_binary Ops.(( *@ )) x

let einsum (operands : ([< `const | `dual ] t * string) list) return
  : [ `const | `dual ] t
  =
  let tangent_id = 'x' in
  assert (not (String.contains return tangent_id));
  assert (
    List.fold operands ~init:true ~f:(fun accu (_, eq) ->
      accu && not (String.contains eq tangent_id)));
  let primal =
    Ops.einsum (List.map operands ~f:(fun (x, eq) -> to_tensor x, eq)) return
  in
  let tangent =
    List.foldi operands ~init:None ~f:(fun i accu (E op, eq) ->
      match op with
      | C _ -> accu
      | D (op, dop) ->
        let ops =
          List.mapi operands ~f:(fun j (op', eq') ->
            if i = j
            then tangent_tensor_of op dop, String.of_char tangent_id ^ eq
            else to_tensor op', eq')
        in
        let return = String.of_char tangent_id ^ return in
        let tangent_contrib = Ops.einsum ops return in
        (match accu with
         | None -> Some tangent_contrib
         | Some a -> Some Tensor.(a + tangent_contrib)))
  in
  match tangent with
  | None -> E (C primal)
  | Some dz -> E (D (primal, Explicit dz))

(* ----------------------------------------------------
   -- Operations on [`const] t 
   ---------------------------------------------------- *)

module C = struct
  let make_unary (z : unary_info) =
    let f (E x : [ `const ] t) : [ `const ] t =
      match x with
      | C x -> E (C (z.f x))
      | D _ -> raise Not_const
    in
    f

  let make_binary (z : binary_info) =
    let f (E x : [ `const ] t) (E y : [ `const ] t) : [ `const ] t =
      match x, y with
      | C x, C y -> E (C (z.f x y))
      | _ -> raise Not_const
    in
    f

  let view ~size = make_unary (Ops.view ~size)
  let reshape ~shape = make_unary (Ops.reshape ~shape)
  let permute ~dims = make_unary (Ops.permute ~dims)
  let squeeze ~dim = make_unary (Ops.squeeze ~dim)
  let unsqueeze ~dim = make_unary (Ops.unsqueeze ~dim)
  let transpose ?dims = make_unary Ops.(transpose ?dims)
  let neg = make_unary Ops.neg
  let trace = make_unary Ops.trace
  let sin = make_unary Ops.sin
  let cos = make_unary Ops.cos
  let sqr = make_unary Ops.sqr
  let sqrt = make_unary Ops.sqrt
  let log = make_unary Ops.log
  let exp = make_unary Ops.exp
  let tanh = make_unary Ops.tanh
  let relu = make_unary Ops.relu
  let sigmoid = make_unary Ops.sigmoid
  let softplus = make_unary Ops.softplus
  let slice ?start ?end_ ?step ~dim = make_unary Ops.(slice ?start ?end_ ?step ~dim)
  let sum = make_unary Ops.sum
  let mean = make_unary Ops.mean
  let sum_dim ?keepdim ~dim = make_unary (Ops.sum_dim ?keepdim ~dim)
  let mean_dim ?keepdim ~dim = make_unary (Ops.mean_dim ?keepdim ~dim)
  let logsumexp ?keepdim ~dim = make_unary (Ops.logsumexp ?keepdim ~dim)
  let ( + ) = make_binary Ops.( + )
  let ( - ) = make_binary Ops.( - )
  let ( * ) = make_binary Ops.( * )
  let ( / ) = make_binary Ops.( / )
  let ( $+ ) z = make_unary Ops.(( $+ ) z)
  let ( $* ) z = make_unary Ops.(( $* ) z)
  let ( $/ ) z = make_unary Ops.(( $/ ) z)
  let ( *@ ) = make_binary Ops.(( *@ ))

  let einsum (operands : ([ `const ] t * string) list) return : [ `const ] t =
    let tangent_id = 'x' in
    assert (not (String.contains return tangent_id));
    assert (
      List.fold operands ~init:true ~f:(fun accu (_, eq) ->
        accu && not (String.contains eq tangent_id)));
    let all_const =
      List.fold operands ~init:true ~f:(fun accu (E x, _) ->
        match x with
        | C _ -> accu
        | _ -> false)
    in
    if not all_const then raise Not_const;
    let z = Ops.einsum (List.map operands ~f:(fun (x, eq) -> to_tensor x, eq)) return in
    E (C z)

  let svd (E x) =
    match x with
    | C x ->
      let u, s, vt = Tensor.svd ~some:true ~compute_uv:true x in
      of_tensor u, of_tensor s, of_tensor vt
    | _ -> raise Not_const

  let qr (E x) =
    match x with
    | C x ->
      let q, r = Tensor.linalg_qr ~a:x ~mode:"complete" in
      of_tensor q, of_tensor r
    | _ -> raise Not_const
end
