open Base
open Torch
include Maths_typ

(*
   We will systematically work with the assumption that for a primal value of shape [n1; n2; ... ],
   the associated tangents have one more dimension, corresponding to tangent batch: [K; n1; n2; ... ]
*)

(* let print s = Stdio.print_endline (Sexp.to_string_hum s) *)
let shape (x, _) = Tensor.shape x

(* int list starting from 1 and ending at the last dim of a *)
let all_dims_but_first a = List.range 1 (List.length (Tensor.shape a))

(* get primal, which is the first element *)
let primal = fst

(* get tangent opt, which is instantiated if direct or not instantiated if lazy *)
let tangent' = function
  | Direct dx -> dx
  | Lazy dx -> dx ()

(* get tangent, which is the second element *)
let tangent (_, t) = Option.map t ~f:tangent'

(* check that the assumption above is satisfied *)
let assert_right_shape label t =
  Option.iter (tangent t) ~f:(fun dx ->
    let sx = Tensor.shape (primal t)
    and sdx = Tensor.shape dx in
    let n = List.length sx in
    let n' = List.length sdx in
    if n' <> n + 1 || Poly.(List.tl_exn sdx <> sx) then raise (Wrong_shape label));
  t

(* constant tensor, i.e. no associated tangents *)
let const x = x, None

(* constant scalar tensor *)
let f x = Tensor.f x, None

(* make dual number of (primal, tangent) *)
let make_dual x ~t = (x, Some t) |> assert_right_shape "make_dual"

(* apply f to tangents dx. *)
let with_tangent dx ~f =
  Option.map dx ~f:(function
    | Direct dx -> Direct (f dx)
    | Lazy dx -> Direct (f (dx ())))

let append_batch ~tangent shape =
  let b = List.hd_exn (Tensor.shape tangent) in
  b :: shape

(** Unary operations *)

(* reshape the size of x to size, and of each batch in dx to size. *)
let view (x, dx) ~size =
  let y = Tensor.view x ~size in
  let dy =
    with_tangent dx ~f:(fun dx ->
      let size = append_batch ~tangent:dx size in
      Tensor.view ~size dx)
  in
  (y, dy) |> assert_right_shape "view"

let permute (x, dx) ~dims =
  let y = Tensor.permute x ~dims in
  let dy =
    with_tangent dx ~f:(fun dx ->
      let tensor_dims = 0 :: List.map dims ~f:(fun i -> i + 1) in
      Tensor.permute dx ~dims:tensor_dims)
  in
  (y, dy) |> assert_right_shape "permute"

(* y = -x, dy = -dx *)
let neg (x, dx) =
  let y = Tensor.neg x in
  let dy = with_tangent dx ~f:Tensor.neg in
  (y, dy) |> assert_right_shape "neg"

let trace (x, dx) =
  assert (
    match Tensor.shape x with
    | [ a; b ] when a = b -> true
    | _ -> false);
  let y = Tensor.(reshape (trace x) ~shape:[ 1 ]) in
  let dy =
    with_tangent dx ~f:(fun dx ->
      Tensor.einsum [ dx ] ~equation:"qii->q" ~path:None
      |> Tensor.reshape ~shape:[ -1; 1 ])
  in
  (y, dy) |> assert_right_shape "trace"

(* y = sin(x), dy = cos(x) dx *)
let sin (x, dx) =
  let y = Tensor.sin x in
  let dy = with_tangent dx ~f:(fun dx -> Tensor.(cos x * dx)) in
  (y, dy) |> assert_right_shape "sin"

(* y = cos(x), dy = -sin(x) dx *)
let cos (x, dx) =
  let y = Tensor.cos x in
  let dy = with_tangent dx ~f:(fun dx -> Tensor.(neg (sin x) * dx)) in
  (y, dy) |> assert_right_shape "cos"

(* y = x^2, dy = 2 x dx *)
let sqr (x, dx) =
  let y = Tensor.square x in
  let dy =
    with_tangent dx ~f:(fun dx ->
      let tmp = Tensor.(x * dx) in
      Tensor.mul_scalar_ tmp (Scalar.f 2.))
  in
  (y, dy) |> assert_right_shape "sqr"

(* y = x^{1/2}, dy = 1/2 x^{-1/2} dx *)
let sqrt (x, dx) =
  let y = Tensor.sqrt x in
  let dy =
    with_tangent dx ~f:(fun dx ->
      let tmp = Tensor.(dx / y) in
      Tensor.mul_scalar_ tmp (Scalar.f 0.5))
  in
  (y, dy) |> assert_right_shape "sqrt"

(* y = log x, dy = dx/x *)
let log (x, dx) =
  let y = Tensor.log x in
  let dy = with_tangent dx ~f:(fun dx -> Tensor.(div dx x)) in
  (y, dy) |> assert_right_shape "log"

(* y = exp(x), dy = exp(x) dx *)
let exp (x, dx) =
  let y = Tensor.exp x in
  let dy = with_tangent dx ~f:(fun dx -> Tensor.(dx * y)) in
  (y, dy) |> assert_right_shape "exp"

(* y = tanh(x), dy = (1 - tanh(x)^2) dx *)
let tanh (x, dx) =
  let y = Tensor.tanh x in
  let dy =
    with_tangent dx ~f:(fun dx ->
      let tmp = Tensor.(square y) in
      let tmp = Tensor.(neg_ tmp) in
      let tmp = Tensor.(add_scalar_ tmp Scalar.(f 1.)) in
      Tensor.mul tmp dx)
  in
  (y, dy) |> assert_right_shape "tanh"

(* invert a square matrix; y = x^-1, dy = - x^-1 dx x^-1 *)
let inv_sqr (x, dx) =
  assert (List.length (Tensor.shape x) = 2);
  assert (List.hd_exn (Tensor.shape x) = List.nth_exn (Tensor.shape x) 1);
  (* let y =
     let u, s, vt = Tensor.svd ~some:true ~compute_uv:true x in
     let tran_2d = Tensor.transpose ~dim0:1 ~dim1:0 in
     Tensor.(matmul (tran_2d vt / s) (tran_2d u))
     in *)
  let y = Tensor.inverse x in
  let dy = with_tangent dx ~f:(fun dx -> Tensor.(neg (matmul y (matmul dx y)))) in
  (y, dy) |> assert_right_shape "inv_sqr"

(* pseudo-inverse of a matrix of size [m x n] where m != n *)
let inv_rectangle ?(rcond = 1e-6) (x, dx) =
  assert (List.length (Tensor.shape x) = 2);
  let y = Tensor.pinverse x ~rcond in
  let dy =
    with_tangent dx ~f:(fun dx ->
      let tran_2d = Tensor.transpose ~dim0:1 ~dim1:0 in
      let xTx = Tensor.(matmul (tran_2d x) x) in
      let tmp1 = Tensor.(matmul (inverse xTx) (Tensor.transpose dx ~dim0:2 ~dim1:1)) in
      let tmp2 = Tensor.(matmul tmp1 (matmul x y)) in
      let tmp3 = Tensor.(matmul y (matmul dx y)) in
      Tensor.(tmp1 - tmp2 - tmp3))
  in
  (y, dy) |> assert_right_shape "inv_rectangle"

let relu (x, dx) =
  let y = Tensor.relu x in
  let dy =
    with_tangent dx ~f:(fun dx ->
      let values = Tensor.(ones ~device:(device x) ~kind:(type_ x) []) in
      Tensor.mul dx (Tensor.heaviside ~values x))
  in
  (y, dy) |> assert_right_shape "relu"

let sigmoid (x, dx) =
  let y = Tensor.sigmoid x in
  let dy = with_tangent dx ~f:(fun dx -> Tensor.(mul dx (mul y (ones_like y - y)))) in
  (y, dy) |> assert_right_shape "sigmoid"

let softplus (x, dx) =
  let y = Tensor.softplus x in
  let dy = with_tangent dx ~f:(fun dx -> Tensor.(mul (sigmoid x) dx)) in
  (y, dy) |> assert_right_shape "softplus"

(* like Torch's slice but not sharing data *)
let slice ~dim ~start ~end_ ~step (x, dx) =
  let y = Tensor.slice_copy x ~dim ~start ~end_ ~step in
  let dy = with_tangent dx ~f:(Tensor.slice_copy ~dim:(dim + 1) ~start ~end_ ~step) in
  (y, dy) |> assert_right_shape "slice"

(* y = sum of x_i, dy = sum of dx_i *)
let sum (x, dx) =
  let y = Tensor.sum x in
  let dy =
    with_tangent dx ~f:(fun dx ->
      (* if x has shape [m, n, k], dim is the int list [1, 2, 3]
         whereas dx has shape [B, m, n, k] *)
      let dim = List.mapi (Tensor.shape x) ~f:(fun i _ -> i + 1) in
      (* retain the batch dimension, sum over the rest *)
      Tensor.(sum_dim_intlist ~keepdim:false ~dtype:(type_ dx) ~dim:(Some dim) dx))
  in
  (y, dy) |> assert_right_shape "sum"

let sum_dim (x, dx) ~dim ~keepdim =
  let y = Tensor.(sum_dim_intlist x ~dim ~keepdim ~dtype:(type_ x)) in
  let dy =
    with_tangent dx ~f:(fun dx ->
      (* make sure to preserve the batch dimension in dx *)
      let dim =
        match dim with
        | Some d -> List.map d ~f:Int.succ
        | None -> List.mapi (Tensor.shape x) ~f:(fun i _ -> i + 1)
      in
      Tensor.(sum_dim_intlist ~dim:(Some dim) ~keepdim ~dtype:(type_ dx) dx))
  in
  (y, dy) |> assert_right_shape "sum_dim"

let mean (x, dx) =
  let y = Tensor.mean x in
  let dy =
    with_tangent dx ~f:(fun dx ->
      let dim = List.mapi (Tensor.shape x) ~f:(fun i _ -> i + 1) in
      Tensor.(mean_dim ~keepdim:false ~dtype:(type_ dx) ~dim:(Some dim) dx))
  in
  (y, dy) |> assert_right_shape "mean"

let mean_dim (x, dx) ~dim ~keepdim =
  let y = Tensor.(mean_dim x ~dim ~keepdim ~dtype:(type_ x)) in
  let dy =
    with_tangent dx ~f:(fun dx ->
      (* make sure to preserve the batch dimension in dx *)
      let dim =
        match dim with
        | Some d -> List.map d ~f:Int.succ
        | None -> List.mapi (Tensor.shape x) ~f:(fun i _ -> i + 1)
      in
      Tensor.(mean_dim ~dim:(Some dim) ~keepdim ~dtype:(type_ dx) dx))
  in
  (y, dy) |> assert_right_shape "mean_dim"

(* let print s = Stdio.printf "%s\n%!" (Base.Sexp.to_string_hum s) *)

let max_2d_dim1 (x, dx) ~keepdim =
  let y, y_indices = Tensor.(max_dim x ~dim:1 ~keepdim) in
  let dy =
    with_tangent dx ~f:(fun dx ->
      let[@warning "-8"] (bs :: n :: _) = Tensor.shape x in
      let num_tangents = List.hd_exn (Tensor.shape dx) in
      let device = Tensor.device x in
      let kind = Tensor.type_ x in
      let y_shape = Tensor.shape y in
      let indices_offset =
        let offset_block =
          Tensor.arange_start
            ~start:(Scalar.int 0)
            ~end_:(Scalar.int bs)
            ~options:(kind, device)
          |> Tensor.reshape ~shape:[ 1; -1 ]
        in
        let second_dim_offset = Tensor.mul_scalar offset_block (Scalar.int n) in
        Tensor.(second_dim_offset + Tensor.view y_indices ~size:[ 1; -1 ])
      in
      (* cast as integer and collapse to one long vector. *)
      let indices_offset =
        Tensor._cast_int indices_offset ~non_blocking:true
        |> Tensor.view ~size:[ 1; -1 ]
        |> Tensor.squeeze
      in
      let dx_collapsed = Tensor.reshape dx ~shape:[ num_tangents; -1 ] in
      let dy = Tensor.index_select dx_collapsed ~dim:1 ~index:indices_offset in
      Tensor.reshape dy ~shape:(num_tangents :: y_shape))
  in
  (y, dy) |> assert_right_shape "max_2d_dim1"

(* y = x^T, dy = dx^T *)
let transpose (x, dx) ~dim0 ~dim1 =
  let y = Tensor.transpose_copy x ~dim0 ~dim1 in
  let dy = with_tangent dx ~f:(Tensor.transpose_copy ~dim0:(dim0 + 1) ~dim1:(dim1 + 1)) in
  (y, dy) |> assert_right_shape "transpose"

(* y = log of sum of exp(x_i), dy = sum of exp (x_i - y) dx_i *)
let logsumexp (x, dx) ~dim ~keepdim =
  let y = Tensor.logsumexp x ~dim ~keepdim in
  (* s is the shape of y if keep dimension, otherwise remove the dimension specified by dim. *)
  let s =
    if keepdim
    then Tensor.shape y
    else
      List.mapi (Tensor.shape x) ~f:(fun i di ->
        if List.mem ~equal:Int.( = ) dim i then 1 else di)
  in
  let dy =
    with_tangent dx ~f:(fun dx ->
      let tmp = Tensor.(x - view ~size:s y) in
      let tmp = Tensor.exp_ tmp in
      let tmp = Tensor.mul dx tmp in
      (* make sure to preserve the batch dimension in dx *)
      let dim = List.map dim ~f:Int.succ in
      Tensor.(sum_dim_intlist ~dim:(Some dim) ~keepdim ~dtype:(type_ x) tmp))
  in
  (y, dy) |> assert_right_shape "logsumexp"

(* x are the categorical probabilities. *)
(* let gumbel_softmax (x, dx) ~tau ~with_noise ~discrete =
   let gumbel_noise =
   if with_noise
   then (
   let uniform_noise = Tensor.uniform x ~from:0. ~to_:1. in
   Some Tensor.(neg (log (neg (log uniform_noise)))))
   else None
   in
   let logits =
   match gumbel_noise with
   | None -> Tensor.(div_scalar (log x) (Scalar.f tau))
   | Some gumbel_noise -> Tensor.(div_scalar (log x + gumbel_noise) (Scalar.f tau))
   in
   let reduce_dim_list = all_dims_but_first x in
   let num_classes = List.hd_exn reduce_dim_list in
   let summed_exp_logits =
   Tensor.(
   sum_dim_intlist
   (exp logits)
   ~dim:(Some reduce_dim_list)
   ~keepdim:true
   ~dtype:(Tensor.type_ x))
   in
   let y = Tensor.(exp (logits - logsumexp ~dim:reduce_dim_list ~keepdim:true logits)) in
   let dy =
   with_tangent dx ~f:(fun dx ->
   let tmp1 = Tensor.(div_scalar (y * dx / x) (Scalar.f tau)) in
   let reduce_dim_list_dx = List.map reduce_dim_list ~f:Int.succ in
   let tmp2 =
   let logits_diff = Tensor.(div_scalar (exp logits * dx / x) (Scalar.f tau)) in
   let logits_diff_summed =
   Tensor.sum_dim_intlist
   logits_diff
   ~dim:(Some reduce_dim_list_dx)
   ~keepdim:true
   ~dtype:(Tensor.type_ dx)
   in
   Tensor.(logits_diff_summed * y / summed_exp_logits)
   in
   Tensor.(tmp1 - tmp2))
   in
   (* if discrete, return one-hot encoded version *)
   let y_final =
   if discrete
   then (
   let pos = Tensor.argmax y ~dim:1 ~keepdim:true in
   Tensor.one_hot pos ~num_classes)
   else y
   in
   (y_final, dy) |> assert_right_shape "gumbel_softmax" *)

(* x are the categorical logits. *)
let gumbel_softmax (x, dx) ~tau ~with_noise ~discrete =
  let gumbel_noise =
    if with_noise
    then (
      let uniform_noise = Tensor.uniform x ~from:0. ~to_:1. in
      Some Tensor.(neg_ (log_ (neg_ (log_ uniform_noise)))))
    else None
  in
  let logits =
    match gumbel_noise with
    | None -> Tensor.(div_scalar x (Scalar.f tau))
    | Some gumbel_noise -> Tensor.(div_scalar (x + gumbel_noise) (Scalar.f tau))
  in
  let reduce_dim_list = all_dims_but_first x in
  let num_classes = List.nth_exn (Tensor.shape x) 1 in
  let summed_exp_logits =
    Tensor.(
      sum_dim_intlist
        (exp logits)
        ~dim:(Some reduce_dim_list)
        ~keepdim:true
        ~dtype:(Tensor.type_ x))
  in
  let y = Tensor.(exp (logits - logsumexp ~dim:reduce_dim_list ~keepdim:true logits)) in
  let dy =
    with_tangent dx ~f:(fun dx ->
      let tmp1 = Tensor.(div_scalar (y * dx) (Scalar.f tau)) in
      let reduce_dim_list_dx = List.map reduce_dim_list ~f:Int.succ in
      let tmp2 =
        let logits_diff = Tensor.(div_scalar (exp logits * dx) (Scalar.f tau)) in
        let logits_diff_summed =
          Tensor.sum_dim_intlist
            logits_diff
            ~dim:(Some reduce_dim_list_dx)
            ~keepdim:true
            ~dtype:(Tensor.type_ dx)
        in
        Tensor.(logits_diff_summed * y / summed_exp_logits)
      in
      Tensor.(tmp1 - tmp2))
  in
  (* if discrete, return one-hot encoded version *)
  let y_final =
    if discrete
    then (
      let pos = Tensor.argmax y ~dim:1 ~keepdim:true in
      Tensor.one_hot pos ~num_classes |> Tensor.squeeze)
    else y
  in
  (y_final, dy) |> assert_right_shape "gumbel_softmax"

let maxpool2d
  ?(padding = 0, 0)
  ?(dilation = 1, 1)
  ?(ceil_mode = false)
  ?stride
  (x, dx)
  ~ksize
  =
  let stride =
    match stride with
    | None -> ksize
    | Some stride -> stride
  in
  let pair_to_list (a, b) = [ a; b ] in
  (* extract the corresponding slice of the tangents given the indices retrieved by maxpool2D *)
  let y, indices =
    Tensor.max_pool2d_with_indices
      ~padding:(pair_to_list padding)
      ~dilation:(pair_to_list dilation)
      ~ceil_mode
      ~stride:(pair_to_list stride)
      x
      ~kernel_size:(pair_to_list ksize)
  in
  let dy =
    with_tangent dx ~f:(fun dx ->
      let[@warning "-8"] (bs :: nc :: rest_in) = Tensor.shape x in
      let num_rest_in = List.fold rest_in ~init:1 ~f:(fun acc i -> acc * i) in
      let y_shape = Tensor.shape y in
      let[@warning "-8"] (_ :: _ :: rest_out) = y_shape in
      let num_tangents = List.hd_exn (Tensor.shape dx) in
      (* collapse last two *)
      let indices_collapse_last_two = Tensor.reshape indices ~shape:[ bs; nc; -1 ] in
      let num_rest_out = List.fold rest_out ~init:1 ~f:(fun acc i -> acc * i) in
      let device = Tensor.device x in
      let kind = Tensor.type_ x in
      (* m1[i, j, k] = i; m1 and m2 has the same shape as indices *)
      let m1 =
        let m1_a =
          Tensor.arange_start
            ~start:(Scalar.int 0)
            ~end_:(Scalar.int bs)
            ~options:(kind, device)
          |> Tensor.reshape ~shape:[ -1; 1 ]
        in
        let m1_a_b =
          Tensor.repeat m1_a ~repeats:[ 1; nc ] |> Tensor.reshape ~shape:[ bs; nc; 1 ]
        in
        Tensor.repeat m1_a_b ~repeats:[ 1; 1; num_rest_out ]
      in
      (* m2[i, j, k] = j *)
      let m2 =
        let m2_a =
          Tensor.arange_start
            ~start:(Scalar.int 0)
            ~end_:(Scalar.int nc)
            ~options:(kind, device)
          |> Tensor.reshape ~shape:[ 1; -1 ]
        in
        let m2_a_b =
          Tensor.repeat m2_a ~repeats:[ bs; 1 ] |> Tensor.reshape ~shape:[ bs; nc; 1 ]
        in
        Tensor.repeat m2_a_b ~repeats:[ 1; 1; num_rest_out ]
      in
      (* indices_offset[i, j, k] = i * (nc * num_rest_in) + j * num_rest_in + indices [i, j, k]. *)
      let indices_offset =
        let first_dim_offset = Tensor.mul_scalar m1 (Scalar.int (nc * num_rest_in)) in
        let second_dim_offset = Tensor.mul_scalar m2 (Scalar.int num_rest_in) in
        Tensor.(first_dim_offset + second_dim_offset + indices_collapse_last_two)
      in
      (* cast as integer and collapse to one long vector. *)
      let indices_offset = Tensor._cast_int indices_offset ~non_blocking:true in
      let indices_collapsed =
        Tensor.reshape indices_offset ~shape:[ 1; -1 ] |> Tensor.squeeze
      in
      let dx_collapsed = Tensor.reshape dx ~shape:[ num_tangents; -1 ] in
      let dy = Tensor.index_select dx_collapsed ~dim:1 ~index:indices_collapsed in
      Tensor.reshape dy ~shape:(num_tangents :: y_shape))
  in
  (y, dy) |> assert_right_shape "maxpool2d"

(** Binary operations *)

(* apply fx to tangents dx. and fy to tangents dy and fxy to tangents dx, dy. *)
let with_tangents dx dy ~fx ~fy ~fxy =
  match dx, dy with
  | None, None -> None
  | Some dx, None -> Some (Direct (fx (tangent' dx)))
  | None, Some dy -> Some (Direct (fy (tangent' dy)))
  | Some dx, Some dy -> Some (Direct (fxy (tangent' dx) (tangent' dy)))

(* z = x + y, dz = dx + dy *)
let ( + ) (x, dx) (y, dy) =
  let z = Tensor.(x + y) in
  let dz = with_tangents dx dy ~fx:Fn.id ~fy:Fn.id ~fxy:Tensor.( + ) in
  (z, dz) |> assert_right_shape "( + )"

(* z = x - y, dz = dx - dy *)
let ( - ) (x, dx) (y, dy) =
  let z = Tensor.(x - y) in
  let dz = with_tangents dx dy ~fx:Fn.id ~fy:Tensor.neg ~fxy:Tensor.( - ) in
  (z, dz) |> assert_right_shape "( - )"

(* z = x * y, dz = y dx + x dy *)
let ( * ) (x, dx) (y, dy) =
  let z = Tensor.(x * y) in
  let dz =
    with_tangents
      dx
      dy
      ~fx:Tensor.(mul y)
      ~fy:Tensor.(mul x)
      ~fxy:(fun dx dy -> Tensor.(add_ (dx * y) (dy * x)))
  in
  (z, dz) |> assert_right_shape "( * )"

(* z = x / y, dz = 1/y dx - x/(y^2) dy *)
let ( / ) (x, dx) (y, dy) =
  let z = Tensor.(x / y) in
  let y2 = Tensor.square y in
  let dz =
    with_tangents
      dx
      dy
      ~fx:(fun dx -> Tensor.(div_ dx y))
      ~fy:(fun dy -> Tensor.(neg_ (div_ (dy * x) y2)))
      ~fxy:(fun dx dy -> Tensor.(div_ (sub_ (dx * y) (dy * x)) y2))
  in
  (z, dz) |> assert_right_shape "( / )"

(* x = x + z *)
let ( $+ ) z (x, dx) = (Tensor.(add_scalar x (Scalar.f z)), dx) |> assert_right_shape "$+"

(* x = x *z *)
let ( $* ) z (x, dx) =
  let y = Tensor.(mul_scalar x (Scalar.f z)) in
  let dy = with_tangent dx ~f:(fun dx -> Tensor.(mul_scalar dx Scalar.(f z))) in
  (y, dy) |> assert_right_shape "$*"

let ( $/ ) x (y, dy) =
  let z = Tensor.(mul_scalar (reciprocal y) (Scalar.f x)) in
  let y2 = Tensor.square y in
  let dz =
    with_tangent dy ~f:(fun dy -> Tensor.(neg_ (div_ (mul_scalar dy (Scalar.f x)) y2)))
  in
  (z, dz) |> assert_right_shape "( $/ )"

let ( /$ ) (x, dx) y = Float.(1. / y) $* (x, dx)

(* z = xy, dz = dx y + x dy *)
let ( *@ ) (x, dx) (y, dy) =
  if List.length (Tensor.shape x) < 2 && List.length (Tensor.shape y) < 2
  then failwith "( *@ ) does not operate on two vectors";
  let z = Tensor.matmul x y in
  let dz =
    with_tangents
      dx
      dy
      ~fx:(fun dx -> Tensor.(matmul dx y))
      ~fy:(fun dy -> Tensor.(matmul x dy))
      ~fxy:(fun dx dy -> Tensor.(add_ (matmul dx y) (matmul x dy)))
  in
  (z, dz) |> assert_right_shape "( *@ )"

let __einsum_primal operands return =
  let equation = String.concat ~sep:"," (List.map ~f:snd operands) ^ "->" ^ return in
  Tensor.einsum ~equation (List.map ~f:fst operands) ~path:None

(* einsum [ a, "ij"; b, "jk"; c, "ki" ] "ii" *)
let einsum (operands : (t * string) list) return =
  let tangent_id = 'x' in
  assert (not (String.contains return tangent_id));
  assert (
    List.fold operands ~init:true ~f:(fun accu (_, eq) ->
      accu && not (String.contains eq tangent_id)));
  let primal =
    __einsum_primal (List.map operands ~f:(fun ((primal, _), eq) -> primal, eq)) return
  in
  let tangent =
    List.foldi operands ~init:None ~f:(fun i accu (op, eq) ->
      match tangent op with
      | None -> accu
      | Some dop ->
        let ops =
          List.mapi operands ~f:(fun j (op', eq') ->
            if i = j then dop, String.of_char tangent_id ^ eq else fst op', eq')
        in
        let return = String.of_char tangent_id ^ return in
        let tangent_contrib = __einsum_primal ops return in
        (match accu with
         | None -> Some tangent_contrib
         | Some a -> Some Tensor.(a + tangent_contrib)))
  in
  primal, Option.map tangent ~f:(fun x -> Direct x)

(* solve for ax=b. *)
let linsolve (a, da) (b, db) =
  let z = Tensor.linalg_solve ~a ~b ~left:true in
  let a_shape = Tensor.shape a in
  let b_shape = Tensor.shape b in
  let dz =
    with_tangents
      da
      db
      ~fx:(fun da ->
        let num_tangents_a = List.hd_exn (Tensor.shape da) in
        let a_exp = Tensor.expand a ~size:(num_tangents_a :: a_shape) ~implicit:true in
        let da_z =
          match List.length b_shape with
          | 2 -> Tensor.einsum ~equation:"ijkm,jm->ijk" [ da; z ] ~path:None
          | 1 -> Tensor.einsum ~equation:"ijkm,jmp->ijkp" [ da; z ] ~path:None
          | _ -> assert false
        in
        let dx = Tensor.linalg_solve ~a:a_exp ~b:Tensor.(neg da_z) ~left:true in
        dx)
      ~fy:(fun db ->
        let num_tangents_b = List.hd_exn (Tensor.shape db) in
        let a_exp = Tensor.expand a ~size:(num_tangents_b :: a_shape) ~implicit:true in
        Tensor.linalg_solve ~a:a_exp ~b:db ~left:true)
      ~fxy:(fun da db ->
        let num_tangents_a = List.hd_exn (Tensor.shape da) in
        let num_tangents_b = List.hd_exn (Tensor.shape db) in
        assert (num_tangents_a = num_tangents_b);
        let a_exp = Tensor.expand a ~size:(num_tangents_a :: a_shape) ~implicit:true in
        let da_z =
          match List.length b_shape with
          | 2 -> Tensor.einsum ~equation:"ijkm,jm->ijk" [ da; z ] ~path:None
          | 3 -> Tensor.einsum ~equation:"ijkm,jmp->ijkp" [ da; z ] ~path:None
          | _ -> assert false
        in
        let dx = Tensor.linalg_solve ~a:a_exp ~b:Tensor.(db - da_z) ~left:true in
        dx)
  in
  (z, dz) |> assert_right_shape "linsolve"

let conv2d
  ?(padding = 0, 0)
  ?(dilation = 1, 1)
  ?(groups = 1)
  ~bias:(b, db)
  ~stride
  (x, dx)
  (w, dw)
  =
  (* x has shape [bs x n_channels x w x h], w has shape [out_channels x in_channels x kerl_x x kerl_y] *)
  let z = Tensor.conv2d ~padding ~dilation ~groups x w (Some b) ~stride in
  let maybe_add_db dz =
    match tangent (b, db) with
    | None -> dz
    | Some db ->
      let n_dim = List.length (Tensor.shape dz) in
      let[@warning "-8"] [ num_tangents; c_out ] = Tensor.shape db in
      (* broadcast db *)
      let db =
        Tensor.reshape
          db
          ~shape:
            (List.init n_dim ~f:(function
              | 0 -> num_tangents
              | 2 -> c_out
              | _ -> 1))
      in
      Tensor.(db + dz)
  in
  let dz =
    with_tangents
      dx
      dw
      ~fx:(fun dx ->
        let[@warning "-8"] (num_tangents :: bs :: rest) = Tensor.shape dx in
        (* collapse num_tangents and batch_size dimensions *)
        let dx = Tensor.reshape dx ~shape:(-1 :: rest) in
        let dz = Tensor.conv2d ~padding ~dilation ~groups dx w None ~stride in
        (* un-collapse *)
        let[@warning "-8"] (_ :: rest') = Tensor.shape dz in
        Tensor.reshape dz ~shape:(num_tangents :: bs :: rest') |> maybe_add_db)
      ~fy:(fun dw ->
        let[@warning "-8"] (num_tangents :: nc_out :: rest) = Tensor.shape dw in
        (* collapse num_tangents and num_out_channels dimensions *)
        let dw = Tensor.reshape dw ~shape:(-1 :: rest) in
        let dz = Tensor.conv2d ~padding ~dilation ~groups x dw None ~stride in
        (* un-collapse *)
        let[@warning "-8"] (bs :: _ :: rest') = Tensor.shape dz in
        Tensor.reshape dz ~shape:(bs :: num_tangents :: nc_out :: rest')
        |> Tensor.transpose_ ~dim0:1 ~dim1:0
        |> maybe_add_db)
      ~fxy:(fun dx dw ->
        let part1 =
          let[@warning "-8"] (num_tangents :: bs :: rest) = Tensor.shape dx in
          let dx = Tensor.reshape dx ~shape:(-1 :: rest) in
          let tmp = Tensor.conv2d ~padding ~dilation ~groups dx w None ~stride in
          let[@warning "-8"] (_ :: rest') = Tensor.shape tmp in
          Tensor.reshape tmp ~shape:(num_tangents :: bs :: rest')
        in
        let part2 =
          let[@warning "-8"] (num_tangents :: nc_out :: rest) = Tensor.shape dw in
          (* collapse num_tangents and num_out_channels dimensions *)
          let dw = Tensor.reshape dw ~shape:(-1 :: rest) in
          let tmp = Tensor.conv2d ~padding ~dilation ~groups x dw None ~stride in
          let[@warning "-8"] (bs :: _ :: rest') = Tensor.shape tmp in
          Tensor.reshape tmp ~shape:(bs :: num_tangents :: nc_out :: rest')
          |> Tensor.transpose_ ~dim0:1 ~dim1:0
        in
        Tensor.(part1 + part2) |> maybe_add_db)
  in
  (z, dz) |> assert_right_shape "conv2d"

let concat_list x_list ~dim =
  let z = Tensor.concat (List.map x_list ~f:fst) ~dim in
  let num_tangents =
    List.fold x_list ~init:0 ~f:(fun acc (_, dx) ->
      let num_tangents =
        match dx with
        | None -> acc
        | Some dx ->
          let dx = tangent' dx in
          List.hd_exn (Tensor.shape dx)
      in
      match acc with
      | 0 -> num_tangents
      | acc -> if num_tangents = acc then acc else assert false)
  in
  let dz =
    if num_tangents = 0
    then None
    else (
      let dx_extend_zero_list =
        List.map x_list ~f:(fun (x, dx) ->
          let dx_new =
            match dx with
            | None ->
              let x_shape = Tensor.shape x in
              Tensor.zeros
                ~device:(Tensor.device x)
                ~kind:(Tensor.type_ x)
                (num_tangents :: x_shape)
            | Some dx -> tangent' dx
          in
          dx_new)
      in
      Some (Direct (Tensor.concat dx_extend_zero_list ~dim:Int.(dim + 1))))
  in
  (z, dz) |> assert_right_shape "concat_list"

let concat (x, dx) (y, dy) ~dim =
  let z = Tensor.concat [ x; y ] ~dim in
  let x_shape, y_shape = Tensor.shape x, Tensor.shape y in
  let tangent_dim = Int.(dim + 1) in
  let dz =
    with_tangents
      dx
      dy
      ~fx:(fun dx ->
        let num_tangents = List.hd_exn (Tensor.shape dx) in
        let y_zeros =
          Tensor.zeros
            ~device:(Tensor.device y)
            ~kind:(Tensor.type_ y)
            (num_tangents :: y_shape)
        in
        Tensor.concat [ dx; y_zeros ] ~dim:tangent_dim)
      ~fy:(fun dy ->
        let num_tangents = List.hd_exn (Tensor.shape dy) in
        let x_zeros =
          Tensor.zeros
            ~device:(Tensor.device x)
            ~kind:(Tensor.type_ x)
            (num_tangents :: x_shape)
        in
        Tensor.concat [ x_zeros; dy ] ~dim:tangent_dim)
      ~fxy:(fun dx dy -> Tensor.concat [ dx; dy ] ~dim:tangent_dim)
  in
  (z, dz) |> assert_right_shape "concat"

let epsilon = 1e-5

let check_grad1 f x =
  (* wrap f around a rng seed setter so that stochasticity is the same *)
  let key = Random.int Int.max_value in
  let f x =
    Torch_core.Wrapper.manual_seed key;
    f x
  in
  (* draw a random direction along which to evaluate derivatives *)
  let sx = Tensor.shape x in
  let v = Tensor.randn ~kind:(Tensor.type_ x) sx in
  (* compute directional derivative as automatically computed by our maths module *)
  let dy_pred =
    let v = Tensor.view_copy v ~size:(1 :: sx) in
    let x : t = make_dual x ~t:(Direct v) in
    let y : t = f x in
    let s = Tensor.shape (primal y) in
    tangent y |> Option.value_exn |> Tensor.view_copy ~size:s
  in
  (* compare with finite-differences *)
  let dy_finite =
    let v = Tensor.mul_scalar v Scalar.(f epsilon) in
    (* guarding against nan, posinf and neginf values.*)
    let yplus = f (const Tensor.(x + v)) |> primal
    and yminus = f (const Tensor.(x - v)) |> primal in
    Tensor.(div_scalar (yplus - yminus) Scalar.(f Float.(2. * epsilon)))
  in
  Tensor.(norm (dy_pred - dy_finite) / norm dy_finite) |> Tensor.to_float0_exn

let check_grad2 f x y =
  (* wrap f around a rng seed setter so that stochasticity is the same *)
  let key = Random.int Int.max_value in
  let f x y =
    Torch_core.Wrapper.manual_seed key;
    f x y
  in
  (* draw a random direction along which to evaluate derivatives *)
  let sx = Tensor.shape x in
  let vx = Tensor.randn ~kind:(Tensor.type_ x) sx in
  let sy = Tensor.shape y in
  let vy = Tensor.randn ~kind:(Tensor.type_ y) sy in
  (* compute directional derivative as automatically computed by our maths module *)
  let dz_pred =
    let vx = Tensor.view_copy vx ~size:(1 :: sx) in
    let vy = Tensor.view_copy vy ~size:(1 :: sy) in
    let x : t = make_dual x ~t:(Direct vx) in
    let y : t = make_dual y ~t:(Direct vy) in
    let z : t = f x y in
    let s = Tensor.shape (primal z) in
    tangent z |> Option.value_exn |> Tensor.view_copy ~size:s
  in
  (* compare with finite-differences *)
  let dz_finite =
    let vx = Tensor.mul_scalar vx Scalar.(f epsilon) in
    let vy = Tensor.mul_scalar vy Scalar.(f epsilon) in
    (* guarding against nan, posinf and neginf values.*)
    let zplusplus = f (const Tensor.(x + vx)) (const Tensor.(y + vy)) |> primal
    and zminusminus = f (const Tensor.(x - vx)) (const Tensor.(y - vy)) |> primal in
    Tensor.(div_scalar (zplusplus - zminusminus) Scalar.(f Float.(2. * epsilon)))
  in
  Tensor.(norm (dz_pred - dz_finite) / norm dz_finite) |> Tensor.to_float0_exn
