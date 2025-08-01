open Base
open Torch
open Forward_torch
open Maths

let n_tests = 100
let device = Torch.Device.Cpu
let kind = Torch_core.Kind.(T f64)

(* generate a random shape with a specified minimum order. *)
let random_shape min_order =
  List.init Int.(min_order + Random.int 3) ~f:(fun _ -> Int.(3 + Random.int 4))

(* generate a random shape with a specified order *)
let random_shape_set order = List.init order ~f:(fun _ -> Int.(2 + Random.int 4))
let rel_tol = Alcotest.float 1e-2

(* two types of contraints: input tensors need to be postive or their orders need to be greater than specified. *)
type input_constr =
  [ `positive
  | `not_all_neg
  | `order_greater_than of int
  | `order_equal_to of int
  | `specified_unary of int list
  | `specified_binary of int list * int list
  | `matmul
  | `linsolve_2d_left_true
  | `linsolve_left_true
  | `linsolve_left_false
  | `linsolve_tri_left_true
  | `linsolve_tri_left_false
  | `pos_def
  ]

(* each unary test test is characterized by a name,
   a (potentially empty) list of constraints on the input,
   and a unary math function to be tested *)
type 'a unary = string * input_constr list * (int list -> 'a some t -> 'a t)

type ('a, 'b) binary =
  string * input_constr list * (int list -> 'a some t -> 'b some t -> any t)

let any_shape f _ x = f x

(* generate tensor according to specified shape and any additional constraint. *)
let generate_tensor ~shape ~input_constr_list =
  let x = Tensor.randn ~kind ~device shape in
  let x =
    if List.mem input_constr_list `pos_def ~equal:Poly.( = )
    then Tensor.(matmul x (transpose x ~dim0:1 ~dim1:0))
    else x
  in
  (* if we require tensor to be positive (for log, sqrt functions etc), we take modulus *)
  let x =
    if List.mem input_constr_list `positive ~equal:Poly.( = )
    then Tensor.(add_scalar (abs x) (Scalar.f 0.1))
    else if
      List.mem input_constr_list `not_all_neg ~equal:Poly.( = )
      (* guard against the all zero case *)
    then (
      let x_abs = Tensor.(abs x) in
      let y = Tensor.randint_like_low_dtype x_abs ~low:(-1) ~high:1 in
      Tensor.(add_scalar (x_abs * y) (Scalar.f 0.5)))
    else x
  in
  x

let check_grad1 f x =
  let module F = Framework.Make (Prms.Single) (Prms.Single) in
  let _, _, e = F.run x ~f in
  e

let check_grad2 f x y =
  let module Two = Prms.Pair (Prms.Single) (Prms.Single) in
  let module F = Framework.Make (Two) (Prms.Single) in
  let _, _, e = F.run (x, y) ~f in
  e

(* this is how we test a unary function *)
let test_unary ((name, input_constr, f) : _ unary) =
  ( name
  , `Quick
  , fun () ->
      List.range 0 n_tests
      |> List.iter ~f:(fun _ ->
        Stdlib.Gc.major ();
        let shape =
          let min_order =
            List.fold input_constr ~init:None ~f:(fun accu c ->
              match accu with
              | Some d -> Some d
              | None ->
                (match c with
                 | `order_greater_than d -> Some d
                 | _ -> accu))
          in
          let set_order =
            List.fold input_constr ~init:None ~f:(fun accu c ->
              match accu with
              | Some d -> Some d
              | None ->
                (match c with
                 | `order_equal_to d -> Some d
                 | _ -> accu))
          in
          let specified_shape =
            List.fold input_constr ~init:None ~f:(fun accu c ->
              match accu with
              | Some d -> Some d
              | None ->
                (match c with
                 | `specified_unary shape -> Some shape
                 | _ -> accu))
          in
          match specified_shape, set_order, min_order with
          | Some specified_shape, _, _ -> specified_shape
          | None, Some d, _ -> random_shape_set d
          | None, None, Some d -> random_shape d
          | None, None, None -> random_shape 1
        in
        let f = f shape in
        let x = generate_tensor ~shape ~input_constr_list:input_constr in
        Alcotest.(check @@ rel_tol) name 0.0 (check_grad1 f (of_tensor x))) )

let inv_sqr x =
  let x_primal = Maths.to_tensor x in
  let x_device = Tensor.device x_primal in
  let x_kind = Tensor.type_ x_primal in
  (* make sure x is positive definite *)
  let x_sym = Maths.(x *@ btr x) in
  let x_size = List.last_exn (Tensor.shape x_primal) in
  let x_final =
    Maths.(x_sym + of_tensor (Tensor.eye ~n:x_size ~options:(x_kind, x_device)))
  in
  Maths.inv_sqr x_final

let unary_tests =
  let test_list : _ unary list =
    [ ( "permute"
      , [ `specified_unary [ 2; 3; 6; 4 ] ]
      , any_shape (permute ~dims:[ 0; 2; 3; 1 ]) )
    ; ( "broadcast_to"
      , [ `specified_unary [ 2; 1; 3; 4 ] ]
      , fun shape_ x ->
          assert (Poly.(shape_ = shape x));
          let size = [ 2; 7; 3; 4 ] in
          broadcast_to ~size x )
    ; ( "broadcast_to_left"
      , [ `specified_unary [ 2; 3; 4 ] ]
      , fun shape_ x ->
          assert (Poly.(shape_ = shape x));
          let size = [ 5; 2; 3; 4 ] in
          broadcast_to ~size x )
    ; ( "reshape"
      , [ `specified_unary [ 2; 3; 6; 4 ] ]
      , any_shape (reshape ~shape:[ 2; 4; 18 ]) )
    ; "squeeze", [ `specified_unary [ 3; 1; 7 ] ], any_shape (squeeze ~dim:1)
    ; "unsqueeze", [], any_shape (unsqueeze ~dim:0)
    ; "sqr", [], any_shape sqr
    ; "neg", [], any_shape neg
    ; "trace", [ `specified_unary [ 10; 10 ] ], any_shape trace
    ; ( "trace_with_einsum"
      , [ `specified_unary [ 10; 10 ] ]
      , any_shape (fun a -> einsum [ a, "ii" ] "") )
    ; ( "transpose_with_einsum"
      , [ `specified_unary [ 10; 10 ] ]
      , any_shape (fun a -> einsum [ a, "ij" ] "ji") )
    ; ( "batch_transpose_with_einsum"
      , [ `specified_unary [ 4; 10; 10 ] ]
      , any_shape (fun a -> einsum [ a, "kij" ] "kji") )
    ; "cos", [], any_shape cos
    ; "sin", [], any_shape sin
    ; "sqrt", [ `positive ], any_shape sqrt
    ; "log", [ `positive ], any_shape log
    ; "exp", [], any_shape exp
    ; "sigmoid", [], any_shape sigmoid
    ; "softplus", [], any_shape softplus
    ; "tanh", [], any_shape tanh
    ; "pdf", [], any_shape pdf (* ; "cdf", [], any_shape cdf *)
    ; "erf", [], any_shape erf
    ; "inv_sqr", [ `specified_unary [ 4; 4 ] ], any_shape inv_sqr
    ; "relu", [ `positive ], any_shape relu
    ; "soft_relu", [], any_shape soft_relu
    ; ( "sum"
      , []
      , fun shape ->
          let n_dims = List.length shape in
          let keepdim = Random.bool () in
          (* randomly choose the dimensions to be summed over.*)
          let dim =
            match Random.bool () with
            | true ->
              Some
                List.(
                  range 0 n_dims |> permute |> sub ~pos:0 ~len:Int.(1 + Random.int n_dims))
            | false -> None
          in
          sum ?dim ~keepdim )
    ; ( "slice"
      , [ `order_greater_than 4 ]
      , fun shape ->
          let n_dims = List.length shape in
          (* randomly choose the dimensions to be slice in .*)
          let dim = Random.int n_dims in
          let dim_length = List.nth_exn shape dim in
          let start = Random.int Int.(dim_length - 2) in
          let end_ =
            let tmp = Int.(1 + start + Random.int dim_length) in
            if tmp > dim_length then None else Some tmp
          in
          let step = Int.(1 + Random.int 1) in
          slice ~dim ~start ?end_ ~step )
    ; ( "mean"
      , []
      , fun shape ->
          let n_dims = List.length shape in
          let keepdim = Random.bool () in
          let dim =
            match Random.bool () with
            | true ->
              Some
                List.(
                  range 0 n_dims |> permute |> sub ~pos:0 ~len:Int.(1 + Random.int n_dims))
            | false -> None
          in
          mean ?dim ~keepdim )
    ; ( "transpose"
      , [ `order_greater_than 2 ]
      , fun shape ->
          let n_dims = List.length shape in
          let dims = List.(permute (range 0 n_dims)) in
          transpose ~dims )
    ; "btr", [ `order_greater_than 2 ], any_shape btr
    ; "diagonal", [ `order_greater_than 2 ], any_shape (diagonal ~offset:0)
    ; ( "diag_embed"
      , [ `order_greater_than 2 ]
      , any_shape (diag_embed ~offset:0 ~dim1:0 ~dim2:1) )
    ; "tril", [ `order_greater_than 2 ], any_shape (tril ~_diagonal:0)
    ; ( "logsumexp"
      , []
      , fun shape ->
          let n_dims = List.length shape in
          let dim =
            List.(
              range 0 n_dims |> permute |> sub ~pos:0 ~len:Int.(1 + Random.int n_dims))
          in
          let keepdim = Random.bool () in
          logsumexp ~dim ~keepdim )
    ; ( "cholesky"
      , [ `specified_unary [ 5; 5 ] ]
      , any_shape (fun x ->
          let xxt = x *@ transpose x in
          cholesky xxt) )
    ; ( "batch_cholesky"
      , [ `specified_unary [ 3; 5; 5 ] ]
      , any_shape (fun x ->
          let xxt = einsum [ x, "ijk"; x, "ilk" ] "ijl" in
          cholesky xxt) )
    ]
  in
  List.map ~f:test_unary test_list

(* generate the shape of 2 by 2 matrices where A *@ B is possible. *)
let random_mult_matrix_shapes () =
  let open Int in
  let first_dim = 1 + Random.int 3 in
  let second_dim = 1 + Random.int 3 in
  let third_dim = 1 + Random.int 3 in
  [ first_dim; second_dim ], [ second_dim; third_dim ]

(* generate 1. if left, the shape of A of shape [m x n x n] and B of shape [m x n x p] or B of shape [m x n]. n needs to be at least 2.
   2. if left is false, the shape of A of shape [m x n x n] and B of shape [m x p x n] or B of shape [m x n]. *)

let random_linsolve_matrix_shapes ~left =
  let open Int in
  let m = 3 + Random.int 5 in
  let n = 3 + Random.int 5 in
  let p = 3 + Random.int 5 in
  if left then [ m; n; n ], [ m; n; p ] else [ m; n; n ], [ m; p; n ]

let random_linsolve_2d_matrix_shapes () =
  let open Int in
  let m = 3 + Random.int 5 in
  let n = 3 + Random.int 5 in
  [ m; n; n ], [ m; n ]

(* this is how we test a binary function *)
(* for simplicity apply same constraint on both tensors. *)
let test_binary ((name, input_constr, f) : (_, _) binary) =
  ( name
  , `Quick
  , fun () ->
      List.range 0 n_tests
      |> List.iter ~f:(fun _ ->
        let shape =
          let min_order =
            List.fold input_constr ~init:None ~f:(fun accu c ->
              match accu with
              | Some d -> Some d
              | None ->
                (match c with
                 | `order_greater_than d -> Some d
                 | _ -> accu))
          in
          let set_order =
            List.fold input_constr ~init:None ~f:(fun accu c ->
              match accu with
              | Some d -> Some d
              | None ->
                (match c with
                 | `order_equal_to d -> Some d
                 | _ -> accu))
          in
          let specified_shape =
            List.fold input_constr ~init:None ~f:(fun accu c ->
              match accu with
              | Some d -> Some d
              | None ->
                (match c with
                 | `specified_binary shape -> Some shape
                 | _ -> accu))
          in
          let matmul_shape =
            List.fold input_constr ~init:None ~f:(fun accu c ->
              match accu with
              | Some d -> Some d
              | None ->
                (match c with
                 | `matmul -> Some random_mult_matrix_shapes
                 | _ -> accu))
          in
          let linsolve_shape =
            List.fold input_constr ~init:None ~f:(fun accu c ->
              match accu with
              | Some d -> Some d
              | None ->
                (match c with
                 | `linsolve_2d_left_true -> Some (random_linsolve_2d_matrix_shapes ())
                 | `linsolve_left_true -> Some (random_linsolve_matrix_shapes ~left:true)
                 | `linsolve_left_false ->
                   Some (random_linsolve_matrix_shapes ~left:false)
                 | _ -> accu))
          in
          match specified_shape, matmul_shape, linsolve_shape, set_order, min_order with
          | Some specified_shape, None, None, None, None -> specified_shape
          | None, Some matmul_shape, None, _, _ -> matmul_shape ()
          | None, None, Some linsolve_shape, _, _ -> linsolve_shape
          | None, None, None, Some d, _ ->
            let shape = random_shape_set d in
            shape, shape
          | None, None, None, None, Some d ->
            let shape = random_shape d in
            shape, shape
          | None, None, None, None, None ->
            let shape = random_shape 1 in
            shape, shape
          | _ -> assert false
        in
        let f = f (fst shape) in
        (* x and y same shape *)
        let x = generate_tensor ~shape:(fst shape) ~input_constr_list:input_constr in
        let y = generate_tensor ~shape:(snd shape) ~input_constr_list:input_constr in
        Alcotest.(check @@ rel_tol)
          name
          0.0
          (check_grad2 (fun (x, y) -> f x y) (of_tensor x) (of_tensor y))) )

let matmul_with_einsum a b = Maths.einsum [ a, "ij"; b, "jk" ] "ik"
let batch_matmul_with_einsum_33 a b = Maths.einsum [ a, "mij"; b, "mjk" ] "mik"
let batch_matmul_with_einsum_32 a b = Maths.einsum [ a, "mij"; b, "mj" ] "mi"
let batch_trans_matmul_with_einsum a b = Maths.(einsum [ a, "mij"; b, "mik" ] "mjk")
let batch_vecmat_with_einsum a b = Maths.(einsum [ a, "mi"; b, "mij" ] "mj")
let batch_vecmat_trans_with_einsum a b = Maths.(einsum [ a, "mi"; b, "mji" ] "mj")

let linsolve ~left a b =
  let a_primal = Maths.to_tensor a in
  let a_device = Tensor.device a_primal in
  let a_kind = Tensor.type_ a_primal in
  let n = List.last_exn (Tensor.shape a_primal) in
  (* improve condition number of a *)
  let a =
    Maths.(
      of_tensor
        Tensor.(
          a_primal
          + Tensor.(
              mul_scalar
                (eye ~n ~options:(a_kind, a_device))
                (Scalar.f Float.(10. *. of_int n)))))
  in
  Maths.linsolve ~left a b

let linsolve_tri ~left ~upper a b =
  let a =
    match upper with
    | false -> Maths.tril a ~_diagonal:0
    | true -> Maths.(btr (tril a ~_diagonal:0))
  in
  Maths.linsolve_triangular a b ~left ~upper

let concat_list a b = Maths.concat [ a; b ] ~dim:0
let block_diag a b = Maths.block_diag [ a; b ]

let binary_tests =
  let test_list : (_, _) binary list =
    [ "plus", [], any_shape Maths.( + )
    ; "minus", [], any_shape Maths.( - )
    ; "mult", [], any_shape Maths.( * )
    ; "div", [ `positive ], any_shape Maths.( / )
    ; "matmul", [ `matmul ], any_shape Maths.( *@ )
    ; "matmul_with_einsum", [ `matmul ], any_shape matmul_with_einsum
    ; ( "batch_matmul_with_einsum_33"
      , [ `specified_binary ([ 3; 4; 5 ], [ 3; 5; 7 ]) ]
      , any_shape batch_matmul_with_einsum_33 )
    ; ( "batch_matmul_with_einsum_32"
      , [ `specified_binary ([ 3; 4; 5 ], [ 3; 5 ]) ]
      , any_shape batch_matmul_with_einsum_32 )
    ; ( "batch_trans_matmul_with_einsum"
      , [ `specified_binary ([ 3; 4; 5 ], [ 3; 4; 7 ]) ]
      , any_shape batch_trans_matmul_with_einsum )
    ; ( "batch_vecmat_with_einsum"
      , [ `specified_binary ([ 3; 4 ], [ 3; 4; 7 ]) ]
      , any_shape batch_vecmat_with_einsum )
    ; ( "batch_vecmat_trans_with_einsum"
      , [ `specified_binary ([ 3; 4 ], [ 3; 8; 4 ]) ]
      , any_shape batch_vecmat_trans_with_einsum )
    ; "linsolve_left_true", [ `linsolve_left_true ], any_shape (linsolve ~left:true)
    ; "linsolve_left_false", [ `linsolve_left_false ], any_shape (linsolve ~left:false)
    ; ( "linsolve_tri_left_true_upper_true"
      , [ `linsolve_left_true ]
      , any_shape (linsolve_tri ~left:true ~upper:true) )
    ; ( "linsolve_tri_left_true_upper_false"
      , [ `linsolve_left_true ]
      , any_shape (linsolve_tri ~left:true ~upper:false) )
    ; ( "linsolve_tri_left_false_upper_true"
      , [ `linsolve_left_false ]
      , any_shape (linsolve_tri ~left:false ~upper:true) )
    ; ( "linsolve_tri_left_false_upper_false"
      , [ `linsolve_left_false ]
      , any_shape (linsolve_tri ~left:false ~upper:false) )
    ; "kron", [ `specified_binary ([ 3; 4 ], [ 5; 7 ]) ], any_shape Maths.kron
    ; "concat", [], any_shape concat_list
    ; "block_diag", [ `order_equal_to 2 ], any_shape block_diag
    ; ( "conv2d"
      , [ `order_equal_to 4 ]
      , fun _ ->
          let stride = Int.(1 + Random.int 1) in
          fun w x ->
            Maths.conv2d
              ~padding:(0, 0)
              ~dilation:(1, 1)
              ~groups:1
              ~bias:None
              ~stride:(stride, stride)
              ~w
              x )
    ]
  in
  List.map ~f:test_binary test_list

let _ =
  Alcotest.run
    "Maths tests"
    [ "Unary operations", unary_tests; "Binary operations", binary_tests ]
