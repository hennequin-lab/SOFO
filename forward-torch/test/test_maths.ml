open Base
open Torch
open Forward_torch

let n_tests = 100
let device = Torch.Device.Cpu
let kind = Torch_core.Kind.(T f64)

(* generate a random shape with a specified minimum order. *)
let random_shape min_order =
  List.init (min_order + Random.int 3) ~f:(fun _ -> 3 + Random.int 4)

(* generate a random shape with a specified order *)
let random_shape_set order = List.init order ~f:(fun _ -> 2 + Random.int 4)
let rel_tol = Alcotest.float 1e-4

(* two types of contraints: input tensors need to be postive or their orders need to be greater than specified. *)
type input_constr =
  [ `positive
  | `not_all_neg
  | `order_greater_than of int
  | `order_equal_to of int
  | `specified_unary of int list
  | `specified_binary of int list * int list
  | `matmul
  | `linsolve_left_true
  | `linsolve_left_false
  | `linsolve_tri_left_true
  | `linsolve_tri_left_false
  | `pos_def
  ]

(* each unary test test is characterized by a name,
   a (potentially empty) list of constraints on the input,
   and a unary math function to be tested *)
type unary = string * input_constr list * (int list -> Maths.t -> Maths.t)

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
    else if List.mem input_constr_list `not_all_neg ~equal:Poly.( = )
            (* guard against the all zero case *)
    then (
      let x_abs = Tensor.(abs x) in
      let y = Tensor.randint_like_low_dtype x_abs ~low:(-1) ~high:1 in
      Tensor.(add_scalar (x_abs * y) (Scalar.f 0.5)))
    else x
  in
  x

(* this is how we test a unary function *)
let test_unary ((name, input_constr, f) : unary) =
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
        Alcotest.(check @@ rel_tol) name 0.0 (Maths.check_grad1 f x)) )

let cholesky_test x =
  let x_primal = Maths.primal x in
  let x_device = Tensor.device x_primal in
  let x_kind = Tensor.type_ x_primal in
  (* make sure x is positive definite *)
  let x_sym = Maths.(0.5 $* x + transpose x ~dim0:1 ~dim1:0) in
  let x_size = List.last_exn (Tensor.shape x_primal) in
  let x_final =
    Maths.(x_sym + const (Tensor.eye ~n:x_size ~options:(x_kind, x_device)))
  in
  Maths.cholesky x_final

let unary_tests =
  let test_list : unary list =
    [ ( "permute"
      , [ `specified_unary [ 2; 3; 6; 4 ] ]
      , any_shape (Maths.permute ~dims:[ 0; 2; 3; 1 ]) )
    ; "sqr", [], any_shape Maths.sqr
    ; "neg", [], any_shape Maths.neg
    ; "trace", [ `specified_unary [ 10; 10 ] ], any_shape Maths.trace
    ; ( "trace_with_einsum"
      , [ `specified_unary [ 10; 10 ] ]
      , any_shape (fun a -> Maths.einsum [ a, "ii" ] "") )
    ; "cos", [], any_shape Maths.cos
    ; "sin", [], any_shape Maths.sin
    ; "sqrt", [ `positive ], any_shape Maths.sqrt
    ; "log", [ `positive ], any_shape Maths.log
    ; "exp", [], any_shape Maths.exp
    ; "inv_sqr", [ `specified_unary [ 10; 10 ] ], any_shape Maths.inv_sqr
    ; "inv_rectangle", [ `specified_unary [ 80; 15 ] ], any_shape Maths.inv_rectangle
    ; "sigmoid", [], any_shape Maths.sigmoid
    ; "softplus", [], any_shape Maths.softplus
    ; "tanh", [], any_shape Maths.tanh
    ; "relu", [ `positive ], any_shape Maths.relu
    ; "sum", [], any_shape Maths.sum
    ; "mean", [], any_shape Maths.mean
    ; "max2d_dim1", [ `order_equal_to 2 ], any_shape (Maths.max_2d_dim1 ~keepdim:false)
    ; ( "gumbel_softmax"
      , [ `positive; `order_equal_to 2 ]
      , any_shape (Maths.gumbel_softmax ~tau:2. ~with_noise:false ~discrete:false) )
    ; ( "sum_dim"
      , []
      , fun shape ->
          let n_dims = List.length shape in
          let keepdim = Random.bool () in
          (* randomly choose the dimensions to be summed over.*)
          let dim =
            List.(range 0 n_dims |> permute |> sub ~pos:0 ~len:(1 + Random.int n_dims))
          in
          Maths.sum_dim ~dim:(Some dim) ~keepdim )
    ; ( "slice"
      , [ `order_greater_than 4 ]
      , fun shape ->
          let n_dims = List.length shape in
          (* randomly choose the dimensions to be slice in .*)
          let dim = Random.int n_dims in
          let dim_length = List.nth_exn shape dim in
          let start = Random.int Int.(dim_length - 2) in
          let some_end =
            let tmp = 1 + start + Random.int dim_length in
            if tmp > dim_length then None else Some tmp
          in
          let step = 1 + Random.int 1 in
          Maths.slice ~dim ~start:(Some start) ~end_:some_end ~step )
    ; ( "mean_dim"
      , []
      , fun shape ->
          let n_dims = List.length shape in
          let keepdim = Random.bool () in
          let dim =
            List.(range 0 n_dims |> permute |> sub ~pos:0 ~len:(1 + Random.int n_dims))
          in
          Maths.mean_dim ~dim:(Some dim) ~keepdim )
    ; ( "transpose"
      , [ `order_greater_than 2 ]
      , fun shape ->
          let n_dims = List.length shape in
          let[@warning "-8"] (dim0 :: dim1 :: _) =
            List.permute (List.init n_dims ~f:Fn.id)
          in
          Maths.transpose ~dim0 ~dim1 )
    ; "cholesky", [ `pos_def; `specified_unary [ 14; 14 ] ], any_shape cholesky_test
    ; ( "logsumexp"
      , []
      , fun shape ->
          let n_dims = List.length shape in
          let dim =
            List.(range 0 n_dims |> permute |> sub ~pos:0 ~len:(1 + Random.int n_dims))
          in
          let keepdim = Random.bool () in
          Maths.logsumexp ~dim ~keepdim )
    ; ( "maxpool2d"
      , [ `order_equal_to 4 ]
      , fun _ ->
          let ksize = 1 + Random.int 2 in
          let stride = 1 + Random.int 1 in
          Maths.maxpool2d
            ~padding:(0, 0)
            ~dilation:(1, 1)
            ~ceil_mode:false
            ~stride:(stride, stride)
            ~ksize:(ksize, ksize) )
    ]
  in
  List.map ~f:test_unary test_list

(* each binary test test is characterized by a name,
   a (potentially empty) list of constraints on the input,
   and a binary math function to be tested *)
type binary = string * input_constr list * (int list -> Maths.t -> Maths.t -> Maths.t)

let print s = Stdio.print_endline (Sexp.to_string_hum s)

(* generate the shape of 2 by 2 matrices where A *@ B is possible. *)
let random_mult_matrix_shapes () =
  let first_dim = 1 + Random.int 3 in
  let second_dim = 1 + Random.int 3 in
  let third_dim = 1 + Random.int 3 in
  [ first_dim; second_dim ], [ second_dim; third_dim ]

(* generate 1. if left, the shape of A of shape [m x n x n] and B of shape [m x n x p] or B of shape [m x n]. n needs to be at least 2.
   2. if left is false, the shape of A of shape [m x n x n] and B of shape [m x p x n] or B of shape [m x n]. *)

let random_linsolve_matrix_shapes ~left =
  let m = 1 + Random.int 3 in
  let n = 2 + Random.int 3 in
  let p = 1 + Random.int 3 in
  if left then [ m; n; n ], [ m; n; p ] else [ m; n; n ], [ m; p; n ]

(* this is how we test a binary function *)
(* for simplicity apply same constraint on both tensors. *)
let test_binary ((name, input_constr, f) : binary) =
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
                 | `linsolve_left_true -> Some (random_linsolve_matrix_shapes ~left:true)
                 | `linsolve_left_false ->
                   Some (random_linsolve_matrix_shapes ~left:false)
                 | _ -> accu))
          in
          match matmul_shape, linsolve_shape, set_order, min_order with
          | Some matmul_shape, None, _, _ -> matmul_shape ()
          | None, Some linsolve_shape, _, _ -> linsolve_shape
          | None, None, Some d, _ ->
            let shape = random_shape_set d in
            shape, shape
          | None, None, None, Some d ->
            let shape = random_shape d in
            shape, shape
          | None, None, None, None ->
            let shape = random_shape 1 in
            shape, shape
          | _ -> assert false
        in
        let f = f (fst shape) in
        (* x and y same shape *)
        let x = generate_tensor ~shape:(fst shape) ~input_constr_list:input_constr in
        let y = generate_tensor ~shape:(snd shape) ~input_constr_list:input_constr in
        Alcotest.(check @@ rel_tol) name 0.0 (Maths.check_grad2 f x y)) )

let matmul_with_einsum a b = Maths.einsum [ a, "ij"; b, "jk" ] "ik"

let matmul_with_einsum2 a b =
  Maths.einsum [ a, "ij"; Maths.transpose ~dim0:0 ~dim1:1 b, "kj" ] "ik"

let linsolve_tri ~left ~upper a b =
  let a_primal = Maths.primal a in
  let a_size = List.last_exn (Tensor.shape a_primal) in
  let a_device = Tensor.device a_primal in
  let a_kind = Tensor.type_ a_primal in
  (* make sure x is positive definite *)
  let a_batch = if List.length (Tensor.shape (Maths.primal a)) = 3 then 1 else 0 in
  let aaT = Maths.(a *@ transpose a ~dim0:Int.(a_batch + 1) ~dim1:a_batch) in
  let a_lower = Maths.cholesky aaT in
  let a_lower =
    Maths.(a_lower + const (Tensor.eye ~n:a_size ~options:(a_kind, a_device)))
  in
  let a_final =
    if upper
    then Maths.transpose a_lower ~dim0:Int.(a_batch + 1) ~dim1:a_batch
    else a_lower
  in
  Maths.linsolve_triangular a_final b ~left ~upper

let binary_tests =
  let test_list : binary list =
    [ "plus", [], any_shape Maths.( + )
    ; "minus", [], any_shape Maths.( - )
    ; "mult", [], any_shape Maths.( * )
    ; "div", [ `positive ], any_shape Maths.( / )
    ; "matmul", [ `matmul ], any_shape Maths.( *@ )
    ; "matmul_with_einsum", [ `matmul ], any_shape matmul_with_einsum
    ; "matmul_with_einsum2", [ `matmul ], any_shape matmul_with_einsum2
    ; "linsolve_left_true", [ `linsolve_left_true ], any_shape (Maths.linsolve ~left:true)
    ; ( "linsolve_left_false"
      , [ `linsolve_left_false ]
      , any_shape (Maths.linsolve ~left:false) )
    ; ( "linsolve_tri_left_true"
      , [ `linsolve_left_true ]
      , any_shape (linsolve_tri ~left:true ~upper:false) )
    ; ( "linsolve_tri_left_false"
      , [ `linsolve_left_false ]
      , any_shape (linsolve_tri ~left:false ~upper:true) )
    ; ( "concat"
      , []
      , fun shape ->
          let n_dims = List.length shape in
          let dim = Random.int n_dims in
          Maths.concat ~dim )
    ; ( "conv2d"
      , [ `order_equal_to 4 ]
      , fun shape ->
          let out_channel = List.hd_exn shape in
          let b = Tensor.zeros ~kind ~device [ out_channel ] in
          let stride = 1 + Random.int 1 in
          Maths.conv2d
            ~padding:(0, 0)
            ~dilation:(1, 1)
            ~groups:1
            ~bias:(b, None)
            ~stride:(stride, stride) )
    ]
  in
  List.map ~f:test_binary test_list

let _ =
  Alcotest.run
    "Maths tests"
    [ "Unary operations", unary_tests; "Binary operations", binary_tests ]
