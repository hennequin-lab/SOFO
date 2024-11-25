open Base
open Torch
open Forward_torch

let n_tests = 100
let device = Torch.Device.Cpu
let kind = Torch_core.Kind.(T f64)
let print s = Stdio.print_endline (Sexp.to_string_hum s)

(* generate a random shape with a specified minimum order. *)
let random_shape min_order =
  List.init (min_order + Random.int 3) ~f:(fun _ -> 3 + Random.int 4)

(* generate a random shape with a specified order *)
let random_shape_set order = List.init order ~f:(fun _ -> 2 + Random.int 4)
let rel_tol = Alcotest.float 1e-3

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
  let x_sym = Maths.(x *@ transpose x ~dim0:2 ~dim1:1) in
  let x_size = List.last_exn (Tensor.shape x_primal) in
  let x_final =
    Maths.(x_sym + const (Tensor.eye ~n:x_size ~options:(x_kind, x_device)))
  in
  Maths.cholesky x_final

let inv_sqr x =
  let x_primal = Maths.primal x in
  let x_device = Tensor.device x_primal in
  let x_kind = Tensor.type_ x_primal in
  (* make sure x is positive definite *)
  let x_sym = Maths.(x *@ transpose x ~dim0:2 ~dim1:1) in
  let x_size = List.last_exn (Tensor.shape x_primal) in
  let x_final =
    Maths.(x_sym + const (Tensor.eye ~n:x_size ~options:(x_kind, x_device)))
  in
  Maths.inv_sqr x_final

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
    ; "inv_sqr", [ `specified_unary [ 3; 10; 10 ] ], any_shape inv_sqr
    ; "inv_rectangle", [ `specified_unary [ 80; 15 ] ], any_shape Maths.inv_rectangle
    ; "sigmoid", [], any_shape Maths.sigmoid
    ; "softplus", [], any_shape Maths.softplus
    ; "tanh", [], any_shape Maths.tanh
    ; "relu", [ `positive ], any_shape Maths.relu
    ; "sum", [], any_shape Maths.sum
    ; "mean", [], any_shape Maths.mean
      (* ; "max2d_dim1", [ `order_equal_to 2 ], any_shape (Maths.max_2d_dim1 ~keepdim:false) *)
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
    ; "cholesky", [ `specified_unary [ 2; 14; 14 ] ], any_shape cholesky_test
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

(* each binary test is characterized by a name,
   a (potentially empty) list of constraints on the input,
   and a binary math function to be tested *)
type binary = string * input_constr list * (int list -> Maths.t -> Maths.t -> Maths.t)

(* generate the shape of 2 by 2 matrices where A *@ B is possible. *)
let random_mult_matrix_shapes () =
  let first_dim = 1 + Random.int 3 in
  let second_dim = 1 + Random.int 3 in
  let third_dim = 1 + Random.int 3 in
  [ first_dim; second_dim ], [ second_dim; third_dim ]

(* generate 1. if left, the shape of A of shape [m x n x n] and B of shape [m x n x p] or B of shape [m x n]. n needs to be at least 2.
   2. if left is false, the shape of A of shape [m x n x n] and B of shape [m x p x n] or B of shape [m x n]. *)

let random_linsolve_matrix_shapes ~left =
  let m = 3 + Random.int 50 in
  let n = 3 + Random.int 50 in
  let p = n - 1 in
  if left then [ m; n; n ], [ m; n; p ] else [ m; n; n ], [ m; p; n ]

let random_linsolve_2d_matrix_shapes () =
  let m = 3 + Random.int 50 in
  let n = 3 + Random.int 50 in
  [ m; n; n ], [ m; n ]

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
        Alcotest.(check @@ rel_tol) name 0.0 (Maths.check_grad2 f x y)) )

let matmul_with_einsum a b = Maths.einsum [ a, "ij"; b, "jk" ] "ik"
let batch_matmul_with_einsum a b = Maths.einsum [ a, "mij"; b, "mjk" ] "mik"
let batch_trans_matmul_with_einsum a b = Maths.(einsum [ a, "mij"; b, "mik" ] "mjk")
let batch_vecmat_with_einsum a b = Maths.(einsum [ a, "mi"; b, "mij" ] "mj")
let batch_vecmat_trans_with_einsum a b = Maths.(einsum [ a, "mi"; b, "mji" ] "mj")

let linsolve ~left a b =
  let a_primal = Maths.primal a in
  let a_device = Tensor.device a_primal in
  let a_kind = Tensor.type_ a_primal in
  let n = List.last_exn (Tensor.shape a_primal) in
  (* improve condition number of a *)
  let a =
    Maths.(
      a
      + const
          Tensor.(
            mul_scalar
              (eye ~n ~options:(a_kind, a_device))
              (Scalar.f Float.(1. *. of_int n))))
  in
  Maths.linsolve ~left a b

let linsolve_tri ~left ~upper a b =
  let a_primal = Maths.primal a in
  let n = List.last_exn (Tensor.shape a_primal) in
  let a_device = Tensor.device a_primal in
  let a_kind = Tensor.type_ a_primal in
  (* make sure x is positive definite *)
  let a_batch = if List.length (Tensor.shape (Maths.primal a)) = 3 then 1 else 0 in
  let aaT =
    Maths.(
      (a *@ transpose a ~dim0:Int.(a_batch + 1) ~dim1:a_batch)
      + const
          Tensor.(
            mul_scalar
              (eye ~n ~options:(a_kind, a_device))
              (Scalar.f Float.(1. *. of_int n))))
  in
  let a_lower = Maths.cholesky aaT in
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
    ; ( "batch_matmul_with_einsum"
      , [ `specified_binary ([ 30; 40; 50 ], [ 30; 50; 70 ]) ]
      , any_shape batch_matmul_with_einsum )
    ; ( "batch_trans_matmul_with_einsum"
      , [ `specified_binary ([ 30; 40; 50 ], [ 30; 40; 70 ]) ]
      , any_shape batch_trans_matmul_with_einsum )
    ; ( "batch_vecmat_with_einsum"
      , [ `specified_binary ([ 30; 40 ], [ 30; 40; 70 ]) ]
      , any_shape batch_vecmat_with_einsum )
    ; ( "batch_vecmat_trans_with_einsum"
      , [ `specified_binary ([ 30; 40 ], [ 30; 80; 40 ]) ]
      , any_shape batch_vecmat_trans_with_einsum )
    ; "linsolve_2d_left_true", [ `linsolve_2d_left_true ], any_shape (linsolve ~left:true)
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

(* -----------------------------------------
   -- Test for lqr operations       ------
   ----------------------------------------- *)

open Base
open Torch
open Forward_torch
open Sofo
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

let batch_size = 1

(* -----------------------------------------
   -- Define Control Problem          ------
   ----------------------------------------- *)

let base = Optimizer.Config.Base.default
let device_ = Torch.Device.Cpu

(* sample a symmetric positive definite matrix of size n *)
let create_sym_pos n =
  let q_1 = Tensor.randn [ n; n ] ~device:device_ in
  let qqT = Tensor.(matmul q_1 (transpose q_1 ~dim0:1 ~dim1:0)) in
  Tensor.(qqT + mul_scalar (eye ~n ~options:(base.kind, device_)) (Scalar.f 10.))
(* Tensor.eye ~n ~options:(base.kind, device_) *)

let generate_state_cost_params () =
  (* define control problem dimension *)
  let module Lds_params_dim = struct
    let a = 10
    let b = 50
    let n_steps = 10
    let kind = base.kind
    let device = device_
  end
  in
  let module Data = Lds_data.Make_LDS (Lds_params_dim) in
  (* for simplicity same q and r across trials and time *)
  let q_list =
    List.init Lds_params_dim.n_steps ~f:(fun _ ->
      let q_inv =
        create_sym_pos Lds_params_dim.a
        |> Tensor.reshape ~shape:[ 1; Lds_params_dim.a; Lds_params_dim.a ]
      in
      let q_list = List.init batch_size ~f:(fun _ -> q_inv) in
      Tensor.concat q_list ~dim:0)
  in
  let r_list =
    List.init Lds_params_dim.n_steps ~f:(fun _ ->
      let r_inv =
        create_sym_pos Lds_params_dim.b
        |> Tensor.reshape ~shape:[ 1; Lds_params_dim.b; Lds_params_dim.b ]
      in
      let r_list = List.init batch_size ~f:(fun _ -> r_inv) in
      Tensor.concat r_list ~dim:0)
  in
  (* returns the x0 mat, list of target mat. targets x go from t=1 to t=T and targets u go from t=0 to t=T-1. *)
  let sample_data bs =
    let batch_lds_params, x0, x_u_list = Data.batch_trajectory bs in
    let targets_list = List.map x_u_list ~f:(fun (x, _, _) -> x) in
    let target_controls_list = List.map x_u_list ~f:(fun (_, u, _) -> u) in
    let f_ts_list = List.map x_u_list ~f:(fun (_, _, f_t) -> f_t) in
    batch_lds_params, x0, targets_list, target_controls_list, f_ts_list
  in
  let batch_lds_params, x0, targets_list, target_controls_list, f_ts_list =
    sample_data batch_size
  in
  (* form state params/cost_params/xu_desired *)
  let state_params : Tensor.t Forward_torch.Lqr_typ.state_params =
    { n_steps = List.length targets_list
    ; x_0 = x0
    ; f_x_list = List.map batch_lds_params ~f:fst
    ; f_u_list = List.map batch_lds_params ~f:snd
    ; f_t_list = Some f_ts_list
    }
  in
  (* form c_x and c_u lists *)
  let batch_vecmat a b = Tensor.einsum [ a; b ] ~equation:"mi,mij->mj" ~path:None in
  let c_x_list =
    let tmp =
      List.map2_exn targets_list q_list ~f:(fun target q ->
        Tensor.(batch_vecmat (neg target) q))
    in
    Some
      (Tensor.zeros [ batch_size; Lds_params_dim.a ] ~device:Lds_params_dim.device :: tmp)
  in
  let c_u_list =
    let tmp =
      List.map2_exn target_controls_list r_list ~f:(fun target r ->
        Tensor.(batch_vecmat (neg target) r))
    in
    Some
      (tmp
       @ [ Tensor.zeros [ batch_size; Lds_params_dim.b ] ~device:Lds_params_dim.device ])
  in
  (* form cost parameters, which all go from step 0 to T.*)
  let cost_params : Tensor.t Forward_torch.Lqr_typ.cost_params =
    { c_xx_list =
        Tensor.zeros
          [ batch_size; Lds_params_dim.a; Lds_params_dim.a ]
          ~device:Lds_params_dim.device
        :: q_list
    ; c_xu_list =
        Some
          (List.init (Lds_params_dim.n_steps + 1) ~f:(fun _ ->
             Tensor.zeros
               [ batch_size; Lds_params_dim.a; Lds_params_dim.b ]
               ~device:Lds_params_dim.device))
    ; c_uu_list =
        r_list
        @ [ Tensor.zeros
              [ batch_size; Lds_params_dim.b; Lds_params_dim.b ]
              ~device:Lds_params_dim.device
          ]
    ; c_x_list
    ; c_u_list
    }
  in
  state_params, cost_params

(* each lqr test test is characterized by a name, the state params, the cost params and an lqr function to be tested *)
type lqr =
  string
  * (state_params:Maths.t Forward_torch.Lqr_typ.state_params
     -> cost_params:Maths.t Forward_torch.Lqr_typ.cost_params
     -> Maths.t list * Maths.t list)

let attach_tangents : Forward_torch.Lqr_typ.attach_tangents =
  { f_x_tan = false
  ; f_u_tan = false
  ; f_t_tan = false
  ; c_xx_tan = false
  ; c_xu_tan = false
  ; c_uu_tan = false
  ; c_x_tan = false
  ; c_u_tan = true
  }

(* this is how we test the lqr function *)
let test_lqr ((name, f) : lqr) =
  ( name
  , `Quick
  , fun () ->
      List.range 0 n_tests
      |> List.iter ~f:(fun _ ->
        (* x and y same shape *)
        let state_params, cost_params = generate_state_cost_params () in
        Alcotest.(check @@ rel_tol)
          name
          0.0
          (Maths.check_grad_lqr f ~state_params ~cost_params ~attach_tangents)) )

let lqr_tests =
  let test_list : lqr list = [ "lqr", Lqr.lqr (* "lqr_sep", Lqr.lqr_sep *) ] in
  List.map ~f:test_lqr test_list

let _ =
  Alcotest.run
    "Maths tests"
    [ (* "Unary operations", unary_tests *)
      "Binary operations", binary_tests 
      (* "LQR operations", lqr_tests *)
    ]
