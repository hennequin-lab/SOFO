open Printf
open Base
open Torch
open Forward_torch
open Sofo
open Motor
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S
open Single_reach_common
open Rnn_typ

let n_input_channels = 3
let max_iter = 5000

let _ =
  Random.init 1999;
  (* Random.self_init (); *)
  Owl_stats_prng.init (Random.int 100000);
  Torch_core.Wrapper.manual_seed (Random.int 100000)

(* ------------------------------------------------
   --- Kronecker approximation of the GGN
   ------------------------------------------------ *)
let _K = 256

type param_name =
  | Init_cond
  | W
  | B
  | F of int
  | C of int

let n_params_init_cond, n_params_w, n_params_b, n_params_f, n_params_c =
  10, Int.(_K - (9 * 10)), 20, 10, 10

let n_params_list =
  [ n_params_init_cond
  ; n_params_w
  ; n_params_b
  ; n_params_f
  ; n_params_f
  ; n_params_f
  ; n_params_f
  ; n_params_c
  ; n_params_c
  ]

let param_names_list = [ Init_cond; W; B; F 1; F 2; F 3; F 4; C 1; C 2 ]

module P = PP.Make (Prms.P)

module GGN : Wrapper.Auxiliary with module P = P = struct
  include struct
    type 'a p =
      { init_cond_left : 'a
      ; init_cond_right : 'a
      ; w_left : 'a
      ; w_right : 'a
      ; b_left : 'a
      ; b_right : 'a
      ; f1_left : 'a
      ; f1_right : 'a
      ; f2_left : 'a
      ; f2_right : 'a
      ; f3_left : 'a
      ; f3_right : 'a
      ; f4_left : 'a
      ; f4_right : 'a
      ; c1_left : 'a
      ; c1_right : 'a
      ; c2_left : 'a
      ; c2_right : 'a
      }
    [@@deriving prms]
  end

  module P = P
  module A = Make (Prms.P)

  type sampling_state = unit

  let init_sampling_state () = ()

  let zero_params ~shape _K =
    Tensor.zeros ~device:base.device ~kind:base.kind (_K :: shape)

  let random_params ~shape _K () =
    Tensor.randn ~device:base.device ~kind:base.kind (_K :: shape)

  let get_shapes (param_name : param_name) =
    match param_name with
    | Init_cond -> [ 1; n ]
    | W -> [ n + 1; n ]
    | B -> [ n_input_channels; n ]
    | F _ -> [ 1; n ]
    | C _ -> [ n; 1 ]

  let get_n_params (param_name : param_name) =
    match param_name with
    | Init_cond -> n_params_init_cond
    | W -> n_params_w
    | B -> n_params_b
    | F _ -> n_params_f
    | C _ -> n_params_c

  let get_n_params_before_after (param_name : param_name) =
    let n_params_prefix_suffix_sums = Convenience.prefix_suffix_sums n_params_list in
    let param_idx =
      match param_name with
      | Init_cond -> 0
      | W -> 1
      | B -> 2
      | F i -> Int.(i + 2)
      | C i -> Int.(i + 6)
    in
    List.nth_exn n_params_prefix_suffix_sums param_idx

  (* approximation defined implicitly via Gv products *)
  let g12v ~(lambda : A.M.t) (v : P.M.t) : P.M.t =
    let open Maths in
    let tmp_einsum left right w = einsum [ left, "in"; w, "aij"; right, "jm" ] "anm" in
    let init_cond = tmp_einsum lambda.init_cond_left lambda.init_cond_right v.init_cond in
    let b = tmp_einsum lambda.b_left lambda.b_right v.b in
    let w = tmp_einsum lambda.w_left lambda.w_right v.w in
    let f1 = tmp_einsum lambda.f1_left lambda.f1_right v.f1 in
    let f2 = tmp_einsum lambda.f2_left lambda.f2_right v.f2 in
    let f3 = tmp_einsum lambda.f3_left lambda.f3_right v.f3 in
    let f4 = tmp_einsum lambda.f4_left lambda.f4_right v.f4 in
    let c1 = tmp_einsum lambda.c1_left lambda.c1_right v.c1 in
    let c2 = tmp_einsum lambda.c2_left lambda.c2_right v.c2 in
    { init_cond; b; w; f1; f2; f3; f4; c1; c2 }

  (* set tangents = zero for other parameters but v for this parameter *)
  let localise ~local ~param_name ~n_per_param v =
    let init_cond = random_params ~shape:(get_shapes Init_cond) n_per_param () in
    let w = random_params ~shape:(get_shapes W) n_per_param () in
    let b = random_params ~shape:(get_shapes B) n_per_param () in
    let _f = random_params ~shape:(get_shapes (F 0)) n_per_param in
    let _c = random_params ~shape:(get_shapes (C 0)) n_per_param in
    let params_tmp =
      PP.
        { init_cond
        ; w
        ; b
        ; f1 = _f ()
        ; f2 = _f ()
        ; f3 = _f ()
        ; f4 = _f ()
        ; c1 = _c ()
        ; c2 = _c ()
        }
    in
    match param_name with
    | Init_cond -> { params_tmp with init_cond = v }
    | W -> { params_tmp with w = v }
    | B -> { params_tmp with b = v }
    | F 1 -> { params_tmp with f1 = v }
    | F 2 -> { params_tmp with f2 = v }
    | F 3 -> { params_tmp with f3 = v }
    | F 4 -> { params_tmp with f4 = v }
    | C 1 -> { params_tmp with c1 = v }
    | C 2 -> { params_tmp with c2 = v }
    | _ -> assert false

  let random_localised_vs _K : P.T.t =
    let random_localised_param_name param_name =
      let w_shape = get_shapes param_name in
      let before, after = get_n_params_before_after param_name in
      let w = random_params ~shape:w_shape (get_n_params param_name) () in
      let zeros_before = zero_params ~shape:w_shape before in
      let zeros_after = zero_params ~shape:w_shape after in
      let final = Tensor.concat [ zeros_before; w; zeros_after ] ~dim:0 in
      final
    in
    { init_cond = random_localised_param_name Init_cond
    ; w = random_localised_param_name W
    ; b = random_localised_param_name B
    ; f1 = random_localised_param_name (F 1)
    ; f2 = random_localised_param_name (F 2)
    ; f3 = random_localised_param_name (F 3)
    ; f4 = random_localised_param_name (F 4)
    ; c1 = random_localised_param_name (C 1)
    ; c2 = random_localised_param_name (C 2)
    }

  let eigenvectors_for_each_params ~local ~lambda ~param_name =
    let left, right, n_per_param =
      match param_name with
      | Init_cond -> lambda.init_cond_left, lambda.init_cond_right, n_params_init_cond
      | W -> lambda.w_left, lambda.w_right, n_params_w
      | B -> lambda.b_left, lambda.b_right, n_params_b
      | F 1 -> lambda.f1_left, lambda.f1_right, n_params_f
      | F 2 -> lambda.f2_left, lambda.f2_right, n_params_f
      | F 3 -> lambda.f3_left, lambda.f3_right, n_params_f
      | F 4 -> lambda.f4_left, lambda.f4_right, n_params_f
      | C 1 -> lambda.c1_left, lambda.c1_right, n_params_c
      | C 2 -> lambda.c2_left, lambda.c2_right, n_params_c
      | _ -> assert false
    in
    let u_left, s_left, _ = Tensor.svd ~some:true ~compute_uv:true Maths.(primal left) in
    let u_right, s_right, _ =
      Tensor.svd ~some:true ~compute_uv:true Maths.(primal right)
    in
    let s_left = Tensor.to_float1_exn s_left |> Array.to_list in
    let s_right = Tensor.to_float1_exn s_right |> Array.to_list in
    let s_all =
      List.mapi s_left ~f:(fun il sl ->
        List.mapi s_right ~f:(fun ir sr -> il, ir, Float.(sl * sr)))
      |> List.concat
      |> List.sort ~compare:(fun (_, _, a) (_, _, b) -> Float.compare b a)
      |> Array.of_list
    in
    (* randomly select the indices *)
    let n_params =
      Convenience.first_dim (Maths.primal left)
      * Convenience.first_dim (Maths.primal right)
    in
    let selection =
      List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    let selection = List.map selection ~f:(fun j -> s_all.(j)) in
    let local_vs =
      List.map selection ~f:(fun (il, ir, _) ->
        let u_left =
          Tensor.(
            squeeze_dim
              ~dim:1
              (slice u_left ~dim:1 ~start:(Some il) ~end_:(Some Int.(il + 1)) ~step:1))
        in
        let u_right =
          Tensor.(
            squeeze_dim
              ~dim:1
              (slice u_right ~dim:1 ~start:(Some ir) ~end_:(Some Int.(ir + 1)) ~step:1))
        in
        let tmp = Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_left; u_right ] in
        Tensor.unsqueeze tmp ~dim:0)
      |> Tensor.concatenate ~dim:0
    in
    local_vs |> localise ~local ~param_name ~n_per_param

  let eigenvectors ~(lambda : A.M.t) () (_K : int) =
    let eigenvectors_each =
      List.map param_names_list ~f:(fun param_name ->
        eigenvectors_for_each_params ~local:true ~lambda ~param_name)
    in
    let vs =
      List.fold eigenvectors_each ~init:None ~f:(fun accu local_vs ->
        match accu with
        | None -> Some local_vs
        | Some a -> Some (P.map2 a local_vs ~f:(fun x y -> Tensor.concat ~dim:0 [ x; y ])))
    in
    Option.value_exn vs, ()

  let init () =
    let init_eye size =
      Mat.(0.1 $* eye size) |> Tensor.of_bigarray ~device:base.device |> Prms.free
    in
    { init_cond_left = init_eye 1
    ; init_cond_right = init_eye n
    ; w_left = init_eye (n + 1)
    ; w_right = init_eye n
    ; b_left = init_eye n_input_channels
    ; b_right = init_eye n
    ; f1_left = init_eye 1
    ; f1_right = init_eye n
    ; f2_left = init_eye 1
    ; f2_right = init_eye n
    ; f3_left = init_eye 1
    ; f3_right = init_eye n
    ; f4_left = init_eye 1
    ; f4_right = init_eye n
    ; c1_left = init_eye n
    ; c1_right = init_eye 1
    ; c2_left = init_eye n
    ; c2_right = init_eye 1
    }
end

(* -----------------------------------------
   --- Training for single reaches
   ----------------------------------------- *)

module Run (X : sig
    module O :
      Optimizer.T with module W.P = W.P and type W.data = W.data and type W.args = W.args

    val config : iter:int -> (float, Bigarray.float32_elt) O.config

    (* val init : O.W.P.t -> O.state *)
    val name : string
    val init : O.state
  end) =
struct
  open X

  let save_samples prms =
    let bs = 8 in
    let trials = List.init bs ~f:(fun _ -> trial ()) in
    let input = W.input_of trials in
    let traj = R.forward ~t_max:t_tot ~prms input |> Array.of_list in
    let to_mat x = Tensor.to_bigarray ~kind:base.ba_kind x in
    let to_mat' x = to_mat (Maths.primal x) in
    List.iteri trials ~f:(fun i trial ->
      let hand =
        Array.map traj ~f:(fun r ->
          let h = Arm.hand_of r.arm in
          let x1 = Mat.get (to_mat' h.Arm.pos.x1) i 0 in
          let x2 = Mat.get (to_mat' h.Arm.pos.x2) i 0 in
          Mat.of_array [| x1; x2 |] 1 2)
        |> Mat.concatenate ~axis:0
      in
      let network =
        if i = 0 || i = 1
        then
          Array.map traj ~f:(fun r -> Mat.row (to_mat' r.network) i)
          |> Mat.concatenate ~axis:0
          |> Option.some
        else None
      in
      let torques =
        Array.map traj ~f:(fun r ->
          Mat.(row (to_mat' (fst r.torques)) i @|| row (to_mat' (snd r.torques)) i))
        |> Mat.concatenate ~axis:0
      in
      let target = Mat.(of_array [| trial.target.pos.x1; trial.target.pos.x2 |] 1 2) in
      Mat.save_txt ~out:(in_dir (sprintf "target%i_%s" i name)) target;
      Option.iter network ~f:(Mat.save_txt ~out:(in_dir (sprintf "x%i_%s" i name)));
      Mat.save_txt ~out:(in_dir (sprintf "h%i_%s" i name)) hand;
      Mat.save_txt ~out:(in_dir (sprintf "tau%i_%s" i name)) torques)

  let rec loop k wallclock state =
    if k < 1001
    then (
      let data = t_tot, List.init bs ~f:(fun _ -> trial ()) in
      let tic = Unix.gettimeofday () in
      let loss, new_state = O.step ~config:(config ~iter:k) ~state ~data () in
      let toc = Unix.gettimeofday () in
      let it_took = Float.(toc - tic) in
      (* guards against spikes in the loss *)
      let wallclock = Float.(wallclock + it_took) in
      print [%message (k : int) (loss : float)];
      Owl.Mat.(
        save_txt
          ~append:true
          ~out:(in_dir name)
          (of_array [| Float.of_int k; wallclock; loss |] 1 3));
      if k % 10 = 0
      then (
        let prms = W.P.value (O.params state) in
        (* if k % 50 = 0 then save_samples (W.P.map ~f:Maths.const prms); *)
        (* save the parameters so they can later be loaded for analysis *)
        prms |> W.P.T.save ~kind:base.ba_kind ~out:(in_dir (sprintf "prms_%s.bin" name)));
      loop (k + 1) wallclock new_state)

  let prms = R.init ~base ~n_input_channels

  let run () =
    Bos.Cmd.(v "rm" % "-f" % in_dir name) |> Bos.OS.Cmd.run |> ignore;
    Bos.Cmd.(v "rm" % "-f" % in_dir "aux") |> Bos.OS.Cmd.run |> ignore;
    loop 0 0. init
end

(* -----------------------------------------
   --- Adam optimiser
   ----------------------------------------- *)

module With_Adam = struct
  module O = Optimizer.Adam (W)

  let config ~iter:_ =
    Optimizer.Config.Adam.
      { default with
        base = Arm.base
      ; learning_rate = Some 0.001
      ; beta_2 = 0.999
      ; eps = 1e-7
      }

  let init = O.init (R.init ~base ~n_input_channels)
  let name = "adam"
end

(* -----------------------------------------
     --- SGN optimiser
     ----------------------------------------- *)
module With_standard = struct
  module O = Optimizer.SOFO (W) (GGN)

  let config ~iter:_ =
    let aux =
      Optimizer.Config.SOFO.
        { (default_aux (in_dir "aux")) with
          config =
            Optimizer.Config.Adam.
              { default with base; learning_rate = Some 1e-2; eps = 1e-8 }
        ; steps = 5
        }
    in
    Optimizer.Config.SOFO.
      { base = Arm.base
      ; learning_rate = Some 0.1
      ; rank_one = false
      ; n_tangents = _K
      ; damping = Some 1e-5
      ; aux = Some aux
      }

  let init = O.init (R.init ~base ~n_input_channels)
  let name = "sofo"
end

(* -----------------------------------------
   --- Run the opt loop
   ----------------------------------------- *)

let _ =
  match Cmdargs.(get_string "-m" |> force ~usage:"-m [adam|sofo]") with
  | "adam" ->
    let module R = Run (With_Adam) in
    R.run ()
  | "sofo" ->
    let module R = Run (With_standard) in
    R.run ()
  | _ -> failwith "bad method option"
