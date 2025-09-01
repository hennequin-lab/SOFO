open Base
open Torch
open Forward_torch

let get_svals_u_left_right left right =
  let u_left, s_left, _ = Tensor.svd ~some:true ~compute_uv:true Maths.(to_tensor left) in
  let u_right, s_right, _ =
    Tensor.svd ~some:true ~compute_uv:true Maths.(to_tensor right)
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
  s_all, u_left, u_right

let get_local_vs ~selection ~s_all ~u_left ~u_right =
  List.map selection ~f:(fun idx ->
    let il, ir, _ = s_all.(idx) in
    let slice_and_squeeze t dim idx =
      Tensor.squeeze_dim
        ~dim
        (Tensor.slice t ~dim ~start:(Some idx) ~end_:(Some (idx + 1)) ~step:1)
    in
    let u_l = slice_and_squeeze u_left 1 il in
    let u_r = slice_and_squeeze u_right 1 ir in
    let tmp = Tensor.einsum ~path:None ~equation:"i,j->ij" [ u_l; u_r ] in
    Tensor.unsqueeze tmp ~dim:0)
  |> Tensor.concatenate ~dim:0

module GGN_Common (Spec : Spec_typ.SPEC) = struct
  let zero_params ~shape ~device ~kind _K = Tensor.zeros ~device ~kind (_K :: shape)
  let random_params ~shape ~device ~kind _K = Tensor.randn ~device ~kind (_K :: shape)

  let get_total_n_params param_name =
    let list_prod l = List.fold l ~init:1 ~f:(fun accu i -> accu * i) in
    list_prod (Spec.shape param_name)

  let get_n_params_before_after param_name =
    let idx =
      List.findi_exn Spec.all ~f:(fun _ p -> Spec.equal_param_name p param_name) |> fst
    in
    let before = List.take Spec.n_params_list idx |> List.sum (module Int) ~f:Fn.id in
    let after =
      List.drop Spec.n_params_list (idx + 1) |> List.sum (module Int) ~f:Fn.id
    in
    before, after

  (* cache storage with a ref to memoize computed results *)
  let s_u_cache = ref []

  (* compute eigenvalues + eigenvectors for given param *)
  let eigenvectors_for_params ~lambda ~get_sides ~param_name =
    let left, right = get_sides lambda param_name in
    get_svals_u_left_right left right

  (* given param name, get eigenvalues and eigenvectors. *)
  let get_s_u ~lambda ~get_sides ~param_name =
    match List.Assoc.find !s_u_cache param_name ~equal:Spec.equal_param_name with
    | Some s -> s
    | None ->
      let s = eigenvectors_for_params ~lambda ~get_sides ~param_name in
      s_u_cache := (param_name, s) :: !s_u_cache;
      s

  let extract_local_vs ~localise ~s_all ~param_name ~u_left ~u_right ~selection =
    let n_per_param = Spec.n_params param_name in
    let local_vs = get_local_vs ~selection ~s_all ~u_left ~u_right in
    localise ~param_name ~n_per_param local_vs

  let eigenvectors_for_each_param
        ~localise
        ~lambda
        ~param_name
        ~get_sides
        ~sampling_state:_
    =
    let n_per_param = Spec.n_params param_name in
    let n_params = get_total_n_params param_name in
    let s_all, u_left, u_right = get_s_u ~lambda ~get_sides ~param_name in
    let selection =
      (* List.init n_per_param ~f:(fun i ->
              ((sampling_state * n_per_param) + i) % n_params) *)
      List.permute (List.range 0 n_params) |> List.sub ~pos:0 ~len:n_per_param
    in
    extract_local_vs ~localise ~s_all ~param_name ~u_left ~u_right ~selection

  let eigenvectors
        ~lambda
        ~(switch_to_learn : bool)
        ~(sampling_state : int)
        ~get_sides
        ~combine
        ~wrap
        ~localise
    =
    let eigenvectors_each =
      List.map Spec.all ~f:(fun param_name ->
        eigenvectors_for_each_param
          ~localise
          ~lambda
          ~param_name
          ~get_sides
          ~sampling_state)
    in
    let vs =
      List.fold eigenvectors_each ~init:None ~f:(fun accu local_vs ->
        match accu with
        | None -> Some local_vs
        | Some a -> Some (combine a local_vs))
    in
    let vs = Option.map vs ~f:wrap in
    if switch_to_learn then s_u_cache := [];
    let new_sampling_state = if switch_to_learn then 0 else sampling_state + 1 in
    Option.value_exn vs, new_sampling_state
end
