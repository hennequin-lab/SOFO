open Base
open Torch
open Forward_torch
open Maths
open Sofo

module Prune (P : Prms.T) = struct
  open Torch

  (* count number of elements given shape *)
  let numel shape = List.fold ~init:1 ~f:Int.( * ) shape
  let numel_tensor x = numel (Tensor.shape x)
  let numel_maths x = numel (Maths.shape x)

  let convert_bool_mask_to_float ~type_ mask =
    P.map mask ~f:(fun x ->
      let x_f = Torch.Tensor.to_type (to_tensor x) ~type_ in
      of_tensor x_f)

  let convert_float_mask_to_bool mask =
    P.map mask ~f:(fun x ->
      let x_f = Torch.Tensor.to_type C.(to_tensor x) ~type_:Torch_core.Kind.(T Bool) in
      of_tensor x_f)

  (* self define mask with [sparsity] values are 1 and [1-sparsity] values are 0 *)
  let mask_p ~sparsity theta =
    P.map theta ~f:(fun x ->
      (* value 1 with probability [sparsity] *)
      let x_t = to_tensor x in
      Torch.Tensor.bernoulli_float_ x_t ~p:sparsity |> of_tensor)

  (* Flatten a P.t into a single vector *)
  let flatten (x : _ some P.t) =
    P.fold x ~init:[] ~f:(fun accu (x, _) ->
      let x_reshaped = Torch.Tensor.reshape (to_tensor x) ~shape:[ -1; 1 ] in
      x_reshaped :: accu)
    |> Torch.Tensor.concat ~dim:0

  (* count number of ones in mask tensor *)
  let count_mask_tensor mask = Torch.Tensor.masked_select mask ~mask |> numel_tensor

  (* count number of ones in the entire mask *)
  let count_remaining mask =
    P.fold mask ~init:0 ~f:(fun accu (mask, _) ->
      let num = count_mask_tensor (to_tensor mask) in
      Int.(accu + num))

  let count_sparsity mask =
    let n_remaining = count_remaining mask in
    let n_total = P.numel mask in
    Float.(of_int n_remaining / of_int n_total)

  (* Select only entries that are 1 in prev tensor *)
  let surviving_values_tensor ~mask_prev values =
    let surviving_values =
      match mask_prev with
      | None -> values
      | Some mask_prev -> Tensor.masked_select values ~mask:mask_prev
    in
    Tensor.reshape surviving_values ~shape:[ -1; 1 ]

  (* Select only entries that are 1 in prev mask and flatten *)
  let surviving_values ~mask_prev values =
    let values_flat = flatten values in
    match mask_prev with
    | None -> values_flat
    | Some mask_prev ->
      let mask_prev_flat = flatten mask_prev in
      Tensor.masked_select values_flat ~mask:mask_prev_flat
      |> Tensor.reshape ~shape:[ -1; 1 ]

  (* Find the pth smallest absolute value in a Tensor *)
  let threshold_of_values ~p ~min_required surviving_values =
    let v = Tensor.reshape surviving_values ~shape:[ -1; 1 ] in
    let n = List.hd_exn (Tensor.shape v) in
    let sorted, _ = Tensor.sort (Tensor.abs v) ~dim:0 ~descending:false in
    let raw_idx = Float.(to_int (of_int n *. p)) in
    (* Enforce the minimum number kept and valid range *)
    let idx = raw_idx |> Int.max Int.(min_required - 1) |> Int.min Int.(n - 1) in
    Tensor.get_float2 sorted idx 0

  let combine_mask_tensor m_prev m_new =
    Tensor.logical_and (to_tensor m_prev) (to_tensor m_new) |> of_tensor

  (* Combine with previous mask *)
  let combine ~mask_prev mask_new =
    match mask_prev with
    | None -> mask_new
    | Some prev -> P.map2 prev mask_new ~f:combine_mask_tensor

  let mask_from_threshold ~threshold theta =
    Tensor.ge (Tensor.abs theta) (Scalar.f threshold) |> of_tensor

  (* prune smallest [1-p] of values in each layer; return new mask. *)
  let pruning_mask_layerwise
        ?(p_surviving_min = 1e-4)
        ?(n_surviving_min_per_layer = 10)
        ~p
        ~mask_prev
        (theta : _ some P.t)
    : const P.t
    =
    let prune_tensor theta mask_prev_opt =
      (* at least 10 value remains in each tensor *)
      let min_required =
        Int.(
          max
            (of_float Float.(p_surviving_min * of_int (numel_maths theta)))
            n_surviving_min_per_layer)
      in
      let theta_t = to_tensor theta in
      let curr_values =
        surviving_values_tensor
          ~mask_prev:
            (Option.map mask_prev_opt ~f:(fun x ->
               Torch.Tensor.to_type (to_tensor x) ~type_:Torch_core.Kind.(T Bool)))
          theta_t
      in
      (* TODO: is there a more principaled way of pruning only 10% iteration-wise in last layer? *)
      let p =
        let n_out = Tensor.shape theta_t |> List.last_exn in
        if n_out = 10 then Float.(p / 2.) else p
      in
      let threshold = threshold_of_values ~p ~min_required curr_values in
      let mask_new = mask_from_threshold ~threshold theta_t in
      match mask_prev_opt with
      | None -> mask_new
      | Some prev -> combine_mask_tensor prev mask_new
    in
    match mask_prev with
    | None -> P.map theta ~f:(fun t -> prune_tensor t None)
    | Some prev -> P.map2 theta prev ~f:(fun t m -> prune_tensor t (Some m))

  (* prune smallest [1-p] of values globally; return new mask. *)
  let pruning_mask_global ?(n_surviving_min = 200) ~p ~mask_prev (theta : _ some P.t)
    : const P.t
    =
    (* Global surviving values *)
    let surviving = surviving_values ~mask_prev theta in
    (* Threshold for global pruning *)
    let threshold = threshold_of_values ~p ~min_required:n_surviving_min surviving in
    (* Build new masks using this threshold *)
    let mask_new =
      P.map theta ~f:(fun theta_t -> mask_from_threshold ~threshold (to_tensor theta_t))
    in
    combine ~mask_prev mask_new
end
