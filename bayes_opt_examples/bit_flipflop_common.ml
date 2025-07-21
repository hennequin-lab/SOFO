open Base
open Owl
open Torch
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

(* -----------------------------------------
   -- Generate fliplop data          ------
   ----------------------------------------- *)

type pulse_state =
  { input : [ `off | `on of float * int | `refr of int ]
  ; output : float
  }

let sample_one_bit ~pulse_prob ~pulse_duration ~pulse_refr n_steps =
  let rec iter k state accu =
    if k = n_steps
    then
      List.rev_map accu ~f:(fun s ->
        ( (match s.input with
           | `on (s, _) -> s
           | _ -> 0.)
        , s.output ))
    else (
      let state =
        match state.input with
        | `off ->
          if Float.(Random.float 1. < pulse_prob)
          then (
            let s = if Random.bool () then 1. else -1. in
            { state with input = `on (s, 0) })
          else state
        | `on (s, 0) ->
          (* target output is set for the NEXT time step so
              the RNN has a chance to recurrently integrate the input *)
          { input = `on (s, 1); output = s }
        | `on (_, d) when d = pulse_duration ->
          (* enter a refractory state at the end of the pulse *)
          { state with input = `refr pulse_refr }
        | `on (s, d) -> { state with input = `on (s, d + 1) }
        | `refr 0 -> { state with input = `off }
        | `refr r -> { state with input = `refr (r - 1) }
      in
      iter (k + 1) state (state :: accu))
  in
  iter 0 { input = `off; output = 0. } []

(* returns time x bs x bits *)
let sample_batch ~pulse_prob ~pulse_duration ~pulse_refr ~n_steps ~b ~device bs =
  let data =
    List.init bs ~f:(fun _ ->
      List.init b ~f:(fun _ ->
        sample_one_bit ~pulse_prob ~pulse_duration ~pulse_refr n_steps))
  in
  let massage extract =
    data
    |> List.map ~f:(fun a ->
      List.map a ~f:(fun b ->
        let b = List.map b ~f:extract in
        Mat.of_array (Array.of_list b) 1 (-1))
      |> Array.of_list
      |> Mat.concatenate ~axis:0
      |> fun x -> Arr.expand x 3)
    |> Array.of_list
    |> Mat.concatenate ~axis:0
    |> Arr.transpose ~axis:[| 2; 0; 1 |]
  in
  let inputs = massage fst
  and outputs = massage snd in
  Tensor.of_bigarray ~device inputs, Tensor.of_bigarray ~device outputs
