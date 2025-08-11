open Base
open Domainslib
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

let lorenz_flow ~sigma ~rho ~beta x _ =
  let x = Mat.get x 0 0
  and y = Mat.get x 0 1
  and z = Mat.get x 0 2 in
  let xdot = sigma *. (y -. x)
  and ydot = (x *. (rho -. z)) -. y
  and zdot = (x *. y) -. (beta *. z) in
  Mat.(of_array [| xdot; ydot; zdot |] 1 3)

(* output is time x trials x 3 *)
let generate_from_long ?(sigma = 10.) ?(rho = 28.) ?(beta = 8. /. 3.) ~n_steps n_trials =
  let tt = n_trials * n_steps * 10 in
  let dt = 0.01 in
  let duration = Float.(dt * of_int Int.(tt - 1)) in
  let tspec = Owl_ode.Types.(T1 { t0 = 0.; duration; dt }) in
  let f = lorenz_flow ~sigma ~rho ~beta in
  let x0 = Mat.gaussian 1 3 in
  let _, xs = Owl_ode.Ode.odeint (module Owl_ode.Native.S.RK4) f x0 tspec () in
  let all = Arr.reshape xs [| tt / n_steps; n_steps; 3 |] in
  let ids =
    List.range 0 (tt / n_steps) |> List.permute |> List.sub ~pos:0 ~len:n_trials
  in
  Arr.get_fancy [ L ids ] all |> Arr.transpose ~axis:[| 1; 0; 2 |]

let data t =
  let pool = Task.setup_pool ~num_domains:7 () in
  let num_domains = Task.get_num_domains pool in
  let n_per_domain = 500 in
  let data = Arr.empty [| t + 1; n_per_domain * num_domains; 3 |] in
  Task.run pool (fun _ ->
    Task.parallel_for pool ~start:0 ~finish:(num_domains - 1) ~body:(fun i ->
      (* time x trials x d *)
      let x = generate_from_long ~n_steps:(t + 1) n_per_domain in
      Arr.set_slice [ []; [ i * n_per_domain; ((i + 1) * n_per_domain) - 1 ] ] data x));
  (* remove the time / trial average and normalize *)
  let data = Arr.reshape data [| -1; 3 |] in
  let data = Arr.((data - mean ~axis:0 data) / std ~axis:0 data) in
  let data = Arr.reshape data [| t + 1; -1; 3 |] in
  data

(* one trial of data used for simulation *)
let data_test t =
  let pool = Task.setup_pool ~num_domains:1 () in
  let num_domains = Task.get_num_domains pool - 1 in
  let n_per_domain = 1 in
  let data = Arr.empty [| t + 1; n_per_domain * num_domains; 3 |] in
  Task.run pool (fun _ ->
    Task.parallel_for pool ~start:0 ~finish:(num_domains - 1) ~body:(fun i ->
      (* time x trials x d *)
      let x = generate_from_long ~n_steps:(t + 1) n_per_domain in
      Arr.set_slice [ []; [ i * n_per_domain; ((i + 1) * n_per_domain) - 1 ] ] data x));
  (* remove the time average and normalize *)
  let data = Arr.reshape data [| -1; 3 |] in
  let data = Arr.((data - mean ~axis:0 data) / std ~axis:0 data) in
  data

let get_batch data =
  let n_trials = Arr.(shape data).(1) in
  let ids = List.range 0 n_trials in
  fun n ->
    let slice = List.(sub ~pos:0 ~len:n (permute ids)) in
    let minibatch = Arr.get_fancy [ R []; L slice ] data in
    let t = Arr.(shape minibatch).(0) in
    List.init t ~f:(fun i -> Arr.(squeeze (get_slice [ [ i ] ] minibatch)))
