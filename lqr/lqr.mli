open Forward_torch
include module type of Lqr_typ

(** Given a set of params, run LQR with directly forward differentiating through it and return solutions. 
In the solution list u goes from 0 to T-1 while x goes from 1 to T. *)
val _solve
  :  (Maths.t option, (Maths.t, Maths.t -> Maths.t) momentary_params list) Params.p
  -> Maths.t Solution.p list

(** Given a set of params, run LQR with implicitly forward differentiating through it and return solutions. 
In the solution list u goes from 0 to T-1 while x goes from 1 to T. *)
val solve
  :  (Maths.t option, (Maths.t, Maths.t prod) momentary_params list) Params.p
  -> Maths.t Solution.p list
