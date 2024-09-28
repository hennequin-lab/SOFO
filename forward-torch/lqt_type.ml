open Torch

(* A, B and f_t goes from 0 to T-1; f_t list is optional. if a/b/q/r list contains only one entry then it means the cost is the same across all timesteps. if f_t_list only contains one entry then it has to be none. *)
type state_params =
  { n_steps : int
  ; a_list : Maths_typ.t list
  ; b_list : Maths_typ.t list
  ; f_t_list : Maths_typ.t option list
  }

type state_params_tensor =
  { n_steps : int
  ; a_list : Tensor.t list
  ; b_list : Tensor.t list
  ; f_t_list : Tensor.t option list
  }

(* Q goes from 1 to T and R goes from 0 to T-1 *)
type cost_params =
  { q_list : Maths_typ.t list
  ; r_list : Maths_typ.t list
  }

type cost_params_tensor =
  { q_list : Tensor.t list
  ; r_list : Tensor.t list
  }

(* desired trajectory x_1 to x_T and u_0 to u_{T-1}. If x_d_list only contains 1 entry it has to be None. If u_d_list only contains 1 entry it has to be None. *)
type x_u_desired =
  { x_0 : Maths.t
  ; x_d_list : Maths.t option list
  ; u_d_list : Maths.t option list
  }

type x_u_desired_tensor =
  { x_0 : Tensor.t
  ; x_d_list : Tensor.t option list
  ; u_d_list : Tensor.t option list
  }
