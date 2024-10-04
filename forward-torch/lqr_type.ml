open Torch

(* f_x, f_u and f_t goes from 0 to T-1; f_t list is optional. if f_x/f_u/f_t list contains only one entry then it means the system is time-varying. if f_t_list is optional the dynamics does not have a constant term. x_0 is the starting state *)
type state_params =
  { n_steps : int
  ; x_0 : Maths_typ.t
  ; f_x_list : Maths_typ.t list
  ; f_u_list : Maths_typ.t list
  ; f_t_list : Maths_typ.t list option
  }

type state_params_tensor =
  { n_steps : int
  ; x_0 : Tensor.t
  ; f_x_list : Tensor.t list
  ; f_u_list : Tensor.t list
  ; f_t_list : Tensor.t list option
  }

(* c lists fo from 0 to T. Note that c_xx at and c_x at time 0 need to be 0; c_uu and c_u at time T need to be 0. The leading dimension is the batch dimension. *)
type cost_params =
  { c_xx_list : Maths_typ.t list
  ; c_xu_list : Maths_typ.t list option
  ; c_uu_list : Maths_typ.t list
  ; c_x_list : Maths_typ.t list option
  ; c_u_list : Maths_typ.t list option
  }

type cost_params_tensor =
  { c_xx_list : Tensor.t list
  ; c_xu_list : Tensor.t list option
  ; c_uu_list : Tensor.t list
  ; c_x_list : Tensor.t list option
  ; c_u_list : Tensor.t list option
  }
