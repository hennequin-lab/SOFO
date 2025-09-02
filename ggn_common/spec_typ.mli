module type SPEC = sig
  type param_name

  (* type params  *)
  val all : param_name list
  val equal_param_name : param_name -> param_name -> bool
  val shape : param_name -> int list
  val n_params : param_name -> int
  val n_params_list : int list
end
