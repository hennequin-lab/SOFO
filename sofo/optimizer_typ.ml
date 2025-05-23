(** Basic optimizer type. *)
module type T = sig
  (** W is the wrapper type that defines the forward computational graph. *)
  module W : Wrapper.T

  type ('a, 'b) config
  type state
  type ('a, 'b, 'c) init_opts

  (** Extract parameters from state. *)
  val params : state -> W.P.tagged

  (** Initialise state from parameters. *)
  val init : (state, 'a, 'b) init_opts

  (** Given [config], current [state], [data] and additional [args], return loss and updated state. *)
  val step
    :  config:('a, 'b) config
    -> state:state
    -> data:W.data
    -> args:W.args
    -> float * state
end

module Config = struct
  (** Basic config, specifying device, kind and BigArray kind.*)
  module Base = struct
    type ('a, 'b) t =
      { device : Torch_core.Device.t
      ; kind : Torch_core.Kind.packed
      ; ba_kind : ('a, 'b) Bigarray.kind
      }

    (** Default option for the base config. *)
    let default =
      { device = Torch.Device.cuda_if_available ()
      ; kind = Torch_core.Kind.(T f32)
      ; ba_kind = Bigarray.float32
      }
  end

  module SOFO = struct
    type ('a, 'b) t =
      { base : ('a, 'b) Base.t
      ; learning_rate : float option
      ; n_tangents : int
      ; rank_one : bool
      ; damping : float option
      ; momentum : float option
      }

    let default =
      { base = Base.default
      ; learning_rate = None
      ; n_tangents = 10
      ; rank_one = false
      ; damping = None
      ; momentum = None
      }
  end

  module FGD = struct
    type ('a, 'b) t =
      { base : ('a, 'b) Base.t
      ; learning_rate : float option
      ; n_tangents : int
      ; rank_one : bool
      ; momentum : float option
      }

    let default =
      { base = Base.default
      ; learning_rate = None
      ; n_tangents = 10
      ; rank_one = false
      ; momentum = None
      }
  end

  module SGD = struct
    type ('a, 'b) t =
      { base : ('a, 'b) Base.t
      ; learning_rate : float option
      ; momentum : float option
      }

    let default = { base = Base.default; learning_rate = None; momentum = None }
  end

  module Adam = struct
    type ('a, 'b) t =
      { base : ('a, 'b) Base.t
      ; learning_rate : float option
      ; beta_1 : float
      ; beta_2 : float
      ; eps : float
      ; weight_decay : float option
      }

    let default =
      { base = Base.default
      ; learning_rate = None
      ; beta_1 = 0.9
      ; beta_2 = 0.999
      ; eps = 1e-8
      ; weight_decay = None
      }
  end
end
