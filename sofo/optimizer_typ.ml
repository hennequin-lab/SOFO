open Forward_torch

(** Basic optimizer type. *)
module type T = sig
  (* type of parameter set we're optimising *)
  module P : Prms.T

  type state
  type info
  type ('a, 'b) config
  type ('a, 'b, 'c) init_opts

  val params : state -> P.param
  val init : ('a, 'b, state) init_opts
  val clone_state : state -> state
  val step : config:('a, 'b) config -> info:info -> state -> state

  val manual_state_update
    :  state
    -> (Maths.const Maths.t P.p -> Maths.const Maths.t P.p)
    -> state
end

type 'v sofo_info =
  { loss : Maths.any Maths.t
  ; ggn : Maths.const Maths.t
  ; tangents : 'v
  }

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
      ; damping : [ `none | `relative_from_top of float | `relative_from_bottom of float ]
      }

    let default =
      { base = Base.default; learning_rate = None; n_tangents = 10; damping = `none }
  end

  module SGDm = struct
    type ('a, 'b) t =
      { base : ('a, 'b) Base.t
      ; learning_rate : float option
      ; momentum : float
      }

    let default = { base = Base.default; learning_rate = None; momentum = 0.9 }
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
