open Forward_torch
module Mat = Owl.Dense.Matrix.S

module PP = struct
  type 'a p =
    { init_cond : 'a
    ; w : 'a
    ; bias : 'a
    ; b : 'a
    ; f1 : 'a
    ; f2 : 'a
    ; f3 : 'a
    ; f4 : 'a
    ; c1 : 'a
    ; c2 : 'a
    }
  [@@deriving prms]
end

type 'a result =
  { network : 'a
  ; arm : 'a Arm.state
  ; torques : 'a * 'a
  }
