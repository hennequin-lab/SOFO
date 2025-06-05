open Base
open Torch
include Prms_typ
open Maths

(* ------------------------------------------------------------
   -- Basic functions for dealing with parameters
   ------------------------------------------------------------ *)

let value = function
  | Pinned x -> of_tensor x
  | Free x -> of_tensor x
  | Bounded x -> of_tensor x.v

let enforce_bounds ?lb ?ub x =
  let x =
    match lb with
    | None -> x
    | Some lb -> Tensor.max x lb
  in
  match ub with
  | None -> x
  | Some ub -> Tensor.min x ub

(** Path specified as a list of strings. *)
type path = String.t List.t

let cat label =
  if String.(label = "")
  then Fn.id
  else
    function
    | None -> Some [ label ]
    | Some p -> Some (label :: p)

(* ------------------------------------------------------------
   -- Functor used by the ppx
   ------------------------------------------------------------ *)

module Make (B : Basic) : T with type 'a p = 'a B.p = struct
  include B

  type 'a elt = 'a t
  type nonrec 'a t = 'a elt p
  type nonrec param = param p

  let iter x ~f = fold ?path:None x ~init:() ~f:(fun () (x, _) -> f x)
  let iter2 x y ~f = fold2 ?path:None x y ~init:() ~f:(fun () (x, y, _) -> f x y)
  let value = map ~f:value
  let any = map ~f:any
  let of_tensor = map ~f:of_tensor
  let to_tensor = map ~f:to_tensor
  let const = map ~f:const
  let numel x = fold x ~init:0 ~f:(fun accu (x, _) -> Int.(accu + numel x))
  let tangent_exn = map ~f:tangent_exn
  let dual ~tangent:dx x = map2 x dx ~f:(fun x dx -> dual ~tangent:dx x)

  let dual_on_demand ~tangent:dx x =
    map2 x dx ~f:(fun x dx -> dual_on_demand ~tangent:dx x)

  let zeros_like = map ~f:zeros_like
  let zeros_like_k ~k = map ~f:(zeros_like_k ~k)
  let ones_like = map ~f:ones_like
  let rand_like = map ~f:rand_like
  let randn_like = map ~f:randn_like
  let randn_like_k ~k = map ~f:(randn_like_k ~k)

  let dot_prod x y =
    fold2 x y ~init:None ~f:(fun accu (x, y, _) ->
      let z = sum (x * y) in
      match accu with
      | None -> Some z
      | Some a -> Some Maths.(a + z))
    |> Option.value_exn

  let ( + ) = map2 ~f:( + )
  let ( - ) = map2 ~f:( - )
  let ( * ) = map2 ~f:( * )
  let ( / ) = map2 ~f:( / )
  let ( $+ ) z = map ~f:(( $+ ) z)
  let ( $* ) z = map ~f:(( $* ) z)

  module C = struct
    let dot_prod x y =
      fold2 x y ~init:None ~f:(fun accu (x, y, _) ->
        let z = C.(sum (x * y)) in
        match accu with
        | None -> Some z
        | Some a -> Some C.(a + z))
      |> Option.value_exn

    let ( + ) = map2 ~f:C.( + )
    let ( - ) = map2 ~f:C.( - )
    let ( * ) = map2 ~f:C.( * )
    let ( / ) = map2 ~f:C.( / )
    let ( $+ ) z = map ~f:C.(( $+ ) z)
    let ( $* ) z = map ~f:C.(( $* ) z)

    let save m ~kind ~out:filename =
      let m = map m ~f:(fun x -> Tensor.to_bigarray (Maths.to_tensor x) ~kind) in
      let output = Stdio.Out_channel.create filename in
      Stdlib.Marshal.to_channel output m [ Stdlib.Marshal.No_sharing ];
      Stdio.Out_channel.close output

    let load ?device filename =
      let input = Stdio.In_channel.create filename in
      let m = Stdlib.Marshal.from_channel input in
      Stdio.In_channel.close input;
      map m ~f:(fun x -> Maths.of_tensor (Tensor.of_bigarray ?device x))

    let save_npz ?prefix ~kind ~out prms =
      let prms = map prms ~f:(fun x -> Tensor.to_bigarray (Maths.to_tensor x) ~kind) in
      let path = Option.map prefix ~f:(fun s -> [ s ]) in
      let file = Npy.Npz.open_out out in
      fold ?path prms ~init:() ~f:(fun () (prm, path) ->
        let descr =
          match path with
          | None -> assert false
          | Some p -> String.concat ~sep:"/" (List.rev p)
        in
        Npy.Npz.write file descr prm);
      Npy.Npz.close_out file
  end
end

(* ------------------------------------------------------------
   -- Single parameter 
   ------------------------------------------------------------ *)

module Single = struct
  include Make (struct
      type 'a p = 'a

      let map x ~f = f x
      let map2 x y ~f = f x y
      let fold ?path x ~init ~f = f init (x, path)
      let fold2 ?path x y ~init ~f = f init (x, y, path)
    end)

  let pinned x = Pinned (to_tensor x)
  let free x = Free (to_tensor x)

  let bounded ?above:lb ?below:ub x =
    Bounded
      { v = to_tensor x
      ; lb = Option.map ~f:to_tensor lb
      ; ub = Option.map ~f:to_tensor ub
      }
end

(* ------------------------------------------------------------
   -- The empty parameter
   ------------------------------------------------------------ *)

module None : T with type 'a p = unit = Make (struct
    type 'a p = unit

    let map () ~f:_ = ()
    let map2 () () ~f:_ = ()
    let fold ?path:_ () ~init ~f:_ = init
    let fold2 ?path:_ () () ~init ~f:_ = init
  end)

(* ------------------------------------------------------------
   -- Parameter pairs
   ------------------------------------------------------------ *)

module Pair (P1 : T) (P2 : T) = Make (struct
    type 'a p = 'a P1.p * 'a P2.p

    let map (x1, x2) ~f = P1.map x1 ~f, P2.map x2 ~f
    let map2 (x1, x2) (y1, y2) ~f = P1.map2 x1 y1 ~f, P2.map2 x2 y2 ~f

    let fold ?path (x1, x2) ~init ~f =
      let init = P1.fold ?path x1 ~init ~f in
      P2.fold ?path x2 ~init ~f

    let fold2 ?path (x1, x2) (y1, y2) ~init ~f =
      let init = P1.fold2 ?path x1 y1 ~init ~f in
      P2.fold2 ?path x2 y2 ~init ~f
  end)

(* ------------------------------------------------------------
   -- Optional parameters
   ------------------------------------------------------------ *)

module Option (P : T) = Make (struct
    type 'a p = 'a P.p Option.t

    let map x ~f =
      match x with
      | Some x -> Some (P.map ~f x)
      | None -> None

    let map2 x y ~f =
      match x, y with
      | Some x, Some y -> Some (P.map2 ~f x y)
      | _ -> None

    let fold ?path x ~init ~f =
      match x with
      | Some x -> P.fold ?path x ~f ~init
      | None -> init

    let fold2 ?path x y ~init ~f =
      match x, y with
      | Some x, Some y -> P.fold2 ?path x y ~init ~f
      | _ -> init
  end)

(* ------------------------------------------------------------
   -- List of parameters
   ------------------------------------------------------------ *)

module List (P : T) = Make (struct
    type 'a p = 'a P.p List.t

    let map x ~f = List.map x ~f:(P.map ~f)
    let map2 x y ~f = List.map2_exn x y ~f:(P.map2 ~f)

    let fold ?path x ~init ~f =
      List.foldi x ~init ~f:(fun i init w ->
        P.fold ?path:(cat (Int.to_string i) path) ~init ~f w)

    let fold2 ?path x y ~init ~f =
      let _, result =
        List.fold2_exn x y ~init:(0, init) ~f:(fun (i, init) w1 w2 ->
          Int.(i + 1), P.fold2 ?path:(cat (Int.to_string i) path) ~init ~f w1 w2)
      in
      result
  end)
