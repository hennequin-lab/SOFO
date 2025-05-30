open Base
open Torch
include Prms_typ
open Maths

(* ------------------------------------------------------------
   -- Basic functions for dealing with parameters
   ------------------------------------------------------------ *)

let value = function
  | Pinned x -> x
  | Free x -> x
  | Bounded x -> x.v

let maybe_apply_bounds = function
  | Pinned x -> Pinned x
  | Free x -> Free x
  | Bounded { v; lb; ub } ->
    let x = primal v in
    let x = Option.value_map lb ~default:x ~f:(fun lb -> Tensor.max x (primal lb)) in
    let x = Option.value_map ub ~default:x ~f:(fun ub -> Tensor.min x (primal ub)) in
    let v =
      match v with
      | Const _ -> Const x
      | Dual (_, dx) -> Dual (x, dx)
    in
    Bounded { v; lb; ub }

(** Path specified as a list of strings. *)
type path = String.t List.t

module type Basic = sig
  (** If you want to make your own custom parameter structure
      which for a reason or another cannot be automated using [ppx-prms]
      (e.g. if it is not a record type), then you will need to use
      {!Forward_torch.Prms.Make} and provide a module of this type. *)

  type 'a p

  (** Apply function [f] on all elements in x. *)
  val map : 'a p -> f:('a -> 'b) -> 'b p

  (** Apply function [f] on all elements in x and y. *)
  val map2 : 'a p -> 'b p -> f:('a -> 'b -> 'c) -> 'c p

  (** Fold x with [init] and function [f].
    [?path] optionally contains a record of the path (a list of strings, in reverse order) taken to
      arrive at the current value, if it is nested within a broader structure. Make sure to use this
      if you want to attach string labels to the various components of your custom ['a p] type
      (e.g. use for saving to files, see {!Forward_torch.Prms.module-type-T.T.save_npy}). *)
  val fold : ?path:path -> 'a p -> init:'c -> f:('c -> 'a * path option -> 'c) -> 'c

  (** Fold x and y with [init] and function [f]. *)
  val fold2
    :  ?path:path
    -> 'a p
    -> 'b p
    -> init:'c
    -> f:('c -> 'a * 'b * path option -> 'c)
    -> 'c
end

(* ------------------------------------------------------------
   -- Main type for parameter structures
   ------------------------------------------------------------ *)

module type T = sig
  include Basic

  (** Apply function [f] to each element in x. *)
  val iter : 'a p -> f:('a -> unit) -> unit

  (** Apply function [f] to each element in x and y. *)
  val iter2 : 'a p -> 'b p -> f:('a -> 'b -> unit) -> unit

  type 'a elt = 'a t
  type nonrec 'a t = 'a elt p
  type nonrec param = param p

  (** Extract value as Tensor.t in all elements in x. *)
  val value : param -> const t

  val as_const : _ any t -> const t
  val as_dual_exn : _ any t -> dual t
  val const : Tensor.t p -> const t
  val dual : dx:const t -> const t -> dual t
  val dual_lazy : dx:(Device.t -> const Maths.t) p -> const t -> dual t
  val primal : _ any t -> Tensor.t p
  val tangent : dual t -> Tensor.t p
  val zeros_like : _ any t -> const t
  val ones_like : _ any t -> const t
  val rand_like : _ any t -> const t
  val randn_like : _ any t -> const t

  (** Returns the number of parameters in x. *)
  val numel : _ any t -> int

  (** Returns the dot product between x and y. *)
  val dot_prod : 'a any t -> 'b any t -> 'c any elt

  (** Element-wise multiplication of two parameter sets *)
  val ( * ) : 'a any t -> 'b any t -> 'c any t

  (** Element-wise addition of two parameter sets *)
  val ( + ) : 'a any t -> 'b any t -> 'c any t

  (** Element-wise subtraction of two parameter sets *)
  val ( - ) : 'a any t -> 'b any t -> 'c any t

  (** Element-wise division of two parameter sets *)
  val ( / ) : 'a any t -> 'b any t -> 'c any t

  (** Multiplication of a parameter set with a scalar *)
  val ( $* ) : float -> 'a any t -> 'a any t

  (** Adds a scalar to a parameter set *)
  val ( $+ ) : float -> 'a any t -> 'a any t

  (** Save x as [kind] in [out]. *)
  val save : const t -> kind:('a, 'b) Bigarray.kind -> out:string -> unit

  (** Load params from file onto [device]. *)
  val load : ?device:Device.t -> string -> const t

  (** Save x as [kind] in [out] with [prefix] as .npy file. *)
  val save_npz
    :  ?prefix:string
    -> kind:('a, 'b) Bigarray.kind
    -> out:string
    -> const t
    -> unit
end

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
  let as_const = map ~f:as_const
  let as_dual_exn = map ~f:as_dual_exn
  let const = map ~f:const
  let dual ~dx x = map2 x dx ~f:(fun x dx -> dual ~dx x)
  let dual_lazy ~dx x = map2 x dx ~f:(fun x dx -> dual_lazy ~dx x)
  let primal = map ~f:primal
  let tangent = map ~f:tangent
  let zeros_like = map ~f:zeros_like
  let ones_like = map ~f:ones_like
  let rand_like = map ~f:rand_like
  let randn_like = map ~f:randn_like
  let numel x = fold x ~init:0 ~f:(fun accu (x, _) -> Int.(accu + numel x))

  let dot_prod x y =
    fold2 x y ~init:None ~f:(fun accu (x, y, _) ->
      let z = sum (x * y) in
      match accu with
      | None -> Some z
      | Some a -> Some (a + z))
    |> Option.value_exn

  let ( + ) = map2 ~f:( + )
  let ( - ) = map2 ~f:( - )
  let ( * ) = map2 ~f:( * )
  let ( / ) = map2 ~f:( / )
  let ( $+ ) z = map ~f:(( $+ ) z)
  let ( $* ) z = map ~f:(( $* ) z)

  let save m ~kind ~out:filename =
    let m = map m ~f:(fun x -> Tensor.to_bigarray (Maths.primal x) ~kind) in
    let output = Stdio.Out_channel.create filename in
    Stdlib.Marshal.to_channel output m [ Stdlib.Marshal.No_sharing ];
    Stdio.Out_channel.close output

  let load ?device filename =
    let input = Stdio.In_channel.create filename in
    let m = Stdlib.Marshal.from_channel input in
    Stdio.In_channel.close input;
    map m ~f:(fun x -> Maths.const (Tensor.of_bigarray ?device x))

  let save_npz ?prefix ~kind ~out prms =
    let prms = map prms ~f:(fun x -> Tensor.to_bigarray (Maths.primal x) ~kind) in
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

(* ------------------------------------------------------------
   -- Singleton parameter 
   ------------------------------------------------------------ *)

module Singleton_basic : Basic with type 'a p = 'a = struct
  type 'a p = 'a

  let map x ~f = f x
  let map2 x y ~f = f x y
  let fold ?path x ~init ~f = f init (x, path)
  let fold2 ?path x y ~init ~f = f init (x, y, path)
end

module Singleton = struct
  include Make (Singleton_basic)

  let pinned x = Pinned x
  let free (x : const t) = Free x
  let bounded ?above:lb ?below:ub (x : const t) = Bounded { v = x; lb; ub }
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

(* ------------------------------------------------------------
   -- Arrays of parameters
   ------------------------------------------------------------ *)

module Array (P : T) = Make (struct
    type 'a p = 'a P.p array

    let map x ~f = Array.map x ~f:(P.map ~f)
    let map2 x y ~f = Array.map2_exn x y ~f:(P.map2 ~f)

    let fold ?path x ~init ~f =
      Array.foldi x ~init ~f:(fun i init w ->
        P.fold ?path:(cat (Int.to_string i) path) ~init ~f w)

    let fold2 ?path x y ~init ~f =
      let _, result =
        Array.fold2_exn x y ~init:(0, init) ~f:(fun (i, init) w1 w2 ->
          Int.(i + 1), P.fold2 ?path:(cat (Int.to_string i) path) ~init ~f w1 w2)
      in
      result
  end)
