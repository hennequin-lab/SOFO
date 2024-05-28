open Base
open Ppxlib
open Ast_builder.Default

(* tag each field whose type is a type variable (e.g. 'a)
   with a matching module name (e.g. A) *)
let tag_polymorphic_fields fields =
  List.map fields ~f:(fun ld ->
    let name =
      match ld.pld_type.ptyp_desc with
      | Ptyp_var s -> Some (String.uppercase s)
      | _ -> None
    in
    ld, name)

let mod_params ~loc typ =
  let (module Ast) = Ast_builder.make loc in
  List.filter_map typ.ptype_params ~f:(function
    | { ptyp_desc = Ptyp_var s; _ }, _ -> Some (String.uppercase s)
    | _ -> None)

(* type 'a p = ('a A.p, 'a B.p, ...) p *)
let build_typ ~loc typ =
  let (module Ast) = Ast_builder.make loc in
  let typ_name = Ast.Located.map lident typ.ptype_name in
  let t =
    match mod_params ~loc typ with
    | [] -> None
    | tp ->
      Some
        (Ast.ptyp_constr
           typ_name
           (List.map tp ~f:(fun name ->
              let a = Ast.Located.lident (name ^ ".p") in
              Ast.ptyp_constr a [ Ast.ptyp_var "a" ])))
  in
  Option.map t ~f:(fun t -> [%stri type nonrec 'a p = [%t t]])

let build_map ~loc fields =
  let (module Ast) = Ast_builder.make loc in
  let new_record =
    Ast.pexp_record
      (List.map fields ~f:(fun (ld, name) ->
         let field_id = Ast.Located.lident ld.pld_name.txt in
         (* lident_of_field ld in *)
         let z = Ast.pexp_field [%expr x] field_id in
         let field_expr =
           match name with
           | None -> (* irrelevant field *) [%expr [%e z]]
           | Some n ->
             let mod_name = Ast.pmod_ident (Ast.Located.lident n) in
             [%expr
               let module XXXXX = [%m mod_name] in
               XXXXX.map [%e z] ~f]
         in
         field_id, field_expr))
      None
  in
  [%stri let map x ~f = [%e new_record]]

let build_map2 ~loc fields =
  let (module Ast) = Ast_builder.make loc in
  let new_record =
    Ast.pexp_record
      (List.map fields ~f:(fun (ld, name) ->
         let field_id = Ast.Located.lident ld.pld_name.txt in
         let zx = Ast.pexp_field [%expr x] field_id in
         let zy = Ast.pexp_field [%expr y] field_id in
         let field_expr =
           match name with
           | None ->
             (* irrelevant field *)
             [%expr [%e zx]]
           | Some n ->
             let mod_name = Ast.pmod_ident (Ast.Located.lident n) in
             [%expr
               let module XXXXX = [%m mod_name] in
               XXXXX.map2 [%e zx] [%e zy] ~f]
         in
         field_id, field_expr))
      None
  in
  [%stri let map2 x y ~f = [%e new_record]]

let build_fold ~loc fields =
  let (module Ast) = Ast_builder.make loc in
  let fold_expr =
    List.fold_right
      fields
      ~f:(fun (ld, name) accu ->
        let field_id = Ast.Located.lident ld.pld_name.txt in
        let field_name = Ast.pexp_constant (Pconst_string (ld.pld_name.txt, loc, None)) in
        let z = Ast.pexp_field [%expr x] field_id in
        match name with
        | None -> accu
        | Some n ->
          let mod_name = Ast.pmod_ident (Ast.Located.lident n) in
          [%expr
            let module XXXXX = [%m mod_name] in
            let init = XXXXX.fold ?path:(Prms.cat [%e field_name] path) [%e z] ~init ~f in
            [%e accu]])
      ~init:[%expr init]
  in
  [%stri let fold ?path x ~init ~f = [%e fold_expr]]

let build_fold2 ~loc fields =
  let (module Ast) = Ast_builder.make loc in
  let fold_expr =
    List.fold_right
      fields
      ~f:(fun (ld, name) accu ->
        let field_id = Ast.Located.lident ld.pld_name.txt in
        let field_name = Ast.pexp_constant (Pconst_string (ld.pld_name.txt, loc, None)) in
        let zx = Ast.pexp_field [%expr x] field_id in
        let zy = Ast.pexp_field [%expr y] field_id in
        match name with
        | None -> accu
        | Some n ->
          let mod_name = Ast.pmod_ident (Ast.Located.lident n) in
          [%expr
            let module XXXXX = [%m mod_name] in
            let init =
              XXXXX.fold2 ?path:(Prms.cat [%e field_name] path) [%e zx] [%e zy] ~init ~f
            in
            [%e accu]])
      ~init:[%expr init]
  in
  [%stri let fold2 ?path x y ~init ~f = [%e fold_expr]]

let generate_impl ~ctxt (_rec_flag, type_declarations) functor_name =
  let loc = Expansion_context.Deriver.derived_item_loc ctxt in
  List.map type_declarations ~f:(fun (td : type_declaration) ->
    match td with
    | { ptype_kind = Ptype_abstract | Ptype_variant _ | Ptype_open; ptype_loc; _ } ->
      let ext =
        Ppxlib.Location.error_extensionf
          ~loc:ptype_loc
          "Cannot derive prms for non record types"
      in
      [ Ast_builder.Default.pstr_extension ~loc ext [] ]
    | { ptype_kind = Ptype_record fields; _ } ->
      let fields = tag_polymorphic_fields fields in
      let str_items =
        [ build_map ~loc fields
        ; build_map2 ~loc fields
        ; build_fold ~loc fields
        ; build_fold2 ~loc fields
        ]
      in
      let str_items =
        match build_typ ~loc td with
        | Some a -> a :: str_items
        | None -> str_items
      in
      let (module Ast) = Ast_builder.make loc in
      let mod_type = Ast.pmty_ident (Ast.Located.lident "Prms.T") in
      (* functor definition:
         functor (A) -> (functor (B) -> (... -> inside)) *)
      let m =
        let make = Ast.pmod_ident (Ast.Located.lident "Prms.Make") in
        Ast.pmod_apply make (Ast.pmod_structure str_items)
      in
      let f =
        (* fold_right because we go inside out *)
        List.fold_right
          (mod_params ~loc td)
          ~f:(fun name e ->
            Ast.pmod_functor (Named (Ast.Located.mk (Some name), mod_type)) e)
          ~init:m
      in
      (* module Make = [... functor definition ...] *)
      let functor_name = Option.value ~default:"Make" functor_name in
      [ module_binding ~loc ~name:(Located.mk ~loc (Some functor_name)) ~expr:f
        |> pstr_module ~loc
      ])
  |> List.concat

let args = Deriving.Args.(empty +> arg "name" (estring __))
let impl_generator = Deriving.Generator.V2.make args generate_impl
let my_deriver = Deriving.add "prms" ~str_type_decl:impl_generator
