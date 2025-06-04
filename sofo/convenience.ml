open Base

let print s = Stdio.print_endline (Sexp.to_string_hum s)
