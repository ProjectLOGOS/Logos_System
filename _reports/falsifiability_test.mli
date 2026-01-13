
type __ = Obj.t

type bool =
| True
| False

val negb : bool -> bool

type nat =
| O
| S of nat

type 'a option =
| Some of 'a
| None

type ('a, 'b) prod =
| Pair of 'a * 'b

type 'a list =
| Nil
| Cons of 'a * 'a list

val length : 'a1 list -> nat

val eqb : bool -> bool -> bool

val existsb : ('a1 -> bool) -> 'a1 list -> bool

val forallb : ('a1 -> bool) -> 'a1 list -> bool

val filter : ('a1 -> bool) -> 'a1 list -> 'a1 list

type ascii =
| Ascii of bool * bool * bool * bool * bool * bool * bool * bool

val eqb0 : ascii -> ascii -> bool

type string =
| EmptyString
| String of ascii * string

val eqb1 : string -> string -> bool

type modal_context = { mc_world : string; mc_accessible : string list;
                       mc_valuation : (string -> bool) }

type modal_prop =
| MProp of string
| MNeg of modal_prop
| MConj of modal_prop * modal_prop
| MDisj of modal_prop * modal_prop
| MImpl of modal_prop * modal_prop
| MBox of modal_prop
| MDia of modal_prop
| MBot

val eval_modal : modal_context -> modal_prop -> bool

val make_context :
  string -> string list -> (string, bool) prod list -> modal_context

val generate_countermodel_modal :
  string -> string list -> (string, bool) prod list -> modal_prop ->
  modal_prop option

val verify_countermodel : modal_prop -> bool

type falsifiable = __

type unfalsifiable = __

type verifiable = __

type tautology = __

type contradiction = __

val falsifiability_coverage : modal_prop list -> nat

val test_proposition_set : modal_prop list

val runtime_check_falsifiable :
  string -> string list -> (string, bool) prod list -> modal_prop -> bool

val runtime_check_tautology :
  string -> string list -> (string, bool) prod list -> modal_prop -> bool
