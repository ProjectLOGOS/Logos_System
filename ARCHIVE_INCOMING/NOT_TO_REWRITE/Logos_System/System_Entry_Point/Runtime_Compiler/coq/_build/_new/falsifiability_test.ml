
type __ = Obj.t

type bool =
| True
| False

(** val negb : bool -> bool **)

let negb = function
| True -> False
| False -> True

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

(** val length : 'a1 list -> nat **)

let rec length = function
| Nil -> O
| Cons (_, l') -> S (length l')

(** val eqb : bool -> bool -> bool **)

let eqb b1 b2 =
  match b1 with
  | True -> b2
  | False -> (match b2 with
              | True -> False
              | False -> True)

(** val existsb : ('a1 -> bool) -> 'a1 list -> bool **)

let rec existsb f = function
| Nil -> False
| Cons (a, l0) -> (match f a with
                   | True -> True
                   | False -> existsb f l0)

(** val forallb : ('a1 -> bool) -> 'a1 list -> bool **)

let rec forallb f = function
| Nil -> True
| Cons (a, l0) -> (match f a with
                   | True -> forallb f l0
                   | False -> False)

(** val filter : ('a1 -> bool) -> 'a1 list -> 'a1 list **)

let rec filter f = function
| Nil -> Nil
| Cons (x, l0) ->
  (match f x with
   | True -> Cons (x, (filter f l0))
   | False -> filter f l0)

type ascii =
| Ascii of bool * bool * bool * bool * bool * bool * bool * bool

(** val eqb0 : ascii -> ascii -> bool **)

let eqb0 a b =
  let Ascii (a0, a1, a2, a3, a4, a5, a6, a7) = a in
  let Ascii (b0, b1, b2, b3, b4, b5, b6, b7) = b in
  (match match match match match match match eqb a0 b0 with
                                       | True -> eqb a1 b1
                                       | False -> False with
                                 | True -> eqb a2 b2
                                 | False -> False with
                           | True -> eqb a3 b3
                           | False -> False with
                     | True -> eqb a4 b4
                     | False -> False with
               | True -> eqb a5 b5
               | False -> False with
         | True -> eqb a6 b6
         | False -> False with
   | True -> eqb a7 b7
   | False -> False)

type string =
| EmptyString
| String of ascii * string

(** val eqb1 : string -> string -> bool **)

let rec eqb1 s1 s2 =
  match s1 with
  | EmptyString ->
    (match s2 with
     | EmptyString -> True
     | String (_, _) -> False)
  | String (c1, s1') ->
    (match s2 with
     | EmptyString -> False
     | String (c2, s2') ->
       (match eqb0 c1 c2 with
        | True -> eqb1 s1' s2'
        | False -> False))

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

(** val eval_modal : modal_context -> modal_prop -> bool **)

let rec eval_modal ctx = function
| MProp s -> ctx.mc_valuation s
| MNeg q -> negb (eval_modal ctx q)
| MConj (a, b) ->
  (match eval_modal ctx a with
   | True -> eval_modal ctx b
   | False -> False)
| MDisj (a, b) ->
  (match eval_modal ctx a with
   | True -> True
   | False -> eval_modal ctx b)
| MImpl (a, b) ->
  (match negb (eval_modal ctx a) with
   | True -> True
   | False -> eval_modal ctx b)
| MBox q ->
  (match eval_modal ctx q with
   | True -> forallb (fun _ -> eval_modal ctx q) ctx.mc_accessible
   | False -> False)
| MDia q ->
  (match eval_modal ctx q with
   | True -> True
   | False -> existsb (fun _ -> eval_modal ctx q) ctx.mc_accessible)
| MBot -> False

(** val make_context :
    string -> string list -> (string, bool) prod list -> modal_context **)

let make_context world accessible valuations =
  let lookup =
    let rec lookup xs s =
      match xs with
      | Nil -> False
      | Cons (p, t) ->
        let Pair (k, v) = p in
        (match eqb1 s k with
         | True -> v
         | False -> lookup t s)
    in lookup
  in
  { mc_world = world; mc_accessible = accessible; mc_valuation =
  (lookup valuations) }

(** val generate_countermodel_modal :
    string -> string list -> (string, bool) prod list -> modal_prop ->
    modal_prop option **)

let generate_countermodel_modal _ _ _ p =
  Some p

(** val verify_countermodel : modal_prop -> bool **)

let verify_countermodel _ =
  True

type falsifiable = __

type unfalsifiable = __

type verifiable = __

type tautology = __

type contradiction = __

(** val falsifiability_coverage : modal_prop list -> nat **)

let falsifiability_coverage test_set =
  length (filter (fun _ -> True) test_set)

(** val test_proposition_set : modal_prop list **)

let test_proposition_set =
  Cons ((MProp (String ((Ascii (False, False, False, False, True, True, True,
    False)), EmptyString))), (Cons ((MNeg (MProp (String ((Ascii (False,
    False, False, False, True, True, True, False)), EmptyString)))), (Cons
    ((MConj ((MProp (String ((Ascii (False, False, False, False, True, True,
    True, False)), EmptyString))), (MProp (String ((Ascii (True, False,
    False, False, True, True, True, False)), EmptyString))))), (Cons ((MDisj
    ((MProp (String ((Ascii (False, False, False, False, True, True, True,
    False)), EmptyString))), (MProp (String ((Ascii (True, False, False,
    False, True, True, True, False)), EmptyString))))), (Cons ((MImpl ((MProp
    (String ((Ascii (False, False, False, False, True, True, True, False)),
    EmptyString))), (MProp (String ((Ascii (True, False, False, False, True,
    True, True, False)), EmptyString))))), (Cons ((MBox (MProp (String
    ((Ascii (False, False, False, False, True, True, True, False)),
    EmptyString)))), (Cons ((MDia (MProp (String ((Ascii (False, False,
    False, False, True, True, True, False)), EmptyString)))), (Cons (MBot,
    Nil)))))))))))))))

(** val runtime_check_falsifiable :
    string -> string list -> (string, bool) prod list -> modal_prop -> bool **)

let runtime_check_falsifiable world accessible valuations prop =
  let ctx = make_context world accessible valuations in
  negb (eval_modal ctx prop)

(** val runtime_check_tautology :
    string -> string list -> (string, bool) prod list -> modal_prop -> bool **)

let runtime_check_tautology world accessible valuations prop =
  let ctx = make_context world accessible valuations in eval_modal ctx prop
