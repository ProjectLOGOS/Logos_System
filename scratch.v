Set Implicit Arguments.

Definition Lvl := nat.
Parameter Atom : Type.

Inductive Form : Lvl -> Type :=
| FAtom  : forall level, Atom -> Form level
| FTop   : forall level, Form level
| FBot   : forall level, Form level
| FNot   : forall level, Form level -> Form level
| FTruth : forall level, Form level -> Form (S level).

Require Import Logic.JMeq.
Require Import Program.Equality.

Lemma test (level : Lvl) (phi : Form level) (psi : Form level) :
  JMeq psi (FTruth phi) -> False.
Proof.
  intro H.
  dependent destruction H.
