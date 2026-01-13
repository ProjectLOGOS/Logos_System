# Bridge Axioms Status Report

✅ **STATUS: COMPLETED** (2025-12-21)  
**All 4 bridge axioms eliminated via definitional upgrade**

**Date:** 2025-12-20  
**Context:** Phase 1 Track 2 - Bridge axiom derivability analysis  
**Outcome:** ✅ PImp := (→) and MEquiv := (↔) defined, all 4 bridge axioms now trivial

---

## Executive Summary

**Status:** ❌ **NOT DERIVABLE** (currently)

The 4 bridge axioms (`ax_imp_intro`, `ax_imp_elim`, `ax_mequiv_intro`, `ax_mequiv_elim`) **cannot be derived** because:
1. `PImp` (⟹) is a **Parameter** (primitive), not defined
2. `MEquiv` (⩪) is a **Parameter** (primitive), not defined

These axioms establish isomorphisms between Coq's built-in logical connectives (`->`, `<->`) and PXL's modal connectives (`⟹`, `⩪`). Without explicit definitions of the PXL connectives, the isomorphisms must be asserted axiomatically.

---

## Detailed Analysis

### 1. ax_imp_intro

```coq
Axiom ax_imp_intro : forall p q : Prop, (p -> q) -> p ⟹ q.
```

**Classification:** BRIDGE  
**Derivability:** ❌ PRIMITIVE  
**Reason:** `PImp` is a Parameter without definition

**Current declaration:**
```coq
Parameter PImp : Prop -> Prop -> Prop.
Infix " ⟹ " := PImp (at level 90, right associativity).
```

**Why it's an axiom:**
- `PImp` has no computational content or semantic interpretation
- It's a "black box" relation between Props
- The axiom asserts that Coq's `->` can be embedded into `⟹`

---

### 2. ax_imp_elim

```coq
Axiom ax_imp_elim : forall p q : Prop, p ⟹ q -> p -> q.
```

**Classification:** BRIDGE  
**Derivability:** ❌ PRIMITIVE  
**Reason:** `PImp` is a Parameter without definition

**Why it's an axiom:**
- Inverse of `ax_imp_intro`
- Asserts that `⟹` can be projected back to Coq's `->`
- Together with `ax_imp_intro`, establishes `(p -> q) <-> (p ⟹ q)`

---

### 3. ax_mequiv_intro

```coq
Axiom ax_mequiv_intro : forall p q : Prop, (p <-> q) -> p ⩪ q.
```

**Classification:** BRIDGE  
**Derivability:** ❌ PRIMITIVE  
**Reason:** `MEquiv` is a Parameter without definition

**Current declaration:**
```coq
Parameter MEquiv : Prop -> Prop -> Prop.
Infix " ⩪ " := MEquiv (at level 80).
```

**Why it's an axiom:**
- `MEquiv` has no definition - it's a primitive modal equivalence
- The axiom embeds Coq's `<->` into the modal `⩪`
- Unlike classical equivalence, `⩪` may have modal/metaphysical semantics

---

### 4. ax_mequiv_elim

```coq
Axiom ax_mequiv_elim : forall p q : Prop, p ⩪ q -> p <-> q.
```

**Classification:** BRIDGE  
**Derivability:** ❌ PRIMITIVE  
**Reason:** `MEquiv` is a Parameter without definition

**Why it's an axiom:**
- Inverse of `ax_mequiv_intro`
- Asserts that modal equivalence `⩪` implies classical equivalence `<->`
- Together with `ax_mequiv_intro`, establishes `(p <-> q) <-> (p ⩪ q)`

---

## Path to Elimination

To eliminate these 4 axioms, one of the following strategies is required:

### Strategy A: Define PImp and MEquiv Semantically

**Option 1:** Direct definitional equality
```coq
Definition PImp (p q : Prop) : Prop := p -> q.
Definition MEquiv (p q : Prop) : Prop := p <-> q.
```

**Consequence:**
- All 4 axioms become trivial lemmas (proven by `reflexivity` or `unfold`)
- **BUT:** Loses the distinction between PXL connectives and Coq connectives
- May violate design intent if `⟹` and `⩪` are meant to have richer semantics

---

**Option 2:** Modal/semantic interpretation
```coq
Definition PImp (p q : Prop) : Prop := □ (p -> q).
Definition MEquiv (p q : Prop) : Prop := □ (p <-> q).
```

**Consequence:**
- Preserves modal character of PXL connectives
- Axioms become provable IF we can show `□` distributes appropriately
- May require additional modal axioms (like K, T, Nec - which we have as theorems)

---

**Option 3:** Kripke/possible-worlds semantics
```coq
Definition PImp (p q : Prop) : Prop := 
  forall (w : W), R w w0 -> (p -> q).
Definition MEquiv (p q : Prop) : Prop := 
  forall (w : W), R w w0 -> (p <-> q).
```

**Consequence:**
- Rich semantic interpretation
- Requires Kripke frame infrastructure (already exists for modal axioms)
- Axioms provable via semantic soundness theorems

---

### Strategy B: Embed via Modal Logic

Instead of defining PImp/MEquiv directly, prove they satisfy specific modal laws, then derive intro/elim from those laws.

**Example:**
```coq
(* If we can prove: *)
Lemma PImp_respects_implication : 
  forall p q, (p -> q) -> PImp p q.
Lemma PImp_implies : 
  forall p q, PImp p q -> (p -> q).

(* Then ax_imp_intro and ax_imp_elim are just restatements *)
```

**Blocker:** Still requires some axiomatic characterization of PImp behavior.

---

## Recommendation

**Current stance:** **Keep as axioms**

**Rationale:**
1. **Design clarity:** These axioms document the intended relationship between Coq and PXL connectives
2. **Minimal overhead:** 4 axioms that establish clean interfaces
3. **No circularity:** They don't block other derivations
4. **Future flexibility:** Preserves option to later refine PImp/MEquiv semantics without breaking existing proofs

**Future work (Phase 2+):**
- Investigate whether PImp/MEquiv have an intended semantic interpretation in the PXL metaphysics
- If yes, define them and eliminate the axioms
- If no, document that these are fundamental interface axioms (not eliminable)

---

## Current Kernel Status

**16 axioms** (after modal elimination):
- **Identity (3):** ax_ident_refl, ax_ident_symm, ax_ident_trans
- **Interaction (1):** ax_inter_comm
- **Bridge (4):** ax_imp_intro, ax_imp_elim, ax_mequiv_intro, ax_mequiv_elim ← **ANALYZED HERE**
- **Metaphysical (8):** A2_noncontradiction, A7_triune_necessity, modus_groundens, triune_dependency_substitution, privative_collapse, grounding_yields_entails, coherence_lifts_entailment, entails_global_implies_truth

**Elimination potential:**
- ✅ Modal (3): ELIMINATED (ax_K, ax_T, ax_Nec are theorems in semantic variant)
- ❌ Bridge (4): NOT ELIMINABLE without definitions
- 🟡 Identity (3): Potentially eliminable if Ident gets an explicit definition (Phase 2)
- 🟡 Interaction (1): Potentially eliminable if Inter gets an explicit definition (Phase 2)
- 🔒 Metaphysical (8): Likely irreducible (genuine postulates)

**Realistic minimum:** ~11-12 axioms (if identity/interaction get definitions)

---

## Proposed Definitional Upgrade (Optional)

If the design intent is to make PImp/MEquiv derivable:

### Minimal Change (Strategy A, Option 1):

```coq
(* Replace Parameters with Definitions *)
Definition PImp (p q : Prop) : Prop := p -> q.
Definition MEquiv (p q : Prop) : Prop := p <-> q.

(* Delete axioms, replace with trivial lemmas *)
Lemma ax_imp_intro : forall p q : Prop, (p -> q) -> p ⟹ q.
Proof. intros; unfold PImp; assumption. Qed.

Lemma ax_imp_elim : forall p q : Prop, p ⟹ q -> p -> q.
Proof. intros; unfold PImp in *; assumption. Qed.

Lemma ax_mequiv_intro : forall p q : Prop, (p <-> q) -> p ⩪ q.
Proof. intros; unfold MEquiv; assumption. Qed.

Lemma ax_mequiv_elim : forall p q : Prop, p ⩪ q -> p <-> q.
Proof. intros; unfold MEquiv in *; assumption. Qed.
```

**Impact:** 16 → 12 axioms (instant 4-axiom reduction)

**Risk:** May conflict with intended modal semantics of `⟹` and `⩪`

---

## Conclusion

**Track 2 verdict:** Bridge axioms are **primitive** in the current architecture.

**Options:**
1. **Keep as axioms** (current recommendation)
2. **Define PImp/MEquiv** and eliminate (requires design decision on semantics)
3. **Defer to Phase 2** (after clarifying PXL connective semantics)
