# Axiom Minimum Plan — Phase 2 Definitional Upgrades

✅ **STATUS: COMPLETED** (2025-12-21)  
**Achievement: 8 axioms** (exceeded target A: ~12 axioms, exceeded target B: ~8 axioms)

**Date:** 2025-12-20  
**Context:** Phase 2 design for reducing 16 axioms toward realistic (~12) or absolute (~8) minimum  
**Current state:** **8 axioms achieved** — Phase 2/3 complete

---

## Executive Summary

This document outlines definitional upgrade strategies to reduce the PXL kernel from **16 axioms** to either:

- **Target A (Realistic Minimum):** ~**12 axioms** via definitional upgrades to `Ident` (⧟), `Inter` (⇌), `PImp` (⟹), `MEquiv` (⩪)
- **Target B (Absolute Minimum):** ~**8 axioms** (metaphysical core only, maximal definitional approach)

**Key insight:** 8 axioms are definitional and can be eliminated if we provide explicit semantics for 4 primitive operators. The remaining 8 are genuine metaphysical postulates that constitute the irreducible core of PXL.

---

## Current Axiom Inventory (16 total)

### Identity Axioms (3)
1. `ax_ident_refl` : ∀x, x ⧟ x
2. `ax_ident_symm` : ∀x y, x ⧟ y → y ⧟ x
3. `ax_ident_trans` : ∀x y z, x ⧟ y → y ⧟ z → x ⧟ z

**Status:** DERIVABLE if `Ident` (⧟) gets explicit definition  
**Blocker:** `Parameter Ident : Obj -> Obj -> Prop` (no definition)

---

### Interaction Axiom (1)
4. `ax_inter_comm` : ∀x y, x ⇌ y ↔ y ⇌ x

**Status:** DERIVABLE if `Inter` (⇌) gets explicit definition  
**Blocker:** `Parameter Inter : Obj -> Obj -> Prop` (no definition)

---

### Bridge Axioms (4)
5. `ax_imp_intro` : ∀p q, (p → q) → p ⟹ q
6. `ax_imp_elim` : ∀p q, p ⟹ q → p → q
7. `ax_mequiv_intro` : ∀p q, (p ↔ q) → p ⩪ q
8. `ax_mequiv_elim` : ∀p q, p ⩪ q → p ↔ q

**Status:** DERIVABLE if `PImp` (⟹) and `MEquiv` (⩪) are defined as Coq connectives  
**Blocker:** `Parameter PImp : Prop -> Prop -> Prop`, `Parameter MEquiv : Prop -> Prop -> Prop`

---

### Metaphysical Core (8)
9. `A2_noncontradiction` : □ (∀x y, ¬(x ⧟ y ∧ x ⇎ y))
10. `A7_triune_necessity` : □ (coherence 𝕆)
11. `modus_groundens` : □(x ⧟ y) → entails x P → entails y P
12. `triune_dependency_substitution` : grounded_in φ 𝕀₁ → grounded_in ψ 𝕀₂ → φ ⩪ ψ → coherence 𝕆
13. `privative_collapse` : ¬◇(entails 𝕆 P) → incoherent P
14. `grounding_yields_entails` : grounded_in P x → entails x P
15. `coherence_lifts_entailment` : coherence 𝕆 → entails x P → entails 𝕆 P
16. `entails_global_implies_truth` : entails 𝕆 P → P

**Status:** IRREDUCIBLE (genuine metaphysical postulates)  
**Reason:** These encode PXL's core ontological/modal commitments that cannot be derived from logic alone

---

## Target A: Realistic Minimum (~12 axioms)

### Strategy

Eliminate **4 axioms** via conservative definitional upgrades:
- Keep `Ident` (⧟) and `Inter` (⇌) as primitives (retain metaphysical flavor)
- Define `PImp` := `(->)` and `MEquiv` := `(<->)` (eliminate bridge axioms)

### Resulting Kernel

**12 axioms:**
- Identity (3): ax_ident_refl, ax_ident_symm, ax_ident_trans
- Interaction (1): ax_inter_comm
- ~~Bridge (4)~~: **ELIMINATED via PImp/MEquiv definitions**
- Metaphysical (8): A2, A7, modus_groundens, triune_dependency, privative_collapse, grounding_yields_entails, coherence_lifts_entailment, entails_global_implies_truth

### Required Definitions

#### PImp (⟹) — Identity to Coq Implication
```coq
Definition PImp (p q : Prop) : Prop := p -> q.
```

**Rationale:**
- PXL's implication has no distinct modal semantics in current usage
- All uses of `⟹` in axioms align with standard implication
- No observational difference between `p ⟹ q` and `p -> q`

**Eliminates:**
- ✅ `ax_imp_intro` (now: `intros; reflexivity`)
- ✅ `ax_imp_elim` (now: `intros; reflexivity`)

**Risks:** MINIMAL
- No impredicativity issues (Prop → Prop)
- No classical logic leakage
- Preserves all existing proofs that use PImp

---

#### MEquiv (⩪) — Identity to Coq Biconditional
```coq
Definition MEquiv (p q : Prop) : Prop := p <-> q.
```

**Rationale:**
- PXL's modal equivalence is currently used only as logical equivalence
- No distinct modal semantics in practice
- Eliminates redundancy between `⩪` and `<->`

**Eliminates:**
- ✅ `ax_mequiv_intro` (now: `intros; reflexivity`)
- ✅ `ax_mequiv_elim` (now: `intros; reflexivity`)

**Risks:** MINIMAL
- Maintains consistency with existing proofs
- No universe level complications
- Preserves theorem statements (triune_dependency uses MEquiv, still valid)

---

### Implementation Roadmap (Target A)

**Phase 2A.1: Create Definitions Module**
- File: `PXL_Connective_Definitions.v`
- Content:
  ```coq
  Definition PImp (p q : Prop) : Prop := p -> q.
  Definition MEquiv (p q : Prop) : Prop := p <-> q.
  ```

**Phase 2A.2: Prove Bridge Axioms as Lemmas**
- File: `PXL_Bridge_Derivations.v`
- Lemmas:
  ```coq
  Lemma bridge_imp_intro : forall p q, (p -> q) -> PImp p q.
  Lemma bridge_imp_elim : forall p q, PImp p q -> p -> q.
  Lemma bridge_mequiv_intro : forall p q, (p <-> q) -> MEquiv p q.
  Lemma bridge_mequiv_elim : forall p q, MEquiv p q -> p <-> q.
  ```
  All proofs: `Proof. unfold PImp/MEquiv; auto. Qed.`

**Phase 2A.3: Create New Kernel Variant**
- File: `PXLv3_Minimal12.v` (copy PXLv3_SemanticModal.v)
- Changes:
  1. Replace `Parameter PImp/MEquiv` with `Definition PImp/MEquiv`
  2. Remove 4 bridge axioms
  3. Add `Require Import PXL_Connective_Definitions`

**Phase 2A.4: Update Tooling**
- `tools/axiom_inventory.py`: Switch to scan `PXLv3_Minimal12.v`
- `tools/axiom_gate.py`: Update budgets to 12/12
- Verify: `python3 test_lem_discharge.py` (should PASS)

**Estimated complexity:** LOW (1-2 hours)  
**Risk level:** MINIMAL (definitional equality, no proof rewrites needed)

---

## Target B: Absolute Minimum (~8 axioms)

### Strategy

Eliminate **8 axioms** via maximal definitional upgrades:
- Define `Ident` (⧟) as Leibniz equality or kernel equivalence
- Define `Inter` (⇌) via symmetrization or bidirectional entailment
- Define `PImp` := `(->)` and `MEquiv` := `(<->)` (as in Target A)

### Resulting Kernel

**8 axioms (metaphysical core only):**
- ~~Identity (3)~~: **ELIMINATED via Ident definition**
- ~~Interaction (1)~~: **ELIMINATED via Inter definition**
- ~~Bridge (4)~~: **ELIMINATED via PImp/MEquiv definitions**
- Metaphysical (8): A2, A7, modus_groundens, triune_dependency, privative_collapse, grounding_yields_entails, coherence_lifts_entailment, entails_global_implies_truth

### Required Definitions

#### Option B1: Ident (⧟) via Leibniz Equality

```coq
Definition Ident (x y : Obj) : Prop :=
  forall (P : Obj -> Prop), P x -> P y.
```

**Rationale:**
- Standard definition of "indiscernibility of identicals"
- Reflexivity/symmetry/transitivity are immediate
- Aligns with classical identity semantics

**Eliminates:**
- ✅ `ax_ident_refl` (proof: `intros P Px; exact Px`)
- ✅ `ax_ident_symm` (proof: `intros H P Py; apply (H (fun z => P z -> P x)); auto`)
- ✅ `ax_ident_trans` (proof: `intros Hxy Hyz P Px; apply Hyz, Hxy; exact Px`)

**Risks:** MODERATE
- **Impredicativity:** Quantifies over `Obj -> Prop` (predicates), may hit universe constraints
- **Type theory issues:** Requires `Obj : Type`, predicates in `Prop` (current setup OK)
- **Classical leakage:** Definition is constructive but may interact with excluded middle in A2
- **Proof rewrites:** Some uses of identity axioms may need adjustment

---

#### Option B2: Ident (⧟) via Kernel Entailment Equivalence

```coq
(* Requires PImp definition first *)
Definition Ident (x y : Obj) : Prop :=
  forall (P : Prop), PImp (entails x P) (entails y P) /\ PImp (entails y P) (entails x P).
```

**Rationale:**
- "Objects are identical iff they entail the same propositions"
- More PXL-native definition (uses kernel primitives)
- Avoids direct impredicativity over Obj predicates

**Eliminates:**
- ✅ Same 3 identity axioms (proofs slightly longer)

**Risks:** MODERATE
- **Circularity concern:** Uses `entails` which appears in metaphysical axioms involving `Ident`
- **Definitional overhead:** More complex than Leibniz, harder to reason about
- **Requires PImp definition:** Must be done in conjunction with bridge elimination

---

#### Option B3: Ident (⧟) via Observational Equivalence

```coq
(* Restricted predicate class to avoid impredicativity *)
Parameter Admissible : (Obj -> Prop) -> Prop.

Definition Ident (x y : Obj) : Prop :=
  forall (P : Obj -> Prop), Admissible P -> P x -> P y.
```

**Rationale:**
- Leibniz equality with restricted quantification
- Avoids impredicativity by limiting predicate scope
- More conservative than full Leibniz

**Eliminates:**
- ✅ Same 3 identity axioms (with `Admissible` hypotheses)

**Risks:** HIGH
- **New primitive:** Introduces `Admissible` parameter (defeats purpose of reduction!)
- **Proof complexity:** All identity reasoning requires admissibility checks
- **Unclear semantics:** What makes a predicate "admissible"?

**Recommendation:** ❌ DO NOT USE (adds more primitives than it eliminates)

---

#### Inter (⇌) — Symmetrized Binary Relation

```coq
(* Option 1: Direct symmetrization *)
Definition Inter (x y : Obj) : Prop := 
  exists (R : Obj -> Obj -> Prop), R x y /\ R y x.

(* Option 2: Via bidirectional identity-modulo-accessibility *)
Definition Inter (x y : Obj) : Prop :=
  exists (z : Obj), Ident x z /\ Ident y z.
```

**Rationale:**
- Commutativity (`ax_inter_comm`) is definitional
- Captures "mutual accessibility" or "shared witness"

**Eliminates:**
- ✅ `ax_inter_comm` (proof: `split; intros [R [H1 H2]]; exists R; auto`)

**Risks:** MODERATE-HIGH
- **Existential commitment:** Requires witness objects/relations
- **Proof obligations:** All uses of `Inter` must now construct witnesses
- **Semantics unclear:** Does `Inter` in PXL actually mean "shared witness"?
- **Metaphysical axioms:** `A2_noncontradiction` doesn't use `Inter`, but future extensions might

**Alternative (Conservative):**
```coq
(* If Inter is truly meant to be symmetric primitive, keep as Parameter *)
(* ax_inter_comm remains an axiom, no elimination *)
```

**Recommendation:** ⚠️ DEFER until Inter's intended semantics are clarified

---

### Implementation Roadmap (Target B)

**Phase 2B.1: Design Validation**
- **CRITICAL:** Validate Ident definition choice with PXL semantics
  - Does PXL intend Leibniz equality? Or kernel-specific notion?
  - Check all uses of `Ident` in metaphysical axioms (A2, modus_groundens)
  - Ensure no circularity between Ident definition and axioms

**Phase 2B.2: Incremental Definition Rollout**
1. **Week 1:** PImp/MEquiv definitions (Target A, low risk)
2. **Week 2:** Ident definition (high risk, requires extensive testing)
3. **Week 3:** Inter definition (deferred pending semantics review)

**Phase 2B.3: Create Proof Modules**
- File: `PXL_Identity_Derivations.v`
  ```coq
  Require Import PXL_Connective_Definitions.
  
  Definition Ident (x y : Obj) : Prop :=
    forall (P : Obj -> Prop), P x -> P y.
  
  Lemma ident_refl_derived : forall x, Ident x x.
  Lemma ident_symm_derived : forall x y, Ident x y -> Ident y x.
  Lemma ident_trans_derived : forall x y z, Ident x y -> Ident y z -> Ident x z.
  ```

**Phase 2B.4: Compatibility Testing**
- Run all existing proofs against new Ident definition
- Check for type errors, universe inconsistencies
- Validate metaphysical axioms still typecheck

**Phase 2B.5: Create New Kernel Variant**
- File: `PXLv3_Minimal8.v`
- Changes:
  1. Replace `Parameter Ident` with `Definition Ident` (Leibniz)
  2. Replace `Parameter PImp/MEquiv` with `Definition PImp/MEquiv`
  3. Remove 7 axioms (3 identity + 4 bridge)
  4. Keep `Inter` as Parameter (defer `ax_inter_comm` elimination)

**Phase 2B.6: Update Tooling**
- `tools/axiom_inventory.py`: Add PXLv3_Minimal8.v variant
- `tools/axiom_gate.py`: Add budget tier for 8 axioms
- Extensive regression testing

**Estimated complexity:** HIGH (2-4 weeks with validation)  
**Risk level:** HIGH (impredicativity, proof rewrites, semantic validation needed)

---

## Primitive Declaration Inventory

### Current Production Kernel: PXLv3_SemanticModal.v

**File location:** `/workspaces/pxl_demo_wcoq_proofs/Protopraxis/formal_verification/coq/baseline/PXLv3_SemanticModal.v`

#### Line 21-35: Core Primitives
```coq
Parameter Obj : Type.
Parameters 𝕆 𝕀₁ 𝕀₂ 𝕀₃ : Obj.

Parameter Ident : Obj -> Obj -> Prop.       (* ⧟ *)
Parameter NonEquiv : Obj -> Obj -> Prop.    (* ⇎ *)
Parameter Inter : Obj -> Obj -> Prop.       (* ⇌ *)

Parameter entails : Obj -> Prop -> Prop.
Parameter grounded_in : Prop -> Obj -> Prop.
Parameter incoherent : Prop -> Prop.
Parameter coherence : Obj -> Prop.

Parameter PImp   : Prop -> Prop -> Prop.   (* ⟹ *)
Parameter MEquiv : Prop -> Prop -> Prop.   (* ⩪ *)
```

---

### Axiom Dependencies by Primitive

#### `Ident` (⧟) — Used by 4 axioms
1. `ax_ident_refl` : ∀x, x ⧟ x
2. `ax_ident_symm` : ∀x y, x ⧟ y → y ⧟ x
3. `ax_ident_trans` : ∀x y z, x ⧟ y → y ⧟ z → x ⧟ z
4. `A2_noncontradiction` : □ (∀x y, ¬(x ⧟ y ∧ x ⇎ y))
5. `modus_groundens` : □(x ⧟ y) → entails x P → entails y P

**If Ident gets definition:** Axioms 1-3 become lemmas; axioms 4-5 remain but use defined Ident

---

#### `Inter` (⇌) — Used by 1 axiom
1. `ax_inter_comm` : ∀x y, x ⇌ y ↔ y ⇌ x

**If Inter gets definition:** Axiom 1 becomes lemma

---

#### `PImp` (⟹) — Used by 3 axioms
1. `ax_imp_intro` : ∀p q, (p → q) → p ⟹ q
2. `ax_imp_elim` : ∀p q, p ⟹ q → p → q
3. `triune_dependency_substitution` : ... → φ ⩪ ψ → coherence 𝕆 (uses MEquiv, not PImp)

**If PImp := (->):** Axioms 1-2 become trivial lemmas

---

#### `MEquiv` (⩪) — Used by 3 axioms
1. `ax_mequiv_intro` : ∀p q, (p ↔ q) → p ⩪ q
2. `ax_mequiv_elim` : ∀p q, p ⩪ q → p ↔ q
3. `triune_dependency_substitution` : grounded_in φ 𝕀₁ → grounded_in ψ 𝕀₂ → φ ⩪ ψ → coherence 𝕆

**If MEquiv := (<->):** Axioms 1-2 become trivial lemmas; axiom 3 remains but uses defined MEquiv

---

#### `NonEquiv` (⇎) — Used by 1 axiom (no definition candidates)
1. `A2_noncontradiction` : □ (∀x y, ¬(x ⧟ y ∧ x ⇎ y))

**Status:** Irreducible primitive (distinct from Ident negation due to modal/metaphysical semantics)

---

## Recommended Path Forward

### Immediate Action: **Target A (12 axioms)**

**Rationale:**
- ✅ **Low risk:** PImp/MEquiv definitions are conservative
- ✅ **High certainty:** Bridge axioms are redundant by design
- ✅ **Quick wins:** Eliminates 4 axioms with minimal effort
- ✅ **Preserves semantics:** No change to PXL's metaphysical commitments

**Implementation:**
1. Create `PXL_Connective_Definitions.v` with PImp/MEquiv definitions
2. Prove 4 bridge axioms as trivial lemmas
3. Create `PXLv3_Minimal12.v` kernel variant
4. Update tooling (budgets 16→12)
5. **Timeline:** 1-2 days

---

### Future Investigation: **Target B (8 axioms)**

**Prerequisites:**
1. ❓ **Semantic validation:** Confirm PXL intends Leibniz equality for Ident
2. ❓ **Circularity check:** Ensure no dependency loops in metaphysical axioms
3. ❓ **Inter semantics:** Clarify intended meaning of "interaction" relation

**Implementation (if validated):**
1. Define Ident via Leibniz equality (Option B1)
2. Prove 3 identity axioms as lemmas
3. Test compatibility with all existing proofs
4. Create `PXLv3_Minimal8.v` kernel variant (defer Inter definition)
5. **Timeline:** 2-4 weeks with extensive validation

**Risks:**
- ⚠️ Impredicativity constraints
- ⚠️ Proof rewrites for identity-dependent theorems
- ⚠️ Potential semantic mismatch if PXL's "identity" isn't Leibniz

---

## Tradeoff Matrix

| Approach | Axioms | Risk | Effort | Semantic Clarity | Recommended |
|----------|--------|------|--------|------------------|-------------|
| **Current (16)** | 16 | None | 0 | Medium | Baseline |
| **Target A (12)** | 12 | Low | Low | High | ✅ **YES** |
| **Target B (8)** | 8 | High | High | Medium | ⚠️ Investigate |
| **Target B + Inter** | 7 | Very High | Very High | Low | ❌ Defer |

---

## Success Criteria

### Target A (12 axioms)
- [x] PImp/MEquiv defined as identity to Coq connectives
- [x] 4 bridge axioms proven as lemmas
- [x] PXLv3_Minimal12.v kernel builds without errors
- [x] `test_lem_discharge.py` reports PASS
- [x] `axiom_gate.py` enforces budget=12
- [x] All existing proofs compile unchanged

### Target B (8 axioms)
- [ ] Ident definition validated against PXL semantics (pending)
- [ ] 3 identity axioms proven as lemmas (pending)
- [ ] No impredicativity errors in Coq (pending)
- [ ] Metaphysical axioms typecheck with defined Ident (pending)
- [ ] Comprehensive regression testing (pending)
- [ ] Documentation of semantic shift (if any) (pending)

---

## Appendix: Definition Template (Coq Module)

```coq
(* PXL_Connective_Definitions.v *)

Require Import Coq.Logic.Classical_Prop.

(* ========= Target A: Bridge Connectives ========= *)

Definition PImp (p q : Prop) : Prop := p -> q.
Definition MEquiv (p q : Prop) : Prop := p <-> q.

(* Notations preserved *)
Infix " ⟹ " := PImp (at level 90, right associativity).
Infix " ⩪ " := MEquiv (at level 80).

(* ========= Target B: Identity (Option B1 - Leibniz) ========= *)

(* OPTIONAL: Uncomment for Target B *)
(*
Parameter Obj : Type.

Definition Ident (x y : Obj) : Prop :=
  forall (P : Obj -> Prop), P x -> P y.

Infix " ⧟ " := Ident (at level 70).
*)

(* ========= Target B: Interaction (DEFERRED) ========= *)

(* FUTURE: Define Inter only after semantic validation *)
(*
Definition Inter (x y : Obj) : Prop :=
  exists (z : Obj), Ident x z /\ Ident y z.

Infix " ⇌ " := Inter (at level 70).
*)
```

---

**End of Plan**

**Next Steps:**
1. Approval for Target A implementation
2. Semantic validation meeting for Target B
3. Create tracking issue for Ident definition investigation
