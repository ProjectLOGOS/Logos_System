# Primitive Declarations — Complete Inventory

**Date:** 2025-12-20  
**Context:** Phase 2 design support — mapping all Parameters to their axiom dependencies  
**Production kernel:** PXLv3_SemanticModal.v

---

## File Locations

### Production Kernel
- **Primary:** [PXLv3_SemanticModal.v](Protopraxis/formal_verification/coq/baseline/PXLv3_SemanticModal.v) (lines 21-35, 90-123)
- **Legacy:** [PXLv3.v](Protopraxis/formal_verification/coq/baseline/PXLv3.v) (not counted by axiom_inventory.py)
- **Development:** [PXLv3_head.v](Protopraxis/formal_verification/coq/baseline/PXLv3_head.v) (diagnostic excerpt)

---

## Primitive Operators (Parameters)

### 1. Obj — Object Domain Type

**Declaration:**
```coq
Parameter Obj : Type.
```

**File:** PXLv3_SemanticModal.v (line 21)

**Dependencies:** All axioms use `Obj` (domain of discourse)

**Status:** **IRREDUCIBLE** (foundational type parameter)

**Notes:**
- Universe: `Type` (not `Prop`)
- Cannot be defined within Coq (external domain)
- Represents PXL's ontological objects (𝕆, 𝕀₁, 𝕀₂, 𝕀₃)

---

### 2. Ident (⧟) — Identity Relation

**Declaration:**
```coq
Parameter Ident : Obj -> Obj -> Prop.
Notation "x ⧟ y" := (Ident x y) (at level 70).
```

**File:** PXLv3_SemanticModal.v (line 24)

**Axioms using Ident (5 total):**
1. `ax_ident_refl` : ∀x, x ⧟ x
2. `ax_ident_symm` : ∀x y, x ⧟ y → y ⧟ x
3. `ax_ident_trans` : ∀x y z, x ⧟ y → y ⧟ z → x ⧟ z
4. `A2_noncontradiction` : □ (∀x y, ¬(x ⧟ y ∧ x ⇎ y))
5. `modus_groundens` : □(x ⧟ y) → entails x P → entails y P

**Elimination potential:**
- ✅ **HIGH** if defined as Leibniz equality: `∀(P : Obj → Prop), P x → P y`
- Eliminates axioms 1-3 (identity laws become lemmas)
- Axioms 4-5 remain but use defined Ident

**Risks:**
- Impredicativity (quantifies over `Obj → Prop`)
- Potential semantic mismatch (is PXL's identity Leibniz?)

---

### 3. NonEquiv (⇎) — Non-Equivalence Relation

**Declaration:**
```coq
Parameter NonEquiv : Obj -> Obj -> Prop.
Notation "x ⇎ y" := (NonEquiv x y) (at level 70).
```

**File:** PXLv3_SemanticModal.v (line 25)

**Axioms using NonEquiv (1 total):**
1. `A2_noncontradiction` : □ (∀x y, ¬(x ⧟ y ∧ x ⇎ y))

**Elimination potential:**
- ❌ **NONE** (appears only in metaphysical axiom)
- Could define as `¬(Ident x y)` but loses modal/metaphysical nuance
- A2 is irreducible, so NonEquiv remains primitive

**Status:** **IRREDUCIBLE** (metaphysical primitive)

**Notes:**
- Distinct from `¬(x ⧟ y)` due to PXL's privative logic
- Encodes "positive incompatibility" not just "lack of identity"

---

### 4. Inter (⇌) — Interaction Relation

**Declaration:**
```coq
Parameter Inter : Obj -> Obj -> Prop.
Notation "x ⇌ y" := (Inter x y) (at level 70).
```

**File:** PXLv3_SemanticModal.v (line 26)

**Axioms using Inter (1 total):**
1. `ax_inter_comm` : ∀x y, x ⇌ y ↔ y ⇌ x

**Elimination potential:**
- ⚠️ **MODERATE** if semantics clarified
- Candidate definition: `∃(z : Obj), x ⧟ z ∧ y ⧟ z` (shared witness)
- Eliminates `ax_inter_comm` (commutativity is definitional)

**Risks:**
- Unclear semantics (what does "interaction" mean in PXL?)
- Existential commitment may not match intended use
- No other axioms use Inter (low validation surface)

**Status:** **DEFER** pending semantic validation

---

### 5. PImp (⟹) — PXL Implication

**Declaration:**
```coq
Parameter PImp : Prop -> Prop -> Prop.
Notation "p ⟹ q" := (PImp p q) (at level 90, right associativity).
```

**File:** PXLv3_SemanticModal.v (line 33)

**Axioms using PImp (2 total):**
1. `ax_imp_intro` : ∀p q, (p → q) → p ⟹ q
2. `ax_imp_elim` : ∀p q, p ⟹ q → p → q

**Elimination potential:**
- ✅ **MAXIMAL** — Define as identity to Coq implication: `p → q`
- Eliminates both bridge axioms (become trivial: `reflexivity`)
- Used in `triune_dependency_substitution` but definition preserves semantics

**Recommended definition:**
```coq
Definition PImp (p q : Prop) : Prop := p -> q.
```

**Risks:** **MINIMAL**
- No observational difference between `⟹` and `→` in current usage
- All proofs using PImp remain valid
- No impredicativity or universe issues

**Status:** ✅ **READY FOR IMPLEMENTATION** (Target A)

---

### 6. MEquiv (⩪) — PXL Modal Equivalence

**Declaration:**
```coq
Parameter MEquiv : Prop -> Prop -> Prop.
Notation "p ⩪ q" := (MEquiv p q) (at level 80).
```

**File:** PXLv3_SemanticModal.v (line 34)

**Axioms using MEquiv (3 total):**
1. `ax_mequiv_intro` : ∀p q, (p ↔ q) → p ⩪ q
2. `ax_mequiv_elim` : ∀p q, p ⩪ q → p ↔ q
3. `triune_dependency_substitution` : grounded_in φ 𝕀₁ → grounded_in ψ 𝕀₂ → φ ⩪ ψ → coherence 𝕆

**Elimination potential:**
- ✅ **MAXIMAL** — Define as identity to Coq biconditional: `p ↔ q`
- Eliminates axioms 1-2 (become trivial: `reflexivity`)
- Axiom 3 remains but uses defined MEquiv (no semantic change)

**Recommended definition:**
```coq
Definition MEquiv (p q : Prop) : Prop := p <-> q.
```

**Risks:** **MINIMAL**
- MEquiv used only as logical equivalence in practice
- No distinct modal semantics observed
- Preserves `triune_dependency_substitution` statement

**Status:** ✅ **READY FOR IMPLEMENTATION** (Target A)

---

## Non-Eliminable Primitives (Metaphysical Core)

These Parameters appear only in metaphysical axioms and cannot be defined within Coq:

### entails : Obj → Prop → Prop
**Used by:** `modus_groundens`, `grounding_yields_entails`, `coherence_lifts_entailment`, `entails_global_implies_truth`, `privative_collapse`

**Status:** IRREDUCIBLE (core PXL operator)

---

### grounded_in : Prop → Obj → Prop
**Used by:** `triune_dependency_substitution`, `grounding_yields_entails`

**Status:** IRREDUCIBLE (metaphysical primitive)

---

### coherence : Obj → Prop
**Used by:** `A7_triune_necessity`, `triune_dependency_substitution`, `coherence_lifts_entailment`

**Status:** IRREDUCIBLE (modal/metaphysical)

---

### incoherent : Prop → Prop
**Used by:** `privative_collapse`

**Status:** IRREDUCIBLE (privative logic primitive)

---

### Box (□), Dia (◇) — Modal Operators

**Status in PXLv3_SemanticModal.v:**
- ✅ **ALREADY ELIMINATED** (now Definitions via Kripke semantics)
- Lines 50-52:
  ```coq
  Definition Box (p : Prop) : Prop := PXL_Box W R p.
  Definition Dia (p : Prop) : Prop := PXL_Dia W R p.
  ```

**Used by:** `A2_noncontradiction`, `A7_triune_necessity`, `modus_groundens`, `privative_collapse`

---

## Elimination Summary by Category

### Category 1: READY (Target A — 4 axioms)
- ✅ PImp → `Definition PImp p q := p -> q`
- ✅ MEquiv → `Definition MEquiv p q := p <-> q`
- **Eliminates:** `ax_imp_intro`, `ax_imp_elim`, `ax_mequiv_intro`, `ax_mequiv_elim`

---

### Category 2: INVESTIGATE (Target B — 3 axioms)
- ⚠️ Ident → `Definition Ident x y := ∀P, P x → P y` (Leibniz)
- **Eliminates:** `ax_ident_refl`, `ax_ident_symm`, `ax_ident_trans`
- **Risks:** Impredicativity, semantic validation needed

---

### Category 3: DEFER (1 axiom)
- ⏸️ Inter → Requires semantic clarification
- **Eliminates:** `ax_inter_comm`
- **Risk:** Unclear what "interaction" means in PXL

---

### Category 4: IRREDUCIBLE (8 axioms)
- ❌ NonEquiv, entails, grounded_in, coherence, incoherent (metaphysical)
- **Status:** Genuine postulates, cannot be defined

---

## Dependency Graph

```
Obj (Type)
 ├─ Ident (⧟)
 │   ├─ ax_ident_refl       [DERIVABLE if Ident defined]
 │   ├─ ax_ident_symm       [DERIVABLE if Ident defined]
 │   ├─ ax_ident_trans      [DERIVABLE if Ident defined]
 │   ├─ A2_noncontradiction [IRREDUCIBLE - uses NonEquiv]
 │   └─ modus_groundens     [IRREDUCIBLE - uses entails]
 │
 ├─ NonEquiv (⇎)
 │   └─ A2_noncontradiction [IRREDUCIBLE]
 │
 ├─ Inter (⇌)
 │   └─ ax_inter_comm       [DERIVABLE if Inter defined]
 │
 └─ entails
     ├─ modus_groundens               [IRREDUCIBLE]
     ├─ grounding_yields_entails      [IRREDUCIBLE]
     ├─ coherence_lifts_entailment    [IRREDUCIBLE]
     ├─ entails_global_implies_truth  [IRREDUCIBLE]
     └─ privative_collapse            [IRREDUCIBLE]

Prop
 ├─ PImp (⟹)
 │   ├─ ax_imp_intro   [DERIVABLE → Define as (->)]
 │   └─ ax_imp_elim    [DERIVABLE → Define as (->)]
 │
 ├─ MEquiv (⩪)
 │   ├─ ax_mequiv_intro            [DERIVABLE → Define as (<->)]
 │   ├─ ax_mequiv_elim             [DERIVABLE → Define as (<->)]
 │   └─ triune_dependency_substitution [IRREDUCIBLE but uses defined MEquiv]
 │
 ├─ grounded_in
 │   ├─ triune_dependency_substitution [IRREDUCIBLE]
 │   └─ grounding_yields_entails       [IRREDUCIBLE]
 │
 ├─ coherence
 │   ├─ A7_triune_necessity            [IRREDUCIBLE]
 │   ├─ triune_dependency_substitution [IRREDUCIBLE]
 │   └─ coherence_lifts_entailment     [IRREDUCIBLE]
 │
 └─ incoherent
     └─ privative_collapse [IRREDUCIBLE]
```

---

## Next Actions

1. **Immediate (Target A):**
   - Create `PXL_Connective_Definitions.v` with PImp/MEquiv definitions
   - Prove 4 bridge axioms as lemmas
   - Update kernel to PXLv3_Minimal12.v

2. **Investigation (Target B):**
   - Validate Ident semantics: Is Leibniz equality PXL's intent?
   - Check A2_noncontradiction/modus_groundens for circularity with defined Ident
   - Prototype Ident definition in test module

3. **Deferred:**
   - Inter semantic validation meeting
   - Document intended meaning of "interaction" in PXL
   - Revisit ax_inter_comm elimination in Phase 3

---

**End of Inventory**
