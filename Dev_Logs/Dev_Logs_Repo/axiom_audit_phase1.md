# Axiom Audit — Phase 3 Complete ✅

**Generated:** 2025-12-20 | **Updated:** 2025-12-21  
**Current State:** **8 core axioms (Phase 2/3 complete)**  
**Eliminated:** 12 axioms proven as lemmas via definitional upgrades & semantic modal logic

---

## Phase 3: 8 Irreducible Axioms

### Current Kernel (PXLv3.v): 8 total

### Current Kernel (PXLv3.v): 8 total

#### **Core Metaphysical Axioms (2)**

```coq
1. A2_noncontradiction      : □ (∀x y, ¬(x ⧟ y ∧ x ⇎ y))
2. A7_triune_necessity      : □ (coherence 𝕆)
```

#### **Bridging Principles (6)**

```coq
3. modus_groundens                : ∀x y P, □(x ⧟ y) → entails x P → entails y P
4. triune_dependency_substitution : ∀φ ψ, grounded_in φ 𝕀₁ → grounded_in ψ 𝕀₂ → φ ⩪ ψ → coherence 𝕆
5. privative_collapse             : ∀P, ¬(◇(entails 𝕆 P)) → incoherent P
6. grounding_yields_entails       : ∀x P, grounded_in P x → entails x P
7. coherence_lifts_entailment     : ∀x P, coherence 𝕆 → entails x P → entails 𝕆 P
8. entails_global_implies_truth   : ∀P, entails 𝕆 P → P
```

---

## Phase 2/3 Eliminated Axioms (Now Proven Lemmas)

### **Category A: Modal Frame Conditions** (3 eliminated)

**Eliminated via Semantic Modal Logic** ([PXL_Modal_Axioms_Semantic.v](Protopraxis/formal_verification/coq/baseline/PXL_Modal_Axioms_Semantic.v)):

```coq
✅ ax_K   : □(p → q) → □p → □q     [NOW: Kripke frame condition]
✅ ax_T   : □p → p                 [NOW: frame_reflexivity]  
✅ ax_Nec : p → □p                 [NOW: frame_necessitation]
```

These are not arbitrary axioms but **semantic properties** derivable from S5 Kripke frame structure. The modal operators (□, ◇) are grounded in reflexive, symmetric, transitive accessibility relations.

These are not arbitrary axioms but **semantic properties** derivable from S5 Kripke frame structure. The modal operators (□, ◇) are grounded in reflexive, symmetric, transitive accessibility relations.

### **Category B: Structural Properties** (5 eliminated)

**Eliminated via Definitional Upgrades** ([PXL_Derivations_Phase2.v](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v)):

```coq
✅ ax_ident_refl  : ∀x, x ⧟ x              [NOW: ident_refl_derived from Leibniz definition]
✅ ax_ident_symm  : ∀x y, x ⧟ y → y ⧟ x   [NOW: ident_symm_derived]
✅ ax_ident_trans : ∀x y z, x ⧟ y → y ⧟ z → x ⧟ z  [NOW: ident_trans_derived]
✅ ax_inter_comm  : ∀x y, x ⇌ y ↔ y ⇌ x   [NOW: inter_comm_derived from symmetric witness]
✅ ax_nonequiv_irrefl : ∀x, ¬(x ⇎ x)      [NOW: nonequiv_irrefl_derived](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L104-L121)
```

**Key Innovation:** `Ident` (⧟) is now defined as Leibniz equality, not a primitive parameter. This makes reflexivity, symmetry, and transitivity theorems, not axioms.

### **Category C: Bridge Axioms** (4 eliminated)

**Eliminated via Connective Definitions** ([PXL_Derivations_Phase2.v](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v)):

```coq
✅ ax_imp_intro    : ∀p q, (p → q) → p ⟹ q   [NOW: PImp := (→) definitionally]
✅ ax_imp_elim     : ∀p q, p ⟹ q → p → q    [NOW: definitional unfolding]
✅ ax_mequiv_intro : ∀p q, (p ↔ q) → p ⩪ q   [NOW: MEquiv := (↔) definitionally]
✅ ax_mequiv_elim  : ∀p q, p ⩪ q → p ↔ q    [NOW: definitional unfolding]
```

**Key Innovation:** PXL connectives (⟹, ⩪) are now defined as Coq connectives (→, ↔), making bridge axioms trivial definitional equalities.

**Key Innovation:** PXL connectives (⟹, ⩪) are now defined as Coq connectives (→, ↔), making bridge axioms trivial definitional equalities.

---

## Axiom Reduction Timeline

| Phase | Axiom Count | Eliminated | Method |
|-------|-------------|------------|--------|
| **Initial (PXLv3 pre-Phase 1)** | 20 | — | Baseline |
| **Phase 1 (Semantic Modal)** | 17 | 3 modal | Kripke semantics |
| **Phase 2 (Definitional)** | 12 | 5 structural | Leibniz Ident, symmetric Inter |
| **Phase 3 (Bridge elimination)** | **8** | 4 bridge | PImp := (→), MEquiv := (↔) |

**Achievement:** **60% reduction** (20 → 8 axioms) while maintaining full proof power.

---

## Why These 8 Axioms Are Irreducible

### **A2_noncontradiction** & **A7_triune_necessity**
- **Genuinely metaphysical**: Postulate the structure of 𝕆 (necessary being) and the coherence of trinitarian identities
- **Cannot be derived**: These are foundational commitments about the nature of being itself
- **Comparable to**: ZFC's axiom of infinity, Peano's axioms for arithmetic

### **6 Bridging Principles**
- **Ground entailment semantics**: Link grounding, entailment, and coherence
- **Not eliminable without**: Defining richer semantic structures (possible future Phase 4)
- **Currently irreducible**: Would require a complete semantics of `entails`, `grounded_in`, `coherence`, `incoherent`

**Note:** Further reduction to ~6 axioms may be possible if entailment/grounding predicates receive explicit semantic definitions, but this would require substantial foundational work.

---

**Note:** Further reduction to ~6 axioms may be possible if entailment/grounding predicates receive explicit semantic definitions, but this would require substantial foundational work.

---

## HISTORICAL CONTEXT: Phase 1 Planning (Completed)

*The sections below document the original Phase 1 planning. All targets have been exceeded.*

### Original Phase 1 Target: 17 → ~12-14 Axioms ✅ EXCEEDED

**Actual Achievement: 17 → 8 axioms** (Phase 2/3 combined)

### Original Classification Analysis

#### **CATEGORY A: STRUCTURAL** ✅ ALL ELIMINATED

**1. ax_ident_refl** ✅ ELIMINATED
```coq
Axiom ax_ident_refl : forall x : Obj, x ⧟ x.
```coq
Axiom ax_ident_refl : forall x : Obj, x ⧟ x.
```
- **Status:** ✅ ELIMINATED (now `ident_refl_derived` in PXL_Derivations_Phase2.v)

- **Status:** ✅ ELIMINATED (now `ident_refl_derived` in PXL_Derivations_Phase2.v)

**2. ax_ident_symm** ✅ ELIMINATED
```coq
Axiom ax_ident_symm : forall x y : Obj, x ⧟ y -> y ⧟ x.
```
- **Status:** ✅ ELIMINATED (now `ident_symm_derived`)

**3. ax_ident_trans** ✅ ELIMINATED
```coq
Axiom ax_ident_trans : forall x y z : Obj, x ⧟ y -> y ⧟ z -> x ⧟ z.
```
- **Status:** ✅ ELIMINATED (now `ident_trans_derived`)

**4. ax_nonequiv_irrefl** ✅ ELIMINATED
```coq
Axiom ax_nonequiv_irrefl : forall x : Obj, ~ (x ⇎ x).
```
- **Status:** ✅ ELIMINATED (now `nonequiv_irrefl_derived` in PXL_Derivations_Phase2.v)

**5. ax_inter_comm** ✅ ELIMINATED
```coq
Axiom ax_inter_comm : forall x y : Obj, x ⇌ y <-> y ⇌ x.
```
- **Status:** ✅ ELIMINATED (now `inter_comm_derived`)

---

#### **CATEGORY B: BRIDGE AXIOMS** ✅ ALL ELIMINATED

**6. ax_imp_intro** ✅ ELIMINATED
```coq
Axiom ax_ident_symm : forall x y : Obj, x ⧟ y -> y ⧟ x.
```
- **Classification:** STRUCTURAL (symmetry)
- **Derivation likelihood:** EASY
- **Candidate location (historic):** `PXL_Structural_Derivations.v` → realized in `PXL_Derivations_Phase2.v`
- **Dependencies:** Identity definition
- **Strategy:** Symmetry follows from equality properties

**3. ax_ident_trans**
```coq
Axiom ax_ident_trans : forall x y z : Obj, x ⧟ y -> y ⧟ z -> x ⧟ z.
```
- **Classification:** STRUCTURAL (transitivity)
- **Derivation likelihood:** EASY
- **Candidate location (historic):** `PXL_Structural_Derivations.v` → realized in `PXL_Derivations_Phase2.v`
- **Dependencies:** Identity definition
- **Strategy:** Transitivity from equivalence relation structure

**4. ax_nonequiv_irrefl**
```coq
Axiom ax_nonequiv_irrefl : forall x : Obj, ~ (x ⇎ x).
```
- **Classification:** STRUCTURAL (irreflexivity)
- **Derivation likelihood:** MEDIUM
- **Candidate location (historic):** `PXL_Structural_Derivations.v` → realized in `PXL_Derivations_Phase2.v`
- **Dependencies:** A2_noncontradiction, identity axioms
- **Strategy:** Derive from "~ (x ⧟ x /\ x ⇎ x)" + reflexivity

**5. ax_inter_comm**
```coq
Axiom ax_inter_comm : forall x y : Obj, x ⇌ y <-> y ⇌ x.
```
- **Classification:** STRUCTURAL (commutativity)
- **Derivation likelihood:** EASY
- **Candidate location (historic):** `PXL_Structural_Derivations.v` → realized in `PXL_Derivations_Phase2.v`
- **Dependencies:** Inter definition
- **Strategy:** Commutativity is typically definitional for symmetric operations

---

#### **CATEGORY B: BRIDGE AXIOMS (MEDIUM DIFFICULTY)**

**6. ax_imp_intro** ✅ ELIMINATED
```coq
Axiom ax_imp_intro : forall p q : Prop, (p -> q) -> p ⟹ q.
```
- **Status:** ✅ ELIMINATED (PImp := → by definition)

**7. ax_imp_elim** ✅ ELIMINATED
```coq
Axiom ax_imp_elim : forall p q : Prop, p ⟹ q -> p -> q.
```
- **Status:** ✅ ELIMINATED (definitional unfolding)

**8. ax_mequiv_intro** ✅ ELIMINATED
```coq
Axiom ax_mequiv_intro : forall p q : Prop, (p <-> q) -> p ⩪ q.
```
- **Status:** ✅ ELIMINATED (MEquiv := ↔ by definition)

**9. ax_mequiv_elim** ✅ ELIMINATED
```coq
Axiom ax_mequiv_elim : forall p q : Prop, p ⩪ q -> p <-> q.
```
- **Status:** ✅ ELIMINATED (definitional unfolding)

---

#### **CATEGORY C: METAPHYSICAL/ONTOLOGICAL** ✅ RETAINED (Irreducible)

**10. A2_noncontradiction** ✅ RETAINED
**10. A2_noncontradiction** ✅ RETAINED
```coq
Axiom A2_noncontradiction : □ (forall x y : Obj, ~ (x ⧟ y /\ x ⇎ y)).
```
- **Status:** ✅ RETAINED (irreducible metaphysical axiom)

**11-17.** All other metaphysical/bridging axioms ✅ RETAINED

See current 8-axiom listing at top of document for final kernel state.

---

## Phase 3 Success Criteria ✅ ALL MET

- ✅ 12 axioms eliminated (20 → 8)
- ✅ All eliminations proven in PXL_Definitions.v, PXL_Derivations_Phase2.v, PXL_Modal_Axioms_Semantic.v
- ✅ All existing proofs still compile
- ✅ `pxl_excluded_middle`, `trinitarian_optimization` remain assumption-free
- ✅ Repository verification tests pass with 8-axiom kernel

---

## References

- [PXLv3.v](Protopraxis/formal_verification/coq/baseline/PXLv3.v) — Current 8-axiom kernel
- [PXL_Definitions.v](Protopraxis/formal_verification/coq/baseline/PXL_Definitions.v) — Ident, Inter, PImp, MEquiv definitions
- [PXL_Derivations_Phase2.v](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v) — 9 axioms proven as lemmas
- [PXL_Modal_Axioms_Semantic.v](Protopraxis/formal_verification/coq/baseline/PXL_Modal_Axioms_Semantic.v) — 3 modal axioms as frame conditions
- [AXIOM_MINIMUM_PLAN.md](AXIOM_MINIMUM_PLAN.md) — Phase 2 design document
- [BRIDGE_AXIOMS_STATUS.md](BRIDGE_AXIOMS_STATUS.md) — Bridge axiom analysis
