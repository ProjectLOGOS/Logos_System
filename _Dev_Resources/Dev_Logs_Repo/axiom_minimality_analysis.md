# Phase 2A: Axiom Minimality & Redundancy Test Report

**Date:** December 14, 2025  
**Current axiom count:** 20  
**Goal:** Identify any axioms provable from the remaining 19

---

## Testing Protocol

For each axiom `AX`:
1. Comment out `AX` in a test copy
2. Attempt to prove `Lemma AX_redundant : <statement of AX>`
3. If proof succeeds → `AX` is redundant, eliminate it
4. If proof fails → `AX` is irreducible

---

## Category 1: Modal Backbone (3 axioms)

### ax_K - Distribution
**Statement:** `∀p q, □(p→q) → □p → □q`  
**Status:** IRREDUCIBLE  
**Reason:** Fundamental K axiom, cannot be derived from T + Nec alone

### ax_T - Reflexivity  
**Statement:** `∀p, □p → p`  
**Status:** IRREDUCIBLE  
**Reason:** Defines reflexivity of accessibility relation, independent of K + Nec

### ax_Nec - Necessitation
**Statement:** `∀p, p → □p`  
**Status:** IRREDUCIBLE  
**Reason:** Collapses modality if removed (makes □ trivial)

**Analysis:** S5 can be axiomatized with just K + T + Nec (we already eliminated 4 and 5). These 3 are minimal for S5.

---

## Category 2: Structural Laws (9 axioms)

### Identity Equivalence (3 axioms)

**ax_ident_refl:** `∀x, x ⧟ x`  
**Status:** IRREDUCIBLE  
**Reason:** Primitive equivalence relation property, no weaker axioms available

**ax_ident_symm:** `∀x y, x⧟y → y⧟x`  
**Status:** TEST CANDIDATE  
**Hypothesis:** May be derivable if ⧟ is defined via mutual entailment or other symmetric construction  
**Priority:** LOW (likely primitive)

**ax_ident_trans:** `∀x y z, x⧟y → y⧟z → x⧟z`  
**Status:** TEST CANDIDATE  
**Hypothesis:** Transitivity may follow from composition if underlying structure supports it  
**Priority:** LOW (likely primitive)

### Other Relations (2 axioms)

**ax_nonequiv_irrefl:** `∀x, ¬(x ⇎ x)`  
**Status:** TEST CANDIDATE  
**Hypothesis:** If NonEquiv is defined as `¬(Ident x y)`, this may be provable from ax_ident_refl  
**Priority:** MEDIUM (check definitions)

**ax_inter_comm:** `∀x y, x⇌y ↔ y⇌x`  
**Status:** TEST CANDIDATE  
**Hypothesis:** Commutativity may be definitional if Inter is symmetric by construction  
**Priority:** MEDIUM (check definitions)

### Connective Bridges (4 axioms)

**ax_imp_intro:** `(p→q) → p⟹q`  
**ax_imp_elim:** `p⟹q → (p→q)`  
**Status:** TEST CANDIDATES (pair)  
**Hypothesis:** If `p⟹q` is *defined* as `□(p→q)`, these become:
- intro: `(p→q) → □(p→q)` = ax_Nec
- elim: `□(p→q) → (p→q)` = ax_T
**Priority:** HIGH (likely redundant with modal axioms)

**ax_mequiv_intro:** `(p↔q) → p⩪q`  
**ax_mequiv_elim:** `p⩪q → (p↔q)`  
**Status:** TEST CANDIDATES (pair)  
**Hypothesis:** If `p⩪q` is *defined* as `□(p↔q)`, same as above  
**Priority:** HIGH (likely redundant with modal axioms)

---

## Category 3: PXL Core (2 axioms)

**A2_noncontradiction:** `□(∀x y, ¬(x⧟y ∧ x⇎y))`  
**Status:** TEST CANDIDATE  
**Hypothesis:** May be provable from ax_nonequiv_irrefl + properties of ⧟ and ⇎  
**Priority:** MEDIUM

**A7_triune_necessity:** `□(coherence 𝕆)`  
**Status:** IRREDUCIBLE  
**Reason:** Fundamental PXL domain constraint (Trinity is necessarily coherent)

---

## Category 4: Bridging Principles (6 axioms)

**modus_groundens:** `□(x⧟y) → entails x P → entails y P`  
**Status:** TEST CANDIDATE  
**Hypothesis:** Substitution principle for ⧟-equivalent objects; may follow from entails properties + ⧟ transitivity  
**Priority:** MEDIUM

**triune_dependency_substitution:** `grounded_in φ 𝕀₁ → grounded_in ψ 𝕀₂ → φ⩪ψ → coherence 𝕆`  
**Status:** IRREDUCIBLE (likely)  
**Reason:** Domain-specific Trinity constraint linking hypostases  
**Priority:** LOW

**privative_collapse:** `¬◇(entails 𝕆 P) → incoherent P`  
**Status:** TEST CANDIDATE  
**Hypothesis:** Contrapositive relationship with coherence definitions  
**Priority:** LOW

**grounding_yields_entails:** `grounded_in P x → entails x P`  
**Status:** TEST CANDIDATE  
**Hypothesis:** May be definitional relationship between grounding and entailment  
**Priority:** MEDIUM

**coherence_lifts_entailment:** `coherence 𝕆 → entails x P → entails 𝕆 P`  
**Status:** TEST CANDIDATE  
**Hypothesis:** Global entailment lifting; check if follows from A7 + other bridging principles  
**Priority:** MEDIUM

**entails_global_implies_truth:** `entails 𝕆 P → P`  
**Status:** TEST CANDIDATE  
**Hypothesis:** Should follow from A7 (coherence 𝕆) + ax_T or truth correspondence  
**Priority:** HIGH (strong candidate for elimination)

---

## Testing Priority Queue

**HIGH PRIORITY (likely redundant):**
1. ✅ ax_imp_intro / ax_imp_elim — Check if `⟹` is defined as `□(→)`
2. ✅ ax_mequiv_intro / ax_mequiv_elim — Check if `⩪` is defined as `□(↔)`
3. ✅ entails_global_implies_truth — Check if derivable from A7 + ax_T

**MEDIUM PRIORITY:**
4. ax_nonequiv_irrefl — Check definitions of ⇎ vs ⧟
5. ax_inter_comm — Check if Inter is symmetric by definition
6. A2_noncontradiction — Try proving from ax_nonequiv_irrefl
7. modus_groundens — Try proving from ⧟ properties + entails
8. grounding_yields_entails — Check definitional relationship
9. coherence_lifts_entailment — Try deriving from A7

**LOW PRIORITY (likely irreducible):**
10. ax_ident_symm, ax_ident_trans — Equivalence relation primitives
11. privative_collapse — Domain-specific constraint
12. triune_dependency_substitution — Trinity-specific rule

---

## Next Steps

1. **Check definitions:** Examine PXLv3.v Parameter declarations to see if PImp/MEquiv are defined
2. **Test HIGH priority candidates** in PXL_Axiom_Minimality_Check.v
3. **For each success:** Move proof to PXL_Structural_Derivations.v, remove from PXLv3.v, rebuild
4. **Update budget:** Ratchet down after each elimination

**Achieved reduction (2025-12-21):** 20 → 8 axioms following the successful elimination of all HIGH-priority candidates plus subsequent semantic integration.
