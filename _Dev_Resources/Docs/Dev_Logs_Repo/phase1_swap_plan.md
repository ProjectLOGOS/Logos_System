# Phase 1 Axiom Swap Plan

✅ **HISTORICAL DOCUMENT — Phase 3 Complete**  
**Original Goal:** 20 axioms → 15 axioms  
**Actual Achievement:** 20 axioms → **8 axioms** (Phase 2/3, 2025-12-21)

**Goal:** Reduce PXLv3.v from 20 axioms → 15 axioms (then eventually → 12)  
**Method:** Replace axioms with proven lemmas in the structural derivation modules (PXL_Structural_Derivations.v, PXL_Derivations_Phase2.v)
**Status:** ✅ EXCEEDED — Achieved 8 axioms (60% reduction)  
**Status:** READY (4 axioms already proven, 1 in progress)

---

## Already Completed (Previous Work)

### Swap 1: A1_identity → A1_from_refl

```
## Phase 1E Result: ax_nonequiv_irrefl → nonequiv_irrefl_derived

- ✅ **Status:** Eliminated.
- **Proof location:** [`nonequiv_irrefl_derived`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L104-L121) derives the lemma constructively, exporting the compatibility alias [`ax_nonequiv_irrefl`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L164-L169).
- **Inputs used:** semantic T-frame law (`ax_T`) plus the definitional identity lemmas, sourced from [`PXL_Kernel_Axioms.v`](Protopraxis/formal_verification/coq/baseline/PXL_Kernel_Axioms.v#L24-L49).
- **Removal:** The axiom declaration is no longer present in [`PXLv3.v`](Protopraxis/formal_verification/coq/baseline/PXLv3.v), which now re-exports the lemma-based alias.

This closes the last open item on the Phase 1 structural roadmap; no `Admitted.` stubs remain in the elimination files.

---

## Phase 1F-G Result: Identity & Interaction Axioms

- ✅ [`ident_refl_derived`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L29-L38), [`ident_symm_derived`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L51-L60), and [`ident_trans_derived`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L69-L77) collectively replace the three Ident axioms via Leibniz equality.
- ✅ [`inter_comm_derived`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L92-L102) proves the interaction axiom using the shared-witness definition of `Inter`.

All four compatibility aliases live alongside the lemmas, so existing code can continue importing the historical `ax_*` names.

---

## Bridge Axioms (Phase 2 Progress)

The bridge connectives are already definitionally reduced:

- [`bridge_imp_intro`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L134-L137) / [`bridge_imp_elim`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L139-L142)
- [`bridge_mequiv_intro`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L144-L147) / [`bridge_mequiv_elim`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L149-L152)

Each lemma unfolds the Phase 3 definitions of `PImp` and `MEquiv`, fully eliminating the corresponding axioms.

---

## Metaphysical Core (Phase 3+)

The eight metaphysical postulates are central to the LOGOS architecture and remain as axioms in [`PXL_Kernel_Axioms.v`](Protopraxis/formal_verification/coq/baseline/PXL_Kernel_Axioms.v#L37-L49). No changes planned here.

---

## Summary: Phase 1 Reduction Status

| Axiom | Status | Replacement |
|-------|--------|-------------|
| A1_identity | ✅ Eliminated | [`A1_from_refl`](Protopraxis/formal_verification/coq/baseline/PXL_Structural_Derivations.v#L34-L43) |
| ax_4 | ✅ Eliminated | [`ax_4_derived`](Protopraxis/formal_verification/coq/baseline/PXL_Structural_Derivations.v#L49-L56) |
| ax_5 | ✅ Eliminated | [`ax_5_derived`](Protopraxis/formal_verification/coq/baseline/PXL_Structural_Derivations.v#L59-L66) |
| A4_distinct_instantiation | ✅ Eliminated | [`A4_derived`](Protopraxis/formal_verification/coq/baseline/PXL_Structural_Derivations.v#L72-L83) |
| ax_nonequiv_irrefl | ✅ Eliminated | [`nonequiv_irrefl_derived`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L104-L121) |
| ax_ident_refl | ✅ Eliminated | [`ident_refl_derived`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L29-L38) |
| ax_ident_symm | ✅ Eliminated | [`ident_symm_derived`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L51-L60) |
| ax_ident_trans | ✅ Eliminated | [`ident_trans_derived`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L69-L77) |
| ax_inter_comm | ✅ Eliminated | [`inter_comm_derived`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L92-L102) |
| 4 bridge axioms | ✅ Eliminated | [`bridge_imp_*`, `bridge_mequiv_*`](Protopraxis/formal_verification/coq/baseline/PXL_Derivations_Phase2.v#L134-L152) |
| 8 metaphysical | 🔒 Keep | [`PXL_Kernel_Axioms.v`](Protopraxis/formal_verification/coq/baseline/PXL_Kernel_Axioms.v#L37-L49) |

**Current structural axiom count:** 11 (all realized as lemmas)  
**Residual axioms (2025-12-22):** 8 metaphysical postulates only
**Current axiom:** Line 46 in PXLv3.v  
**Status:** KEEP AS AXIOM (foundational)

**Reason:** Same as ax_ident_refl - needs explicit Ident definition.

---

### DEFERRED: ax_ident_trans

**Current axiom:** Line 47 in PXLv3.v  
**Status:** KEEP AS AXIOM (foundational)

**Reason:** Same as ax_ident_refl - needs explicit Ident definition.

---

### DEFERRED: ax_inter_comm

**Current axiom:** Line 50 in PXLv3.v  
**Status:** KEEP AS AXIOM (foundational)

**Reason:** `Inter` is a Parameter without definition. Commutativity cannot be
derived without knowing what interaction means formally.

**Future path:** Define `Inter` explicitly with symmetric structure, then derive
commutativity from that definition.

---

## Bridge Axioms (Future Phase 2)

These MAY be derivable if connectives have explicit definitions:

- **ax_imp_intro** (line 52)
- **ax_imp_elim** (line 53)
- **ax_mequiv_intro** (line 54)
- **ax_mequiv_elim** (line 55)

**Condition:** Requires explicit definitions of `PImp` and `MEquiv` in terms of
Coq's `->` and `<->`, or semantic interpretations.

**Strategy:** Investigate whether these connectives can be defined rather than
being primitive Parameters. If yes, the intro/elim axioms become lemmas.

---

## Metaphysical Core (Phase 3+)

These are likely **irreducible genuine postulates** of PXL:

- A2_noncontradiction
- A7_triune_necessity
- modus_groundens
- triune_dependency_substitution
- privative_collapse
- grounding_yields_entails
- coherence_lifts_entailment
- entails_global_implies_truth

**Plan:** Keep as axioms. These encode the essential metaphysical structure of PXL.

---

## Summary: Phase 1 Reduction Path

| Axiom | Status | Action |
|-------|--------|--------|
| A1_identity | ✅ ELIMINATED | Already removed, replaced by A1_from_refl |
| ax_4 | ✅ ELIMINATED | Already removed, replaced by ax_4_derived |
| ax_5 | ✅ ELIMINATED | Already removed, replaced by ax_5_derived |
| A4_distinct_instantiation | ✅ ELIMINATED | Already removed, replaced by A4_derived |
| ax_nonequiv_irrefl | 🔨 IN PROGRESS | Complete proof, then swap out |
| ax_ident_refl | ⏸️ DEFERRED | Foundational - needs Ident definition |
| ax_ident_symm | ⏸️ DEFERRED | Foundational - needs Ident definition |
| ax_ident_trans | ⏸️ DEFERRED | Foundational - needs Ident definition |
| ax_inter_comm | ⏸️ DEFERRED | Foundational - needs Inter definition |
| 4 bridge axioms | 🔮 FUTURE | Phase 2 - investigate PImp/MEquiv definitions |
| 8 metaphysical | 🔒 KEEP | Core PXL postulates |

**Historical snapshot (pre-Phase 2):** 20 axioms  
**After ax_nonequiv_irrefl swap (historical):** 19 axioms  
**Final baseline (2025-12-22):** 8 axioms with all bridge/identity swaps complete

---

## CI Pipeline After Each Swap

```bash
#!/bin/bash
# Run this after each axiom swap

set -e

echo "=== Rebuilding Coq kernel ==="
make -f CoqMakefile clean
make -f CoqMakefile -j$(nproc)

echo "=== Running verification suite ==="
python3 test_lem_discharge.py

echo "=== Updating axiom footprint ==="
python3 tools/axiom_inventory.py

echo "=== Checking axiom budget gate ==="
python3 tools/axiom_gate.py

echo "=== Verifying zero-assumption theorems ==="
python3 scripts/boot_aligned_agent.py

echo "=== All checks passed! ==="
```

**Expected outputs:**
- ✅ Coq compilation succeeds
- ✅ test_lem_discharge.py: "Overall status: PASS"
- ✅ axiom_inventory.py: Updated counts
- ✅ axiom_gate.py: All PASS with updated budget
- ✅ scripts/boot_aligned_agent.py: "Current status: ALIGNED"
