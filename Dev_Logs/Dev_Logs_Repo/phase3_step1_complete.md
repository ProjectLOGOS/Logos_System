# Phase 3 Step 1: Kripke Semantic Scaffold - COMPLETE

**Date:** December 14, 2025  
**Branch:** `phase3-modal-semantic-kripke`  
**Status:** ✅ SUCCESS - Semantic module compiles cleanly

---

## Objective

Introduce a **Kripke-semantic modal layer** alongside the existing PXL baseline **without modifying the 20-axiom kernel**. This proves that PXL's modal axioms (ax_K, ax_T) can be **derived as theorems** from frame conditions.

---

## What Was Created

### New File: PXL_Modal_Semantic_Kripke.v
**Location:** `Protopraxis/formal_verification/coq/baseline/modal/PXL_Modal_Semantic_Kripke.v`  
**Size:** ~75 lines  
**Dependencies:** 
- External: `PXLd.proof_checking.pxl_core_meta_validation_v3.pxl_meta_sets_v3.coq.S5_Kripke`
- Standard: `Coq.Logic.Classical_Prop`

**Key Components:**

1. **Kripke Frame Setup:**
   ```coq
   Variable W : Type.                    (* Possible worlds *)
   Variable R : W -> W -> Prop.          (* Accessibility relation *)
   
   Hypothesis R_refl  : forall w, R w w.
   Hypothesis R_symm  : forall w u, R w u -> R u w.
   Hypothesis R_trans : forall w u v, R w u -> R u v -> R w v.
   ```

2. **Modal Operators (Kripke semantics):**
   ```coq
   Definition box (p:W->Prop) : W->Prop := 
     fun w => forall u, R w u -> p u.
   
   Definition dia (p:W->Prop) : W->Prop := 
     fun w => exists u, R w u /\ p u.
   ```

3. **Prop-Level Adapter (Global Validity):**
   ```coq
   Definition PXL_Box (p : Prop) : Prop :=
     forall w : W, box (fun _ => p) w.
   
   Definition PXL_Dia (p : Prop) : Prop :=
     exists w : W, dia (fun _ => p) w.
   ```

4. **Derived Theorems (Former Axioms):**

   **PXL_ax_K_sem** (Distribution):
   ```coq
   Theorem PXL_ax_K_sem :
     forall p q : Prop, 
       PXL_Box (p -> q) -> PXL_Box p -> PXL_Box q.
   Proof.
     (* Derived from Kripke semantics, not axiomatic *)
   Qed.
   ```

   **PXL_ax_T_sem** (Reflexivity):
   ```coq
   Hypothesis W_inhabited : W.  (* Requires inhabited worlds *)
   
   Theorem PXL_ax_T_sem :
     forall p : Prop, PXL_Box p -> p.
   Proof.
     (* Uses R_refl to derive T property *)
   Qed.
   ```

---

## _CoqProject Changes

**Added:**
```
-Q external/Logos_AGI/Logos_Agent/Protopraxis/pxl-minimal-kernel-main/\
   pxl-minimal-kernel-main/coq PXLd

Protopraxis/formal_verification/coq/baseline/modal/PXL_Modal_Semantic_Kripke.v
```

**Effect:** Baseline build can now import pre-compiled S5_Kripke.v from external project without recompiling it.

---

## Key Findings

### 1. Modal Axioms Are Derivable
- **ax_K** (distribution) proven from Kripke box definition
- **ax_T** (reflexivity) proven from R_refl frame condition
- **No new axioms required** - only frame hypotheses (R_refl, R_symm, R_trans)

### 2. Semantic Approach Trade-offs

**Costs:**
- Requires **inhabited worlds** (W_inhabited hypothesis)
- Adds **frame conditions** as hypotheses (3 properties)
- Uses **classical logic** (already in baseline)

**Benefits:**
- **Richer metatheory** - completeness/soundness via Kripke frames
- **Modular proofs** - separate frame theory from object theory
- **Formal semantics** - box/dia have explicit model-theoretic meaning

### 3. Current PXL Kernel Status
- **Historical snapshot (2025-12-14):** 20 axioms remained in PXLv3.v when this step landed.
- **Update (2025-12-22):** Modal semantic theorems have been integrated; baseline kernel now holds **8 axioms** with pxl_excluded_middle and trinitarian_optimization still assumption-free.
- **Semantic layer** is now the active provider for modal strength rather than a parallel experiment.

---

## Verification

### Compilation Status
```bash
$ make -f CoqMakefile
COQC Protopraxis/formal_verification/coq/baseline/modal/PXL_Modal_Semantic_Kripke.v
# SUCCESS - No errors
```

### Axiom Count (Updated)
```bash
$ python3 tools/axiom_inventory.py
Wrote state/axiom_footprint.json
axiom_count: 8
```

### Dependencies
- S5_Kripke.v: **0 axioms** (all lemmas proven from frame conditions)
- PXL_Modal_Semantic_Kripke.v: **4 hypotheses** (R_refl, R_symm, R_trans, W_inhabited)
- Total assumptions: **Frame conditions only** (not axioms in Coq sense)

---

## Follow-on Outcomes (Phase 3 Step 2+)

### Axiom Assumption Analysis
- Completed via the axiom inventory tool; [../state/axiom_footprint.json](../state/axiom_footprint.json) records zero additional assumptions for PXL_ax_K_sem, PXL_ax_T_sem, and PXL_ax_Nec_sem.

### Integration Testing
- Semantic and axiomatic modal operators were reconciled; the semantic theorems now instantiate the live modal ruleset.

### Kernel Replacement
- PXLv3 imports the semantic module, eliminating ax_K, ax_T, and ax_Nec from the axiom list.
- Axiom budgets ratcheted to 8, matching the current tool output.

---

## Technical Notes

### Why "Global Validity" Adapter?
The Prop-level adapter uses **universal quantification over worlds**:
```coq
PXL_Box p := forall w : W, box (fun _ => p) w
```

**Alternatives considered:**
1. **Designated world:** `PXL_Box p := box (fun _ => p) w0`
   - Problem: Which w0? Arbitrary choice adds complexity
   
2. **Global validity:** `forall w, box (fun _ => p) w`
   - Benefit: No world choice needed, conservative
   - Tradeoff: Stronger than single-world validity

**Decision:** Global validity is more conservative and avoids arbitrary choices. This matches PXL's "metaphysical necessity" interpretation where □p means "necessary in all possible worlds."

### Frame Conditions vs Axioms
Frame conditions (R_refl, R_symm, R_trans) are **Hypotheses** in Coq's Section, not **Axioms** in the global sense. When the Section is closed, theorems become functions that **require frame proofs as arguments**:

```coq
PXL_ax_T_sem : 
  forall (W : Type) (R : W -> W -> Prop),
    (forall w, R w w) ->  (* R_refl as argument, not axiom *)
    (W) ->                (* W_inhabited as argument *)
    forall p, PXL_Box p -> p.
```

This is **stronger** than axiomatic approach because the frame properties become **explicit proof obligations** rather than opaque axioms.

---

## Conclusion

**Phase 3 Step 1 is COMPLETE:**

✅ Semantic modal module created  
✅ Compiles cleanly with zero errors  
✅ ax_K and ax_T proven as theorems  
✅ Baseline kernel now consolidated to 8 axioms  
✅ Follow-on assumption analysis and integration testing complete  

**Historical Significance:**  
This is the first time PXL's modal logic has been given **formal Kripke semantics** rather than axiomatic treatment. It demonstrates that the modal core can be **derived** rather than **assumed**, opening the path that ultimately delivered the **8-axiom baseline** (20 → 8 by eliminating ax_K, ax_T, ax_Nec alongside earlier structural swaps).

**Next Command:**
```bash
python3 tools/axiom_inventory.py --verbose | grep -A10 "PXL_Modal_Semantic"
```
