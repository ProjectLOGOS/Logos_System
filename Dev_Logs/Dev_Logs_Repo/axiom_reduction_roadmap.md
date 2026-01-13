# Axiom Reduction Roadmap — Operational Plan

## Phase 0 — Measure + Gate (CURRENT)
**Artifacts**
- `state/axiom_footprint.json` — authoritative Print Assumptions inventory
- `tools/axiom_inventory.py` — generator
- `tools/axiom_gate.py` — hard gate (assumption-free invariants)

**Hard invariants**
- `pxl_excluded_middle` must remain assumption-free.
- `trinitarian_optimization` must remain assumption-free.

**Next**
- Add privative collapse theorem name to the invariant list.
- Add axiom-count extraction and enforce a cap (initially 49).

## Phase 1 — Convert Structural Axioms (S4) → Lemmas
- Identify "structural axioms" in `PXLv3.v`.
- Prove them in `PXL_Structural_Derivations.v`.
- Delete corresponding `Axiom` declarations.
- Re-run `tools/axiom_inventory.py` and ensure invariants remain intact.

## Phase 2 — Modal Basis Minimization
- Derive redundant S5 properties as lemmas.
- Shrink modal axioms to a minimal basis consistent with your encoding.

## Phase 3 — PXL Core Minimization (A1–A7)
- Attempt to derive A3–A6 from {A1, A2, A7} + definitions.
- Isolate genuinely metaphysical postulates as explicit kernel assumptions.

## Phase 4 — Kernelization
- `PXL_Kernel.v`: minimal axioms only
- `PXL_Derived.v`: all else derived
- CI: axiom budget must never increase
