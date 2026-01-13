# Structure and Gated Surface

## Layer sketch (text)
cli → `scripts/golden_run.sh` → `coq_makefile`/`make` → `test_lem_discharge.py` (gate) → fingerprints in `state/`.

## Core modules (purpose)
- PXL_Kernel_Axioms: base objects, modal axioms.
- PXL_Triune_Principles: triune axiom set and coherence witness.
- PXL_Omni_Properties: omni property predicates (Truth/Wills/Good/etc.).
- PXL_Omni_Bridges: modal bridges OP1–OP3 and power axiom.
- PXL_OntoGrid: axes, axis targets/principles, domain profiles, coherence/no-drift.
- PXL_OntoGrid_OmniHooks: axis-to-omni hooks and grounding-to-hook lemma.
- PXL_OmniKernel / PXL_OmniProofs: omni aggregation and coherence proofs.

## Gated constants (required by `test_lem_discharge.py`)
- PXL_Kernel_Axioms: `PrincipleOf`, `NB_modal_power`.
- PXL_Triune_Principles: `SignMind_Bridge`, `Triune_Principles_Coherent`.
- PXL_Omni_Bridges: `OP1_truth_to_K_box`, `OP3_present_all_worlds`.
- PXL_OmniProofs: `Omni_set_is_coherent`.
- PXL_OntoGrid: `DomainAxisGround`, `DomainProfile_sound`, `AxisTarget`, `AxisTarget_has_Principle`, `OntoGrid_Coherent`, `OntoGrid_NoDrift`, `OntoGrid_NoDrift_holds`.
- PXL_OntoGrid_OmniHooks: `AxisOmniHook`, `DomainAxisGround_implies_AxisOmniHook`, `Truth_axis_hook`, `Goodness_axis_hook`, `Presence_axis_hook`, `Power_axis_hook`.
