## Axiom Footprint Summary (Omni Layer)

Kernel axioms (in force):
- existing kernel axioms
- Perfect_self_knowledge
- NB_modal_power

Omni semantic axioms (irreducible under current semantics):
- OP1_truth_grounded_I1
- OP2_wills_implies_good_box
- OP3_present_all_worlds (permanent unless ⇌ semantics changes)

Derived / aliased bridges (non-axiomatic):
- OP1_grounded_leads_K_I1 (derived via Perfect_self_knowledge)
- OP1_truth_to_K_box (derived)
- OP4_coherent_implies_possible (kernel-aliased to NB_modal_power)

Gate Enforcement:
- test_lem_discharge.py requires .vo artifacts for: PXL_Omni_Properties, PXL_Omni_Bridges, PXL_OmniKernel, PXL_OmniProofs.
- It asserts presence of: OP1_truth_to_K_box, OP3_present_all_worlds, NB_modal_power, Omni_set_is_coherent. Missing artifacts/constants fail the gate.

| Property | Source theorem | Source file | Dependencies | Identity (I1/I2/I3/SOP) | Target lemma in PXL_OmniProofs.v | Status | Proof file / lemma |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OP1-TRANSP | Perfect_self_knowledge | PXL_Kernel_Axioms.v | grounded_in, K | Any x | (supports OP1 bridges) | Axiom | PXL_Kernel_Axioms.v (Perfect_self_knowledge) |
| OP4-KERNEL | NB_modal_power | PXL_Kernel_Axioms.v | incoherent, entails, ◇ | NB / 𝕆 | (supports OP4 bridge) | Axiom | PXL_Kernel_Axioms.v (NB_modal_power) |
| OP1-DEF | Omniscient | PXL_Omni_Properties.v (Omniscient) | Truth, grounded_in, K | I1 / 𝕆 | prove_I1_omniscience | Derived (only Truth→grounding axiomatic) | PXL_Omni_Bridges.v (OP1_truth_grounded_I1 [axiom], OP1_grounded_leads_K_I1 [lemma], OP1_truth_to_K_box [lemma]) |
| OP2-DEF | Omnibenevolent | PXL_Omni_Properties.v (Omnibenevolent) | Wills, Good | I2 / 𝕆 | prove_I2_omnibenevolence | Derived via granular bridge | PXL_Omni_Bridges.v (OP2_wills_implies_good_box) |
| OP3-DEF | Omnipresent | PXL_Omni_Properties.v (Omnipresent) | ⇌ | I3 / 𝕆 | prove_I3_omnipresence | Axiom (bridge; permanent unless ⇌ semantics change) | PXL_Omni_Bridges.v (OP3_present_all_worlds) |
| OP4-DEF | Omnipotent | PXL_Omni_Properties.v (Omnipotent) | Coherent, entails, ◇ | NB / 𝕆 | prove_NB_omnipotence | Derived via kernel-aliased bridge | PXL_Omni_Bridges.v (OP4_coherent_implies_possible := NB_modal_power) |

Foundational source module: PXL_Imp_Nothing.v (Impossibility of Nothing / Necessary Being foundation).
