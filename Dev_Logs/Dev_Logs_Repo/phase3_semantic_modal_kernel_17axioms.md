# Phase 3 Milestone — Semantic Modal Kernel (17 Axioms)

⚠️ **INTERMEDIATE MILESTONE** — Further reduced to **8 axioms** (2025-12-21)  
See [axiom_audit_phase1.md](axiom_audit_phase1.md) for the live roster.

**Date:** 2025-12-14  \
**Branch:** phase3-modal-semantic-kripke  

## Claim
PXL now supports a semantic modal kernel profile where modal behavior is derived rather than axiomatized:
- Box/Dia are definitions (Kripke adapter).
- ax_K, ax_T, ax_Nec are theorems.
- Historical snapshot (2025-12-14): semantic kernel packaged 17 axioms, while baseline remained at 20.
- Current status (2025-12-22): semantic theorems are fully integrated and the live baseline holds 8 axioms total.

## Verified Metrics
- Historical baseline axiom count (after Step 1): 20
- Historical semantic profile axiom count: 17
- Current baseline axiom count: 8 (see [../state/axiom_footprint.json](../state/axiom_footprint.json))
- Tracked theorems (semantic K/T/Nec, LEM, triune optimization) are checked by the axiom inventory tool and enforced by the axiom gate script.

## Entry Point
Semantic profile suite file:
- [../Protopraxis/formal_verification/coq/baseline/PXL_Semantic_Profile_Suite.v](../Protopraxis/formal_verification/coq/baseline/PXL_Semantic_Profile_Suite.v)

This file compiles the semantic kernel and semantic ports and checks:
- ax_K, ax_T, ax_Nec
- pxl_excluded_middle

## Reproducible Command
Run:
```bash
./tools/run_semantic_profile.sh
```
This compiles the semantic suite and runs the dual-budget and theorem-assumption gates.

## Extensions Policy
Any additional axioms beyond Kernel17 are quarantined as explicit extensions. Example:
- [../Protopraxis/formal_verification/coq/baseline/PXL_Semantic_Extensions_DomainProduct.v](../Protopraxis/formal_verification/coq/baseline/PXL_Semantic_Extensions_DomainProduct.v)

The semantic suite entrypoint does not import extensions by default.
