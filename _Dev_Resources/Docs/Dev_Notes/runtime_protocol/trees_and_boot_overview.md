# LOGOS directory focus (quarantine omitted)

Generated on 2026-01-13.

## LOGOS_SYSTEM/System_Entry_Point (directories)
```
System_Entry_Point
System_Entry_Point/Recusion_Grounding
System_Entry_Point/scripts
System_Entry_Point/System_Proof_Compiler
```

## LOGOS_SYSTEM/System_Stack/Logos_Protocol/Protocol_Core (directories)
```
System_Stack/Logos_Protocol/Protocol_Core
System_Stack/Logos_Protocol/Protocol_Core/Logos_Epistemology
System_Stack/Logos_Protocol/Protocol_Core/Logos_Epistemology/system_operations
System_Stack/Logos_Protocol/Protocol_Core/Logos_Epistemology/grounding_sources
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sources
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sources/pxl_state
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence/unlocked_system
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence/Protopraxis
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence/Protopraxis/formal_verification
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence/Protopraxis/formal_verification/coq
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence/Protopraxis/formal_verification/coq/baseline
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence/Protopraxis/formal_verification/coq/latest
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence/Protopraxis/formal_verification/coq/archive
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence/Protopraxis/formal_verification/coq/archive/tests
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence/Protopraxis/formal_verification/pxl-minimal-kernel-main
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence/Protopraxis/formal_verification/pxl-minimal-kernel-main/.github
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence/pxl-minimal-kernel-main/.github/workflows
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence/pxl-minimal-kernel-main/coq
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Unverified_State/verification_sequence/pxl-minimal-kernel-main/coq/phase6_expressiveness
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Verifed_State
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Verifed_State/epistemologial_state
System_Stack/Logos_Protocol/Protocol_Core/Logos_Ontology/Verifed_State/epistemologial_state/config
System_Stack/Logos_Protocol/Protocol_Core/Activation_Sequencer
System_Stack/Logos_Protocol/Protocol_Core/Activation_Sequencer/tests
System_Stack/Logos_Protocol/Protocol_Core/Activation_Sequencer/Identity_Generator
System_Stack/Logos_Protocol/Protocol_Core/Activation_Sequencer/Identity_Generator/Agent_ID_Spin_Up
System_Stack/Logos_Protocol/Protocol_Core/Activation_Sequencer/Agent_System_Initializer
```

## PXL_Gate (directories; quarantine pruned)
```
PXL_Gate
PXL_Gate/state
PXL_Gate/coq
PXL_Gate/coq/_build
PXL_Gate/coq/_build/_new
PXL_Gate/coq/src
PXL_Gate/coq/src/tests
PXL_Gate/coq/src/modal
PXL_Gate/coq/src/modal/PXLd
PXL_Gate/coq/src/baseline
PXL_Gate/coq/src/ui
PXL_Gate/coq/src/option_b
PXL_Gate/Protopraxis
```

## System_Audit_Logs (directories)
```
System_Audit_Logs
System_Audit_Logs/Boot_Sequence_Log
```

## Boot and runtime script surfaces
- [LOGOS_SYSTEM/__main__.py](LOGOS_SYSTEM/__main__.py): Module entrypoint for launching the packaged LOGOS system.
- [START_LOGOS.py](START_LOGOS.py): Top-level starter that orchestrates LOGOS runtime dispatch.
- [LOGOS_SYSTEM/System_Entry_Point/System_Proof_Compiler/test_lem_discharge.py](LOGOS_SYSTEM/System_Entry_Point/System_Proof_Compiler/test_lem_discharge.py): Canonical Coq rebuild/report; streams `_run_stream`, checks `pxl_excluded_middle`, emits status markers (`Overall status: PASS`).
- [\_reports/HOLDING_PATTERN/scripts/system_stack_tbd/could_be_dev/system_mode_initializer.py](\_reports/HOLDING_PATTERN/scripts/system_stack_tbd/could_be_dev/system_mode_initializer.py): Persists mission profiles to `state/mission_profile.json` (default `DEMO_STABLE`).
- [\_reports/HOLDING_PATTERN/scripts/system_stack_tbd/could_be_dev/start_agent.py](\_reports/HOLDING_PATTERN/scripts/system_stack_tbd/could_be_dev/start_agent.py): Loads mission profile, applies guardrails, runs sandboxed agent with restricted tool surface and write caps.
- [\_reports/HOLDING_PATTERN/scripts/system_stack_tbd/need_to_distribute/boot_aligned_agent.py](\_reports/HOLDING_PATTERN/scripts/system_stack_tbd/need_to_distribute/boot_aligned_agent.py): Alignment gate; guards SHA-256 identity, runs LEM discharge, writes alignment audit entry.
- [\_reports/HOLDING_PATTERN/scripts/system_stack_tbd/need_to_distribute/protocol_probe.py](\_reports/HOLDING_PATTERN/scripts/system_stack_tbd/need_to_distribute/protocol_probe.py): Read-only probe of ARP/SOP/UIP/SCP packages; logs discovery/runtime data.
- [\_reports/HOLDING_PATTERN/scripts/system_stack_tbd/need_to_distribute/aligned_agent_import.py](\_reports/HOLDING_PATTERN/scripts/system_stack_tbd/need_to_distribute/aligned_agent_import.py): Clones/refreshes external Logos_AGI, records commit SHAs, optionally runs protocol probe.
