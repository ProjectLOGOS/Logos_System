# Import Surface & Dependency Notes

## START_LOGOS.py
- **Stdlib**: `argparse`, `asyncio`, `os`, `signal`, `sys`, `threading`, `time`, `pathlib.Path` ([START_LOGOS.py](START_LOGOS.py#L9-L21)).
- **Internal dynamic imports**:
  - `logos_dashboard.metrics` plus `logos_dashboard.app.set_gate_status/start_dashboard/mark_agent/mark_protocol` for telemetry ([START_LOGOS.py](START_LOGOS.py#L24-L109)).
  - PXL gate utilities `PXL_Gate.ui.run_coq_pipeline`, `PXL_Gate.ui.lem_portal`, `PXL_Gate.ui.audit_and_emit`, and Flask app `PXL_Gate.ui.serve_pxl` inside the proof/bootstrap helpers ([START_LOGOS.py](START_LOGOS.py#L54-L83)).
  - UIP manager from `System_Operations_Protocol.startup.uip_startup` when spawning the background asyncio loop ([START_LOGOS.py](START_LOGOS.py#L112-L134)).
- **Outside runtime roots**: none. All custom imports are satisfied by paths injected via `_extend_sys_path()` (repo/PXL_Gate/System_Stack/Logos_AGI).

## System_Stack/System_Operations_Protocol/deployment/configuration/entry.py
- **Stdlib**: `atexit`, `json`, `logging`, `sys`, `datetime`, `pathlib.Path`, typing helpers ([System_Stack/.../entry.py](System_Stack/System_Operations_Protocol/deployment/configuration/entry.py#L1-L35)).
- **Internal**:
  - Governance core: `governance.core.logosc_core.autonomous_learning`, `governance.core.logosc_core.reference_monitor`.
  - Runtime adapters: `.iel_integration`, `.worker_integration`, `.unified_classes` for Bayesian + semantic stacks.
  - NLP fallbacks from `User_Interaction_Protocol` and `Advanced_Reasoning_Protocol` packages.
  - Safety stack `alignment_protocols.safety.integrity_framework.integrity_safeguard`.
- **Outside runtime roots**: none—every import resolves within `System_Stack`.
- **Notes**: All non-stdlib imports are wrapped in `try/except` blocks, providing stub/fallback behaviour when a subcomponent is absent; module-level `_module_init()` also ensures logging/audit directories exist ([System_Stack/.../entry.py](System_Stack/System_Operations_Protocol/deployment/configuration/entry.py#L620-L639)).

## System_Stack/System_Operations_Protocol/deployment/configuration/LOGOS.py
- **Stdlib**: `os`, `subprocess`, `sys`, `time`, `typing` ([System_Stack/.../LOGOS.py](System_Stack/System_Operations_Protocol/deployment/configuration/LOGOS.py#L1-L20)).
- **Internal dynamic import**: `from entry import get_logos_core` within `initialize_core_system()` to avoid circular import costs until launch time ([System_Stack/.../LOGOS.py](System_Stack/System_Operations_Protocol/deployment/configuration/LOGOS.py#L28-L38)).
- **Outside runtime roots**: none.

## PXL_Gate/ui/run_coq_pipeline.py
- **Stdlib**: `subprocess`, `pathlib.Path` ([PXL_Gate/ui/run_coq_pipeline.py](PXL_Gate/ui/run_coq_pipeline.py#L14-L28)).
- **Internal usage**: resolves repository paths into `System_Stack/Logos_AGI/Logos_Agent/Logos_Agent_Core/Protopraxis` to run Coq toolchain commands but does not import further modules.
- **Outside runtime roots**: none (only `subprocess` + filesystem access).

## PXL_Gate/ui/serve_pxl.py
- **Stdlib**: `glob`, `hashlib`, `queue`, `signal`, `subprocess`, `sys`, `threading`, `time`, `uuid` ([PXL_Gate/ui/serve_pxl.py](PXL_Gate/ui/serve_pxl.py#L1-L25)).
- **Third-party**: `flask` (`Flask`, `jsonify`, `request`) supplies the HTTP interface ([PXL_Gate/ui/serve_pxl.py](PXL_Gate/ui/serve_pxl.py#L17-L20)). This is the only import that leaves the three runtime roots and requires an installed dependency.
- **Internal resources**: interacts with compiled `.vo` artifacts via `glob` and shells out to `sertop` but does not import other repo modules.
