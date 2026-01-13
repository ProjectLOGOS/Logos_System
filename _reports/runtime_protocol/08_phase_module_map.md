| Phase | Function / Hook | Module | Role | Modality |
| --- | --- | --- | --- | --- |
| PHASE_0_PATH_SETUP | `ensure_runtime_roots()` | `System_Stack.Logos_Protocol.Protocol_Core.runtime_protocol` | audit/state (root verification log) | Passive |
| PHASE_0_PATH_SETUP | `_extend_sys_path()` | `START_LOGOS` adding `REPO_ROOT`, `PXL_Gate`, `System_Stack/Logos_AGI` to `sys.path` | internal logic | Passive |
| PHASE_1_PROOF_GATE | `_run_proofs()` | `PXL_Gate.ui.run_coq_pipeline` | proof-related (Coq build + checks) | Passive |
| PHASE_1_PROOF_GATE | `_run_proofs()` | `PXL_Gate.ui.lem_portal` (`open_identity_portal`) | proof-related (LEM portal) | Passive |
| PHASE_1_PROOF_GATE | `_run_proofs()` | `PXL_Gate.ui.audit_and_emit` (`main`) | audit/state (deterministic audit rewrite) | Passive |
| PHASE_2_IDENTITY_AUDIT | `runtime_protocol.mark_phase` | `System_Stack.Logos_Protocol.Protocol_Core.runtime_protocol` | audit/state (records proof+identity completion) | Passive |
| PHASE_3_TELEMETRY_DASHBOARD | `_start_dashboard_from_env()` | `logos_dashboard.app` (`start_dashboard`, `mark_agent`, `mark_protocol`) | telemetry/dashboard | Passive |
| PHASE_3_TELEMETRY_DASHBOARD | `_start_dashboard_from_env()` | `logos_dashboard.metrics` | telemetry/dashboard | Passive |
| PHASE_4_UI_SERVICES | `_start_flask()` | `PXL_Gate.ui.serve_pxl` (Flask app) | UI / user interaction | Active |
| PHASE_4_UI_SERVICES | `runtime_protocol.assert_can_enter_active()` | `System_Stack.Logos_Protocol.Protocol_Core.runtime_protocol` | audit/state guard | Passive → Active boundary |
| PHASE_4_UI_SERVICES | `runtime_protocol.mark_modality(MODALITY_ACTIVE)` | `System_Stack.Logos_Protocol.Protocol_Core.runtime_protocol` | audit/state | Active |
| PHASE_5_STACK_LOAD | `_start_uip_background()` | `System_Operations_Protocol.startup.uip_startup.UIPManager` | internal logic (agent stack) | Active |
| PHASE_5_STACK_LOAD | `_start_uip_background()` | `logos_dashboard.app` (`mark_protocol`) | telemetry/dashboard | Active |
| PHASE_5_STACK_LOAD | `_start_uip_background()` | `logos_dashboard.app` (UIP status updates) | telemetry/dashboard | Active |
| PHASE_6_SIGNAL_LOOP | `_wait_for_signal()` | `START_LOGOS` | internal logic (event loop) | Active |
| ALL PHASES | `runtime_protocol.mark_phase()` | `System_Stack.Logos_Protocol.Protocol_Core.runtime_protocol` | audit/state logging | Passive pre-Phase 4, Active afterwards |
| ALL PHASES | `runtime_protocol.mark_modality()` | `System_Stack.Logos_Protocol.Protocol_Core.runtime_protocol` | audit/state logging | Passive/Active per mode |
