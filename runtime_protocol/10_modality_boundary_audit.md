# Modality Boundary Audit

## PASSIVE operations
- Proof gate pipeline (`_run_proofs` → `PXL_Gate.ui.run_coq_pipeline`, `lem_portal`, `audit_and_emit`).
- Runtime root checks and path setup (`runtime_protocol.ensure_runtime_roots`, `_extend_sys_path`).
- Dashboard boot telemetry prior to UI services (`_start_dashboard_from_env`).
- Runtime logging (`runtime_protocol.mark_phase`, `mark_modality` before PHASE_4).

## ACTIVE operations
- Flask UI service (`_start_flask` / `PXL_Gate.ui.serve_pxl`).
- UIP async manager (`_start_uip_background` → `System_Operations_Protocol.startup.uip_startup.UIPManager`).
- Signal loop + long-running threads once services are online.

## Boundary crossings
| Operation | Location | Guard / Enforcement |
| --- | --- | --- |
| Transition to ACTIVE modality | `START_LOGOS.main` before PHASE_4 | `runtime_protocol.assert_can_enter_active()` ensures proof + identity flags set |
| Flask service launch | `_start_flask` invoked only after assertion passes | inherits guard above |
| UIP Manager start | `_start_uip_background` called during PHASE_5 after modality already ACTIVE | relies on prior guard (no additional check) |

No other passive code flips modality; dashboard start remains passive because it does not expose user control surfaces.
