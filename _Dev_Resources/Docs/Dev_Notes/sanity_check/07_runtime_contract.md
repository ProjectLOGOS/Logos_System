# Runtime Contract v0

## Allowed Runtime Roots
- `START_LOGOS.py`
- `System_Stack/`
- `PXL_Gate/`

Only these repo-internal roots may be imported or executed by runtime-critical entrypoints. Third-party libraries (e.g., `flask`, `asyncio`) are permitted, but any repo-internal import crossing outside the list constitutes a boundary violation (see "Import Boundary" below).

## Boot Phases
1. `PHASE_0_PATH_SETUP` – Resolver extends `sys.path`, records runtime roots, and confirms PASSIVE modality.
2. `PHASE_1_PROOF_GATE` – Proof pipeline prepared (Coq make/regeneration).
3. `PHASE_2_IDENTITY_AUDIT` – LEM portal verified and deterministic audit write completes.
4. `PHASE_3_TELEMETRY_DASHBOARD` – Dashboard/metrics services start broadcasting state.
5. `PHASE_4_UI_SERVICES` – User-facing endpoints (Flask PXL UI, console/UIP portals) become ACTIVE.
6. `PHASE_5_STACK_LOAD` – UIP/background managers load System_Stack subsystems.
7. `PHASE_6_SIGNAL_LOOP` – Process enters blocking signal wait with graceful shutdown hooks.

Modality definitions:
- `PASSIVE` – No user interaction endpoints are enabled; only internal preparation allowed.
- `ACTIVE` – Any user interaction endpoint (Flask server, dashboard UI, UIP chat bridge) is reachable.

**Boundary rule**: `ACTIVE` modality may only begin after both proof gate success and the identity/audit write complete (`PHASE_2_IDENTITY_AUDIT`).

## Required Artifacts per Phase (best-effort)
| Phase | Artifact(s) |
| --- | --- |
| `PHASE_0_PATH_SETUP` | Runtime roots verification log entry (JSONL) |
| `PHASE_1_PROOF_GATE` | Proof compilation outputs (`PXL_Gate/ui/run_coq_pipeline.py` logs, Coq make artifacts) |
| `PHASE_2_IDENTITY_AUDIT` | LEM portal state, deterministic audit entry (e.g., `_reports` + `state/alignment_LOGOS-AGENT-OMEGA.json`) |
| `PHASE_3_TELEMETRY_DASHBOARD` | Dashboard readiness markers (metrics boot phase, agent/protocol status) |
| `PHASE_4_UI_SERVICES` | Service readiness markers (Flask thread metadata, port bindings) |
| `PHASE_5_STACK_LOAD` | UIP manager status (async loop started, `mark_protocol("UIP", "ready")`) |
| `PHASE_6_SIGNAL_LOOP` | Blocking wait confirmation + signal handler registration |

## Append-Only Runtime Log
- Location: `state/alignment/runtime_protocol.jsonl`
- Format: one JSON object per line `{ "ts": ISO8601, "event": "phase|modality|runtime_roots_checked|...", "payload": {...} }`
- Scope: phase transitions, modality changes, runtime-root verification, and future guardrails.

## Import Boundary
- **Violation definition**: Any repo-internal import performed by runtime entrypoints that resolves outside `START_LOGOS.py`, `System_Stack/`, or `PXL_Gate/` (e.g., importing a module under `JUNK_DRAWER/`).
- **Allowed imports**: Python stdlib and third-party packages installed in the environment.
- **Planned enforcement**: upcoming revisions will add a static scanner that walks `sys.modules` after PHASE_5 and asserts that repo-relative origins stay within the allowed roots (logging boundary violations before entering ACTIVE mode).
