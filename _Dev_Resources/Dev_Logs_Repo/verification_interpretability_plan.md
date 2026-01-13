# Enhanced Verification & Interpretability Plan

## Objectives
- Extend automated verification beyond constructive LEM discharge by integrating health reporting and dataset provenance checks into continuous tests.
- Provide transparent operator-facing artefacts (summaries, diffs, metrics) that explain why the planner produced particular agendas or actions.
- Maintain existing guardrails (`require_safe_interfaces`, audit logs) while exposing richer observability hooks.

## Current Baseline
- `plugins/perception_ingestors.py` exposes `health_report()` and status details for symbolic, training, telemetry, and simulation feeds.
- `demos/perception_planner_demo.py` and `scripts/prototype_grounding_workflow.py` surface READY/WAIT health snapshots alongside observation digests rendered from the broker.
- Planner unit tests (`tests/test_agent_planner.py`) verify agenda derivation when alignment is marked as satisfied.

## Verification Enhancements (In Progress)
1. **Health-contract tests**
   - ✅ Added `tests/test_perception_ingestors.py` covering simulation ingestion summaries, broker health outputs, and runtime telemetry parsing (including anomaly detection).
2. **End-to-end regression harness**
   - ✅ Build a smoke test that runs `scripts/run_simulation_stub.py --log-dir <tmp>` and asserts key log output without polluting sandbox logs.
   - ✅ Documented `python3 tools/run_ci_checks.py` to execute the smoke suite (ingestors + CLI) and match CI expectations.
3. **Proof artefact audits**
   - ⬜ Capture the latest `_CoqProject` rebuild metadata inside `state/alignment_LOGOS-AGENT-OMEGA.json` and surface in demos.

## Interpretability Enhancements (Planned)
1. **Action rationale digest**
   - ⬜ Emit structured JSON alongside demo output summarising agenda priorities and linked observations.
   - ⬜ Provide an optional `--explain` flag in planners to write these digests under `state/` for operator review.
 2. **Observation traceability UI hooks**
   - ✅ Broker now emits `trace_digest()` metadata (last update, sample size, hashes) with demos/scripts printing operator-friendly summaries.
   - ⬜ Document how to stream these summaries into alignment audit logs without breaking append-only rules.
3. **Telemetry trend snapshots**
   - ✅ Extended `RuntimeTelemetryIngestor` to compute rolling averages, trend labels, and anomaly flags surfaced in planner rationales.

## Next Steps
- Prioritise verification tasks (health-contract tests, regression harness) before modifying planner outputs.
- Coordinate with audit pipeline owners to ensure new JSON digests comply with append-only logging requirements.
- Once verification scaffolding is stable, implement interpretability enhancements iteratively, starting with structured action digests.

## CI Smoke Test Workflow
- Execute `python3 tools/run_ci_checks.py` locally before opening pull requests to mirror the CI gating.
- The helper script runs `tests.test_perception_ingestors` and `tests.test_simulation_cli`, ensuring telemetry digests and simulation logs remain healthy.
- Update GitHub Actions workflows to invoke the same entry point so local runs match automation.
