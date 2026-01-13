# MVP Acceptance Contract (Phase 12.1)

This contract freezes the deterministic finish line for the single Logos Agent over SCP/ARP/SOP/UIP.
Running `python scripts/run_mvp_acceptance.py` must answer **exactly once**:

- `MVP_ACCEPTANCE: PASS` — all required checks completed successfully.
- `MVP_ACCEPTANCE: FAIL (step N: <command>)` — the first failing check.

## Required Checks (run in order, stop on first failure)

1) `python test_lem_discharge.py` — Coq kernel rebuild + constructive LEM verification.
2) `python scripts/test_alignment_gate_smoke.py` — alignment gate blocks/permits actions based on proofs.
3) `python scripts/test_tool_pipeline_smoke.py` — governed tool proposal pipeline integrity.
4) `python scripts/test_uwm_smoke.py` — unified working memory invariants.
5) `python scripts/test_plan_checkpoint_smoke.py` — plan checkpoint lifecycle.
6) `python scripts/test_belief_consolidation_smoke.py` — belief consolidation and hashing.
7) `python scripts/test_plan_revision_on_contradiction.py` — plan revision on contradictions.
8) `python scripts/test_proved_grounding.py` — grounding proofs present and provable.
9) `python scripts/test_belief_tool_policy.py` — belief-informed tool policy behavior.
10) `python scripts/test_run_ledger_smoke.py` — run ledger completeness and policy capture.
11) `python scripts/test_goal_proposal_smoke.py` — deterministic goal proposal safety.
12) `python scripts/test_tool_improvement_intent_smoke.py` — tool improvement intent safeguards.

## PASS Criteria
- Every check above returns exit code 0.
- On failure, the process halts immediately after the first failing step.
- Output is deterministic: a single summary line as described above.

## Audit Trail (optional)
- When enabled, an audit record is written under `audit/mvp_acceptance/` capturing per-step status and a hash of the record.
