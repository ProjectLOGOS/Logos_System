# Proof ↔ Runtime Bridge Map

| Bridge | Artifact(s) | Consumer | Direction |
| --- | --- | --- | --- |
| Coq proof pipeline | `PXL_Gate/ui/run_coq_pipeline` outputs `.vo/.glob` proof objects | `_run_proofs()` ensuring successful compilation before runtime | One-way (proof → runtime gate) |
| LEM portal | Identity/LEM discharge state via `PXL_Gate/ui/lem_portal.open_identity_portal()` | `_run_proofs()` raises if gate not satisfied | One-way (proof metadata → runtime) |
| Audit rewrite | Deterministic audit JSON via `PXL_Gate/ui/audit_and_emit` | `_run_proofs()` writes canonical audit entries consumed later by runtime audits | One-way |
| Runtime protocol log | `state/alignment/runtime_protocol.jsonl` | `runtime_protocol` module (Python) records proof-gate success; downstream tooling reads log | One-way |
| UIP Manager loading | Proof-backed configs (identity hashes, mission profiles) referenced via `System_Operations_Protocol.startup.uip_startup` | UIP runtime enforces proof-derived guardrails as it loads | One-way |

Currently there are no Python operations that push data back into Coq proof artifacts (runtime does not mutate `.v` or `.vo` files), so all bridges remain proof → runtime.
