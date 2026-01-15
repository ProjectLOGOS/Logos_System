# Audit & Normalize — Automation Contract (Phases 1–5)

This document is the authoritative contract for all automation
supporting the Audit & Normalize workflow.

Automation MUST enforce these rules. Violation = hard failure.

---

## Global Rules (All Phases)

- set -euo pipefail is mandatory.
- Fail-closed on missing scripts, paths, or env.
- No improvisation or inferred replacements.
- All outputs logged under:
  _Reports/Audit_Normalize/<phase>/<timestamped_run>/
- Capture environment at start:
  pwd, PATH, python -V, venv, opam switch (if applicable).
- Coq / proof artifacts are immutable unless explicitly authorized.

---

## Phase 1 — Audit Data Generation

Allowed:
- scan_* tooling
- read-only crawls
- large outputs (must be logged)

Forbidden:
- any filesystem mutation

Exit Gate:
- Required audit artifacts exist and are indexed.

---

## Phase 2 — Audit Input Indexing

Allowed:
- index generation
- cross-artifact resolution

Forbidden:
- guessing paths
- regenerating audit data

Exit Gate:
- Canonical Audit_Input_Index present and referenced.

---

## Phase 3 — Synthesis / Planning

Allowed:
- planning documents
- resolution plans
- disposition decisions

Forbidden:
- code rewrites
- file moves

Exit Gate:
- Explicit plan approved and recorded under _Reports/.

---

## Phase 4 — Crawl (Read-Only Enrichment)

Allowed:
- scoped read-only inspection
- classification enrichment

Forbidden:
- mutation of inspected files

Exit Gate:
- Crawl outputs logged and referenced by plan.

---

## Phase 5 — Controlled Execution

Allowed:
- batch-scoped rewrites or moves
- DRY-RUN first (mandatory)
- Execute only after human confirmation

Forbidden:
- execution without dry-run
- touching unresolved blockers
- proof/Coq mutation

Exit Gate:
- Verification complete
- Logs and manifests written
- Batch explicitly closed before next begins.

---

## Known Blockers (As of Beta)

- Canonical tool registry loader unresolved.
- Import rewrite tooling is beta and dry-run only.

---

## Enforcement Principle

Automation exists to enforce this contract.
If a step cannot be proven safe, it must not run.
