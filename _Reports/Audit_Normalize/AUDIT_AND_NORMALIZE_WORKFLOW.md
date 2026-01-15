# Audit & Normalize — Workflow

This is the canonical, procedural guide for the Audit & Normalize meta task. It is evergreen and avoids progress state (tracked separately in `AUDIT_AND_NORMALIZE_LOG.md`).

## Purpose
- Define a fail-closed, traceable, reversible, proof-safe workflow for repo-wide audit/normalize work.
- Ensure session resilience: new sessions can resume by reading this guide, the input index, and the log.

## Core Principles
- Audit before action: no moves/renames/merges/deletes without audit artifacts.
- Read-only first: initial crawls/classifications are non-destructive.
- Governance before automation: establish naming/protocol/safety constraints before refactors.
- Proof immutability: `.v` files are never modified; documentation is sidecar-only.
- Single source of truth: all audit inputs are indexed via the canonical index.
- Batch execution: small, reversible batches with verification between batches.

## Directory & Artifact Conventions
- All audit/normalize artifacts live under `_Reports/Audit_Normalize/Runtime_Contents/` (authoritative for this task).

## Canonical Phases
1) Governance & Constraints (Phase 0)
   - Record naming conventions (Title_Case_With_Underscores), protocol mappings (SOP, SCP, ARP, MTP, Logos_Protocol, PXL_Gate), fail-closed posture, Coq immutability + sidecar rule, header/sidecar documentation rules.
   - No code changes in this phase.

2) Audit Data Generation (Phase 1)
   - Generate/import: naming violations (files/dirs/modules), duplicate files/dirs, semantic duplicates, orphaned/unreachable files, runtime surface (entrypoints/servers/CLI tools), side effects (filesystem/network).
   - Outputs are read-only snapshots.

3) Audit Input Indexing (Phase 2)
   - Create canonical index resolving all audit artifacts to stable paths: `Audit_Input_Index.json`.
   - Downstream steps must not guess paths.

4) Synthesis (Planning) (Phase 3)
   - From indexed data, generate planning artifacts: `Refactor_Plan (Pass N)`, `Rename_Queue (Pass N)`, `Move_Queue (Pass N)`, `Merge_Queue (Pass N)`.
   - Plans describe intent only; no filesystem mutations.

5) Crawl (Read-Only Inspection) (Phase 4)
   - Scoped, read-only inspections to enrich planning data (e.g., legacy/dev scripts, orphan-classified files, protocol-specific families like ETGC/SCP tooling).
   - Outputs: classifications, risk indicators (side effects), recommended disposition buckets.
   - No moves/renames/deletes.

6) Controlled Execution (Phase 5)
   - Execute changes in small, explicit batches (rename/move/merge), then verify imports, runtime entrypoints, and proof gates (if relevant).
   - Record results in the audit log; proceed only after verification.

7) Verification & Exit (Phase 6)
   - Confirm: no unresolved imports (beyond true unresolved), entrypoints unchanged unless planned, Coq artifacts untouched, logs complete.
   - Remove this workflow addendum only after meta-task completion and approval.

## Documentation Rules (Summary)
- `.py`: standardized inline header (deferred until Script_Crawl phase).
- `.json/.yaml/.yml/.toml/.ini`: sidecar `*.header.md`.
- `.v`: sidecar only; never modify file contents.
- Legacy protocol names in headers are documentation debt, not runtime signals.

## Session Continuity
- Any new session must read: this workflow, `Audit_Input_Index.json`, and `AUDIT_AND_NORMALIZE_LOG.md`, then resume at the correct phase without re-auditing completed work.
