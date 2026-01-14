# Scripts Overview

This directory hosts the Python entry points that orchestrate LOGOS proof
verification, agent bootstrapping, and investor-facing demos. All scripts are
standard-library only and share the `_bootstrap.py` helper to resolve the
repository root.

## Alignment & Verification

- `boot_aligned_agent.py` – rebuilds the Coq kernel, verifies the constructive
  Law of Excluded Middle, and records audit entries before releasing sandbox
  tooling.
- `test_lem_discharge.py` (at repo root) feeds into these scripts; import its
  helpers instead of duplicating proof checks.
- `system_mode_initializer.py` – persists mission profiles under `state/` and
  seeds the sandbox write cap used by runtime agents.
- `aligned_agent_import.py` / `protocol_probe.py` – refresh external protocol
  packages and capture read-only probe results once alignment gates pass.

## Agent Runtime

- `start_agent.py` – wires guarded tools into the sandboxed agent loop.
- `logos_chat.py` and `logos_interface.py` – interactive shells that load the
  runtime backend via `_bootstrap.py` and the shared tool registry.
- `llm_interface_suite/reasoning_agent.py` – scripted reasoning harness that
  mirrors the interactive behavior for automated experiments.
- `sync_genesis_corpus.py` – copies repo-level training_data into the Logos
  Agent genesis capsule (dry-run and delete modes available).
- `genesis_capsule.py` – validates the genesis capsule manifest and referenced
  pointers used by `start_agent.py --bootstrap-genesis`.

## Demonstrations & Dashboards

- `investor_demo.py` – narrates the proof-before-autonomy story for live demos.
- `investor_dashboard.py` – renders the executive summary consumed by the
  integration workflow.
- `run_recursive_immersion_cycle.py` – experimental loop running reflection and
  gap-filling cycles inside the sandbox.

## Usage Notes

1. Always import `REPO_ROOT` from `_bootstrap.py` when resolving file paths.
2. Keep audit writes append-only; follow the JSON structures already present in
   `state/` directories.
3. When adding new scripts, document them here and ensure they pass through the
   same verification gates before attempting to mutate sandbox state.

## Operator Guide

See _reports/logos_agi_audit/60_PHASE_B_OPERATOR_GUIDE.md for the approved entrypoints and constrained run instructions.

## Operator acknowledgment guardrail (high-risk scripts)

Some operator-only scripts refuse to run unless you explicitly acknowledge risk:

- Set env: `LOGOS_OPERATOR_OK=1`
- Provide a script-specific flag:
  - `--allow-git` for git/network operations (imports/clones)
  - `--serve` for scripts that start local servers
  - `--allow-audit-write` for scripts that write proposal artifacts or append audit logs

This is intentional to reduce accidental destructive operations.

Test note:
- Some smoke tests may redirect audit outputs to a temp directory via `LOGOS_AUDIT_DIR` to avoid touching tracked `audit/`.

## Repo tools

- `repo_tools/lock_coq_stacks.sh` keeps Coq-related stacks read-only by default.
  - Guarded paths: `/workspaces/Logos_System/PXL_Gate` and `/workspaces/Logos_System/Logos_System/System_Entry_Point/Runtime_Compiler`.
  - Status: `./_Dev_Resources/Dev_Scripts/repo_tools/lock_coq_stacks.sh status`.
  - Lock (default): `./_Dev_Resources/Dev_Scripts/repo_tools/lock_coq_stacks.sh lock`.
  - Unlock (requires explicit override): `./_Dev_Resources/Dev_Scripts/repo_tools/lock_coq_stacks.sh unlock "edit coq stacks"`.
