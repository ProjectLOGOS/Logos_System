# Runtime state policy (Option A)

## What this is
`state/` contains runtime artifacts (identity, attestations, agent state, logs) that may change during normal runs.

## Rule
Do not commit `state/` changes unless explicitly approved as part of a governance action.

## Default workflow
- Treat `state/` as runtime churn.
- Keep commits focused on code, manifests, and audited docs.

## Guardrail
This repo may mark tracked `state/` files as `skip-worktree` to prevent accidental commits.
Reapply the guardrail using:

  tools/apply_state_skip_worktree.sh
