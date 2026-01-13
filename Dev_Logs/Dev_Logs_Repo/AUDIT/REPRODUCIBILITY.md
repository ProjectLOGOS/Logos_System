# Reproducibility Contract

## Fingerprint boundary
- Captures `parent_sha` (repo), `logos_agi_sha` (submodule), `vsrocq_sha` (submodule), `coq_version`, `python_version`.
- Emitted at `state/golden_run_fingerprint.txt` by `scripts/golden_run.sh`.

## Determinism claim
- Two consecutive runs of `./scripts/golden_run.sh` should be identical in PASS/FAIL status and toolchain identity (fingerprint fields), with only `parent_sha` advancing when commits change.
- The gate asserts: full Coq rebuild succeeds; required modules compile; required constants exist; no unexpected admits; extra assumptions lists are `<none>`.

## Out of scope
- Network access (submodules must already be fetched).
- OS differences beyond tested Ubuntu 24.04.2 LTS environment.
- Optional editor/IDE tooling; only CLI flow is covered.
