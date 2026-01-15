# External Audit Quickstart

## Prerequisites
- Rocq/Coq 9.1.0 in PATH (matches golden run fingerprint).
- Python 3.12.3 (CPython) and virtualenv at `.venv/` or set `PYTHON` env var.
- Git submodules available (no network once fetched).

## Single-command replay
- From repo root: `./scripts/golden_run.sh`
- This initializes submodules, rebuilds all Coq artifacts, runs `test_lem_discharge.py`, and writes a fingerprint.

## Expected PASS conditions
- `test_lem_discharge.py` prints `Overall status: PASS` and `<none>` for extra assumptions and residual `Admitted.`
- No unexpected admitted theorems or missing required modules/constants.
- Script exits 0 and leaves the tree clean apart from the fingerprint file it manages.

## Fingerprint
- Location: `state/golden_run_fingerprint.txt` (rewritten each run).
- Contains: `parent_sha`, `logos_agi_sha` (submodule), `vsrocq_sha` (submodule), `coq_version`, `python_version`.
- Purpose: captures toolchain and code identities observed during the golden run; deterministic runs should keep these stable apart from `parent_sha` advancing with new commits.
