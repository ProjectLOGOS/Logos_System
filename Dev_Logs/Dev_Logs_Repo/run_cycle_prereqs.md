# Run Cycle Prerequisites

This note captures current blockers that prevent `tools/run_cycle.sh` from operating unattended for extended periods and the steps required to harden the sandbox loop.

## Runtime Dependencies

- **Python packages**
  - `numpy`: required by the fractal toolkit imported during ARP bootstrap. Missing module currently raises repeated warnings and disables analytics.
  - `requests`, `PyYAML`: optional APIs inside ARP emit warnings when absent but do not abort execution. Install to enable external validation hooks.
  - Core sandbox scripts assume a compatible virtualenv at `.venv/`; recreate with `python -m venv .venv && .venv/bin/pip install -r requirements.txt` before scheduling runs.
- **Plugins / toolkits**
  - `external/Logos_AGI` submodules export `LOGOS_V2`, `enhanced_uip_integration_plugin`, `uip_integration_plugin`. These are intentionally stubbed out in the demo environment; unattended loops should tolerate the warnings or ship noop replacements to silence noise.

## Control Flow Constraints

- `tools/run_cycle.sh` drives `scripts/start_agent.py` with a hard `timeout 90s`. Any cycle that needs more than 90 seconds will be terminated. Adjust the timeout or add backoff logic before hands-off execution.
- Consent prompts inside `start_agent.py` are handled by piping eight `'y'` responses. If the agent adds or reorders steps, the canned responses will desynchronize. A non-interactive consent hook is needed for resilient runs.
- Planner archival now runs at the end of each cycle via `scripts/archive_planner_digests.py`, so state growth is bounded. Ensure `state/` remains writable on the host executing the loop.

## Observed Failure Modes

- Missing `numpy` triggers repeated "fractal toolkit" import errors. These do not halt execution but spam logs.
- Reference monitor initialization can throw `'NoneType' object is not callable` when optional plugins are absent. Hardened deployments should either install the dependencies or gate those code paths.
- The agent currently writes at most three files (`--cap-writes 3`). Increase this cap if forthcoming tasks demand more artifacts.

## Immediate Hardening Tasks

1. Add a prerequisite check script that verifies required Python packages are importable before launching `run_cycle.sh`.
2. Provide a configuration switch for `start_agent.py` to auto-approve supervised steps without relying on positional `y` responses.
3. Document the exact command sequence (mode selection, agent goal, archival) so operators can replicate the loop on new hardware.
