# Continuous Run Orchestrator Design

Goal: enable unattended, repeatable execution of the agent cycle with guardrails, archival hooks, and operational visibility.

## Core Requirements

1. **Prerequisite validation**
   - Run `scripts/check_run_cycle_prereqs.py` before each launch.
   - Abort early if required modules (e.g., numpy) are missing.

2. **Loop controller**
   - Python driver script `scripts/run_cycle_loop.py` (proposed) that:
     - Accepts parameters: `--mode`, `--goal`, `--iterations` (or `--continuous` with stop signal), `--sleep` between runs.
     - Calls prerequisite checker; if it fails, log and skip iteration.
     - Executes `tools/run_cycle.sh` via `subprocess.run` with extended timeout (default 300s) and captured stdout/stderr.
     - Records per-iteration metadata (start/end time, exit status, planner archive generated) into `state/run_cycle_history.jsonl`.
     - Implements exponential backoff on consecutive failures (e.g., 1m, 5m, 15m) capped at 30m.
     - Respects `SIGINT`/`SIGTERM` for graceful stop (finish current cycle, flush logs).

3. **Health checks & dependency remediation**
   - Optional integration with `scripts/system_mode_initializer.py` and `scripts/protocol_probe.py` already handled inside `run_cycle.sh`.
   - Add lightweight check post-run to ensure new archive exists; if missing, mark iteration as degraded.

4. **Operator controls**
   - Provide environment variable overrides (e.g., `RUN_CYCLE_MODE`, `RUN_CYCLE_SLEEP`).
   - Allow `--dry-run` to perform prerequisite check and command preview without launching.
   - Support `--max-failures` to abort after N consecutive errors.

5. **Scheduling options**
   - `cron` example: `*/30 * * * * /path/to/.venv/bin/python scripts/run_cycle_loop.py --iterations 1 --sleep 0`
   - `systemd` unit suggestion with Restart=always and Environment overrides.
   - Containerized deployment: wrap script in supervisor (e.g., `forever`, `pm2` for Python) with mounted `state/` volume.

6. **Logging & observability**
   - Log directory `logs/run_cycle/` with per-iteration files (`YYYYMMDDTHHMMSS.log`).
   - Summary log `logs/run_cycle/latest.json` updated atomically to show current status (success/failure count, last planner archive path).
   - Optionally publish metrics to stdout in JSON for external scraping (e.g., `{ "timestamp": "…", "status": "ok" }`).
   - On failure, capture return code, stderr snippet, and prerequisite status for debugging.

7. **Operational safeguards**
   - Enforce write caps by passing `--cap-writes` (configurable) to `scripts/start_agent.py`.
   - Provide `--sandbox-reset` flag to clear `sandbox/` between runs when needed.
   - Include `--stop-file` path; if the file exists, halt before next iteration as an operator kill switch.
   - Verify planner archives: compare latest JSONL timestamp each run and alert if stale across multiple iterations.

8. **Open questions**
   - Should the loop trigger `git status`/`git pull` checks to ensure code freshness before each cycle?
   - Do we need adaptive goals per iteration (mission-driven or randomized) instead of static text?
   - How should we surface consent audit logs when moving to fully unattended execution?

## Next Steps

1. Implement `scripts/run_cycle_loop.py` following the design above.
2. Extend `tools/run_cycle.sh` or `scripts/start_agent.py` with a non-interactive consent option to remove hardcoded `printf 'y'`.
3. Add dry-run and unit tests to validate the loop controller without launching full agent cycles.
4. Document scheduling examples (cron, systemd) and monitoring/rollback instructions.