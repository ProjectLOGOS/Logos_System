# Continuous Run Operations Guide

This guide explains how to launch, monitor, and safely halt the unattended planner cycles.

## 1. Prerequisites

1. Create or update the virtual environment:
   ```bash
   python -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
2. Verify required modules with the built-in diagnostic:
   ```bash
   .venv/bin/python scripts/check_run_cycle_prereqs.py
   ```
   - Install any missing *required* modules (e.g., `pip install numpy`).
   - Optional modules only affect auxiliary integrations and can be deferred.
3. Ensure the repository has write access to `sandbox/`, `state/`, and `logs/`.

## 2. One-Off Execution

Run a single cycle to confirm everything works:
```bash
.venv/bin/python scripts/run_cycle_loop.py --iterations 1 --sleep 0 --timeout 300
```
Key flags:
- `--mode` selects the mission profile (`experimental` by default).
- `--goal` overrides the default objective text.
- `--timeout` caps each cycle’s wall-clock runtime (seconds).

The loop writes:
- Per-run log: `logs/run_cycle/YYYYMMDDTHHMMSS.log`
- Summary pointer: `logs/run_cycle/latest.json`
- History ledger: `state/run_cycle_history.jsonl`

## 3. Continuous Mode

To run indefinitely until stopped or failure limit reached:
```bash
.venv/bin/python scripts/run_cycle_loop.py --continuous --sleep 600 --max-failures 5
```
Recommended defaults:
- `--sleep 600` pauses 10 minutes between successful runs.
- `--max-failures 5` halts after five consecutive failures.
- `--max-backoff 3600` limits exponential backoff to one hour.
- `--stop-file state/run_cycle.stop` provides a kill switch—create the file to abort before the next iteration.

## 4. Scheduling Examples

### Cron
```
*/30 * * * * /path/to/repo/.venv/bin/python /path/to/repo/scripts/run_cycle_loop.py --iterations 1 --sleep 0 --timeout 600 >> /path/to/repo/logs/run_cycle/cron.log 2>&1
```

### systemd (excerpt)
```
[Unit]
Description=PXL Continuous Run Loop
After=network.target

[Service]
WorkingDirectory=/path/to/repo
ExecStart=/path/to/repo/.venv/bin/python scripts/run_cycle_loop.py --continuous --sleep 600 --max-failures 5
Restart=always
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

## 5. Monitoring

- Tail `logs/run_cycle/latest.json` to view last status, consecutive failures, and last archive path.
- Inspect `state/run_cycle_history.jsonl` for a full run ledger.
- Planner archives aggregate under `state/planner_digest_archives/`; each cycle should produce a newer gzip file.

## 6. Recovery & Cleanup

- To stop gracefully, touch the stop file or send `Ctrl+C` if running in foreground.
- If prerequisites begin failing (e.g., missing numpy), fix the environment and restart the loop.
- Logs can be rotated or pruned manually; they are append-only text files.
- Use `scripts/archive_planner_digests.py` manually when troubleshooting digest generation.

## 7. Safety Notes

- `tools/run_cycle.sh` now passes `--assume-yes` to the agent, so supervise changes to the plan pipeline carefully.
- Keep `--cap-writes` tuned for your workload; default is three writes per cycle.
- Planner digests remain append-only; never delete `state/planner_digests.jsonl` unless you have a full backup.
