# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: FORBIDDEN (DRY_RUN_ONLY)
# AUTHORITY: GOVERNED
# INSTALL_STATUS: DRY_RUN_ONLY
# SOURCE_LEGACY: stress_sop_runtime.py

"""
DRY-RUN REWRITE

This file is a structural, governed rewrite candidate generated for
rewrite-system validation only. No execution, no side effects.
"""
"""Stress test for the SOP runtime scheduler and telemetry pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
# Ensure SOP runtime and its local modules are importable when running standalone.
sys.path.append(str(REPO_ROOT / "external"))
sys.path.append(str(REPO_ROOT / "external" / "Logos_AGI"))

try:
    agent_system_module = import_module(
        "Logos_AGI.System_Operations_Protocol.infrastructure.agent_system."
        "logos_agent_system"
    )
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing Logos_AGI SOP runtime; run "
        "`python3 scripts/aligned_agent_import.py --probe` to fetch dependencies."
    ) from exc

AgentRegistry = agent_system_module.AgentRegistry
initialize_agent_system = agent_system_module.initialize_agent_system


def _build_fallback_shared_resources() -> Any:
    base_dir = REPO_ROOT / "state" / "sop_runtime"
    base_dir.mkdir(parents=True, exist_ok=True)

    class _FallbackSharedResources:
        def __init__(self) -> None:
            self.RUNTIME_BASE_DIR = base_dir
            self._runtime_dir = base_dir

        def configure_runtime_state_dir(self, path: Path) -> None:
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            self._runtime_dir = target
            self.RUNTIME_BASE_DIR = target

        def runtime_log_path(self, name: str) -> Path:
            target = self._runtime_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            return target

    return _FallbackSharedResources()


_shared_resources_spec = find_spec(
    "Logos_AGI.System_Operations_Protocol.infrastructure.shared_resources"
)
if _shared_resources_spec is not None:
    sop_shared_resources = import_module(
        "Logos_AGI.System_Operations_Protocol.infrastructure.shared_resources"
    )
else:
    sop_shared_resources = _build_fallback_shared_resources()


async def _spawn_user_requests(
    registry: AgentRegistry,
    total: int,
    text: str,
    concurrency: int,
) -> List[Dict[str, Any]]:
    user_agent = await registry.create_user_agent("stress_user")

    async def _invoke(idx: int) -> Dict[str, Any]:
        payload = f"{text} (sample {idx})"
        last_error: RuntimeError | None = None
        for attempt in range(60):
            try:
                return await user_agent.process_user_input(
                    payload,
                    context={"stress_index": idx, "attempt": attempt},
                )
            except RuntimeError as exc:
                if "Insufficient cpu_slots" not in str(exc):
                    raise
                last_error = exc
                await asyncio.sleep(0.2 * (attempt + 1))
        raise RuntimeError("Failed to process user input after retries") from last_error

    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: List[Dict[str, Any]] = []

    async def _guarded_call(idx: int) -> None:
        async with semaphore:
            results.append(await _invoke(idx))

    tasks = [asyncio.create_task(_guarded_call(i)) for i in range(total)]
    await asyncio.gather(*tasks)
    return results


async def _spawn_meta_cycles(
    registry: AgentRegistry, total: int
) -> List[Dict[str, Any]]:
    system_agent = registry.get_system_agent()
    results: List[Dict[str, Any]] = []

    async def _invoke(idx: int) -> Dict[str, Any]:
        trigger = {
            "type": "stress_autonomous",
            "origin": "stress_tool",
            "sequence": idx,
        }
        last_error: RuntimeError | None = None
        for attempt in range(60):
            try:
                return await system_agent.initiate_cognitive_processing(
                    trigger_data=trigger,
                    processing_type="stress_cycle",
                )
            except RuntimeError as exc:
                if "Insufficient cpu_slots" not in str(exc):
                    raise
                last_error = exc
                await asyncio.sleep(0.2 * (attempt + 1))
        raise RuntimeError("Failed to run meta cycle after retries") from last_error

    for idx in range(total):
        results.append(await _invoke(idx))
    return results


def _tail_jsonl(path: Path, keep: int = 5) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: deque[str] = deque(maxlen=keep)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(line.strip())
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        if not row:
            continue
        try:
            parsed.append(json.loads(row))
        except json.JSONDecodeError:
            parsed.append({"raw": row})
    return parsed


def _load_jsonl_since(path: Path, since: float) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    collected: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = float(record.get("timestamp", 0.0))
            if timestamp >= since:
                collected.append(record)
    return collected


def _prepare_run_directory() -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = sop_shared_resources.RUNTIME_BASE_DIR
    run_dir = base / f"run-{run_id}"
    sop_shared_resources.configure_runtime_state_dir(run_dir)
    for log_name in (
        "resource_events.jsonl",
        "scheduler_history.jsonl",
        "health_snapshots.jsonl",
    ):
        sop_shared_resources.runtime_log_path(log_name)
    return run_dir


def _summarize_log_file(path: Path) -> Dict[str, Any]:
    lines = 0
    first_timestamp: Optional[float] = None
    last_timestamp: Optional[float] = None
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                row = raw.strip()
                if not row:
                    continue
                lines += 1
                try:
                    record = json.loads(row)
                except json.JSONDecodeError:
                    continue
                timestamp = record.get("timestamp")
                if isinstance(timestamp, (int, float)):
                    value = float(timestamp)
                    if first_timestamp is None or value < first_timestamp:
                        first_timestamp = value
                    if last_timestamp is None or value > last_timestamp:
                        last_timestamp = value
    return {
        "lines": lines,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = _prepare_run_directory()
    registry = await initialize_agent_system(
        scp_mode=args.scp_mode,
        enable_autonomy=False,
    )
    orchestrator = registry.protocol_orchestrator
    runtime = orchestrator.runtime
    assert runtime is not None

    scp_initial = await orchestrator.get_scp_status()
    start_time = time.time()
    # Skip active workloads when the demo runtime has no schedulable CPU slots.
    cpu_spec = runtime.manager.capacities.get("cpu_slots")
    cpu_limit = getattr(cpu_spec, "limit", 0)
    cpu_available = max(cpu_limit - runtime.manager.allocations.get("cpu_slots", 0), 0)

    workloads_skipped = args.meta_cycles == 0 and args.user_requests == 0
    skip_reason: str | None = None
    workload_error: str | None = None

    # Temporary stub: defer CPU-intensive workloads until hardware access returns.
    workloads_stubbed = True
    if workloads_stubbed:
        workloads_skipped = True
        skip_reason = (
            "CPU workloads stubbed pending hardware access (TODO: restore spin-ups)"
        )

    if cpu_limit <= 0 or cpu_available <= 0:
        workloads_skipped = True
        skip_reason = (
            skip_reason or "scheduler reports zero cpu_slots in this environment"
        )

    user_results: List[Dict[str, Any]] = []
    meta_results: List[Dict[str, Any]] = []

    if not workloads_skipped:
        try:
            user_results = await _spawn_user_requests(
                registry,
                total=args.user_requests,
                text=args.prompt,
                concurrency=args.concurrency,
            )
        except RuntimeError as exc:
            workload_error = str(exc)
            if "cpu_slots" in workload_error:
                workloads_skipped = True
                skip_reason = (
                    skip_reason
                    or "UIP workload aborted after exhausting available cpu_slots"
                )
            else:
                raise

    if not workloads_skipped and args.meta_cycles > 0:
        try:
            meta_results = await _spawn_meta_cycles(registry, args.meta_cycles)
        except RuntimeError as exc:
            workload_error = workload_error or str(exc)
            if "cpu_slots" in str(exc):
                workloads_skipped = True
                skip_reason = (
                    skip_reason
                    or "Meta-cognitive cycles aborted after exhausting cpu_slots"
                )
                meta_results = []
            else:
                raise

    scp_final = await orchestrator.get_scp_status()

    final_snapshot = runtime.manager.describe()
    latest_health = runtime.monitor.latest()

    resource_log = run_dir / "resource_events.jsonl"
    scheduler_log = run_dir / "scheduler_history.jsonl"
    health_log = run_dir / "health_snapshots.jsonl"

    log_excerpt = {
        "resource_events": _tail_jsonl(resource_log, keep=8),
        "scheduler_history": _tail_jsonl(scheduler_log, keep=8),
        "health_snapshots": _tail_jsonl(health_log, keep=8),
    }

    recent_resource_events = _load_jsonl_since(resource_log, start_time)
    recent_scheduler_events = _load_jsonl_since(scheduler_log, start_time)

    log_summaries = {
        "resource_events.jsonl": _summarize_log_file(resource_log),
        "scheduler_history.jsonl": _summarize_log_file(scheduler_log),
        "health_snapshots.jsonl": _summarize_log_file(health_log),
    }

    peak_cpu = 0
    cumulative_cpu = 0
    for event in recent_resource_events:
        allocations = event.get("allocations", {})
        peak_cpu = max(peak_cpu, int(allocations.get("cpu_slots", 0)))
        if event.get("action") == "allocate":
            cumulative_cpu += int(event.get("cpu_slots", 0))

    scp_scheduler_events = [
        entry
        for entry in recent_scheduler_events
        if str(entry.get("label", "")).startswith("SCP:")
    ]
    scp_jobs_submitted = len(scp_scheduler_events)
    scp_jobs_completed = sum(
        1 for entry in scp_scheduler_events if entry.get("outcome") == "completed"
    )
    scp_last_error = None
    for entry in scp_scheduler_events:
        if "error" in entry:
            scp_last_error = str(entry["error"])

    if args.meta_cycles > 0 and scp_jobs_submitted == 0 and not workloads_skipped:
        raise RuntimeError(
            "SCP scheduling bypass detected: no SCP jobs recorded for meta-cycles"
        )

    if peak_cpu > cpu_limit and cpu_limit > 0:
        raise RuntimeError("Peak concurrent cpu_slots exceeded configured limit")

    log_file_counts = {
        name: summary["lines"] for name, summary in log_summaries.items()
    }
    log_spans = {
        name: {
            "first_timestamp": summary["first_timestamp"],
            "last_timestamp": summary["last_timestamp"],
        }
        for name, summary in log_summaries.items()
    }

    try:
        run_dir_relative = str(run_dir.relative_to(REPO_ROOT))
    except ValueError:
        run_dir_relative = str(run_dir)

    result = {
        "user_requests": len(user_results),
        "meta_cycles": len(meta_results),
        "run_dir": run_dir_relative,
        "log_file_counts": log_file_counts,
        "log_spans": log_spans,
        "workloads": {
            "requested_user": args.user_requests,
            "requested_meta": args.meta_cycles,
            "skipped": workloads_skipped,
            "reason": skip_reason,
            "error": workload_error,
            "capacity": {
                "cpu_limit": cpu_limit,
                "cpu_available": cpu_available,
            },
        },
        "cpu_metrics": {
            "limit": cpu_limit,
            "peak_concurrent_leases": peak_cpu,
            "cumulative_leases": cumulative_cpu,
        },
        "snapshot": final_snapshot,
        "latest_health": latest_health,
        "logs": log_excerpt,
        "scp": {
            "enabled": bool(getattr(orchestrator, "scp_system", None)),
            "mode": orchestrator.scp_mode,
            "connect_ok": bool(scp_final.get("status") == "ok"),
            "jobs_submitted": scp_jobs_submitted,
            "jobs_completed": scp_jobs_completed,
            "last_error": scp_last_error,
            "initial": scp_initial,
            "final": scp_final,
        },
    }

    try:
        return result
    finally:
        if orchestrator.scp_system and hasattr(orchestrator.scp_system, "shutdown"):
            await orchestrator.scp_system.shutdown()
        await runtime.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user-requests",
        type=int,
        default=20,
        help="Total synthetic user interactions to schedule",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent UIP jobs",
    )
    parser.add_argument(
        "--meta-cycles",
        type=int,
        default=5,
        help="Number of system agent meta-cognitive cycles",
    )
    parser.add_argument(
        "--prompt",
        default="Trace the constructive LEM cascade and summarize resource posture.",
        help="Base prompt text for synthetic interactions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON summary",
    )
    parser.add_argument(
        "--scp-mode",
        choices=["local", "off", "remote"],
        default="local",
        help="Select SCP transport mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args))
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()