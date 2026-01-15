#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/scp_local_diagnostic.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

# ============================
# LOGOS FILE HEADER (STANDARD)
# ============================
# File: _Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/scp_local_diagnostic.py
# Generated: 2026-01-14T19:54:10Z (UTC)
#
# SUMMARY:
#   Legacy dev/runtime-adjacent utility script captured during Audit & Normalize.
#   Header-standardized under Rewrite_Sub_Batch_1A.
#
# PURPOSE:
#   Preserve existing behavior; standardize documentation + naming/abbreviation policy markers.
#
# EXECUTION PATH:
#   Developer-invoked utility (audit/registry/diagnostic); not a critical runtime entrypoint.
#
# SIDE EFFECTS:
#   Review below; do not expand side effects. Fail-closed on unsafe operations.
#
# GOVERNANCE:
#   - Fail-closed required.
#   - Do not touch Coq stacks.
#   - Canon abbreviations: ETGC, SMP, MVS, BDN, PXL, IEL, SOP, SCP, ARP, MTP, TLM, UWM, LEM.
#   - Alias forbidden: use ETGC only; legacy variant disallowed.
# ============================

"""Deterministic SCP stress diagnostic without using stress_sop_runtime."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, TYPE_CHECKING
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure local modules import correctly when executed standalone.

sys.path.append(str(REPO_ROOT / "external"))
sys.path.append(str(REPO_ROOT / "external" / "Logos_AGI"))

if TYPE_CHECKING:
    from Logos_AGI.System_Operations_Protocol.infrastructure.agent_system import (
        logos_agent_system,
    )

    AgentRegistryProtocol = logos_agent_system.AgentRegistry
else:
    AgentRegistryProtocol = Any


@lru_cache(maxsize=1)
def _resolve_logos_agi_components() -> tuple[Any, Callable[..., Any], Path]:
    agent_module_name = (
        "Logos_AGI.System_Operations_Protocol.infrastructure.agent_system."
        "logos_agent_system"
    )
    boot_module_name = "Logos_AGI.System_Operations_Protocol.infrastructure.boot_system"

    try:
        agent_module = importlib.import_module(agent_module_name)
        boot_module = importlib.import_module(boot_module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Logos_AGI components are unavailable. Run "
            "'python3 scripts/aligned_agent_import.py --probe' to refresh them."
        ) from exc

    registry = getattr(agent_module, "AgentRegistry", None)
    initializer = getattr(agent_module, "initialize_agent_system", None)
    runtime_dir = getattr(boot_module, "RUNTIME_STATE_DIR", None)

    if registry is None or initializer is None or runtime_dir is None:
        raise RuntimeError("Logos_AGI package is missing required SOP exports.")

    return registry, initializer, runtime_dir


async def _run_user_cycles(
    registry: AgentRegistryProtocol,
    total: int,
    prompt: str,
    *,
    concurrency: int,
) -> List[Dict[str, Any]]:
    user = await registry.create_user_agent("diagnostic_user")

    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: List[Dict[str, Any]] = []

    async def _invoke(index: int) -> None:
        async with semaphore:
            payload = f"{prompt} (sample {index})"
            attempt = 0
            while True:
                try:
                    results.append(
                        await user.process_user_input(
                            payload,
                            context={"diagnostic_index": index, "attempt": attempt},
                        )
                    )
                    break
                except RuntimeError as exc:
                    if "Insufficient cpu_slots" not in str(exc) or attempt >= 5:
                        raise
                    await asyncio.sleep(0.1 * (attempt + 1))
                    attempt += 1

    await asyncio.gather(*[asyncio.create_task(_invoke(i)) for i in range(total)])
    return results


async def _run_meta_cycles(
    registry: AgentRegistryProtocol, total: int
) -> List[Dict[str, Any]]:
    system_agent = registry.get_system_agent()
    results: List[Dict[str, Any]] = []
    for index in range(total):
        trigger = {
            "type": "diagnostic_autonomous",
            "origin": "scp_local_diagnostic",
            "sequence": index,
        }
        attempt = 0
        while True:
            try:
                results.append(
                    await system_agent.initiate_cognitive_processing(
                        trigger_data=trigger,
                        processing_type="diagnostic",
                    )
                )
                break
            except RuntimeError as exc:
                if "Insufficient cpu_slots" not in str(exc) or attempt >= 5:
                    raise
                await asyncio.sleep(0.1 * (attempt + 1))
                attempt += 1
    return results


def _tail(path: Path, keep: int = 8) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-keep:]
    records: List[Dict[str, Any]] = []
    for row in lines:
        row = row.strip()
        if not row:
            continue
        try:
            records.append(json.loads(row))
        except json.JSONDecodeError:
            records.append({"raw": row})
    return records


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    _, initialize_agent_system_fn, runtime_state_dir = _resolve_logos_agi_components()
    registry = await initialize_agent_system_fn(
        scp_mode=args.scp_mode,
        enable_autonomy=False,
    )
    orchestrator = registry.protocol_orchestrator
    runtime = orchestrator.runtime
    assert runtime is not None

    scp_initial = await orchestrator.get_scp_status()

    user_results = await _run_user_cycles(
        registry,
        total=args.user_requests,
        prompt=args.prompt,
        concurrency=args.concurrency,
    )
    meta_results = await _run_meta_cycles(registry, args.meta_cycles)

    scp_final = await orchestrator.get_scp_status()

    snapshot = runtime.manager.describe()
    health = runtime.monitor.latest()

    logs = {
        "resource_events": _tail(runtime_state_dir / "resource_events.jsonl"),
        "scheduler_history": _tail(runtime_state_dir / "scheduler_history.jsonl"),
        "health_snapshots": _tail(runtime_state_dir / "health_snapshots.jsonl"),
    }

    result = {
        "user_requests": len(user_results),
        "meta_cycles": len(meta_results),
        "scp": {
            "mode": orchestrator.scp_mode,
            "initial": scp_initial,
            "final": scp_final,
        },
        "snapshot": snapshot,
        "latest_health": health,
        "logs": logs,
    }

    try:
        return result
    finally:
        if orchestrator.scp_system and hasattr(orchestrator.scp_system, "shutdown"):
            await orchestrator.scp_system.shutdown()
        await runtime.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-requests", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--meta-cycles", type=int, default=3)
    parser.add_argument(
        "--prompt",
        default="Trace the constructive LEM cascade and summarize resource posture.",
    )
    parser.add_argument("--output", type=Path)
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
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
