"""Lightweight daemon stub that emits deterministic telemetry.

This module intentionally avoids the heavyweight historical daemon and instead
produces synthetic telemetry that exercises the end-to-end pipeline tests.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import List


def _build_gap_events(include_gaps: bool) -> List[dict]:
    base_event = {
        "event_type": "daemon_started",
        "timestamp": time.time(),
        "daemon_version": "stub-1.0",
    }
    events = [base_event]
    if include_gaps:
        events.append(
            {
                "event_type": "gap_detected",
                "timestamp": time.time(),
                "gap_id": f"gap-{uuid.uuid4().hex[:8]}",
                "context": {
                    "module": "logos_core.daemon.logos_daemon",
                    "severity": "info",
                },
            }
        )
    else:
        events.append(
            {
                "event_type": "daemon_heartbeat",
                "timestamp": time.time(),
            }
        )
    return events


def run_once(out_path: Path, emit_gaps: bool) -> None:
    events = _build_gap_events(emit_gaps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LOGOS daemon stub once")
    parser.add_argument("--once", action="store_true", help="Run a single iteration and exit")
    parser.add_argument(
        "--emit-gaps",
        action="store_true",
        help="Emit synthetic reasoning gap events",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("metrics/agi_status.jsonl"),
        help="Telemetry output file",
    )
    args = parser.parse_args(argv)

    run_once(args.out, emit_gaps=args.emit_gaps)

    print(
        f"Generated telemetry at {args.out} with"
        f" {'gap' if args.emit_gaps else 'heartbeat'} events"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
