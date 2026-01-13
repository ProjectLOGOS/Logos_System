#!/usr/bin/env python3
"""Run tools/run_cycle.sh repeatedly with health checks and logging."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_CYCLE = REPO_ROOT / "tools" / "run_cycle.sh"
PREREQ_CHECK = REPO_ROOT / "scripts" / "check_run_cycle_prereqs.py"
LOGOS_AGI_PATH = REPO_ROOT / "external" / "Logos_AGI"
STATE_DIR = Path(os.getenv("LOGOS_STATE_DIR", REPO_ROOT / "state"))
ARCHIVE_DIR = STATE_DIR / "planner_digest_archives"
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "run_cycle"
DEFAULT_HISTORY_PATH = STATE_DIR / "run_cycle_history.jsonl"
GOAL_DEFAULT = (
    "Read last artifact, produce 3-bullet recap (facts only), "
    "propose one measurable experiment with success criteria, then write plan.md."
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default="experimental",
        help="Mission mode passed to run_cycle.sh",
    )
    parser.add_argument("--goal", default=GOAL_DEFAULT, help="Agent objective text")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of cycles to run (0 means until stopped)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Ignore --iterations and run until stop signal or failure limit",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=300,
        help="Seconds to sleep between successful runs (default: 300)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="External timeout per cycle in seconds (default: 600)",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="Abort after this many consecutive failures (default: 3)",
    )
    parser.add_argument(
        "--max-backoff",
        type=int,
        default=1800,
        help="Maximum backoff delay after failures in seconds (default: 1800)",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="If this file exists before a new iteration, the loop halts",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for iteration logs (default: logs/run_cycle)",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="Append iteration metadata to this JSONL file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform prerequisite check and exit without running the cycle",
    )
    return parser.parse_args(argv)


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    if current:
        segments = [segment for segment in current.split(os.pathsep) if segment]
    else:
        segments = []

    candidates = [REPO_ROOT, LOGOS_AGI_PATH]
    for candidate in candidates:
        if not candidate.exists():
            continue
        path_str = str(candidate)
        if path_str not in segments:
            segments.append(path_str)

    if segments:
        env["PYTHONPATH"] = os.pathsep.join(segments)
    return env


def _run_prereq() -> Tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, str(PREREQ_CHECK)],
        text=True,
        capture_output=True,
        check=False,
        env=_build_env(),
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output


def _run_cycle(
    mode: str,
    goal: str,
    timeout: int,
) -> Tuple[str, str, Optional[int], str, float]:
    start = time.monotonic()
    try:
        result = subprocess.run(
            [str(RUN_CYCLE), mode, goal],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=_build_env(),
        )
        duration = time.monotonic() - start
        return result.stdout, result.stderr, result.returncode, "ok", duration
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - guard
        duration = time.monotonic() - start
        stdout = exc.output or ""
        stderr = exc.stderr or ""
        return stdout, stderr, None, "timeout", duration
    except subprocess.SubprocessError as exc:  # pragma: no cover - guard
        duration = time.monotonic() - start
        return "", str(exc), None, "error", duration


def _latest_archive() -> Optional[dict[str, object]]:
    if not ARCHIVE_DIR.exists():
        return None
    archives = [
        path
        for path in ARCHIVE_DIR.glob("planner_digests_*.jsonl.gz")
        if path.is_file()
    ]
    if not archives:
        return None
    archives.sort(key=lambda item: item.stat().st_mtime)
    latest = archives[-1]
    stats = latest.stat()
    return {
        "path": latest,
        "mtime": stats.st_mtime,
        "size": stats.st_size,
    }


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_log(
    log_path: Path,
    header: dict[str, object],
    prereq: str,
    stdout: str,
    stderr: str,
) -> None:
    lines = [
        json.dumps(header, default=str),
        "\n",
        "=== Prerequisite Check ===\n",
        prereq or "<none>\n",
    ]
    lines.append("\n=== STDOUT ===\n")
    lines.append(stdout or "<empty>\n")
    lines.append("\n=== STDERR ===\n")
    lines.append(stderr or "<empty>\n")
    log_path.write_text("".join(lines), encoding="utf-8")


def _append_history(history_path: Path, entry: dict[str, object]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")


def _write_summary(log_dir: Path, summary: dict[str, object]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "latest.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        ok, output = _run_prereq()
        print(output.strip())
        return 0 if ok else 1

    if not RUN_CYCLE.exists():
        raise SystemExit(f"Missing run cycle script: {RUN_CYCLE}")

    args.log_dir.mkdir(parents=True, exist_ok=True)

    iterations_remaining = (
        None if args.continuous or args.iterations == 0 else args.iterations
    )
    completed = 0
    consecutive_failures = 0

    while True:
        if iterations_remaining is not None and iterations_remaining <= 0:
            break
        if args.stop_file and args.stop_file.exists():
            print("Stop file detected; halting loop.")
            break

        prereq_ok, prereq_output = _run_prereq()
        start_ts = datetime.now(timezone.utc)
        archive_before = _latest_archive()

        stdout = ""
        stderr = ""
        return_code: Optional[int] = None
        status = "prereq_failed"
        duration = 0.0

        if prereq_ok:
            stdout, stderr, return_code, status, duration = _run_cycle(
                args.mode, args.goal, args.timeout
            )
        else:
            status = "prereq_failed"

        archive_after = _latest_archive()
        archive_created = False
        archive_path = None
        if archive_after:
            archive_path = _relative(archive_after["path"])
            if not archive_before:
                archive_created = True
            elif archive_after["path"] != archive_before["path"]:
                archive_created = True
            elif archive_after["mtime"] != archive_before["mtime"]:
                archive_created = True

        success = prereq_ok and status == "ok" and return_code == 0
        status_label = "success" if success else status

        header = {
            "timestamp": start_ts.isoformat(),
            "mode": args.mode,
            "goal": args.goal,
            "status": status_label,
            "duration_seconds": round(duration, 3),
            "return_code": return_code,
            "archive_created": archive_created,
            "archive_path": archive_path,
        }

        log_path = args.log_dir / f"{start_ts.strftime('%Y%m%dT%H%M%SZ')}.log"
        _write_log(log_path, header, prereq_output, stdout, stderr)

        entry = {
            "timestamp": start_ts.isoformat(),
            "mode": args.mode,
            "goal": args.goal,
            "status": status_label,
            "prerequisites_ok": prereq_ok,
            "return_code": return_code,
            "duration_seconds": round(duration, 3),
            "archive_created": archive_created,
            "archive_path": archive_path,
            "log_path": _relative(log_path),
        }
        _append_history(args.history_path, entry)

        stop_after_failure = False
        if success:
            completed += 1
            consecutive_failures = 0
            sleep_duration = args.sleep
        else:
            consecutive_failures += 1
            if args.max_failures and consecutive_failures >= args.max_failures:
                stop_after_failure = True
            backoff = args.sleep * (2 ** (max(consecutive_failures - 1, 0)))
            sleep_duration = min(args.max_backoff, backoff)

        summary = {
            "timestamp": start_ts.isoformat(),
            "mode": args.mode,
            "goal": args.goal,
            "status": status_label,
            "consecutive_failures": consecutive_failures,
            "last_return_code": return_code,
            "last_log": _relative(log_path),
            "last_archive": archive_path,
            "iterations_completed": completed,
        }
        _write_summary(args.log_dir, summary)

        if stop_after_failure:
            print("Failure limit reached; halting loop.")
            break

        if iterations_remaining is not None:
            iterations_remaining -= 1
            if iterations_remaining <= 0:
                break

        if sleep_duration > 0:
            for _ in range(sleep_duration):
                if args.stop_file and args.stop_file.exists():
                    print("Stop file detected during sleep; halting loop.")
                    return 0
                time.sleep(1)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nLoop interrupted by user.")
        sys.exit(130)
