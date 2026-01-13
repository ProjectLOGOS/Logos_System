#!/usr/bin/env python3
"""Create a compressed snapshot of the planner digest log."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_ROOT = Path(os.getenv("LOGOS_STATE_DIR", REPO_ROOT / "state"))
DEFAULT_LOG = STATE_ROOT / "planner_digests.jsonl"
DEFAULT_ARCHIVE_DIR = STATE_ROOT / "planner_digest_archives"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG,
        help="Path to planner_digests.jsonl (default: state/planner_digests.jsonl).",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help="Directory where compressed snapshots should be written.",
    )
    return parser.parse_args()


def _ensure_repo_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.append(repo_root)


def main() -> int:
    args = _parse_args()
    _ensure_repo_on_path()

    from Protopraxis.agent_planner import snapshot_digest_log

    archive_path = snapshot_digest_log(args.log_path, args.archive_dir)
    if not archive_path:
        print("No snapshot created; log file missing or empty.")
        return 0

    try:
        relative = archive_path.relative_to(REPO_ROOT)
    except ValueError:
        relative = archive_path

    print(f"Snapshot written to {relative}")

    # Write the latest archive path for identity binding
    latest_archive_file = STATE_ROOT / "latest_planner_digest_archive.txt"
    latest_archive_file.parent.mkdir(parents=True, exist_ok=True)
    latest_archive_file.write_text(str(relative))

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nSnapshot interrupted by user.")
        sys.exit(130)
