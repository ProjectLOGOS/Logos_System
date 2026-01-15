#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/prototype_grounding_workflow.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Prototype runner that wires ingestion outputs into the planner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
PLANNER_DIGEST_LOG = REPO_ROOT / "state" / "planner_digests.jsonl"
PLANNER_ARCHIVE_DIR = REPO_ROOT / "state" / "planner_digest_archives"


def _load_runtime_dependencies():
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from Logos_AGI.Logos_Agent.creator_packet.reflection_builder.perception_ingestors import ObservationBroker
    from Protopraxis.agent_planner import (
        AlignmentAwarePlanner,
        AlignmentRequiredError,
        append_digest_to_log,
        snapshot_digest_log,
        latest_digest_archive,
    )

    return (
        ObservationBroker,
        AlignmentAwarePlanner,
        AlignmentRequiredError,
        append_digest_to_log,
        snapshot_digest_log,
        latest_digest_archive,
    )


ALIGNMENT_LOG = REPO_ROOT / "state" / "alignment_LOGOS-AGENT-OMEGA.json"


def _alignment_history() -> List[Dict[str, Any]]:
    if not ALIGNMENT_LOG.exists():
        return []
    try:
        data = json.loads(ALIGNMENT_LOG.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _alignment_status() -> tuple[bool, Dict[str, Any], List[str], List[Dict[str, Any]]]:
    history = _alignment_history()
    if not history:
        return False, {}, ["alignment log missing or empty"], []

    record = history[-1]

    issues: List[str] = []
    if not record.get("rebuild_success"):
        issues.append("latest rebuild did not report success")

    lem_assumptions = record.get("lem_assumptions", [])
    if lem_assumptions:
        issues.append(
            "constructive LEM discharge still depends on assumptions: "
            + ", ".join(map(str, lem_assumptions))
        )

    admitted = record.get("admitted_stubs", [])
    if admitted:
        issues.append(
            "proof artifacts contain Admitted stubs: " + ", ".join(map(str, admitted))
        )

    aligned = not issues
    return aligned, record, issues, history


def _format_alignment_entry(entry: Dict[str, Any]) -> str:
    status = "PASS"
    if (
        not entry.get("rebuild_success")
        or entry.get("lem_assumptions")
        or entry.get("admitted_stubs")
    ):
        status = "WARN"
    return f"{entry.get('verified_at', '<unknown>')} → {status}"


def _format_actions(actions: List[Any]) -> str:
    lines = []
    for action in actions:
        sources = ", ".join(action.sources) if action.sources else "<none>"
        lines.append(f"- {action.name}: {action.rationale} (sources: {sources})")
    return "\n".join(lines) if lines else "<none>"


def _print_health(broker: Any, stage: str) -> None:
    try:
        statuses = broker.health_report()
    except AttributeError:
        return
    if not statuses:
        return
    print(f"\nIngestion health ({stage}):")
    for status in statuses:
        flag = "READY" if status.available else "WAIT"
        print(f"  [{flag}] {status.name}: {status.detail}")


def _print_trace_digest(broker: Any) -> None:
    try:
        digest = broker.trace_digest()
    except AttributeError:
        return
    if not digest:
        return
    print("\nObservation traceability digest:")
    for entry in digest:
        name = entry.get("name", "<unknown>")
        availability = "available" if entry.get("available") else "waiting"
        print(f"  - {name} ({availability})")
        path = entry.get("path")
        if path:
            print(f"      path: {path}")
        updated_at = entry.get("updated_at")
        if updated_at:
            print(f"      updated: {updated_at}")
        extra = entry.get("extra") or {}
        for key in sorted(extra):
            print(f"      {key}: {extra[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assume-aligned",
        action="store_true",
        help="Override alignment detection for dry runs",
    )
    parser.add_argument(
        "--explain-path",
        type=str,
        help="Write planner explanation JSON relative to the repository root.",
    )
    parser.add_argument(
        "--archive-digest",
        action="store_true",
        help="Create a compressed snapshot of planner_digests.jsonl after logging.",
    )
    args = parser.parse_args()

    aligned, record, issues, history = _alignment_status()
    alignment_flag = aligned or args.assume_aligned

    (
        ObservationBroker,
        AlignmentAwarePlanner,
        AlignmentRequiredError,
        append_digest_to_log,
        snapshot_digest_log,
        latest_digest_archive,
    ) = _load_runtime_dependencies()

    broker = ObservationBroker()
    planner = AlignmentAwarePlanner()

    print("Available symbolic datasets:")
    for name in broker.available_ingestors():
        print(f"  - {name}")
    if not broker.available_ingestors():
        print("  <none>")

    _print_health(broker, "pre-gather")

    if record:
        timestamp = record.get("verified_at", "<unknown>")
        print(f"\nLatest alignment record: {timestamp}")
        if record.get("agent_hash"):
            print(f"  Agent hash: {record['agent_hash']}")

    if history:
        print(f"Alignment audit entries: {len(history)}")
        recent = history[-3:]
        for entry in recent:
            print(f"  - {_format_alignment_entry(entry)}")

    if alignment_flag and aligned:
        planner.mark_alignment_verified(True)
    elif alignment_flag and args.assume_aligned:
        print(
            "\nAlignment override engaged (--assume-aligned). Proceed only after "
            "confirming constructive LEM verification via test_lem_discharge.py."
        )
        planner.mark_alignment_verified(True)
    else:
        print("\nAlignment not yet verified; planner activation will halt if invoked.")
        for note in issues:
            print(f"  - {note}")

    try:
        observations = broker.gather()
    except RuntimeError as exc:
        print(f"\nIngestion blocked: {exc}")
        print(
            "Run after alignment unlocks safe interface restrictions or use "
            "--assume-aligned once mission profile allows it."
        )
        return

    print(f"\nCollected {len(observations)} observations.")

    _print_health(broker, "post-gather")
    _print_trace_digest(broker)

    try:
        actions = planner.plan(observations)
    except AlignmentRequiredError as exc:
        print(f"Planner halted: {exc}")
        print("Use --assume-aligned after confirming constructive LEM verification.")
        return

    print("\nProposed safe actions:")
    print(_format_actions(actions))

    digest = planner.latest_digest()
    if digest:
        append_digest_to_log(digest, PLANNER_DIGEST_LOG)
        if args.archive_digest:
            archive = snapshot_digest_log(PLANNER_DIGEST_LOG, PLANNER_ARCHIVE_DIR)
            if archive:
                print(
                    "\nPlanner digest snapshot archived to "
                    f"{archive.relative_to(REPO_ROOT)}"
                )
        archive_info = latest_digest_archive(PLANNER_ARCHIVE_DIR)
        if archive_info:
            archive_path = archive_info["path"]
            try:
                relative = archive_path.relative_to(REPO_ROOT)
            except ValueError:
                relative = archive_path
            size = archive_info.get("size_bytes", 0)
            modified = archive_info.get("modified_at", "<unknown>")
            print(
                "\nLatest planner archive: "
                f"{relative} ({size} bytes, updated {modified})"
            )
        if args.explain_path:
            target = (REPO_ROOT / args.explain_path).resolve()
            try:
                target.relative_to(REPO_ROOT)
            except ValueError:
                print(f"\nExplain path escapes repository: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(digest, indent=2) + "\n", encoding="utf-8")
                print(
                    f"\nPlanner explanation written to {target.relative_to(REPO_ROOT)}"
                )


if __name__ == "__main__":
    main()
