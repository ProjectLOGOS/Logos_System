from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def build_crawl_queue(report_dir: Path) -> dict:
    """
    Phase 4 — Crawl (Read-only)

    Purpose:
      Expand Phase 3 planning outputs into an actionable *crawl queue*.
      This is still NON-MUTATING: no routing, no moves, no deletes.

    Current minimal implementation:
      - Detect Phase 3 planning artifacts under this report directory (best-effort).
      - Emit Phase_4/Crawl_Queue.json as a structured queue stub for future enrichment.
    """
    ph4 = report_dir / "Phase_4"
    ph4.mkdir(parents=True, exist_ok=True)

    # Best-effort discovery: any JSON/JSONL under Phase_3 is considered planning input.
    ph3 = report_dir / "Phase_3"
    planning_files = []
    if ph3.exists():
        planning_files = sorted(
            [str(p) for p in ph3.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl", ".md"}]
        )

    queue = {
        "version": "phase-4-crawl-queue/1",
        "ts": _now(),
        "report_dir": str(report_dir),
        "planning_inputs": planning_files,
        "tasks": [],
        "notes": [
            "Phase 4 is read-only: this file is a queue stub and may be enriched later.",
            "No routing is permitted before Phase 5.",
        ],
    }

    # If nothing to crawl, still return PASS with zero tasks.
    if not planning_files:
        (ph4 / "Crawl_Queue.json").write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (ph4 / "Crawl_Summary.json").write_text(
            json.dumps(
                {
                    "ts": queue["ts"],
                    "planning_inputs_count": len(planning_files),
                    "tasks_count": 0,
                    "status": "PASS",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return {"status": "PASS", "tasks_count": 0}

    # Minimal deterministic expansion:
    # One crawl task per Phase 3 planning artifact
    for p in planning_files:
        queue["tasks"].append({
            "kind": "crawl_planning_artifact",
            "path": p,
            "read_only": True,
        })

    (ph4 / "Crawl_Queue.json").write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ph4 / "Crawl_Summary.json").write_text(
        json.dumps(
            {
                "ts": queue["ts"],
                "planning_inputs_count": len(planning_files),
                "tasks_count": len(queue["tasks"]),
                "status": "PASS",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "status": "PASS",
        "tasks_count": len(queue["tasks"]),
    }
