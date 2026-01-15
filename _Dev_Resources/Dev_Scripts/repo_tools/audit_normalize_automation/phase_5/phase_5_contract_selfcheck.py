"""
Phase 5 contract self-check (Step 1).

Run manually if desired:
  python3 -m audit_normalize_automation.phase_5.phase_5_contract_selfcheck

This file MUST NOT be imported by the runner automatically.
"""

from __future__ import annotations

import json

from .phase_5_types import validate_phase5_queue_obj


def main() -> int:
    sample = {
        "version": "audit_normalize/phase_5_queue/1",
        "generated_utc": "1970-01-01T00:00:00Z",
        "source_phase_4": {"status_path": "_Reports/Audit_Normalize/Phase_4_Status.json"},
        "tasks": [
            {
                "task_id": "T0001",
                "action": "NOOP",
                "src_path": "Some/Repo/Relative/Path.py",
                "dest_path": None,
                "reason": "Contract self-check only.",
                "evidence": {"artifacts": ["_Reports/Audit_Normalize/Phase_4_Status.json"], "notes": "sample"},
            }
        ],
    }
    q = validate_phase5_queue_obj(sample)
    print("OK:", q.version, "tasks=", len(q.tasks))
    print(json.dumps(sample, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
