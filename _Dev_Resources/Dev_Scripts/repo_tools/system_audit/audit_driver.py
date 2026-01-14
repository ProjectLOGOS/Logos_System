#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

SCANS = [
    "scan_imports.py",
    "scan_dependencies.py",
    "scan_symbols.py",
    "scan_duplicates.py",
    "scan_naming.py",
    "scan_runtime_surface.py",
    "scan_side_effects.py",
    "scan_tree.py",
    "scan_dev_scripts.py",
    "scan_dead_code.py",
    "scan_config_and_schema.py",
]

def run_scan(script: Path) -> None:
    subprocess.check_call(["python3", str(script)])

def main() -> int:
    base = Path(__file__).resolve().parent
    for s in SCANS:
        run_scan(base / s)

    # Summary stub: populated after first run so automation has a stable handle
    summary_dir = Path("/workspaces/Logos_System/_Reports/SYSTEM_AUDIT/SUMMARY")
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "automation_readiness.json").write_text(
        '{\n  "status": "generated",\n  "next": ["build rename plan", "build import-fix plan", "apply Title_Case_With_Underscores sweep"]\n}\n',
        encoding="utf-8"
    )
    (summary_dir / "normalization_candidates.json").write_text(
        '{\n  "note": "Populate after reviewing naming_violations + duplicates + orphaned_files + unresolved_imports"\n}\n',
        encoding="utf-8"
    )
    (summary_dir / "refactor_plan.json").write_text(
        '{\n  "note": "Populate after identifying canonical module map and runtime entry surface"\n}\n',
        encoding="utf-8"
    )
    print(str(summary_dir))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
