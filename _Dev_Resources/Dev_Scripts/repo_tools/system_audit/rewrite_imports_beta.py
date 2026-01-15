#!/usr/bin/env python3
"""
BETA: Import Rewrite Helper (Dry-Run Only)

Purpose:
- Prototype controlled import rewrites for Phase 5 (Audit & Normalize)
- Uses patterns discovered by scan_imports.py
- Produces a report; does NOT mutate files unless explicitly enabled later

Status:
- DRY-RUN ONLY
- FAIL-CLOSED
"""

import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="old", required=True)
    ap.add_argument("--to", dest="new", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    matches = []

    for py in root.rglob("*.py"):
        try:
            txt = py.read_text(encoding="utf-8")
        except Exception:
            continue
        if args.old in txt:
            matches.append(str(py))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(matches),
        encoding="utf-8"
    )

    print(f"[DRY-RUN] Found {len(matches)} files containing '{args.old}'")
    print(f"Report written to {out}")

if __name__ == "__main__":
    main()
