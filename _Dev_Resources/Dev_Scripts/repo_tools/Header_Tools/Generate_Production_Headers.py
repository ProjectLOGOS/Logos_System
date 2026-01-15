#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from Header_Lib import (
    build_production_header, enforce_abbrev_policy, iter_py_files, parse_dev_header_v1,
    preserve_preamble, read_text, render_production_header, strip_existing_production_header, unified_diff
)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--exclude", default="PXL_Gate,__pycache__,.git,.venv")
    ap.add_argument("--emit-all", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    diff_dir = report_dir / "Diffs"
    diff_dir.mkdir(parents=True, exist_ok=True)

    files = iter_py_files(root, exclude_dirs=args.exclude.split(","))
    plan = {"root": str(root), "total_py": len(files), "planned": [], "skipped": [], "diff_dir": str(diff_dir)}

    for f in files:
        raw_lines = f.read_text(encoding="utf-8", errors="ignore").splitlines(True)
        dev = parse_dev_header_v1([ln.rstrip("\n") for ln in raw_lines])
        if dev is None and not args.emit_all:
            plan["skipped"].append({"path": str(f), "reason": "no_dev_header_v1"})
            continue

        text = read_text(f)
        enforce_abbrev_policy(text)
        h = build_production_header(f, dev, text)

        pre, rest = preserve_preamble(raw_lines)
        rest = strip_existing_production_header(rest)
        prod = render_production_header(h)
        new_text = "".join(pre) + prod + "".join(rest)

        diff = unified_diff("".join(raw_lines), new_text, fromfile=str(f), tofile=str(f) + " (prod header)")
        diff_path = diff_dir / (f.name + ".prod_header.diff")
        diff_path.write_text(diff, encoding="utf-8")

        plan["planned"].append({"path": str(f), "has_dev_header_v1": dev is not None, "diff": str(diff_path)})

    out_plan = report_dir / "Production_Header_Generation_Plan.json"
    out_plan.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(str(out_plan))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
