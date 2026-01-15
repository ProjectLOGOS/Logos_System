#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from Header_Lib import iter_py_files, parse_dev_header_v1

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--exclude", default="PXL_Gate,__pycache__,.git,.venv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    files = iter_py_files(root, exclude_dirs=args.exclude.split(","))
    rows = []
    count = 0
    for f in files:
        lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        dev = None
        try:
            dev = parse_dev_header_v1(lines)
        except Exception as e:
            raise SystemExit(f"FAIL-CLOSED: malformed dev header in {f}: {e}")
        rows.append({"path": str(f), "has_dev_header_v1": dev is not None})
        if dev is not None: count += 1

    out = {"root": str(root), "total_py": len(files), "dev_header_v1_count": count, "rows": rows}
    out_path = report_dir / "Dev_Header_Scan.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
