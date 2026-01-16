#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import re
import datetime
import os

TARGET_ROOT = Path(os.environ.get("TARGET_ROOT",""))
if not TARGET_ROOT.exists():
    raise SystemExit("FAIL-CLOSED: TARGET_ROOT missing or not set")

HEADER_START = "# LOGOS_HEADER: v1\n"
HEADER_END   = "# END_LOGOS_HEADER\n"

def has_v1(lines: list[str]) -> bool:
    head = "".join(lines[:80])
    return "LOGOS_HEADER: v1" in head

def insert_v1(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    if has_v1(raw):
        return False

    pre = []
    rest = raw[:]
    if rest and rest[0].startswith("#!"):
        pre.append(rest.pop(0))

    utc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    rel = path.as_posix()

    block = []
    block.append(HEADER_START)
    block.append(f"# updated_utc: {utc}\n")
    block.append(f"# path: {rel}\n")
    block.append("# role: dev_tool\n")
    block.append("# phase: audit_normalize\n")
    block.append("# origin: INSPECT_DECIDE\n")
    block.append("# intended_bucket: REWRITE_PROMOTE\n")
    block.append("# side_effects: unknown\n")
    block.append("# entrypoints: unknown\n")
    block.append("# depends_on: \n")
    block.append("# notes: \n")
    block.append(HEADER_END)
    block.append("\n")

    path.write_text("".join(pre + block + rest), encoding="utf-8")
    return True

changed = []
for p in sorted(TARGET_ROOT.rglob("*.py")):
    if not p.is_file():
        continue
    if insert_v1(p):
        changed.append(str(p))

out = {
    "target_root": str(TARGET_ROOT),
    "changed_count": len(changed),
    "changed_files": changed,
}
print(json.dumps(out, indent=2))
