#!/usr/bin/env python3
"""
LOGOS Header Validator
Fail-closed validator for canonical governed headers.
"""

import json
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "Runtime_Module_Header_Schema.json"

def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_header(text: str) -> str:
    text = text.lstrip()
    if not text.startswith('"""'):
        return ""
    return text.split('"""', 2)[1]

def parse_fields(header: str):
    fields = {}
    for line in header.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    return fields

def validate_file(path: Path, schema: dict) -> bool:
    text = path.read_text(encoding="utf-8")
    header = extract_header(text)
    if not header:
        print(f"FAIL: missing header in {path}")
        return False

    fields = parse_fields(header)

    for req in schema["required_fields"]:
        if req not in fields:
            print(f"FAIL: missing field '{req}' in {path}")
            return False

    if schema["field_constraints"]["PROJECT"]["const"] != fields.get("PROJECT"):
        print(f"FAIL: PROJECT mismatch in {path}")
        return False

    if schema["field_constraints"]["PATH"]["must_match_filesystem"]:
        if fields.get("PATH") != str(path):
            print(f"FAIL: PATH mismatch in {path}")
            return False

    return True

def main(paths):
    schema = load_schema()
    ok = True
    for p in paths:
        if p.suffix == ".py":
            ok &= validate_file(p, schema)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    files = [Path(p) for p in sys.argv[1:]]
    if not files:
        files = list(Path(".").rglob("*.py"))
    main(files)
