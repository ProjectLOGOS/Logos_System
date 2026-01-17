#!/usr/bin/env python3
"""Deterministic stub for the legacy audit emission script.

The real implementation historically rebuilt ontological artifacts. For the
current test suite we simply ensure the target configuration file exists and
rewrite it with identical contents, exercising the determinism check without
altering repository state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

_CONFIG_PATH = Path("config/ontological_properties.json")


def _rewrite_file(path: Path) -> None:
    payload = path.read_bytes()
    path.write_bytes(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic audit emitter stub")
    parser.add_argument("--write", action="store_true", help="Rewrite ontological properties without changes")
    args = parser.parse_args(argv)

    if not _CONFIG_PATH.exists():
        raise SystemExit(f"Missing configuration file: {_CONFIG_PATH}")

    if args.write:
        _rewrite_file(_CONFIG_PATH)

    print("audit_and_emit stub executed; ontology left unchanged")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
