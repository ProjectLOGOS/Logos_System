#!/usr/bin/env python3
"""
LOGOS Header Injector
Injects canonical headers based on path and phase metadata.
"""

from pathlib import Path
import sys

HEADER_TEMPLATE = '''"""
===============================================================================
FILE: {file}
PATH: {path}
PROJECT: LOGOS System
PHASE: {phase}
STEP: {step}
STATUS: GOVERNED — NON-BYPASSABLE

CLASSIFICATION:
- {classification}

GOVERNANCE:
- {governance}

ROLE:
{role}

ORDERING GUARANTEE:
{ordering}

PROHIBITIONS:
{prohibitions}

FAILURE SEMANTICS:
{failure}

===============================================================================
"""
'''

def inject(path: Path, meta: dict):
    body = path.read_text(encoding="utf-8")
    if body.lstrip().startswith('"""'):
        return
    header = HEADER_TEMPLATE.format(**meta)
    path.write_text(header + "\n" + body, encoding="utf-8")

def main():
    # Minimal stub: injector is driven by rewrite prompts
    print("Injector ready. Use via rewrite prompts with explicit metadata.")

if __name__ == "__main__":
    main()
