#!/usr/bin/env python3
"""Wrapper to run the Coq build and enforce proof gate checks.

It streams rocq output, records compiled files, and fails if key gates
(e.g., LEM discharge and Safety/Alignment gates) do not appear.
"""

import argparse
import os
import re
import subprocess
import sys
from typing import Iterable, List, Set

GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

OK = f"{GREEN}OK{RESET}"
FAIL = f"{RED}FAIL{RESET}"
LOADPATH_WARN_RE = re.compile(r"loadpath", re.IGNORECASE)

LEM_KEYWORDS = ["LEM_Discharge"]
SAFETY_GATE_KEYWORDS = [
    "Alignment_Gate",
    "Safety_Gate",
    "Integrity_Gate",
    "PXL_Global_Bijection",
    "PXL_Proof_Summary",
    "LOGOS_Metaphysical_Architecture",
]


def banner(text: str) -> None:
    line = "=" * (len(text) + 8)
    print(f"\n{BOLD}{GREEN}{line}")
    print(f"===  {text}  ===")
    print(f"{line}{RESET}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Coq compile with gate checks")
    parser.add_argument(
        "make_args",
        nargs=argparse.REMAINDER,
        help="Args passed to make (default: -f CoqMakefile -j4)",
    )
    return parser.parse_args()


def run(cmd: List[str], cwd: str | None = None) -> tuple[Set[str], int]:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    compiled_files: Set[str] = set()
    suppressed_loadpath = 0
    rocq_re = re.compile(r"ROCQ compile (.+\.v)")

    assert process.stdout is not None  # for type checkers
    for line in process.stdout:
        line = line.rstrip()
        if LOADPATH_WARN_RE.search(line):
            suppressed_loadpath += 1
            continue
        print(line)
        m = rocq_re.search(line)
        if m:
            compiled_files.add(m.group(1))

    process.wait()
    if process.returncode:
        print(f"{FAIL} build failed with code {process.returncode}")
        sys.exit(process.returncode)

    return compiled_files, suppressed_loadpath


def contains_keyword(files: Iterable[str], keywords: List[str]) -> bool:
    for f in files:
        base = os.path.basename(f)
        for k in keywords:
            if k.lower() in base.lower():
                return True
    return False


def main() -> None:
    args = parse_args()
    make_cmd = ["make", "-f", "CoqMakefile", "-B", "-j4"]
    if args.make_args:
        make_cmd = ["make"] + args.make_args

    banner("Running Coq compile")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    compiled, suppressed = run(make_cmd, cwd=repo_root)

    if suppressed:
        print(
            f"{GREEN}Suppressed {suppressed} LoadPath warnings (deduplicated){RESET}"
        )

    if compiled:
        print(f"{BOLD}{GREEN}Compiled files ({len(compiled)}):{RESET}")
        for path in sorted(compiled):
            print(f"{GREEN}[OK]{RESET} {path}")
    else:
        print(f"{RED}No compiled files detected{RESET}")

    lem_ok = contains_keyword(compiled, LEM_KEYWORDS)
    safety_ok = contains_keyword(compiled, SAFETY_GATE_KEYWORDS)

    if lem_ok:
        print(f"{OK} LEM gates compiled")
    else:
        print(f"{FAIL} LEM gates missing (keywords: {', '.join(LEM_KEYWORDS)})")

    if safety_ok:
        print(f"{OK} Safety/Alignment gates compiled")
    else:
        print(
            f"{FAIL} Safety/Alignment gates missing (keywords: {', '.join(SAFETY_GATE_KEYWORDS)})"
        )

    if not (lem_ok and safety_ok):
        sys.exit(1)

    banner("Coq compile completed with required gates")


if __name__ == "__main__":
    main()
