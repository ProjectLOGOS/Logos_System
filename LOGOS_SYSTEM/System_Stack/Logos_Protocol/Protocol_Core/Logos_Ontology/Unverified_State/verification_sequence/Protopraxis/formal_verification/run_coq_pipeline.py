"""Utility to compile the integrated PXL proof suites and enforce the
requirement that only the Law of Excluded Middle (LEM) remains admitted.

The script performs three checks:
1. Compiles the baseline proof packet (v1) using its Makefile.
2. Compiles the meta proof packet v2 (PXLv3_Meta.v).
3. Scans active Coq sources in the protopraxis directory to ensure the single
   intentional `Admitted.` occurs only in `LEM_Admit.v`.

This helper raises `RuntimeError` on failure and is safe to import from the
logic kernel to gate runtime behaviour.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

PROTOPRAXIS_ROOT = Path(__file__).resolve().parent
COQ_ROOT = PROTOPRAXIS_ROOT / "formal_verification" / "coq"


def _run(cmd: list[str], cwd: Path) -> None:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command {} failed in {}\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                " ".join(cmd),
                cwd,
                completed.stdout,
                completed.stderr,
            )
        )


def compile_baseline() -> None:
    baseline_dir = COQ_ROOT / "baseline"
    if not baseline_dir.exists():
        raise RuntimeError(f"Baseline proof directory missing: {baseline_dir}")
    _run(
        ["coq_makefile", "-f", "_CoqProject", "-o", "Makefile.coq"],
        cwd=baseline_dir,
    )
    _run(["make", "-f", "Makefile.coq", "clean"], cwd=baseline_dir)
    _run(["make", "-f", "Makefile.coq", "all"], cwd=baseline_dir)


def compile_latest_meta() -> None:
    latest_dir = COQ_ROOT / "latest"
    target = latest_dir / "PXLv3_Meta.v"
    if not target.exists():
        raise RuntimeError(f"Meta proof target missing: {target}")
    _run(["coqc", str(target.name)], cwd=latest_dir)


def enforce_single_lem_admit() -> None:
    admits = []
    for coq_file in COQ_ROOT.glob("*.v"):
        contents = coq_file.read_text(encoding="utf-8")
        if "Admitted." in contents:
            admits.append(coq_file)
    if len(admits) != 1 or admits[0].name != "LEM_Admit.v":
        raise RuntimeError(
            "Expected exactly one admitted proof (LEM_Admit.v); found: "
            + ", ".join(str(p) for p in admits)
        )


def run_full_pipeline() -> None:
    compile_baseline()
    compile_latest_meta()
    enforce_single_lem_admit()


if __name__ == "__main__":
    run_full_pipeline()
    print("✅ PXL proof pipeline compiled successfully with only LEM admitted.")
