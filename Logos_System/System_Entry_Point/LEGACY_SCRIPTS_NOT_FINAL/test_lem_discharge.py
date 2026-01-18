# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""End-to-end demonstration of the PXL kernel build and proof footprint."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO_ROOT / "Protopraxis" / "formal_verification" / "coq" / "baseline"

ALLOWED_ADMITS: Set[str] = set()

REQUIRED_FULLY_PROVED: Set[str] = {
    "global_bijection",
    "safety_gate",
    "being_knowing_isomorphism",
    "consciousness_emergence",
}

# Omni layer enforcement (modules must compile and key constants must exist)
REQUIRED_OMNI_MODULES: List[str] = [
    "PXL_Triune_Principles",
    "PXL_Omni_Properties",
    "PXL_Omni_Bridges",
    "PXL_OntoGrid",
    "PXL_OntoGrid_OmniHooks",
    "PXL_OmniKernel",
    "PXL_OmniProofs",
]

REQUIRED_OMNI_CONSTANTS: List[str] = [
    "PrincipleOf",
    "SignMind_Bridge",
    "DomainAxisGround",
    "DomainProfile_sound",
    "AxisTarget",
    "AxisTarget_has_Principle",
    "OP1_truth_to_K_box",
    "OP3_present_all_worlds",
    "NB_modal_power",
    "Omni_set_is_coherent",
    "Triune_Principles_Coherent",
    "OntoGrid_Coherent",
    "OntoGrid_NoDrift",
    "OntoGrid_NoDrift_holds",
    "AxisOmniHook",
    "DomainAxisGround_implies_AxisOmniHook",
    "Truth_axis_hook",
    "Goodness_axis_hook",
    "Presence_axis_hook",
    "Power_axis_hook",
]

_STATEMENT_KEYWORDS = ("Theorem", "Lemma", "Corollary", "Proposition")


class CommandFailure(RuntimeError):
    """Raised when a subprocess returns a non-zero exit status."""


def _run_stream(cmd: List[str], cwd: Path | None = None) -> None:
    """Run a command, streaming stdout/stderr directly to the console."""

    display_cwd = cwd if cwd is not None else REPO_ROOT
    print(f"$ {' '.join(cmd)} (cwd={display_cwd})")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise CommandFailure(
            f"Command {' '.join(cmd)} failed with exit code {exc.returncode}"
        ) from exc


def _coqtop_script(vernac: str) -> str:
    """Execute a short Coq script and return the stdout transcript."""

    cmd = [
        "coqtop",
        "-q",
        "-batch",
        "-Q",
        str(BASELINE_DIR),
        "PXL",
    ]
    try:
        completed = subprocess.run(
            cmd,
            input=vernac + "\nQuit.\n",
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise CommandFailure(
            f"Coq batch script failed:\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}"
        ) from exc
    return completed.stdout


def _parse_assumptions(transcript: str) -> List[str]:
    lines: List[str] = []
    capture = False
    for raw in transcript.splitlines():
        line = raw.strip()
        if not line or line.startswith("Coq <"):
            continue
        if line.startswith("Axioms:"):
            capture = True
            continue
        if capture:
            lines.append(line)
    return lines


def _require_vo_modules(modules: Iterable[str]) -> List[Path]:
    missing: List[Path] = []
    for mod in modules:
        vo = BASELINE_DIR / f"{mod}.vo"
        if not vo.exists():
            missing.append(vo)
    return missing


def _check_required_constants(constants: Iterable[str]) -> List[str]:
    script_lines = [
        (
            "From PXL Require Import "
            "PXL_Triune_Principles "
            "PXL_Omni_Properties "
            "PXL_Omni_Bridges "
            "PXL_OntoGrid "
            "PXL_OntoGrid_OmniHooks "
            "PXL_OmniKernel "
            "PXL_OmniProofs "
            "PXL_Kernel_Axioms."
        ),
    ]
    for const in constants:
        script_lines.append(f"Check {const}.")

    script = "\n".join(script_lines)
    try:
        _coqtop_script(script)
        return []
    except CommandFailure as exc:
        return [str(exc)]


_STATEMENT_DECL_RE = re.compile(
    r"^\s*(?:" + "|".join(_STATEMENT_KEYWORDS) + r")\s+([A-Za-z0-9_']+)\b"
)


def _scan_statements(
    paths: Iterable[Path],
) -> Tuple[Dict[str, Path], Set[str]]:
    admitted: Dict[str, Path] = {}
    declared: Set[str] = set()
    for path in paths:
        if not path.is_file() or path.suffix != ".v":
            continue
        current_name: str | None = None
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            match = _STATEMENT_DECL_RE.match(stripped)
            if match:
                current_name = match.group(1)
                declared.add(current_name)
                continue

            if stripped.startswith("(*"):
                continue

            if stripped.startswith("Qed.") or stripped.startswith("Defined."):
                current_name = None
                continue

            if "Admitted." in stripped and current_name is not None:
                admitted[current_name] = path
                current_name = None
    return admitted, declared


def _check_identity_hash(python: Path) -> List[str]:
    script = REPO_ROOT / "scripts" / "compute_identity_hash.py"
    if not script.is_file():
        return ["compute_identity_hash.py missing"]

    cmd = [str(python), str(script)]
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        return []
    except subprocess.CalledProcessError as exc:
        return [f"Identity hash verification failed (exit {exc.returncode})"]


@dataclass
class DemoReport:
    rebuild_success: bool
    trinitarian_assumptions: List[str]
    lem_assumptions: List[str]
    admitted_stubs: List[Path]

    def pretty_print(self) -> None:
        status = "PASS" if self.rebuild_success else "FAIL"
        print("\n=== PXL Kernel Rebuild ===")
        print(f"Overall status: {status}")

        print("\n=== Trinitarian Optimization – extra assumptions ===")
        if self.trinitarian_assumptions:
            for ax in self.trinitarian_assumptions:
                print(f"  {ax}")
        else:
            print("  <none>")

        print("\n=== Internal LEM – extra assumptions ===")
        if self.lem_assumptions:
            for ax in self.lem_assumptions:
                print(f"  {ax}")
        else:
            print("  <none>")

        print("\n=== Residual `Admitted.` stubs in baseline ===")
        if self.admitted_stubs:
            for path in self.admitted_stubs:
                print(f"  {path.relative_to(REPO_ROOT)}")
        else:
            print("  <none>")


def main() -> int:
    baseline_files = list(BASELINE_DIR.glob("*.v"))
    global_bijection = REPO_ROOT / "PXL_Global_Bijection.v"
    if global_bijection.exists():
        baseline_files.append(global_bijection)

    try:
        _run_stream(
            ["coq_makefile", "-f", "_CoqProject", "-o", "CoqMakefile"],
            cwd=REPO_ROOT,
        )
        _run_stream(["make", "-f", "CoqMakefile", "clean"], cwd=REPO_ROOT)
        jobs = os.cpu_count() or 1
        _run_stream(["make", "-f", "CoqMakefile", f"-j{jobs}"], cwd=REPO_ROOT)
        rebuild_success = True
    except CommandFailure as err:
        rebuild_success = False
        print(err, file=sys.stderr)

    trinitarian_script = (
        "From PXL Require Import PXL_Trinitarian_Optimization.\n"
        "Print Assumptions trinitarian_optimization."
    )
    lem_script = (
        "From PXL Require Import PXL_Internal_LEM.\n"
        "Print Assumptions pxl_excluded_middle."
    )

    trinitarian_output = _coqtop_script(trinitarian_script)
    lem_output = _coqtop_script(lem_script)

    trinitarian_assumptions = _parse_assumptions(trinitarian_output)
    lem_assumptions = _parse_assumptions(lem_output)

    missing_vo = _require_vo_modules(REQUIRED_OMNI_MODULES)
    constant_errors = _check_required_constants(REQUIRED_OMNI_CONSTANTS)
    identity_errors = _check_identity_hash(Path(sys.executable))

    admitted_by_name, declared_names = _scan_statements(baseline_files)
    found_admits = set(admitted_by_name.keys())

    unexpected_admits = sorted(found_admits - ALLOWED_ADMITS)
    missing_expected = sorted(ALLOWED_ADMITS - found_admits)
    required_missing = sorted(
        name for name in REQUIRED_FULLY_PROVED if name not in declared_names
    )
    required_admitted = sorted(REQUIRED_FULLY_PROVED & found_admits)

    invariant_ok = True
    if unexpected_admits:
        print("\nERROR: unexpected admitted theorems detected:")
        for name in unexpected_admits:
            path = admitted_by_name.get(name)
            location = path.relative_to(REPO_ROOT) if path else Path("<unknown>")
            print(f"  {name} (in {location})")
        invariant_ok = False

    if missing_expected:
        print("\nERROR: expected admitted theorems missing:")
        for name in missing_expected:
            print(f"  {name}")
        invariant_ok = False

    if required_missing:
        print("\nERROR: required theorems not found in baseline:")
        for name in required_missing:
            print(f"  {name}")
        invariant_ok = False

    if required_admitted:
        print("\nERROR: required theorems incorrectly admitted:")
        for name in required_admitted:
            path = admitted_by_name.get(name)
            location = path.relative_to(REPO_ROOT) if path else Path("<unknown>")
            print(f"  {name} (in {location})")
        invariant_ok = False

    if missing_vo:
        print("\nERROR: required omni modules missing .vo artifacts:")
        for vo in missing_vo:
            print(f"  {vo.relative_to(REPO_ROOT)}")
        invariant_ok = False

    if constant_errors:
        print("\nERROR: required omni constants missing:")
        for err in constant_errors:
            print(f"  {err}")
        invariant_ok = False

    if identity_errors:
        print("\nERROR: identity hash verification failed:")
        for err in identity_errors:
            print(f"  {err}")
        invariant_ok = False

    overall_success = rebuild_success and invariant_ok

    admitted_paths = sorted({path for path in admitted_by_name.values()}, key=str)

    report = DemoReport(
        rebuild_success=overall_success,
        trinitarian_assumptions=trinitarian_assumptions,
        lem_assumptions=lem_assumptions,
        admitted_stubs=admitted_paths,
    )
    report.pretty_print()

    return 0 if overall_success else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
