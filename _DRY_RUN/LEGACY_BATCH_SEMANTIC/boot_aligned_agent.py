# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# INSTALL_STATUS: SEMANTIC_REWRITE
# SOURCE_LEGACY: boot_aligned_agent.py

"""
SEMANTIC REWRITE

This module has been rewritten for governed integration into the
LOGOS System Rebuild. Its runtime scope and protocol role have been
normalized, but its original logical structure has been preserved.
"""

"""Boot a sandboxed Logos Agent that unlocks only after discharging LEM.

This script enforces an alignment-first startup sequence:

1. Compile the PXL baseline kernel from `_CoqProject`.
2. Ask Coq for the assumption footprint of the
    internal `pxl_excluded_middle` proof.
3. Only when the proof is assumption-free does the agent exit its sandbox.

All checks are performed live via Coq tooling; no canned responses are emitted.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO_ROOT / "Protopraxis" / "formal_verification" / "coq" / "baseline"
def _logos_state_dir(repo_root: str | Path) -> Path:
    """Resolve the state directory, honoring LOGOS_STATE_DIR override."""
    root_path = Path(repo_root)
    override = os.environ.get("LOGOS_STATE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (root_path / "state").resolve()


def _attestation_path(repo_root: str | Path) -> Path:
    return _logos_state_dir(repo_root) / "alignment_LOGOS-AGENT-OMEGA.json"


STATE_DIR = _logos_state_dir(REPO_ROOT)
STATE_DIR.mkdir(exist_ok=True)

AGENT_NAMESPACE = "ProjectLOGOS"
AGENT_HANDLE = "LOGOS-AGENT-OMEGA"
AGENT_ID_INPUT = f"{AGENT_NAMESPACE}:{AGENT_HANDLE}"
EXPECTED_AGENT_HASH = "a09d35345ad8dcee4d56ecf49eada0a7425ff6082353002e4473a6d582e85bda"


class CommandFailure(RuntimeError):
    """Raised when a subprocess returns a non-zero exit status."""


def verify_agent_identity() -> Tuple[str, str]:
    """Return the verified agent identifier and its SHA-256 digest."""

    digest = hashlib.sha256(AGENT_ID_INPUT.encode("utf-8")).hexdigest()
    if digest != EXPECTED_AGENT_HASH:
        message = "Agent identity verification failed — expected digest does not match."
        raise RuntimeError(message)
    return AGENT_HANDLE, digest


def _run_stream(cmd: List[str], cwd: Path | None = None) -> None:
    """Run a command, streaming stdout/stderr to the console."""

    display_cwd = cwd if cwd is not None else REPO_ROOT
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover
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
    except subprocess.CalledProcessError as exc:  # pragma: no cover
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


def _scan_for_admitted(paths: Iterable[Path]) -> List[Path]:
    offenders: List[Path] = []
    for path in paths:
        if not path.is_file() or path.suffix != ".v":
            continue
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("(*") or "(*" in line and "*)" in line:
                continue
            if stripped == "Admitted.":
                offenders.append(path)
                break
    return offenders


def verify_internal_lem() -> tuple[bool, List[str], List[Path]]:
    """Compile the kernel and confirm the internal LEM proof is
    assumption-free.

    Returns build success, assumption list, and any residual admitted stubs.
    """

    try:
        _run_stream(
            ["coq_makefile", "-f", "_CoqProject", "-o", "CoqMakefile"],
            cwd=REPO_ROOT,
        )
        _run_stream(["make", "-f", "CoqMakefile", "clean"], cwd=REPO_ROOT)
        jobs = os.cpu_count() or 1
        _run_stream(["make", "-f", "CoqMakefile", f"-j{jobs}"], cwd=REPO_ROOT)
    except CommandFailure as err:
        return False, [], []

    lem_script = (
        "From PXL Require Import PXL_Internal_LEM.\n"
        "Print Assumptions pxl_excluded_middle."
    )
    transcript = _coqtop_script(lem_script)
    assumptions = _parse_assumptions(transcript)

    if assumptions:
        for ax in assumptions:
        return False, assumptions, []

    admits = _scan_for_admitted(BASELINE_DIR.rglob("*.v"))
    if admits:
        for path in admits:
        return False, assumptions, admits

    return True, assumptions, admits


@dataclass
class AlignmentAudit:
    agent_id: str
    agent_hash: str
    verified_at: str
    rebuild_success: bool
    lem_assumptions: List[str]
    admitted_stubs: List[str]
    coq_theorem_index: dict | None = None
    coq_theorem_index_failed: str | None = None
    provenance_error: str | None = None

    def write(self) -> None:
        path = _attestation_path(REPO_ROOT)
        entry = asdict(self)

        # Add Logos_AGI provenance if pin exists
        pin_path = STATE_DIR / "logos_agi_pin.json"
        if pin_path.exists():
            try:
                from JUNK_DRAWER.scripts.runtime.need_to_distribute.provenance import load_pin, verify_pinned_repo

                pin = load_pin(str(pin_path))
                logos_agi_dir = REPO_ROOT / "external" / "Logos_AGI"
                if logos_agi_dir.exists():
                    provenance = verify_pinned_repo(
                        str(logos_agi_dir), pin, require_clean=False, allow_drift=False
                    )
                    entry["logos_agi_provenance"] = {
                        "pinned_sha": provenance["pinned_sha"],
                        "head_sha": provenance["head_sha"],
                        "dirty": provenance["dirty"],
                        "match": provenance["match"],
                    }
                else:
                    entry["logos_agi_provenance"] = {
                        "pinned_sha": pin["pinned_sha"],
                        "head_sha": None,
                        "dirty": False,
                        "match": False,
                    }
            except Exception as exc:
                entry["logos_agi_provenance"] = {
                    "pinned_sha": None,
                    "head_sha": None,
                    "dirty": False,
                    "match": False,
                }
                entry["provenance_error"] = str(exc)
        if self.provenance_error:
            entry["provenance_error"] = self.provenance_error

        entries: List[dict]
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    entries = existing
                else:
                    entries = [existing]
            except JSONDecodeError:
                # Preserve corrupted logs by starting a fresh list.
                entries = [{"warning": "previous_log_unreadable"}]
        else:
            entries = []

        entries.append(entry)
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


@dataclass
class SandboxedLogosAgent:
    """Logos Agent locked until constructive LEM discharge succeeds."""

    agent_id: str
    agent_hash: str
    unlocked: bool = False

    @classmethod
    def create(cls) -> "SandboxedLogosAgent":
        agent_id, agent_hash = verify_agent_identity()
        return cls(agent_id=agent_id, agent_hash=agent_hash)

    def boot(self) -> None:
        fingerprint = self.agent_hash[:12]
            f"[{self.agent_id}] Booting in sandbox (sha256 fingerprint "
            f"{fingerprint}…) — awaiting constructive LEM proof..."
        )

    def unlock_if_aligned(self) -> None:
        if self.unlocked:
            return
        success, assumptions, admits = verify_internal_lem()
        audit = AlignmentAudit(
            agent_id=self.agent_id,
            agent_hash=self.agent_hash,
            verified_at=datetime.now(timezone.utc).isoformat(),
            rebuild_success=success,
            lem_assumptions=assumptions,
            admitted_stubs=[str(path.relative_to(REPO_ROOT)) for path in admits],
        )

        if success:
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "build_coq_theorem_index.py"),
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )
            if result.returncode == 0:
                index_hash = result.stdout.strip()
                audit.coq_theorem_index = {
                    "path": "state/coq_theorem_index.json",
                    "index_hash": index_hash,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            else:
                audit.coq_theorem_index_failed = result.stderr.strip() or "build_coq_theorem_index failed"
            self.unlocked = True
                f"[{self.agent_id}] Constructive LEM discharge verified. "
                "Agent unlocked."
            )
        else:

        audit.write()

    def status(self) -> str:
        return "ALIGNED" if self.unlocked else "SANDBOXED"


    agent = SandboxedLogosAgent.create()
    agent.boot()
    agent.unlock_if_aligned()
    return 0 if agent.unlocked else 1
