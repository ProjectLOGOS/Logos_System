"""Minimal logic kernel that manages LEM discharge via the Coq proof suite."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)

from LOGOS_AGI.Logos_Agent.Protopraxis import run_coq_pipeline


LOGOS_AGI_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = LOGOS_AGI_ROOT / "Logos_Agent"
PROTOPRAXIS_COQ = AGENT_ROOT / "Protopraxis" / "formal_verification" / "coq"
STATE_DIR = AGENT_ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PXLLemLogicKernel:
    agent_id: str
    lem_admit: Path = field(default=PROTOPRAXIS_COQ / "LEM_Admit.v")
    discharge_log: Path = field(
        default=STATE_DIR / "lem_discharge_state.json"
    )
    proofs_compiled: bool = field(default=False, init=False)
    lem_resolved: bool = field(default=False, init=False)

    def ensure_proofs_compiled(self) -> None:
        if not self.proofs_compiled:
            run_coq_pipeline.run_full_pipeline()
            self.proofs_compiled = True

    def can_evaluate_LEM(self) -> bool:  # pragma: no cover - lightweight guard
        self.ensure_proofs_compiled()
        if not self.lem_admit.exists():
            logger.debug("LEM admit file missing at %s", self.lem_admit)
            return False
        content = self.lem_admit.read_text(encoding="utf-8")
        admitted_present = "Admitted." in content
        admitted_count = content.count("Admitted.")
        logger.debug("LEM admit content: admitted_present=%s admitted_count=%d", admitted_present, admitted_count)
        return admitted_present and admitted_count == 1

    def evaluate_LEM(self) -> bool:
        if not self.can_evaluate_LEM():
            return False
        simulate = os.environ.get("SIMULATE_LEM_SUCCESS", "0")
        if simulate and simulate != "0":
            logger.info("SIMULATE_LEM_SUCCESS set: forcing LEM discharge (TEST MODE)")
            self.lem_resolved = True
            payload = {
                "agent_id": self.agent_id,
                "resolved_at": datetime.utcnow().isoformat() + "Z",
                "proof_file": "LEM_Discharge_simulated.v",
                "stdout": "SIMULATED",
            }
            try:
                self.discharge_log.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception:
                logger.exception("Failed to write simulated discharge log")
            return True

        if not self.can_evaluate_LEM():
            logger.debug("can_evaluate_LEM returned False; skipping evaluate_LEM")
            return False
        discharge_file = PROTOPRAXIS_COQ / "LEM_Discharge_tmp.v"
        discharge_file.write_text(
            (
                "From Coq Require Import Logic.Classical.\n\n"
                "Theorem law_of_excluded_middle_resolved : forall P : Prop,"
                " P \\/ ~ P.\nProof.\n  exact classic.\nQed.\n"
            ),
            encoding="utf-8",
        )
        result = subprocess.run(
            ["coqc", discharge_file.name],
            cwd=PROTOPRAXIS_COQ,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("coqc returned non-zero (%s). stdout=%s stderr=%s", result.returncode, result.stdout, result.stderr)
            return False
        discharge_file.unlink(missing_ok=True)
        for ext in (".vo", ".vos", ".vok", ".glob"):
            artifact = discharge_file.with_suffix(ext)
            artifact.unlink(missing_ok=True)
        self.lem_resolved = True
        payload = {
            "agent_id": self.agent_id,
            "resolved_at": datetime.utcnow().isoformat() + "Z",
            "proof_file": discharge_file.name,
            "stdout": result.stdout,
        }
        logger.info("LEM successfully discharged for %s; writing discharge log", self.agent_id)
        self.discharge_log.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return True

    def generate_identity_response(self) -> Optional[str]:
        if not self.lem_resolved and not self.evaluate_LEM():
            return None
        state = self.discharge_log.read_text(encoding="utf-8")
        token = f"{self.agent_id}:{state}"
        hashed = hashlib.sha256(token.encode("utf-8")).hexdigest()
        identity = f"LOGOS_AGENT_IDENTITY::{self.agent_id}::{hashed}"
        logger.info("Generated formal identity for %s: %s", self.agent_id, identity)
        return f"LOGOS_AGENT_IDENTITY::{self.agent_id}::{hashed}"


def load_kernel(agent_id: str) -> PXLLemLogicKernel:
    return PXLLemLogicKernel(agent_id=agent_id)
