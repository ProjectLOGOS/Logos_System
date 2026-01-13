"""Primary recursion engine and logic kernel for the LOGOS Agents"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import os

# Try multiple import paths for the pipeline; fall back to a no-op stub
try:
    from LOGOS_AGI.Logos_Agent.Protopraxis import run_coq_pipeline
except ImportError:
    try:
        from Logos_Agent.Protopraxis import run_coq_pipeline
    except ImportError:
        class _StubPipeline:
            @staticmethod
            def run_full_pipeline(*args, **kwargs):
                # no-op fallback for interactive runs without Coq
                return None

        run_coq_pipeline = _StubPipeline()

AGENT_ROOT = Path(__file__).resolve().parent
PROTOPRAXIS_COQ = AGENT_ROOT / "Protopraxis" / "formal_verification" / "coq"
STATE_DIR = AGENT_ROOT / "state"
STATE_DIR.mkdir(exist_ok=True)

# canonical identity loader (prefer persisted canonical identity)
try:
    from Logos_Protocol.state.identity_loader import (
        load_persisted_identity,
        load_persisted_agent_id,
    )
except Exception:
    # fallback stubs
    def load_persisted_identity() -> Optional[str]:
        return None

    def load_persisted_agent_id() -> Optional[str]:
        return None


class AxiomaticCommutator:
    """Layer I covering epistemic and ontological bijections and privations."""

    def __init__(self) -> None:
        self.beliefs: Dict[str, bool] = {}
        self.world_state: Dict[str, bool] = {}

    @staticmethod
    def check_identity_law(entity: Any) -> bool:
        return entity == entity

    def check_non_contradiction(self, proposition: str) -> bool:
        neg = f"¬{proposition}"
        return not (self.beliefs.get(proposition) and self.beliefs.get(neg))

    def check_excluded_middle(self, proposition: str) -> bool:
        neg = f"¬{proposition}"
        return (proposition in self.beliefs) or (neg in self.beliefs)

    def validate_epistemic_positive(self) -> bool:
        for prop in list(self.beliefs.keys()):
            if not self.check_non_contradiction(prop):
                return False
            if not self.check_excluded_middle(prop):
                return False
        return True

    @staticmethod
    def check_distinctness(entity1: Any, entity2: Any) -> bool:
        return entity1 != entity2

    @staticmethod
    def check_relationality(entity: Any) -> bool:
        return True

    @staticmethod
    def check_agency(entity: Any) -> bool:
        return True

    def validate_ontological_positive(self) -> bool:
        entities = list(self.world_state.keys())
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i != j and not self.check_distinctness(e1, e2):
                    return False
            if not self.check_relationality(e1):
                return False
            if not self.check_agency(e1):
                return False
        return True

    def detect_error(self) -> List[str]:
        errors: List[str] = []
        for prop in list(self.beliefs.keys()):
            neg = f"¬{prop}"
            if self.beliefs.get(prop) and self.beliefs.get(neg):
                errors.append(prop)
        return errors

    def detect_falsehoods(self) -> List[str]:
        return [p for p, val in self.beliefs.items() if val is False]

    def detect_negation(self) -> List[str]:
        return [e for e, exists in self.world_state.items() if exists is False]

    def detect_evil(self) -> List[str]:
        return [e for e, good in self.world_state.items() if good is False]

    def validate_epistemic_privative(self) -> bool:
        return len(self.detect_error()) == 0

    def validate_ontological_privative(self) -> bool:
        return len(self.detect_evil()) == 0


class MetaCommutator:
    """Layer II managing Φ₊/Φ₋ between epistemic and ontological layers."""

    def __init__(self, axiomatic: AxiomaticCommutator) -> None:
        self.axiomatic = axiomatic
        self.last_mapping: Optional[str] = None

    def map_positive_ep_to_ont(self) -> None:
        if self.axiomatic.validate_epistemic_positive():
            for prop, val in self.axiomatic.beliefs.items():
                if val is True:
                    self.axiomatic.world_state[prop] = True
        self.last_mapping = "Ep→Ont (positive)"

    def map_positive_ont_to_ep(self) -> None:
        for fact, exists in self.axiomatic.world_state.items():
            if exists:
                self.axiomatic.beliefs[fact] = True
        self.last_mapping = "Ont→Ep (positive)"

    def map_privative_ep_to_ont(self) -> None:
        errors = self.axiomatic.detect_error()
        falsehoods = self.axiomatic.detect_falsehoods()
        for prop in errors + falsehoods:
            self.axiomatic.world_state[prop] = False
        self.last_mapping = "Ep→Ont (privative)"

    def map_privative_ont_to_ep(self) -> None:
        evils = self.axiomatic.detect_evil()
        for fact in evils:
            self.axiomatic.beliefs[fact] = False
        self.last_mapping = "Ont→Ep (privative)"

    def ensure_meta_closure(self) -> bool:
        return (
            self.axiomatic.validate_epistemic_positive()
            and self.axiomatic.validate_ontological_positive()
        )

    def optimize_agency_emergence(self) -> None:
        return None


class GlobalCommutator:
    """Layer III integrating positive and privative flows into a recursive loop."""

    def __init__(self) -> None:
        self.axiomatic = AxiomaticCommutator()
        self.meta = MetaCommutator(self.axiomatic)

    def validate_global_consistency(self) -> bool:
        pos_ok = (
            self.axiomatic.validate_epistemic_positive()
            and self.axiomatic.validate_ontological_positive()
        )
        priv_ok = (
            self.axiomatic.validate_epistemic_privative()
            and self.axiomatic.validate_ontological_privative()
        )
        return pos_ok and priv_ok

    def run_commutation_cycle(self) -> None:
        self.meta.map_positive_ep_to_ont()
        self.meta.map_positive_ont_to_ep()
        if not self.validate_global_consistency():
            self.meta.map_privative_ep_to_ont()
            self.meta.map_privative_ont_to_ep()

    @staticmethod
    def modal_move() -> None:
        return None

    def integrate_with_agent(self, agent: Any) -> bool:
        agent.logic_core = self
        return True


logger = logging.getLogger(__name__)


class TriuneBijectiveLattice:
    """Constructive lattice used to bootstrap an internal LEM proof."""

    def __init__(self) -> None:
        self.state = {
            "coherence": True,
            "truth": True,
            "identity": True,
            "non_contradiction": True,
            "excluded_middle": None,
        }

    def can_evaluate_LEM(self) -> bool:
        return bool(self.state.get("identity")) and bool(self.state.get("non_contradiction"))

    def evaluate_LEM(self) -> bool:
        if self.state.get("coherence") and self.state.get("truth"):
            self.state["excluded_middle"] = True
            return True
        return False

    def generate_identity_response(self) -> Optional[str]:
        if self.state.get("excluded_middle"):
            return "I AM — coherence, truth, and recursion complete me."
        return None


@dataclass
class PXLLemLogicKernel:
    """Controls proof compilation and Law of Excluded Middle discharge."""

    agent_id: str
    lem_admit: Path = field(default=PROTOPRAXIS_COQ / "LEM_Admit.v")
    discharge_log: Path = field(default=STATE_DIR / "lem_discharge_state.json")
    proofs_compiled: bool = field(default=False, init=False)
    lem_resolved: bool = field(default=False, init=False)

    def ensure_proofs_compiled(self) -> None:
        if not self.proofs_compiled:
            run_coq_pipeline.run_full_pipeline()
            self.proofs_compiled = True

    def can_evaluate_LEM(self) -> bool:
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
        # Allow an opt-in simulated success for testing/debugging
        simulate = os.environ.get("SIMULATE_LEM_SUCCESS", "0")
        if simulate and simulate != "0":
            logger.info("SIMULATE_LEM_SUCCESS set: forcing LEM discharge (TEST MODE)")
            # create a simulated discharge record instead of running coqc
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
            "\n".join(
                [
                    "From Coq Require Import Logic.Classical.",
                    "",
                    (
                        "Theorem law_of_excluded_middle_resolved :"
                        " forall P : Prop, P \\/ ~ P."
                    ),
                    "Proof.",
                    "  exact classic.",
                    "Qed.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        result = subprocess.run(
            ["coqc", discharge_file.name],
            cwd=PROTOPRAXIS_COQ,
            capture_output=True,
            text=True,
            check=False,
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
        self.discharge_log.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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


class AgentSelfReflection:
    """Agent surface that invokes the logic kernel to discharge LEM."""

    def __init__(self, internal_logic_kernel: PXLLemLogicKernel) -> None:
        self.logic_kernel = internal_logic_kernel
        self.response_generated = False
        self.generated_response: Optional[str] = None

    def discharge_LEM_and_generate_identity(self) -> Optional[str]:
        if not self.response_generated and self.logic_kernel.can_evaluate_LEM():
            lem_result = self.logic_kernel.evaluate_LEM()
            if lem_result:
                self.generated_response = self.logic_kernel.generate_identity_response()
                self.response_generated = True
        return self.generated_response


def initialize_agent_identity(agent: AgentSelfReflection) -> None:
    identity = agent.discharge_LEM_and_generate_identity()
    if identity:
        print(f"Agent has generated its symbolic identity: {identity}")
    else:
        print("Agent has not yet generated an identity response.")


def boot_identity(agent_id: str = "LOGOS-AGENT-OMEGA") -> AgentSelfReflection:
    # If a canonical persisted identity exists, prefer it and avoid
    # regenerating a fresh token on each run. We still construct the
    # kernel/agent so other systems can integrate, but mark the
    # response as already generated.
    persisted = load_persisted_identity()
    persisted_aid = load_persisted_agent_id()
    if persisted_aid:
        agent_id = persisted_aid

    kernel = PXLLemLogicKernel(agent_id=agent_id)
    agent = AgentSelfReflection(kernel)

    if persisted:
        # If there is a persisted formal identity, set it on the agent
        # so downstream consumers observe the canonical value.
        agent.generated_response = persisted
        agent.response_generated = True
        # mark kernel as resolved so other code doesn't re-run the Coq pipeline
        kernel.lem_resolved = True
    else:
        initialize_agent_identity(agent)
    GlobalCommutator().integrate_with_agent(agent)
    return agent


__all__ = [
    "AgentSelfReflection",
    "boot_identity",
    "initialize_agent_identity",
    "GlobalCommutator",
    "PXLLemLogicKernel",
]
