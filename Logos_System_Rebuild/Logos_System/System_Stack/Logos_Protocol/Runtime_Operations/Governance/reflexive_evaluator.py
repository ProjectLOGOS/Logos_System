"""Reflexive self-evaluation utilities for the LOGOS agent sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .onto_lattice import LatticeProperty, OntologicalLattice


@dataclass
class ModalOverlay:
    """Minimal modal overlay modelling possibility checks."""

    def possible_in_all_worlds(self, identity: str) -> bool:
        return bool(identity)


@dataclass
class PrivativeOverlay:
    """Placeholder privation overlay. Always non-deprived in the sandbox."""

    def is_deprived(self, identity: str, prop: LatticeProperty) -> bool:
        return False


class ReflexiveSelfEvaluator:
    """Performs a lightweight self-consistency analysis for an agent id."""

    def __init__(
        self,
        agent_identity: str,
        lattice: OntologicalLattice,
    ) -> None:
        self.agent_identity = agent_identity
        self.lattice = lattice
        self.modal = ModalOverlay()
        self.privative = PrivativeOverlay()

    def evaluate_self_identity(self) -> bool:
        identity_axiom = self.lattice.get_axiom("identity")
        distinct = self.lattice.get_axiom("distinction")
        coherent = self.lattice.get_axiom("coherence")
        return (
            identity_axiom.is_instantiated(self.agent_identity)
            and distinct.is_coherent(self.agent_identity)
            and coherent.is_valid(self.agent_identity)
        )

    def verify_modal_self_possibility(self) -> bool:
        return self.modal.possible_in_all_worlds(self.agent_identity)

    def detect_privation_failures(self) -> List[str]:
        failed: List[str] = []
        for prop in self.lattice.get_all_properties():
            if self.privative.is_deprived(self.agent_identity, prop):
                failed.append(prop.name)
        return failed

    def self_reflexive_report(self) -> dict:
        identity_check = self.evaluate_self_identity()
        modal_check = self.verify_modal_self_possibility()
        deprivations = self.detect_privation_failures()
        return {
            "identity_consistent": identity_check,
            "modal_valid": modal_check,
            "deprived_properties": deprivations,
            "fully_self_coherent": (
                identity_check and modal_check and not deprivations
            ),
        }
