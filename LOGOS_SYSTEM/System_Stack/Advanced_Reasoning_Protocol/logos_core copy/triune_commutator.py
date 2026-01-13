"""Triune commutator implementing axiomatic, meta, and global logic layers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class AxiomaticCommutator:
    """Layer I covering epistemic and ontological bijections and privations."""

    def __init__(self) -> None:
        self.beliefs: Dict[str, bool] = {}
        self.world_state: Dict[str, bool] = {}

    # Epistemic positive laws -------------------------------------------------
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

    # Ontological positive principles -----------------------------------------
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

    # Privative detections ------------------------------------------------
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
    """Layer III integrating positive and privative flows into a
    recursive loop."""

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
