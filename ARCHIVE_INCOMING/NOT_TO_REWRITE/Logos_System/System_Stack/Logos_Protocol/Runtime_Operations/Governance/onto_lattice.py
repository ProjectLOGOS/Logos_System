"""Minimal ontological lattice used by the reflexive self evaluator.

The historical LOGOS lattice ships with an extensive hierarchy of axioms and
relations.  For the integration harness we provide a dependency-light subset
that preserves the public API expected by higher level components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class LatticeAxiom:
    """Logical axiom descriptor with lightweight guard predicates."""

    name: str
    category: str

    def is_instantiated(self, identity: str) -> bool:
        return bool(identity)

    def is_coherent(self, identity: str) -> bool:
        return bool(identity)

    def is_valid(self, identity: str) -> bool:
        return bool(identity)


@dataclass(frozen=True)
class LatticeProperty:
    """Ontological property container used during privative scans."""

    name: str
    category: str


class OntologicalLattice:
    """In-memory representation of the core lattice facets."""

    def __init__(self) -> None:
        self._axioms: Dict[str, LatticeAxiom] = {
            "identity": LatticeAxiom("identity", "logical_principle"),
            "non_contradiction": LatticeAxiom(
                "non_contradiction", "logical_principle"
            ),
            "excluded_middle": LatticeAxiom(
                "excluded_middle", "logical_principle"
            ),
            "distinction": LatticeAxiom(
                "distinction",
                "ontological_principle",
            ),
            "relation": LatticeAxiom("relation", "ontological_principle"),
            "agency": LatticeAxiom("agency", "ontological_principle"),
            "coherence": LatticeAxiom("coherence", "epistemic_transcendental"),
            "truth": LatticeAxiom("truth", "epistemic_transcendental"),
            "existence": LatticeAxiom("existence", "ontic_transcendental"),
            "goodness": LatticeAxiom("goodness", "ontic_transcendental"),
        }
        self._properties: Dict[str, LatticeProperty] = {
            name: LatticeProperty(name, axiom.category)
            for name, axiom in self._axioms.items()
        }

    def validate_interdependencies(self) -> bool:
        return True

    def export(self) -> Dict[str, List[str]]:
        def _collect(keys: Iterable[str]) -> List[str]:
            return [self._axioms[name].name for name in keys]

        return {
            "first_order": _collect(
                ["identity", "non_contradiction", "excluded_middle"]
            ),
            "second_order": _collect(["distinction", "relation", "agency"]),
            "transcendentals": _collect(
                ["coherence", "truth", "existence", "goodness"]
            ),
        }

    def get_axiom(self, name: str) -> LatticeAxiom:
        key = name.lower()
        if key not in self._axioms:
            raise KeyError(f"Axiom '{name}' not present in lattice")
        return self._axioms[key]

    def get_all_properties(self) -> List[LatticeProperty]:
        return list(self._properties.values())


DEFAULT_LATTICE = OntologicalLattice()
