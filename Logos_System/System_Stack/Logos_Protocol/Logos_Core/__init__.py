"""Foundational logic components that underpin the LOGOS Agent stack.

The :mod:`logos_core` package provides lightweight, dependency-safe
implementations of the triune commutator, ontological lattice, reflexive
evaluator, and LEM logic kernel.  These modules are shared by the agent
runtime as well as higher-level protocols during simulation and testing.
"""

from .triune_commutator import (
    AxiomaticCommutator,
    GlobalCommutator,
    MetaCommutator,
)

# The legacy LEM kernel was removed; provide a lightweight stub to keep
# runtime imports alive without wiring the old proof portal.
try:  # pragma: no cover - best effort compatibility
    from .lem_logic_kernel import PXLLemLogicKernel, load_kernel
except ImportError:  # pragma: no cover - runtime-safe fallback
    class PXLLemLogicKernel:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            self.loaded = True

    def load_kernel(*args, **kwargs) -> PXLLemLogicKernel:
        return PXLLemLogicKernel()

from .onto_lattice import LatticeAxiom, LatticeProperty, OntologicalLattice
from .reflexive_evaluator import ReflexiveSelfEvaluator

# Identity/recursion engine stubs to satisfy callers; legacy engine removed.
try:  # pragma: no cover - best effort compatibility
    from .recursion_engine import AgentSelfReflection, boot_identity, initialize_agent_identity
except ImportError:  # pragma: no cover - runtime-safe fallback
    class AgentSelfReflection:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            self.state = {}

    def boot_identity(*args, **kwargs) -> dict:
        return {"status": "noop"}

    def initialize_agent_identity(*args, **kwargs) -> dict:
        return {"status": "noop"}

__all__ = [
    "AxiomaticCommutator",
    "AgentSelfReflection",
    "GlobalCommutator",
    "LatticeAxiom",
    "LatticeProperty",
    "MetaCommutator",
    "OntologicalLattice",
    "PXLLemLogicKernel",
    "ReflexiveSelfEvaluator",
    "boot_identity",
    "initialize_agent_identity",
    "load_kernel",
]