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
from .lem_logic_kernel import PXLLemLogicKernel, load_kernel
from .onto_lattice import LatticeAxiom, LatticeProperty, OntologicalLattice
from .reflexive_evaluator import ReflexiveSelfEvaluator
from .recursion_engine import AgentSelfReflection, boot_identity, initialize_agent_identity

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