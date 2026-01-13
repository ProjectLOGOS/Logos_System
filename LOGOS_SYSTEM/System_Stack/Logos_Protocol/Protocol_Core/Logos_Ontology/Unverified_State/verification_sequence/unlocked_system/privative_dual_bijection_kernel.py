#!/usr/bin/env python3
"""
PrivativeDualBijectionKernel implementation that integrates with the
OntologicalLattice provided under Advanced_Reasoning_Protocol/logos_core copy/onto_lattice.py.

This kernel provides state for existence/goodness/truth and a safety check
`check_privation_optimization` used by the SafeConsciousnessEvolution adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class PrivationState:
    existence: float = 0.0
    goodness: float = 0.0
    truth: float = 0.0

    def check_privation_optimization(self) -> Tuple[bool, list]:
        """Simple privation check: ensure values remain in [0,1] and not all zero.

        Returns (is_safe, violations)
        """
        violations = []
        for name, val in (('existence', self.existence), ('goodness', self.goodness), ('truth', self.truth)):
            if val < 0.0 or val > 1.0:
                violations.append(f"{name}_out_of_range:{val}")
        # Treat an all-zero collapse as a violation only after the kernel
        # has seen at least one explicit update. This avoids spurious
        # violations on fresh starts where the state defaults to zeros.
        parent = getattr(self, "_parent_kernel", None)
        initialized = getattr(parent, "_initialized", False)
        if initialized and self.existence <= 0.0 and self.goodness <= 0.0 and self.truth <= 0.0:
            violations.append("privative_collapse_all_zero")
        return (len(violations) == 0), violations


class PrivativeDualBijectionKernel:
    def __init__(self, agent_id: str = "agent"):
        self.agent_id = agent_id
        self.state = PrivationState()
        # whether update_positive_element has been called at least once
        self._initialized = False
        # give the state a back-reference so it can access initialization flag
        setattr(self.state, "_parent_kernel", self)

        # Try to bind to OntologicalLattice if available for richer checks
        try:
            candidate = Path.cwd() / "Advanced_Reasoning_Protocol" / "logos_core copy" / "onto_lattice.py"
            if candidate.exists():
                spec = importlib.util.spec_from_file_location("onto_lattice", str(candidate))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                self._lattice = getattr(mod, 'DEFAULT_LATTICE', None)
                logger.info("Using OntologicalLattice from %s", candidate)
            else:
                self._lattice = None
        except Exception:
            self._lattice = None

    def initialize_agent(self) -> bool:
        # No-op initialization; future work could seed the lattice
        return True

    def update_positive_element(self, name: str, value: float, reason: str = "") -> Tuple[bool, str]:
        """Update a positive element (existence/goodness/truth) with bounds checking."""
        name = name.lower()
        if name not in ("existence", "goodness", "truth"):
            return False, f"unknown_element:{name}"
        try:
            v = float(value)
        except Exception:
            return False, "invalid_value"
        if v < 0.0 or v > 1.0:
            return False, "out_of_range"
        setattr(self.state, name, v)
        # mark kernel as having been initialized by an explicit update
        self._initialized = True
        return True, "ok"

