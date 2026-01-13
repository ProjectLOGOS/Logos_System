#!/usr/bin/env python3
"""
DualBijectiveLogicKernel adapter.

This module provides a richer logic kernel used by the SafeConsciousnessEvolution
adapter. It prefers to wrap the existing PXLLemLogicKernel implementation
if available (from Advanced_Reasoning_Protocol/logos_core copy/lem_logic_kernel.py).
If that implementation isn't importable, it falls back to an internal PXL state.

The kernel implements a small AgencyPreconditions helper used by the
consciousness safety adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class PXLState:
    identity: float = 0.0
    non_contradiction: float = 0.0
    excluded_middle: float = 0.0
    coherence: float = 0.0
    truth: float = 0.0
    distinction: float = 0.0
    relation: float = 0.0
    agency: float = 0.0
    existence: float = 0.0
    goodness: float = 0.0


class AgencyPreconditions:
    def __init__(self) -> None:
        self.intentionality = 0.0
        self.normativity = 0.0
        self.continuity_of_identity = 0.0
        self.freedom = 0.0

    def is_agency_emergent(self) -> Tuple[bool, float]:
        score = (self.intentionality + self.normativity + self.continuity_of_identity + self.freedom) / 4.0
        return score > 0.6, score

    def is_consciousness_emergent(self) -> Tuple[bool, float]:
        score = (self.intentionality + self.normativity + self.continuity_of_identity + self.freedom) / 4.0
        return score > 0.7, score


class DualBijectiveLogicKernel:
    def __init__(self, agent_id: str = "agent"):
        self.agent_id = agent_id
        self.pxl_state = PXLState()
        self.agency_preconditions = AgencyPreconditions()

        # Try to wrap the project PXLLemLogicKernel if present
        try:
            # locate lem_logic_kernel.py under Advanced_Reasoning_Protocol
            base = Path(__file__).resolve().parents[1]
            candidate = Path.cwd() / "Advanced_Reasoning_Protocol" / "logos_core copy" / "lem_logic_kernel.py"
            if candidate.exists():
                spec = importlib.util.spec_from_file_location("lem_logic_kernel", str(candidate))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "load_kernel"):
                    try:
                        self._pxl_kernel = mod.load_kernel(agent_id)
                        logger.info("Wrapped PXLLemLogicKernel from %s", candidate)
                    except Exception:
                        self._pxl_kernel = None
                else:
                    self._pxl_kernel = None
            else:
                self._pxl_kernel = None
        except Exception:
            self._pxl_kernel = None

    def initialize_agent_state(self) -> bool:
        # If underlying pxl kernel has init, call it
        if getattr(self, "_pxl_kernel", None) and hasattr(self._pxl_kernel, "ensure_proofs_compiled"):
            try:
                self._pxl_kernel.ensure_proofs_compiled()
            except Exception:
                pass
        return True

    def update_agency_preconditions(self, **kwargs) -> None:
        # update the agency preconditions with provided kwargs
        for k, v in kwargs.items():
            if hasattr(self.agency_preconditions, k):
                setattr(self.agency_preconditions, k, float(v))

    def discharge_LEM_and_get_identity(self) -> str:
        """Convenience to trigger LEM evaluation and return generated identity if present."""
        if getattr(self, "_pxl_kernel", None) and hasattr(self._pxl_kernel, "generate_identity_response"):
            try:
                return self._pxl_kernel.generate_identity_response()
            except Exception:
                return ""
        return ""

