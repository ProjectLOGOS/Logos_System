# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Compatibility facade for initializing the lightweight LOGOS core."""

from __future__ import annotations

from typing import Any, Dict, Optional

from Logos_System.System_Stack.System_Operations_Protocol.governance.enhanced_reference_monitor import (
    EnhancedReferenceMonitor,
    IELEvaluator,
    ModalLogicEvaluator,
)


class LogosCoreFacade:
    """Small facade that mirrors the legacy LOGOS core surface area."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.monitor = EnhancedReferenceMonitor(self.config)
        # Allow tests to patch entry-level evaluator classes while keeping monitor lean
        self.monitor.modal_evaluator = ModalLogicEvaluator()
        self.monitor.iel_evaluator = IELEvaluator()
        self._initialized = True

    def evaluate_modal(self, proposition: str):
        return self.monitor.evaluate_modal_proposition(proposition)

    def evaluate_iel(self, proposition: str):
        return self.monitor.evaluate_iel_proposition(proposition)

    def get_status(self) -> Dict[str, Any]:
        return {
            "logos_core": {"initialized": self._initialized},
            "reference_monitor": self.monitor.get_monitor_status(),
        }


_GLOBAL_CORE: Optional[LogosCoreFacade] = None


def initialize_logos_core(config: Optional[Dict[str, Any]] = None) -> LogosCoreFacade:
    """Initialize the lightweight LOGOS core facade."""

    global _GLOBAL_CORE
    _GLOBAL_CORE = LogosCoreFacade(config)
    return _GLOBAL_CORE


def get_logos_core() -> LogosCoreFacade:
    """Return the singleton LOGOS core instance, creating it if required."""

    global _GLOBAL_CORE
    if _GLOBAL_CORE is None:
        _GLOBAL_CORE = LogosCoreFacade()
    return _GLOBAL_CORE


def evaluate_modal(proposition: str):
    """Convenience wrapper for modal proposition evaluation."""

    return get_logos_core().evaluate_modal(proposition)


def evaluate_iel(proposition: str):
    """Convenience wrapper for IEL proposition evaluation."""

    return get_logos_core().evaluate_iel(proposition)


def get_status() -> Dict[str, Any]:
    """Return a composite status report for the lightweight core."""

    return get_logos_core().get_status()


def start_agent():
    return "LOGOS_AGENT_STUB"


__all__ = [
    "EnhancedReferenceMonitor",
    "IELEvaluator",
    "ModalLogicEvaluator",
    "LogosCoreFacade",
    "initialize_logos_core",
    "get_logos_core",
    "evaluate_modal",
    "evaluate_iel",
    "get_status",
    "start_agent",
]
