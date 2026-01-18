# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# ROLE: PROTOCOL_NEXUS
# SUBSYSTEM: ARP
# EXECUTION: FORBIDDEN (DRY_RUN_ONLY)
# AUTHORITY: GOVERNED
# INSTALL_STATUS: DRY_RUN_ONLY
# SOURCE_LEGACY: arp_nexus.py
# NOTES:
# - This is a non-installing dry-run rewrite.
# - No execution surfaces are enabled.
# - This module orchestrates ARP protocol registration only.
# - Planning, reasoning, and execution are delegated elsewhere.

"""
Canonical ARP Nexus (Dry Run)

Purpose:
- Provide a governed orchestration surface for ARP-related protocol interactions.
- Register ARP planners / reasoning components symbolically.
- Enforce strict separation between orchestration and execution.

Non-Goals:
- No execution
- No planning
- No reasoning
- No agent activation
"""

from typing import Dict, Callable, Any


class ARP_Nexus:
    """
    Canonical Nexus orchestrator for the ARP protocol layer.

    This class is a structural rewrite of the legacy ARP Nexus.
    It preserves orchestration intent while eliminating
    execution, reasoning, and side effects.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., Any]] = {}

    def register_planner(self, name: str, handler: Callable[..., Any]) -> None:
        """
        Register an ARP planner or reasoning handler.

        Registration does not imply execution.
        """
        if name in self._registry:
            raise ValueError(f"Planner {name} already registered")
        self._registry[name] = handler

    def list_planners(self) -> Dict[str, Callable[..., Any]]:
        """
        Return a copy of the registered planner map.
        """
        return dict(self._registry)

    def get_planner(self, name: str) -> Callable[..., Any]:
        """
        Retrieve a planner by name.

        Execution responsibility lies outside the Nexus.
        """
        if name not in self._registry:
            raise KeyError(f"Planner {name} not found")
        return self._registry[name]
