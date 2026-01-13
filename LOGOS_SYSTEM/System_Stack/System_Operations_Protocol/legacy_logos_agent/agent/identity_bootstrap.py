"""Legacy wrapper redirecting to the consolidated Logos recursion engine."""

from __future__ import annotations

from LOGOS_AGI.Logos_Agent.Logos_Core_Recursion_Engine import (
    AgentSelfReflection,
    boot_identity,
    initialize_agent_identity,
)

__all__ = ["AgentSelfReflection", "boot_identity", "initialize_agent_identity"]
