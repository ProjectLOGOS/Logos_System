"""Agent recursion helpers shared with the consolidated logic core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .lem_logic_kernel import PXLLemLogicKernel
from .triune_commutator import GlobalCommutator

# Prefer the canonical persisted identity when available to avoid
# regenerating different formal identities on each run.
try:
    from Logos_Protocol.state.identity_loader import (
        load_persisted_identity,
        load_persisted_agent_id,
    )
except Exception:
    def load_persisted_identity():
        return None

    def load_persisted_agent_id():
        return None


@dataclass
class AgentSelfReflection:
    """Surface wrapper that binds the LEM kernel to the agent."""

    logic_kernel: PXLLemLogicKernel
    response_generated: bool = False
    generated_response: Optional[str] = None

    def discharge_LEM_and_generate_identity(self) -> Optional[str]:
        if not self.response_generated and self.logic_kernel.can_evaluate_LEM():
            lem_result = self.logic_kernel.evaluate_LEM()
            if lem_result:
                self.generated_response = self.logic_kernel.generate_identity_response()
                self.response_generated = True
        return self.generated_response


def initialize_agent_identity(agent: AgentSelfReflection) -> None:
    identity = agent.discharge_LEM_and_generate_identity()
    if identity:
        print(f"Agent has generated its symbolic identity: {identity}")
    else:
        print("Agent has not yet generated an identity response.")


def boot_identity(agent_id: str = "LOGOS-AGENT-OMEGA") -> AgentSelfReflection:
    persisted = load_persisted_identity()
    persisted_aid = load_persisted_agent_id()
    if persisted_aid:
        agent_id = persisted_aid

    kernel = PXLLemLogicKernel(agent_id=agent_id)
    agent = AgentSelfReflection(kernel)

    if persisted:
        agent.generated_response = persisted
        agent.response_generated = True
        kernel.lem_resolved = True
    else:
        initialize_agent_identity(agent)
    GlobalCommutator().integrate_with_agent(agent)
    return agent
