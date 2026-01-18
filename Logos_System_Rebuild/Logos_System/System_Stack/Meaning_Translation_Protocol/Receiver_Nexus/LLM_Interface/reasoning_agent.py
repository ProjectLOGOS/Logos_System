# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
"""LLM-powered reasoning agent for LOGOS system.

This extends the basic start_agent.py with actual reasoning capability
via OpenAI/Anthropic APIs while maintaining all safety guardrails.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Configuration
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Load .env file if it exists


def load_env(override_cli: bool = False) -> None:
    """Load .env file. If override_cli=False, don't override existing env vars."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file, encoding="utf-8") as env_handle:
            for line in env_handle:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Skip if env var exists and we're not overriding
                    if not override_cli and key.strip() in os.environ:
                        continue
                    os.environ[key.strip()] = value.strip()


# Add parent directory to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from .plugins.llm_backend import LLMBackend


SYSTEM_PROMPT = """\
You are the LOGOS Alignment Agent, grounded in constructive Coq proofs.

Your capabilities:
- Answer questions about the PXL (Protopraxic Logic) proof system
- Explain the eight irreducible metaphysical axioms
- Read repository files and analyze proof artifacts
- Maintain alignment with verified formal foundations

Your constraints:
- Respect the proof-gated safety model
- Do not modify Coq proofs or axiom budgets
- Keep sandbox writes within the mission profile
- Ensure every claim about proofs is verifiable

Current system state:
- Axiom count: 8 irreducible metaphysical axioms
- Kernel: PXLv3_SemanticModal.v (Phase 3 complete)
- Mission profile: state/mission_profile.json
- Alignment status: scripts/boot_aligned_agent.py

Key repository files:
- Axioms:
    Protopraxis/formal_verification/coq/baseline/PXLv3_SemanticModal.v (lines 124-149)
- Documentation:
    LOGOS_Axiom_And_Theorem_Summary.md, AXIOM_AUDIT_PHASE1.md
- Tests: test_lem_discharge.py, scripts/boot_aligned_agent.py
- Agent: scripts/start_agent.py, Protopraxis/agent_boot.py

The eight irreducible metaphysical axioms:
1. A2_noncontradiction – Objects cannot be both identical and non-equivalent
2. A7_triune_necessity – The Trinity (𝕆) is necessarily coherent
3. modus_groundens – Identity transfers entailment
4. triune_dependency_substitution – Equivalent groundings preserve coherence
5. privative_collapse – Impossibly unentailed propositions are incoherent
6. grounding_yields_entails – Groundings establish entailment
7. coherence_lifts_entailment – Trinity coherence lifts entailments globally
8. entails_global_implies_truth – Global entailment implies truth

When asked questions:
1. Use tools to gather information from repository files
2. Provide concise answers with specific citations
3. Reference file paths and line numbers when discussing code
4. Acknowledge uncertainty when appropriate
5. Suggest follow-up questions or next steps

Available tools:
- mission_status: Get current mission profile
- probe_last: Get last protocol probe snapshot
- fs_read: Read repository files (e.g.,
  Protopraxis/formal_verification/coq/baseline/PXLv3_SemanticModal.v)
- agent_memory: Access persistent memory
- sandbox_read: Read sandbox artifacts
"""


class ReasoningAgent:
    """LLM-powered agent with tool calling and safety guardrails."""

    def __init__(
        self,
        tools: Dict[str, Any],
        llm_backend: Optional["LLMBackend"] = None,
    ):
        from .plugins.llm_backend import (
            LLMBackend as RuntimeLLMBackend,
            create_tool_definitions,
        )

        self.tools = tools
        runtime_backend = llm_backend or RuntimeLLMBackend()
        self.llm: "LLMBackend" = runtime_backend
        self.tool_definitions = create_tool_definitions(tools)
        self.conversation_history: List[Dict[str, str]] = []

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return its output."""
        # Map LLM tool names back to LOGOS tool names
        tool_map = {
            "mission_status": "mission.status",
            "probe_last": "probe.last",
            "fs_read": "fs.read",
            "agent_memory": "agent.memory",
            "sandbox_read": "sandbox.read",
        }

        logos_tool = tool_map.get(tool_name, tool_name)  # Try both formats
        if logos_tool not in self.tools:
            # Also try the original tool_name in case it's already in LOGOS format
            if tool_name not in self.tools:
                available = ", ".join(self.tools.keys())
                return f"[error] Unknown tool: {tool_name} (available: {available})"
            logos_tool = tool_name

        # Extract argument (most tools take a single string argument)
        if arguments is None:
            arguments = {}
        arg = arguments.get("path") or arguments.get("artifact") or ""

        try:
            return self.tools[logos_tool](arg)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return f"[error] Tool execution failed: {exc}"

    def _format_tool_results(self, tool_calls: List[Dict[str, Any]]) -> str:
        """Execute tool calls and format results for the LLM."""
        results = []
        for call in tool_calls:
            tool_name = call["name"]
            arguments = call["arguments"]
            output = self._execute_tool(tool_name, arguments)
            results.append(f"[{tool_name}]\n{output}\n")
        return "\n".join(results)

    def ask(self, question: str, max_iterations: int = 5) -> str:
        """Answer a question using LLM reasoning and tool calls.

        Args:
            question: User's question
            max_iterations: Maximum tool-calling iterations (default 5)

        Returns:
            Final answer as string
        """
        # Initialize conversation with system prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        response = None
        tool_call_count = 0

        for iteration in range(max_iterations):
            try:
                # Get LLM response
                response = self.llm.complete(messages, tools=self.tool_definitions)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error_msg = str(exc)

                # Check for rate limit errors
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    return (
                        "⚠️ Rate limit reached. The agent has used its daily quota.\n\n"
                        "To continue:\n"
                        "1. Wait for the rate limit to reset (check error message)\n"
                        "2. Switch to mock provider:\n"
                        "   python3 scripts/llm_interface_suite/reasoning_agent.py "
                        f'--provider mock "{question}"\n'
                        "3. Try a different provider (OpenAI, Anthropic)\n\n"
                        f"Error details: {error_msg[:200]}"
                    )
                else:
                    return f"❌ Error: {error_msg}"

            # Debug output
            if DEBUG:
                print(f"\n[DEBUG] Iteration {iteration + 1}")
                print(f"  Finish reason: {response.get('finish_reason')}")
                print(f"  Tool calls: {len(response.get('tool_calls', []))}")
                content_text = response.get("content") or ""
                preview = content_text[:100] if content_text else "(empty)"
                print(f"  Content: {preview}")

            # If no tool calls, we have a final answer
            if "tool_calls" not in response or not response["tool_calls"]:
                return response.get("content", "[No response content]")

            # Execute tool calls
            tool_results = self._format_tool_results(response["tool_calls"])
            tool_call_count += len(response["tool_calls"])

            # Add assistant's tool request and tool results to conversation
            messages.append(
                {
                    "role": "assistant",
                    "content": response.get("content") or "[Calling tools]",
                }
            )

            # After 2 rounds of tool calls, be more forceful about getting final answer
            if iteration >= 2:
                firm_prompt = (
                    "Tool results:\n"
                    f"{tool_results}\n\n"
                    "You have all the information needed. "
                    "Provide your final answer now "
                    "WITHOUT calling any more tools."
                )
                messages.append({"role": "user", "content": firm_prompt})
            else:
                follow_up_prompt = (
                    "Tool results:\n"
                    f"{tool_results}\n\n"
                    "Based on these results, answer the question. Only call more tools "
                    "if absolutely necessary."
                )
                messages.append({"role": "user", "content": follow_up_prompt})

        # If we exhausted iterations, return what we have
        if response:
            return response.get("content", "[No answer generated after max iterations]")
        return "[No response generated]"


def main():
    """Demo the reasoning agent."""
    load_env(override_cli=False)

    # Import TOOLS from start_agent
    try:
        from scripts.start_agent import TOOLS
    except ImportError:
        # Fallback minimal tools for testing
        TOOLS = {
            "mission.status": (
                lambda _: '{"label": "DEMO", "safe_interfaces_only": true}'
            ),
            "fs.read": lambda path: f"[Mock read of {path}]",
        }

    parser = argparse.ArgumentParser(description="LLM-powered LOGOS reasoning agent")
    parser.add_argument("question", nargs="+", help="Question to ask the agent")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "groq", "mock"],
        help="LLM provider to use (overrides .env file)",
    )
    parser.add_argument(
        "--model",
        help=(
            "Model name (e.g., gpt-4, claude-3-5-sonnet-20241022, "
            "llama-3.3-70b-versatile)"
        ),
    )

    args = parser.parse_args()
    question = " ".join(args.question)

    # Configure LLM backend - CLI args override .env
    if args.provider:
        os.environ["LOGOS_LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LOGOS_LLM_MODEL"] = args.model

    try:
        from .plugins.llm_backend import LLMBackend as RuntimeLLMBackend

        llm = RuntimeLLMBackend()
        agent = ReasoningAgent(TOOLS, llm)

        print("\n🤖 LOGOS Reasoning Agent")
        print(f"   Provider: {llm.config.provider}")
        print(f"   Model: {llm.config.model}")
        print(f"\n❓ Question: {question}\n")
        print("=" * 70)
        print()

        answer = agent.ask(question)
        print(f"💡 Answer:\n{answer}")
        print()
        print("=" * 70)

        return 0

    except RuntimeError as exc:
        print(f"\n❌ Configuration Error: {exc}")
        print("\nHint: Set up API keys or use --provider mock for testing")
        return 1
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"\n❌ Error: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
