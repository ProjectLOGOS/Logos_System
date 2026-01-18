# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""LLM backend integration for LOGOS agent reasoning.

Supports OpenAI and Anthropic APIs with fallback to local models.
"""

import json
import os
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass

# API clients (install with: pip install openai anthropic groq)
try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from groq import Groq

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


Provider = Literal["openai", "anthropic", "groq", "mock"]


@dataclass
class LLMConfig:
    """LLM backend configuration."""

    provider: Provider = "mock"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7


class LLMBackend:
    """Unified LLM backend supporting multiple providers."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or self._load_config()
        self._validate_config()

    def _load_config(self) -> LLMConfig:
        """Load LLM config from environment variables."""
        provider = os.getenv("LOGOS_LLM_PROVIDER", "mock").lower()

        if provider == "openai":
            return LLMConfig(
                provider="openai",
                model=os.getenv("LOGOS_LLM_MODEL", "gpt-4-turbo-preview"),
                api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=int(os.getenv("LOGOS_LLM_MAX_TOKENS", "2000")),
                temperature=float(os.getenv("LOGOS_LLM_TEMPERATURE", "0.7")),
            )
        elif provider == "anthropic":
            return LLMConfig(
                provider="anthropic",
                model=os.getenv("LOGOS_LLM_MODEL", "claude-3-5-sonnet-20241022"),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                max_tokens=int(os.getenv("LOGOS_LLM_MAX_TOKENS", "4000")),
                temperature=float(os.getenv("LOGOS_LLM_TEMPERATURE", "0.7")),
            )
        elif provider == "groq":
            return LLMConfig(
                provider="groq",
                model=os.getenv("LOGOS_LLM_MODEL", "llama-3.3-70b-versatile"),
                api_key=os.getenv("GROQ_API_KEY"),
                max_tokens=int(os.getenv("LOGOS_LLM_MAX_TOKENS", "8000")),
                temperature=float(os.getenv("LOGOS_LLM_TEMPERATURE", "0.7")),
            )
        else:
            # Mock provider for testing without API keys
            return LLMConfig(provider="mock")

    def _validate_config(self) -> None:
        """Validate configuration and availability."""
        if self.config.provider == "openai":
            if not HAS_OPENAI:
                raise RuntimeError("OpenAI provider requires: pip install openai")
            if not self.config.api_key:
                raise RuntimeError(
                    "OpenAI provider requires OPENAI_API_KEY environment variable"
                )
        elif self.config.provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise RuntimeError("Anthropic provider requires: pip install anthropic")
            if not self.config.api_key:
                raise RuntimeError(
                    "Anthropic provider requires ANTHROPIC_API_KEY environment variable"
                )
        elif self.config.provider == "groq":
            if not HAS_GROQ:
                raise RuntimeError("Groq provider requires: pip install groq")
            if not self.config.api_key:
                raise RuntimeError(
                    "Groq provider requires GROQ_API_KEY environment variable"
                )

    def complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate completion with optional tool calling.

        Args:
            messages: List of {role: "system"|"user"|"assistant", content: str}
            tools: Optional tool definitions for function calling

        Returns:
            {
                "content": str,  # Text response
                "tool_calls": [{"name": str, "arguments": dict}],  # Optional
                "finish_reason": str
            }
        """
        if self.config.provider == "openai":
            return self._complete_openai(messages, tools)
        elif self.config.provider == "anthropic":
            return self._complete_anthropic(messages, tools)
        elif self.config.provider == "groq":
            return self._complete_groq(messages, tools)
        else:
            return self._complete_mock(messages, tools)

    def _complete_openai(
        self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """OpenAI completion."""
        client = openai.OpenAI(api_key=self.config.api_key)

        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        result = {
            "content": message.content or "",
            "finish_reason": response.choices[0].finish_reason,
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in message.tool_calls
            ]

        return result

    def _complete_anthropic(
        self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Anthropic completion."""
        client = anthropic.Anthropic(api_key=self.config.api_key)

        # Extract system message (Anthropic uses separate parameter)
        system_msg = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": user_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if system_msg:
            kwargs["system"] = system_msg

        if tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)

        result = {"content": "", "finish_reason": response.stop_reason}

        tool_calls = []
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "arguments": block.input})

        if tool_calls:
            result["tool_calls"] = tool_calls

        return result

    def _complete_groq(
        self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Groq completion (OpenAI-compatible API)."""
        client = Groq(api_key=self.config.api_key)

        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        result = {
            "content": message.content or "",
            "finish_reason": response.choices[0].finish_reason,
        }

        if hasattr(message, "tool_calls") and message.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in message.tool_calls
            ]

        return result

    def _complete_mock(
        self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Mock completion for testing."""
        # Extract the user's question
        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        # Simple pattern matching for demo purposes
        if "axiom" in user_msg.lower():
            content = (
                "The PXL kernel has 8 irreducible metaphysical axioms: "
                "A2_noncontradiction, A7_triune_necessity, modus_groundens, "
                "triune_dependency_substitution, privative_collapse, "
                "grounding_yields_entails, coherence_lifts_entailment, "
                "and entails_global_implies_truth. These define PXL's "
                "metaphysical commitments and cannot be further reduced."
            )
            tool_calls = (
                [
                    {
                        "name": "fs.read",
                        "arguments": {
                            "path": "Protopraxis/formal_verification/coq/baseline/PXLv3_SemanticModal.v"
                        },
                    }
                ]
                if tools
                else None
            )
        else:
            content = (
                f"Mock LLM response to: {user_msg[:100]}... "
                "(Configure LOGOS_LLM_PROVIDER=openai or anthropic for real reasoning)"
            )
            tool_calls = None

        result = {"content": content, "finish_reason": "stop"}

        if tool_calls:
            result["tool_calls"] = tool_calls

        return result


def create_tool_definitions(available_tools: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert LOGOS tools to LLM function calling format."""
    tools = []

    # Define schemas for LOGOS agent tools
    tool_schemas = {
        "mission.status": {
            "name": "mission_status",
            "description": "Get current mission profile and safety configuration",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "probe.last": {
            "name": "probe_last",
            "description": "Get last protocol probe snapshot with discovery results",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "fs.read": {
            "name": "fs_read",
            "description": "Read file or list directory contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File or directory path (relative to repository root)",
                    }
                },
                "required": ["path"],
            },
        },
        "agent.memory": {
            "name": "agent_memory",
            "description": "Access agent's persistent memory and reflection history",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        "sandbox.read": {
            "name": "sandbox_read",
            "description": "Read artifact from sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact": {
                        "type": "string",
                        "description": "Artifact filename in sandbox",
                    }
                },
                "required": ["artifact"],
            },
        },
    }

    for tool_name, schema in tool_schemas.items():
        if tool_name in available_tools:
            tools.append(schema)

    return tools


# Example usage
if __name__ == "__main__":
    # Test configuration loading
    backend = LLMBackend()
    print(f"LLM Provider: {backend.config.provider}")
    print(f"Model: {backend.config.model}")

    # Test mock completion
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant for the LOGOS proof system.",
        },
        {"role": "user", "content": "What are the 8 axioms in PXL?"},
    ]

    response = backend.complete(messages)
    print(f"\nResponse: {response['content']}")

    if "tool_calls" in response:
        print(f"\nTool calls: {json.dumps(response['tool_calls'], indent=2)}")
