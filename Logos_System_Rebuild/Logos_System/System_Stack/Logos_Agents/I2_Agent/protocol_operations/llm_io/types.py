from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# OpenAI-compatible chat message shape
@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}

@dataclass(frozen=True)
class ChatRequest:
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ChatResponse:
    text: str
    raw: Dict[str, Any]
    provider: str
    model: str
