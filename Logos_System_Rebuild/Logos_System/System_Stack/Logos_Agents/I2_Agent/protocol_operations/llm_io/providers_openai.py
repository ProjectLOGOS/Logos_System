# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from .types import ChatMessage, ChatRequest, ChatResponse

class OpenAIProvider:
    """
    OpenAI-compatible Chat Completions over HTTP.
    No external dependencies.
    """

    def __init__(self, *, api_key: str, base_url: str, default_model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for openai backend.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    def chat(self, req: ChatRequest) -> ChatResponse:
        model = req.model or self.default_model
        url = f"{self.base_url}/v1/chat/completions"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in req.messages],
            "temperature": req.temperature,
        }
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        payload.update(req.extra or {})

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                out = json.loads(raw)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(f"OpenAIProvider HTTPError {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"OpenAIProvider URLError: {e}") from e

        try:
            text = out["choices"][0]["message"]["content"]
        except Exception:
            text = ""

        return ChatResponse(
            text=text or "",
            raw=out,
            provider="openai",
            model=model,
        )
