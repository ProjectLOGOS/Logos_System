# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

from .types import ChatRequest, ChatResponse

class LlamaProvider:
    """
    LLaMA backend:
    - Preferred: OpenAI-compatible HTTP endpoint (LLAMA_ENDPOINT)
      e.g. llama.cpp server, vLLM, text-generation-webui, etc. exposing /v1/chat/completions
    - Fallback: local llama_cpp (if installed) for basic chat completion behavior
    """

    def __init__(self, *, endpoint: Optional[str], default_model: Optional[str], default_temperature: float):
        self.endpoint = endpoint
        self.default_model = default_model
        self.default_temperature = default_temperature

    def chat(self, req: ChatRequest) -> ChatResponse:
        # 1) Endpoint path (recommended)
        if self.endpoint:
            model = req.model or self.default_model or "llama"
            url = self.endpoint

            payload: Dict[str, Any] = {
                "model": model,
                "messages": [m.to_dict() for m in req.messages],
                "temperature": req.temperature if req.temperature is not None else self.default_temperature,
            }
            if req.max_tokens is not None:
                payload["max_tokens"] = req.max_tokens
            payload.update(req.extra or {})

            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}

            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=120) as resp:
                    raw = resp.read().decode("utf-8")
                    out = json.loads(raw)
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8")
                except Exception:
                    pass
                raise RuntimeError(f"LlamaProvider HTTPError {e.code}: {body}") from e
            except urllib.error.URLError as e:
                raise RuntimeError(f"LlamaProvider URLError: {e}") from e

            try:
                text = out["choices"][0]["message"]["content"]
            except Exception:
                text = ""

            return ChatResponse(
                text=text or "",
                raw=out,
                provider="llama",
                model=model,
            )

        # 2) Local llama_cpp fallback
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "LlamaProvider: No LLAMA_ENDPOINT set and llama_cpp is not installed. "
                "Set LLAMA_ENDPOINT to an OpenAI-compatible server OR install llama-cpp-python."
            ) from e

        model_path = (req.extra or {}).get("model_path")
        if not model_path:
            raise RuntimeError(
                "LlamaProvider local mode requires req.extra['model_path'] pointing to a GGUF file."
            )

        llm = Llama(model_path=model_path)

        prompt_parts = []
        for m in req.messages:
            prompt_parts.append(f"{m.role.upper()}: {m.content}")
        prompt = "\n".join(prompt_parts) + "\nASSISTANT:"

        out = llm(
            prompt,
            temperature=req.temperature if req.temperature is not None else self.default_temperature,
            max_tokens=req.max_tokens or 512,
        )

        text = ""
        try:
            text = out["choices"][0]["text"]
        except Exception:
            text = ""

        model = req.model or self.default_model or "llama_cpp_local"

        return ChatResponse(
            text=text or "",
            raw=out,
            provider="llama",
            model=model,
        )
