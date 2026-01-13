from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class LLMConfig:
    backend: str  # "openai" | "llama"

    # OpenAI-compatible
    openai_api_key: Optional[str]
    openai_base_url: str
    openai_model: str

    # LLaMA
    llama_endpoint: Optional[str]  # OpenAI-compatible endpoint recommended
    llama_model: Optional[str]
    llama_temperature: float

def load_config() -> LLMConfig:
    backend = os.getenv("LLM_BACKEND", "openai").strip().lower()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    llama_endpoint = os.getenv("LLAMA_ENDPOINT")
    llama_model = os.getenv("LLAMA_MODEL")
    llama_temperature = float(os.getenv("LLAMA_TEMPERATURE", "0.2"))

    if backend not in {"openai", "llama"}:
        raise ValueError(f"Unsupported LLM_BACKEND={backend}. Use 'openai' or 'llama'.")

    return LLMConfig(
        backend=backend,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        llama_endpoint=llama_endpoint,
        llama_model=llama_model,
        llama_temperature=llama_temperature,
    )
