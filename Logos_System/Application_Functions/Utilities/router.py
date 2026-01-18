# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from ..input_output_processing.parser import ParsedPrompt, PromptParser, parser
from ..metadata_filters.response_engine import ResponseEngine, ResponsePayload, response_engine
from ..core_processing.sanitizer import SanitizationResult, Sanitizer, sanitizer

logger = logging.getLogger(__name__)


@dataclass
class InteractionConfig:
    log_path: Path = Path("logs/agent_response_log.jsonl")


class InteractionRouter:
    """Consolidated interaction layer orchestrating prompt → response flow."""

    def __init__(
        self,
        sanitizer: Sanitizer = sanitizer,
        parser: PromptParser = parser,
        response_engine: ResponseEngine = response_engine,
        config: InteractionConfig | None = None,
    ) -> None:
        self.sanitizer = sanitizer
        self.parser = parser
        self.response_engine = response_engine
        self.config = config or InteractionConfig()
        self.agent_identity: Optional[str] = None
        self.logic_kernel: Optional[Any] = None
        self.bridge: Optional[Any] = None
        self._prompt_thread: Optional[threading.Thread] = None
        self._loop_active = False
        self._prompt_queue: queue.Queue[str] = queue.Queue()
        self._ensure_log_directory()
        self.logger = logging.getLogger(f"{__name__}.InteractionRouter")

    def ensure_initialized(self, agent_identity: str, logic_kernel: Any, bridge: Any | None = None) -> None:
        if self.agent_identity == agent_identity and self.logic_kernel is logic_kernel:
            return
        self.agent_identity = agent_identity
        self.logic_kernel = logic_kernel
        self.bridge = bridge
        self.logger.info("Interaction router initialised for %s", agent_identity)

    def is_active(self) -> bool:
        return self.agent_identity is not None and self.logic_kernel is not None

    def start_prompt_loop(
        self,
        prompt_source: Callable[[], str] | None = None,
        auto_start: bool = True,
    ) -> None:
        if not self.is_active():
            self.logger.debug("Prompt loop requested before router initialised; ignoring")
            return
        if not auto_start:
            self.logger.debug("Auto-start disabled; prompt loop not launched")
            return
        if self._loop_active:
            return

        if prompt_source is None:
            prompt_source = self._stdin_prompt_source

        self._loop_active = True
        self._prompt_thread = threading.Thread(
            target=self._prompt_loop,
            args=(prompt_source,),
            name="logos-interaction-loop",
            daemon=True,
        )
        self._prompt_thread.start()
        self.logger.info("Interaction prompt loop started")

    def enqueue_prompt(self, prompt: str) -> Optional[ResponsePayload]:
        if not self.is_active():
            raise RuntimeError("Interaction router not initialised")
        return self._handle_prompt(prompt)

    def process_prompt(self, prompt: str) -> ResponsePayload:
        if not self.is_active():
            raise RuntimeError("Interaction router not initialised")
        return self._handle_prompt(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _stdin_prompt_source(self) -> Optional[str]:  # pragma: no cover - CLI helper
        if not sys.stdin.isatty():
            self.logger.debug("stdin not interactive; prompt loop idle")
            try:
                self._prompt_queue.get(timeout=5)
            except queue.Empty:
                pass
            return None
        try:
            return input("LOGOS › ")
        except EOFError:
            return None

    def _prompt_loop(self, prompt_source: Callable[[], Optional[str]]) -> None:  # pragma: no cover - background loop
        while self._loop_active:
            prompt = prompt_source()
            if prompt is None:
                continue
            prompt = prompt.strip()
            if not prompt:
                continue
            try:
                response = self._handle_prompt(prompt)
                print(response.text)
            except Exception:  # noqa: BLE001
                self.logger.exception("Failed to process prompt")

    def _handle_prompt(self, prompt: str) -> ResponsePayload:
        sanitised = self.sanitizer.sanitize(prompt)
        parsed = self.parser.parse(sanitised.cleaned_text, sanitised.issues)
        payload = self.response_engine.generate(
            sanitised.cleaned_text,
            parsed,
            self.logic_kernel,
            self.agent_identity or "UNKNOWN",
            bridge=self.bridge,
        )
        self._log_interaction(sanitised, parsed, payload)
        return payload

    def _log_interaction(
        self,
        sanitised: SanitizationResult,
        parsed: ParsedPrompt,
        payload: ResponsePayload,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": sanitised.cleaned_text,
            "sanitizer_issues": sanitised.issues,
            "metadata": sanitised.metadata,
            "parsed": {
                "intent": parsed.intent,
                "symbolic_form": parsed.symbolic_form,
                "keywords": parsed.keywords,
                "confidence": parsed.confidence,
                "notes": parsed.notes,
            },
            "response": payload.data,
        }
        try:
            with self.config.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:  # pragma: no cover - file system failure
            self.logger.exception("Failed to write interaction log")

    def _ensure_log_directory(self) -> None:
        try:
            self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - file system failure
            logger.exception("Unable to create log directory for interaction router")


interaction_router = InteractionRouter()

__all__ = ["InteractionRouter", "InteractionConfig", "interaction_router"]
