# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Non-authoritative LLM advisor for LOGOS runtime.

The advisor only returns structured proposals; it cannot execute tools.
Supports a stub mode (default if OPENAI_API_KEY missing) and an OpenAI
Responses API mode using stdlib urllib.
"""

from __future__ import annotations

import json
import os
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from urllib import request

try:
    from scripts.evidence import normalize_evidence_refs
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.evidence import normalize_evidence_refs


@dataclass
class AdvisorNotes:
    provider: str
    model: str
    mode: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "reason": self.reason,
        }


class LLMAdvisor:
    def __init__(
        self,
        provider: str,
        model: str,
        tools_schema: Dict[str, Any],
        truth_rules: Dict[str, Any],
        timeout_sec: int = 10,
    ) -> None:
        self.provider = provider or "stub"
        self.model = model or "stub"
        self.tools_schema = tools_schema
        self.truth_rules = truth_rules
        self.timeout_sec = timeout_sec
        self.api_key = (
            os.getenv("OPENAI_API_KEY") if self.provider == "openai" else None
        )
        self.anthropic_key = (
            os.getenv("ANTHROPIC_API_KEY") if self.provider == "anthropic" else None
        )
        self.stub_mode = self.provider == "stub" or (
            self.provider == "openai" and not self.api_key
        ) or (self.provider == "anthropic" and not self.anthropic_key)

    def _stub_response(
        self, user_objective: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload_env = os.getenv("LLM_ADVISOR_STUB_PAYLOAD")
        proposals: List[Dict[str, Any]] = []
        claims: List[Dict[str, Any]] = []
        reply: Optional[str] = None
        errors: List[str] = []
        _ = context  # explicitly acknowledge context to avoid unused-argument warnings
        if payload_env:
            try:
                data = json.loads(payload_env)
                if isinstance(data, dict):
                    proposals = data.get("proposals", [])
                    reply = data.get("reply")
                    claims = data.get("claims", [])
            except json.JSONDecodeError as exc:
                errors.append(f"invalid stub payload: {exc}")
        return {
            "proposals": proposals,
            "claims": self._sanitize_claims(claims, user_objective),
            "reply": reply,
            "notes": AdvisorNotes(
                provider=self.provider,
                model=self.model,
                mode="stub",
                reason="stub payload" if payload_env else "default stub",
            ).to_dict(),
            "errors": errors,
        }

    def _call_openai(
        self, user_objective: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.api_key:
            return self._stub_response(user_objective, context)
        url = "https://api.openai.com/v1/responses"
        prompt = {
            "role": "system",
            "content": (
                "You are a non-authoritative advisor. "
                "Output ONLY JSON with a 'reply' string "
                "and a 'proposals' array of tool suggestions. "
                "Do not execute tools. Respect truth rules and risk constraints."
            ),
        }
        user_msg = {
            "role": "user",
            "content": json.dumps(
                {
                    "objective": user_objective,
                    "context": context,
                    "tools": self.tools_schema,
                    "truth_rules": self.truth_rules,
                }
            ),
        }
        body = {
            "model": self.model,
            "input": [prompt, user_msg],
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "proposals": [],
                "reply": "ok (fallback)",
                "notes": AdvisorNotes(
                    provider=self.provider,
                    model=self.model,
                    mode="stub",
                    reason=str(exc),
                ).to_dict(),
                "errors": [str(exc)],
            }

        try:
            output = parsed.get("output") or parsed.get("response") or parsed
            if isinstance(output, str):
                output = json.loads(output)
            proposals = output.get("proposals", []) if isinstance(output, dict) else []
            claims = output.get("claims", []) if isinstance(output, dict) else []
            reply = output.get("reply") if isinstance(output, dict) else None
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "proposals": [],
                "claims": [],
                "reply": "ok (fallback)",
                "notes": AdvisorNotes(
                    provider=self.provider,
                    model=self.model,
                    mode="stub",
                    reason=f"parse_error:{exc}",
                ).to_dict(),
                "errors": [f"parse_error:{exc}"],
            }

        return {
            "proposals": proposals,
            "claims": claims,
            "reply": reply,
            "notes": AdvisorNotes(
                provider=self.provider, model=self.model, mode="real"
            ).to_dict(),
            "errors": [],
        }

    def _call_anthropic(
        self, user_objective: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.anthropic_key:
            return self._stub_response(user_objective, context)
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        system_prompt = (
            "You are a non-authoritative advisor. Respond only with JSON containing "
            "'reply' and 'proposals'. Do not execute tools."
        )
        body = {
            "model": self.model,
            "max_tokens": 512,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "objective": user_objective,
                            "context": context,
                            "tools": self.tools_schema,
                            "truth_rules": self.truth_rules,
                        }
                    ),
                }
            ],
        }
        data = json.dumps(body).encode("utf-8")
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "proposals": [],
                "reply": "ok (fallback)",
                "notes": AdvisorNotes(
                    provider=self.provider,
                    model=self.model,
                    mode="stub",
                    reason=str(exc),
                ).to_dict(),
                "errors": [str(exc)],
            }

        try:
            content_blocks = parsed.get("content") or []
            combined = "".join(
                block.get("text", "")
                for block in content_blocks
                if isinstance(block, dict)
            )
            output = json.loads(combined) if combined else {}
            proposals = output.get("proposals", []) if isinstance(output, dict) else []
            claims = output.get("claims", []) if isinstance(output, dict) else []
            reply = output.get("reply") if isinstance(output, dict) else None
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "proposals": [],
                "claims": [],
                "reply": "ok (fallback)",
                "notes": AdvisorNotes(
                    provider=self.provider,
                    model=self.model,
                    mode="stub",
                    reason=f"parse_error:{exc}",
                ).to_dict(),
                "errors": [f"parse_error:{exc}"],
            }

        return {
            "proposals": proposals,
            "claims": claims,
            "reply": reply,
            "notes": AdvisorNotes(
                provider=self.provider, model=self.model, mode="real"
            ).to_dict(),
            "errors": [],
        }

    def _sanitize_proposals(
        self, proposals: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        clean: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for prop in proposals or []:
            if not isinstance(prop, dict):
                continue
            forbidden_keys = {"execute", "run", "shell", "code"}
            if any(k in prop for k in forbidden_keys):
                rejected.append(
                    {"proposal": prop, "reason": "direct_execution_attempt"}
                )
                continue
            tool = str(prop.get("tool", ""))
            args = prop.get("args", "")
            rationale = str(prop.get("rationale", ""))
            truth_annotation = prop.get("truth_annotation") or {
                "truth": "HEURISTIC",
                "evidence": {
                    "type": "none",
                    "ref": None,
                    "details": "llm_advisor_default",
                },
            }
            confidence = float(prop.get("confidence", 0.5) or 0.5)
            clean.append(
                {
                    "tool": tool,
                    "args": args,
                    "rationale": rationale,
                    "truth_annotation": truth_annotation,
                    "confidence": confidence,
                }
            )
        return {"clean": clean, "rejected": rejected}

    def _sanitize_claims(
        self, claims: List[Dict[str, Any]], default_text: str
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        verified_types = {"file", "url", "schema", "hash"}
        for claim in claims or []:
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("text", "")).strip() or default_text
            truth = str(claim.get("truth", "HEURISTIC"))
            notes = claim.get("notes", "")
            refs_raw = (
                claim.get("evidence_refs")
                if isinstance(claim.get("evidence_refs"), list)
                else []
            )
            refs = normalize_evidence_refs(refs_raw)
            if truth == "PROVED":
                truth = (
                    "VERIFIED"
                    if any(r.get("type") in verified_types for r in refs)
                    else "HEURISTIC"
                )
            if truth == "VERIFIED":
                if not refs or not any(r.get("type") in verified_types for r in refs):
                    truth = "HEURISTIC"
            normalized.append(
                {
                    "text": text,
                    "truth": truth,
                    "evidence_refs": refs,
                    "notes": str(notes) if notes is not None else "",
                }
            )
        if not normalized:
            normalized.append(
                {
                    "text": default_text,
                    "truth": "HEURISTIC",
                    "evidence_refs": [],
                    "notes": "advisor_default",
                }
            )
        return normalized

    def _stream_single_shot(
        self, user_objective: str, context: Dict[str, Any]
    ) -> Generator[str, None, Dict[str, Any]]:
        """Fallback stream: yield the full reply once and return full result."""
        result = self.propose(user_objective, context)
        reply_text = result.get("reply") or f"Acknowledged: {user_objective}"

        def _gen() -> Generator[str, None, Dict[str, Any]]:
            yield str(reply_text)
            return result

        return _gen()

    def _openai_stream(
        self, user_objective: str, context: Dict[str, Any]
    ) -> Optional[Generator[str, None, Dict[str, Any]]]:
        if not self.api_key:
            return None
        try:
            spec = importlib.util.find_spec("openai")
            if spec is None:
                return None
            openai = importlib.import_module("openai")
        except Exception:
            return None
        client_cls = getattr(openai, "OpenAI", None)
        if client_cls is None:
            return None
        try:
            client = client_cls(api_key=self.api_key, timeout=self.timeout_sec)
        except Exception:
            return None

        system_prompt = (
            "You are a non-authoritative advisor. Stream only the assistant reply text."
            " Do not emit JSON. Do not propose or execute tools."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "objective": user_objective,
                        "context": context,
                        "tools": self.tools_schema,
                        "truth_rules": self.truth_rules,
                    }
                ),
            },
        ]
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                timeout=self.timeout_sec,
            )
        except Exception:
            return None

        def _gen() -> Generator[str, None, Dict[str, Any]]:
            chunks: List[str] = []
            for event in stream:
                try:
                    choices = getattr(event, "choices", [])
                    delta = choices[0].delta if choices else None  # type: ignore[index]
                    content = getattr(delta, "content", None) if delta else None
                    if isinstance(content, list):
                        text_delta = "".join(
                            part for part in content if isinstance(part, str)
                        )
                    elif isinstance(content, str):
                        text_delta = content
                    else:
                        text_delta = ""
                except Exception:
                    text_delta = ""
                if text_delta:
                    chunks.append(text_delta)
                    yield text_delta

            proposals_result = self.propose(user_objective, context)
            streamed_reply = "".join(chunks).strip()
            if streamed_reply:
                proposals_result["reply"] = streamed_reply
            return proposals_result

        return _gen()

    def propose_stream(
        self, user_objective: str, context: Dict[str, Any]
    ) -> Generator[str, None, Dict[str, Any]]:
        """Stream reply chunks; return final structured result with proposals.

        - Stub/default: yield one chunk and return stub result.
        - OpenAI: use SDK streaming when available; otherwise fall back to one-shot.
        - Anthropic: fall back to one-shot for now.
        """

        if self.stub_mode:
            return self._stream_single_shot(user_objective, context)

        if self.provider == "openai":
            stream = self._openai_stream(user_objective, context)
            if stream:
                return stream
            return self._stream_single_shot(user_objective, context)

        # Anthropic (or other) fallback: reuse non-streaming behavior
        return self._stream_single_shot(user_objective, context)

    def propose(self, user_objective: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.stub_mode:
            result = self._stub_response(user_objective, context)
        else:
            if self.provider == "openai":
                result = self._call_openai(user_objective, context)
            elif self.provider == "anthropic":
                result = self._call_anthropic(user_objective, context)
            else:
                result = self._stub_response(user_objective, context)

        sanitized = self._sanitize_proposals(result.get("proposals", []))
        result["proposals"] = sanitized.get("clean", [])
        result["rejected"] = sanitized.get("rejected", [])
        result["claims"] = self._sanitize_claims(
            result.get("claims", []), user_objective
        )
        reply = result.get("reply")
        if not reply:
            reply = f"Acknowledged: {user_objective}"
        result["reply"] = str(reply)
        if not result.get("notes"):
            result["notes"] = AdvisorNotes(
                self.provider, self.model, "stub", reason="fallback"
            ).to_dict()
        return result


def build_tool_schema(tools_registry: Dict[str, Any]) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"tools": []}
    high_impact = {"tool_proposal_pipeline", "start_agent", "retrieve.web"}
    for name in sorted(tools_registry.keys()):
        schema["tools"].append(
            {
                "name": name,
                "risk_level": "HIGH" if name in high_impact else "LOW",
                "requires_uip": name in high_impact,
            }
        )
    return schema
