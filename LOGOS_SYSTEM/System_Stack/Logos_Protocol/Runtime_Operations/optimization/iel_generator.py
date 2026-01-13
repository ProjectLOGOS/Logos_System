"""Compatibility proxy for the historical IEL generator implementation."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, List

_TARGET_MODULE = (
    "System_Operations_Protocol.code_generator.meta_reasoning.iel_generator"
)
_delegate: ModuleType | None = None
__all__: List[str] = []


def _load_delegate() -> ModuleType:
    global _delegate, __all__
    if _delegate is None:
        _delegate = importlib.import_module(_TARGET_MODULE)
        exported = getattr(_delegate, "__all__", None)
        __all__ = list(exported) if exported is not None else [
            name for name in dir(_delegate) if not name.startswith("_")
        ]
    return _delegate


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper
    return getattr(_load_delegate(), name)


def __dir__() -> List[str]:  # pragma: no cover - thin wrapper
    loaded = _load_delegate()
    return sorted(set(globals()) | set(dir(loaded)))


def _parse_gap_events(log_path: Path) -> List[dict]:
    if not log_path.exists():
        return []
    events: List[dict] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _render_candidate(proof_name: str, premises: Iterable[str], conclusion: str, obligations: Iterable[str]) -> str:
    premise_chain = " -> ".join(premises) or "True"
    obligations_text = "\n".join(f"  (* - {item} *)" for item in obligations) or "  (* - review obligations *)"
    return f"""(* Generated IEL Candidate *)
(* Rule: {proof_name} *)

Lemma {proof_name} :
  {premise_chain} -> {conclusion}.
Proof.
{obligations_text}
  (* Auto-generated - requires manual verification *)
  Admitted.
Qed.
"""


def main(argv: List[str] | None = None) -> int:  # pragma: no cover - exercised by tests
    parser = argparse.ArgumentParser(description="Generate stub IEL candidates from telemetry logs")
    parser.add_argument("--from-log", type=Path, help="Telemetry JSONL file")
    parser.add_argument("--out", type=Path, help="Destination .v file")
    args = parser.parse_args(argv)

    if not args.from_log or not args.out:
        parser.print_help()
        return 1

    events = _parse_gap_events(args.from_log)
    if not events:
        print("No events found in telemetry log")
        return 1

    delegate = _load_delegate()
    generator_cls = getattr(delegate, "IELGenerator", None)
    gap_cls = getattr(delegate, "ReasoningGap", None)

    candidates: List[Any] = []
    if generator_cls and gap_cls:
        generator = generator_cls()
        first_gap = None
        for event in events:
            if event.get("event_type") in {"gap_detected", "reasoning_gap_detected"}:
                data = event.get("data", {})
                first_gap = gap_cls(
                    gap_type=data.get("type", "auto_gap"),
                    domain=data.get("domain", "auto_domain"),
                    description=data.get("description", "Auto generated gap"),
                    severity=float(data.get("severity", 0.5) or 0.5),
                    required_premises=data.get("premises", []) or ["P1", "P2"],
                    expected_conclusion=data.get("conclusion", "Q"),
                    confidence=float(data.get("confidence", 0.5) or 0.5),
                )
                break
        if first_gap is not None:
            try:
                candidates = generator.generate_candidates_for_gap(first_gap)
            except Exception:  # pragma: no cover - defensive guard
                candidates = []

    if candidates:
        candidate = candidates[0]
        proof_name = getattr(candidate, "rule_name", "auto_rule")
        premises = getattr(candidate, "premises", ["P1", "P2"])
        conclusion = getattr(candidate, "conclusion", "Q")
        obligations = getattr(candidate, "proof_obligations", ["Review obligations"])
    else:
        proof_name = "auto_generated_gap"
        premises = ["P1", "P2"]
        conclusion = "Q"
        obligations = ["Provide formal proof"]

    content = _render_candidate(proof_name, premises, conclusion, obligations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    print(f"Generated IEL candidate at {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    raise SystemExit(main())
