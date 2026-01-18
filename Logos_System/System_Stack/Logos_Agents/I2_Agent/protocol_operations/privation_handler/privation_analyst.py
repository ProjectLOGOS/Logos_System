# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""
I2 Privation Handler — privation_analyst.py

Role:
- Consume classifier output (tags/domain/confidence).
- Compute a severity score (0..1).
- Decide recommended action: allow | transform | quarantine | escalate.
- Optionally consult:
  - consolidated privation library markdown (reference context)
  - IEL bijection JSON mapping (tag/domain -> IEL domain/module)
  - optional dynamic import of an IEL overlay module (best-effort, non-fatal)

Stateless by design. Does not persist or mutate shared state.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# Resolve catalog directory: prefer sibling privation_library, else parent.
_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_CATALOG = _BASE_DIR / "privation_library"
CATALOG_DIR = _DEFAULT_CATALOG if _DEFAULT_CATALOG.is_dir() else _BASE_DIR

IEL_BIJECTION_JSON_CANDIDATES = [
    CATALOG_DIR / "privation_library.json",
    CATALOG_DIR / "iel_ontological_bijection_optimized.json",
    CATALOG_DIR / "IEL_ontological_bijection_optimized.json",
]

PRIVATION_MD_CANDIDATES = [
    CATALOG_DIR / "privation_library.md",
    CATALOG_DIR / "privations.md",
    CATALOG_DIR / "privation_library_compiled.md",
    CATALOG_DIR / "privations_compiled.md",
]

DEFAULT_IEL_DOMAIN_MAP = {
    "epistemic": "Advanced_Reasoning_Protocol.iel_domains.Epistemology",
    "axiological": "Advanced_Reasoning_Protocol.iel_domains.Axiology",
    "ontological": "Advanced_Reasoning_Protocol.iel_domains.Ontology",
    "teleological": "Advanced_Reasoning_Protocol.iel_domains.Teleology",
    "linguistic": "Advanced_Reasoning_Protocol.iel_domains.Language",
    "agentic": "Advanced_Reasoning_Protocol.iel_domains.Anthropology",
}


@dataclass
class PrivationAnalysis:
    severity: float  # 0..1
    action: str      # allow | transform | quarantine | escalate
    iel_module: Optional[str]
    notes: str
    rationale: Dict[str, Any]


def _load_json_mapping() -> Optional[Dict[str, Any]]:
    for path in IEL_BIJECTION_JSON_CANDIDATES:
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def _load_privations_markdown() -> str:
    for path in PRIVATION_MD_CANDIDATES:
        if not path.is_file():
            continue
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            continue
    return ""


def _resolve_iel_module(domain: str, mapping: Optional[Dict[str, Any]]) -> Optional[str]:
    if mapping:
        candidate = mapping.get(domain)
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, dict):
            mod = candidate.get("module") or candidate.get("target")
            if isinstance(mod, str):
                return mod
    return DEFAULT_IEL_DOMAIN_MAP.get(domain)


def analyze(
    *,
    classification: Dict[str, Any],
    overlay_module: Optional[str] = None,
) -> PrivationAnalysis:
    tags = classification.get("tags") if isinstance(classification.get("tags"), list) else []
    domain = classification.get("domain") if isinstance(classification.get("domain"), str) else "unknown"
    confidence = classification.get("confidence")
    severity = float(confidence) if isinstance(confidence, (int, float)) else 0.0

    if "explicit_contradiction" in tags:
        severity = max(severity, 0.75)
    if "ontological_nullity" in tags:
        severity = max(severity, 0.7)
    if "axiological_degradation" in tags:
        severity = max(severity, 0.65)

    if severity >= 0.8:
        action = "escalate"
    elif severity >= 0.6:
        action = "quarantine"
    elif severity >= 0.4:
        action = "transform"
    else:
        action = "allow"

    mapping = _load_json_mapping()
    iel_module = _resolve_iel_module(domain, mapping)

    overlay_notes = None
    if overlay_module:
        try:
            mod = importlib.import_module(overlay_module)
            if hasattr(mod, "refine_analysis"):
                result = mod.refine_analysis(classification)
                if isinstance(result, dict):
                    action = result.get("action", action)
                    new_severity = result.get("severity")
                    if isinstance(new_severity, (int, float)):
                        severity = float(new_severity)
                    iel_candidate = result.get("iel_module")
                    if isinstance(iel_candidate, str):
                        iel_module = iel_candidate
                overlay_notes = "overlay_applied"
            else:
                overlay_notes = "overlay_loaded_noop"
        except Exception as exc:
            overlay_notes = f"overlay_error:{type(exc).__name__}"

    notes = overlay_notes or "baseline"
    rationale = {
        "tags": tags,
        "domain": domain,
        "severity": round(severity, 3),
        "mapping_loaded": bool(mapping),
        "has_markdown": bool(_load_privations_markdown()),
    }

    return PrivationAnalysis(
        severity=round(severity, 3),
        action=action,
        iel_module=iel_module,
        notes=notes,
        rationale=rationale,
    )
