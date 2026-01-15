#!/usr/bin/env python3
from __future__ import annotations
import argparse, dataclasses, datetime as _dt, difflib, hashlib, json, re
from pathlib import Path
from typing import Iterable, Optional

DEV_HEADER_BEGIN = re.compile(r"^#\s*LOGOS_HEADER:\s*v1\s*$")
DEV_HEADER_END = re.compile(r"^#\s*END_LOGOS_HEADER\s*$")
PROD_FENCE_BEGIN = re.compile(r"^#\s*=+\s*$")
PROD_TAG = re.compile(r"^#\s*LOGOS PRODUCTION HEADER\s*$")

SIDE_EFFECT_PATTERNS = {
    "filesystem": re.compile(r"\b(open\(|Path\(|write_text\(|read_text\(|unlink\(|remove\(|rmtree\(|mkdir\(|chmod\(|rename\(|shutil\.)", re.IGNORECASE),
    "subprocess": re.compile(r"\b(subprocess\.|os\.system\(|Popen\(|run\(|check_output\()", re.IGNORECASE),
    "network": re.compile(r"\b(requests\.|httpx\.|urllib\.|socket\.|aiohttp\.)", re.IGNORECASE),
}
ENTRYPOINT_PATTERNS = {
    "main_guard": re.compile(r"if\s+__name__\s*==\s*[\"']__main__[\"']\s*:", re.IGNORECASE),
    "argparse": re.compile(r"\bargparse\b", re.IGNORECASE),
    "click": re.compile(r"\bclick\b", re.IGNORECASE),
    "typer": re.compile(r"\btyper\b", re.IGNORECASE),
}

CANON_ABBREVIATIONS = {"ETGC","SMP","MVS","BDN","PXL","IEL","SOP","SCP","ARP","MTP","TLM","UWM","LEM"}
_BAD_ALIAS = "".join(["E", "G", "TC"])  # avoid embedding forbidden token verbatim
FORBIDDEN_ALIASES = {_BAD_ALIAS: "ETGC"}

@dataclasses.dataclass(frozen=True)
class DevHeaderV1:
    updated_utc: str
    path: str
    role: str
    phase: str
    origin: str
    intended_bucket: str
    side_effects: str
    entrypoints: str
    depends_on: str
    notes: str

@dataclasses.dataclass(frozen=True)
class ProductionHeader:
    file: str
    role: str
    layer: str
    tags: list[str]
    purpose: str
    inputs: list[str]
    outputs: list[str]
    runtime_status: str
    side_effects: list[str]
    governance: list[str]
    generated_utc: str

def utc_now() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def read_text(path: Path, max_bytes: int = 800_000) -> str:
    b = path.read_bytes()
    if len(b) > max_bytes:
        b = b[:max_bytes]
    return b.decode("utf-8", errors="ignore")

def parse_dev_header_v1(lines: list[str]) -> Optional[DevHeaderV1]:
    begin = None
    end = None
    for i, ln in enumerate(lines[:300]):
        if begin is None and DEV_HEADER_BEGIN.match(ln.strip()):
            begin = i
            continue
        if begin is not None and DEV_HEADER_END.match(ln.strip()):
            end = i
            break
    if begin is None:
        return None
    if end is None:
        raise ValueError("DEV_HEADER_BEGIN found but END_LOGOS_HEADER missing")

    kv: dict[str,str] = {}
    for ln in lines[begin+1:end]:
        s = ln.strip()
        if not s.startswith("#"):
            continue
        s = s.lstrip("#").strip()
        if not s or ":" not in s:
            continue
        k, v = s.split(":", 1)
        kv[k.strip()] = v.strip()

    required = ["updated_utc","path","role","phase","origin","intended_bucket","side_effects","entrypoints","depends_on","notes"]
    missing = [k for k in required if k not in kv]
    if missing:
        raise ValueError(f"DEV_HEADER_V1 missing keys: {missing}")
    return DevHeaderV1(**{k: kv[k] for k in required})

def preserve_preamble(lines: list[str]) -> tuple[list[str], list[str]]:
    pre: list[str] = []
    rest = lines[:]
    if rest and rest[0].startswith("#!"):
        pre.append(rest.pop(0))
    if rest:
        if "coding" in rest[0]:
            pre.append(rest.pop(0))
        elif len(rest) > 1 and "coding" in rest[1]:
            pre.append(rest.pop(0))
            pre.append(rest.pop(0))
    return pre, rest

def strip_existing_production_header(lines: list[str]) -> list[str]:
    tag_idx = None
    for i, ln in enumerate(lines[:250]):
        if PROD_TAG.match(ln.strip()):
            tag_idx = i
            break
    if tag_idx is None:
        return lines
    start = 0
    for j in range(tag_idx, -1, -1):
        if PROD_FENCE_BEGIN.match(lines[j].strip()):
            start = j
            break
    end = None
    for k in range(tag_idx, min(len(lines), tag_idx + 400)):
        if lines[k].strip() == "":
            end = k + 1
            break
    if end is None:
        end = min(len(lines), tag_idx + 80)
    return lines[:start] + lines[end:]

def infer_role_layer_tags(path: Path) -> tuple[str,str,list[str]]:
    p = path.as_posix().lower()
    tags: list[str] = []
    if "advanced_reasoning_protocol" in p or "/arp" in p:
        tags.append("ARP"); layer = "ARP"
    elif "synthetic_cognition_protocol" in p or "/scp" in p:
        tags.append("SCP"); layer = "SCP"
    elif "meaning_translation_protocol" in p or "/mtp" in p:
        tags.append("MTP"); layer = "MTP"
    elif "system_operations_protocol" in p or "/sop" in p:
        tags.append("SOP"); layer = "SOP"
    elif "logos_protocol" in p:
        tags.append("LOGOS_PROTOCOL"); layer = "LOGOS_PROTOCOL"
    else:
        layer = "DEV_RESOURCES"
    role = "Utility Script"
    name = path.name.lower()
    if "/system_audit/" in p or "scan_" in name or "audit" in name:
        role = "Repository Audit Utility"; tags.append("AUDIT")
    if "smoke" in p or name.startswith("test_"):
        role = "Smoke Test Utility"; tags.append("TEST")
    if "tool" in p and "registry" in p:
        role = "Tool Registry Utility"; tags.append("TOOL_REGISTRY")
    if "diagnostic" in p or "trace" in p:
        tags.append("DIAGNOSTICS")
    tags = sorted(set(t.upper() for t in tags))
    return role, layer, tags

def infer_side_effects(text: str) -> list[str]:
    hits = [n for n,pat in SIDE_EFFECT_PATTERNS.items() if pat.search(text)]
    return hits or ["none_detected"]

def infer_entrypoints(text: str) -> list[str]:
    eps = []
    if ENTRYPOINT_PATTERNS["main_guard"].search(text): eps.append("__main__")
    if ENTRYPOINT_PATTERNS["argparse"].search(text): eps.append("argparse")
    if ENTRYPOINT_PATTERNS["click"].search(text): eps.append("click")
    if ENTRYPOINT_PATTERNS["typer"].search(text): eps.append("typer")
    return eps or ["unknown"]

def infer_purpose(text: str) -> str:
    m = re.match(r'^\s*(?:"""|\'\'\')\s*(.+?)\s*(?:"""|\'\'\')', text, flags=re.DOTALL)
    if m:
        first = m.group(1).strip().splitlines()[0].strip()
        if first:
            return first[:220]
    return "TODO: describe module responsibility."

def enforce_abbrev_policy(text: str) -> None:
    for bad, good in FORBIDDEN_ALIASES.items():
        if re.search(rf"\b{re.escape(bad)}\b", text):
            raise ValueError(f"Forbidden alias '{bad}' found; must use '{good}'")

def build_production_header(path: Path, dev: Optional[DevHeaderV1], text: str) -> ProductionHeader:
    role, layer, tags = infer_role_layer_tags(path)
    purpose = infer_purpose(text)
    side_effects = infer_side_effects(text)
    entrypoints = infer_entrypoints(text)
    inputs = ["(auto) CLI args / environment (if applicable)"]
    outputs = ["(auto) filesystem artifacts (if applicable)"]
    runtime_status = "Not required for core runtime execution; developer-invoked utility."
    if layer == "LOGOS_PROTOCOL":
        runtime_status = "Runtime-adjacent; ensure changes are reviewed against runtime surface maps."
    governance = [
        "Fail-closed on missing dependencies or invalid state.",
        "Do not modify Coq stacks unless explicitly unlocked by policy.",
        f"Canonical abbreviations enforced: {', '.join(sorted(CANON_ABBREVIATIONS))}.",
    ]
    if dev:
        if dev.side_effects.strip().lower() not in {"unknown",""}:
            side_effects = [s.strip() for s in dev.side_effects.split(",") if s.strip()] or side_effects
        if dev.entrypoints.strip().lower() not in {"unknown",""}:
            entrypoints = [e.strip() for e in dev.entrypoints.split(",") if e.strip()] or entrypoints
        tags = sorted(set(tags + [dev.intended_bucket.upper(), dev.origin.upper()]))
    tags = sorted(set(tags + [f"ENTRY_{e.upper()}" for e in entrypoints]))
    return ProductionHeader(
        file=str(path),
        role=role,
        layer=layer,
        tags=tags,
        purpose=purpose,
        inputs=inputs,
        outputs=outputs,
        runtime_status=runtime_status,
        side_effects=side_effects,
        governance=governance,
        generated_utc=utc_now(),
    )

def render_production_header(h: ProductionHeader) -> str:
    lines = []
    lines.append("# ============================\n")
    lines.append("# LOGOS PRODUCTION HEADER\n")
    lines.append("# ============================\n")
    lines.append(f"# File: {Path(h.file).as_posix()}\n")
    lines.append(f"# Role: {h.role}\n")
    lines.append(f"# Layer: {h.layer}\n")
    lines.append(f"# Tags: {', '.join(h.tags)}\n")
    lines.append("#\n")
    lines.append("# Purpose:\n")
    lines.append(f"#   {h.purpose}\n")
    lines.append("#\n")
    lines.append("# Inputs:\n")
    for x in h.inputs: lines.append(f"#   - {x}\n")
    lines.append("#\n")
    lines.append("# Outputs:\n")
    for x in h.outputs: lines.append(f"#   - {x}\n")
    lines.append("#\n")
    lines.append("# Runtime Status:\n")
    lines.append(f"#   {h.runtime_status}\n")
    lines.append("#\n")
    lines.append("# Side Effects:\n")
    lines.append(f"#   {', '.join(h.side_effects)}\n")
    lines.append("#\n")
    lines.append("# Governance:\n")
    for g in h.governance: lines.append(f"#   - {g}\n")
    lines.append("#\n")
    lines.append(f"# Generated: {h.generated_utc} (UTC)\n")
    lines.append("# ============================\n\n")
    return "".join(lines)

def unified_diff(a: str, b: str, fromfile: str, tofile: str) -> str:
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), fromfile=fromfile, tofile=tofile))

def iter_py_files(root: Path, exclude_dirs: Iterable[str]) -> list[Path]:
    ex = set(exclude_dirs)
    out: list[Path] = []
    for p in root.rglob("*.py"):
        if not p.is_file(): continue
        if any(part in ex for part in p.parts): continue
        out.append(p)
    return sorted(out)
