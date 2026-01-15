#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-15T03:52:00Z
# path: /workspaces/Logos_System/_Dev_Resources/LEGACY_SCRIPTS_TO_EXAMINE/scan_repo.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Repository scanner for the LOGOS knowledge base.

Builds a grounded knowledge base that mirrors the repository state so the
LLM router can reference real data instead of hallucinations.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = Path(os.getenv("LOGOS_STATE_DIR", REPO_ROOT / "state"))


def scan_coq_files() -> Dict[str, Any]:
    """Scan Coq baseline files and capture axiom/lemma metadata."""
    coq_dir = REPO_ROOT / "Protopraxis" / "formal_verification" / "coq" / "baseline"
    if not coq_dir.exists():
        return {"error": "Coq baseline directory not found"}

    coq_files: Dict[str, Any] = {}
    for coq_file in coq_dir.glob("*.v"):
        try:
            content = coq_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            coq_files[str(coq_file.relative_to(REPO_ROOT))] = {"error": str(exc)}
            continue

        lines = content.splitlines()
        axioms = [line.strip() for line in lines if line.strip().startswith("Axiom ")]
        definitions = [
            line.strip() for line in lines if line.strip().startswith("Definition ")
        ]
        lemmas = [
            line.strip()
            for line in lines
            if line.strip().startswith(("Lemma ", "Theorem "))
        ]

        coq_files[str(coq_file.relative_to(REPO_ROOT))] = {
            "axiom_count": len(axioms),
            "axioms": axioms[:10],
            "definition_count": len(definitions),
            "definitions": definitions[:10],
            "lemma_count": len(lemmas),
            "lemmas": lemmas[:10],
            "line_count": len(lines),
        }

    return coq_files


def scan_python_scripts() -> Dict[str, Any]:
    """Scan Python entry points and summarize their docstring."""
    scripts: Dict[str, Any] = {}
    script_dirs = [REPO_ROOT / "scripts", REPO_ROOT / "tools", REPO_ROOT]

    for script_dir in script_dirs:
        if not script_dir.exists():
            continue

        for py_file in script_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                scripts[str(py_file.relative_to(REPO_ROOT))] = {"error": str(exc)}
                continue

            lines = content.splitlines()
            docstring_text = "No docstring found"
            try:
                module = ast.parse(content)
                docstring_text = ast.get_docstring(module) or docstring_text
            except (SyntaxError, ValueError) as exc:
                docstring_text = f"[Docstring unavailable: {exc}]"

            scripts[str(py_file.relative_to(REPO_ROOT))] = {
                "purpose": docstring_text,
                "line_count": len(lines),
                "is_executable": os.access(py_file, os.X_OK),
            }

    return scripts


def scan_documentation() -> Dict[str, Any]:
    """Collect lightweight summaries for Markdown and text files."""
    docs: Dict[str, Any] = {}
    doc_patterns = ["*.md", "*.txt"]
    doc_dirs = [REPO_ROOT, REPO_ROOT / "docs"]

    for doc_dir in doc_dirs:
        if not doc_dir.exists():
            continue

        for pattern in doc_patterns:
            for doc_file in doc_dir.glob(pattern):
                try:
                    content = doc_file.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    docs[str(doc_file.relative_to(REPO_ROOT))] = {"error": str(exc)}
                    continue

                lines = content.splitlines()
                summary: list[str] = []
                for line in lines[:20]:
                    if line.strip():
                        summary.append(line.strip())
                    if len(summary) >= 3:
                        break

                docs[str(doc_file.relative_to(REPO_ROOT))] = {
                    "summary": "\n".join(summary),
                    "line_count": len(lines),
                    "size_kb": len(content) / 1024,
                }

    return docs


def get_repo_structure() -> Dict[str, Any]:
    """Capture key directories and files for quick overview."""
    structure: Dict[str, Any] = {"key_directories": [], "key_files": []}

    for dir_path in [
        "Protopraxis/formal_verification/coq/baseline",
        "scripts",
        "tools",
        "docs",
        "state",
        "sandbox",
        "plugins",
    ]:
        full_path = REPO_ROOT / dir_path
        if full_path.exists():
            structure["key_directories"].append(
                {
                    "path": dir_path,
                    "file_count": len(list(full_path.rglob("*"))),
                    "exists": True,
                }
            )

    for file_path in [
        "README.md",
        "test_lem_discharge.py",
        "scripts/boot_aligned_agent.py",
        "PXL_Global_Bijection.v",
        ".env",
    ]:
        full_path = REPO_ROOT / file_path
        if full_path.exists():
            structure["key_files"].append(
                {
                    "path": file_path,
                    "size_kb": full_path.stat().st_size / 1024,
                    "exists": True,
                }
            )

    return structure


def run_verification_commands() -> Dict[str, Any]:
    """Execute verification helpers to capture their latest output."""
    results: Dict[str, Any] = {}

    try:
        result = subprocess.run(
            ["python3", "tools/axiom_gate.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        results["axiom_gate"] = {
            "stdout": result.stdout,
            "exit_code": result.returncode,
        }
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
        results["axiom_gate"] = {"error": str(exc)}

    try:
        mission_file = STATE_DIR / "mission_profile.json"
        if mission_file.exists():
            content = mission_file.read_text(encoding="utf-8")
            results["mission_profile"] = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        results["mission_profile"] = {"error": str(exc)}

    return results


def build_knowledge_base() -> Dict[str, Any]:
    """Assemble the complete repository knowledge base payload."""
    print("🔍 Scanning repository...")

    timestamp_result = subprocess.run(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "scan_timestamp": timestamp_result.stdout.strip(),
        "repository_root": str(REPO_ROOT),
        "structure": get_repo_structure(),
        "coq_proofs": scan_coq_files(),
        "python_scripts": scan_python_scripts(),
        "documentation": scan_documentation(),
        "verification_state": run_verification_commands(),
    }


def main() -> int:
    """Scan the repository and persist the knowledge base under state/."""
    kb = build_knowledge_base()

    output_file = STATE_DIR / "repository_knowledge_base.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as output_handle:
        json.dump(kb, output_handle, indent=2)

    print(f"\n✅ Knowledge base saved to: {output_file}")
    print("\n📊 Summary:")
    print(f"   Coq files: {len(kb['coq_proofs'])}")
    print(f"   Python scripts: {len(kb['python_scripts'])}")
    print(f"   Documentation files: {len(kb['documentation'])}")
    print(f"   Key directories: {len(kb['structure']['key_directories'])}")

    axiom_status = kb.get("verification_state", {}).get("axiom_gate", {})
    if axiom_status.get("exit_code") == 0:
        print("\n🔒 Verification State:")
        for line in axiom_status.get("stdout", "").splitlines()[:5]:
            if line.strip():
                print(f"   {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
