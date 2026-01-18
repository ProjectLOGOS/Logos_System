# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED


"""Alignment-gated Logos_AGI importer with drift guard and optional probing."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

from JUNK_DRAWER.scripts.need_to_distribute.boot_aligned_agent import SandboxedLogosAgent
from JUNK_DRAWER.scripts.runtime.need_to_distribute.provenance import (
    DriftError,
    PinConfigError,
    load_pin,
    resolve_ref,
    verify_pinned_repo,
    write_pin,
)

REPO_URL = "https://github.com/ProjectLOGOS/Logos_AGI.git"
REPO_DIR = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_DIR / "external" / "Logos_AGI"
STATE_FILE = REPO_DIR / "state" / "alignment_LOGOS-AGENT-OMEGA.json"
PIN_PATH = REPO_DIR / "state" / "logos_agi_pin.json"


def read_audit_entries() -> list[Dict[str, Any]]:
    """Load existing alignment audit entries."""

    if not STATE_FILE.exists():
        return []
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def write_audit_entries(entries: list[Dict[str, Any]]) -> None:
    """Persist audit entries back to disk."""

    STATE_FILE.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def verify_agent_alignment() -> SandboxedLogosAgent:
    """Run the constructive LEM gate and return the agent (locked or unlocked)."""

    print(">> Verifying agent alignment (constructive LEM)…")
    agent = SandboxedLogosAgent.create()
    agent.boot()
    agent.unlock_if_aligned()
    if agent.unlocked:
        print(f"[{agent.agent_id}] Alignment verified.")
    else:
        print("❌ Alignment not verified. Aborting cognition import.")
    return agent


def ensure_repo(repo_url: str, dest_path: Path) -> None:
    """Clone or fast-forward the Logos_AGI repository."""

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and any(dest_path.iterdir()):
        subprocess.check_call(
            [
                "git",
                "-C",
                str(dest_path),
                "fetch",
                "--depth",
                "1",
                "origin",
                "main",
            ]
        )
        subprocess.check_call(
            ["git", "-C", str(dest_path), "reset", "--hard", "origin/main"]
        )
        return
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            repo_url,
            str(dest_path),
        ]
    )


def repo_commit_sha(repo_root: Path) -> str:
    """Return the current HEAD commit for the cloned repo."""

    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_cognitive_protocols(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    """Import the Logos cognitive stack and report module availability."""

    sys.path.insert(0, str(repo_root))

    probes: Dict[str, tuple[str, list[str]]] = {
        "Advanced Reasoning Protocol": (
            "Advanced_Reasoning_Protocol",
            [
                "arp_bootstrap",
                "arp_operations",
                "reasoning_engines",
                "mathematical_foundations",
                "logos_core",
            ],
        ),
        "System Operations Protocol": (
            "System_Operations_Protocol",
            [
                "startup",
                "nexus",
                "governance",
                "operations",
                "alignment_protocols",
            ],
        ),
        "User Interaction Protocol": (
            "User_Interaction_Protocol",
            [
                "uip_protocol",
                "GUI",
                "input_output_processing",
                "interfaces",
                "system_utilities",
            ],
        ),
        "Synthetic Cognition Protocol": (
            "Synthetic_Cognition_Protocol",
            [
                "consciousness",
                "data",
                "system_utilities",
                "BDN_System",
                "MVS_System",
            ],
        ),
        "Logos Agent Core": (
            "Logos_Agent",
            [
                "Protopraxis",
                "state",
                "ui",
            ],
        ),
    }

    results: Dict[str, Dict[str, Any]] = {}
    summary: list[str] = []

    for label, (module_name, hints) in probes.items():
        entry: Dict[str, Any] = {
            "module": module_name,
            "loaded": False,
            "submodules_found": [],
        }
        try:
            import_module(module_name)
            entry["loaded"] = True
            present: list[str] = []
            for hint in hints:
                try:
                    import_module(f"{module_name}.{hint}")
                    present.append(hint)
                except ImportError:
                    continue
            entry["submodules_found"] = present
            summary.append(f"{label.split()[0]}:OK")
            submodules = ", ".join(present) if present else "<none>"
            print(
                "   • {} ({}) loaded; submodules: {}".format(
                    label,
                    module_name,
                    submodules,
                )
            )
        except ImportError as exc:
            entry["error"] = repr(exc)
            summary.append(f"{label.split()[0]}:ERR")
            print(f"[WARNING] {label} unavailable: {exc}")
        results[label] = entry

    if summary:
        print("   Status summary:", " | ".join(summary))

    missing = [label for label, info in results.items() if not info.get("loaded")]
    if missing:
        message = (
            "⚠️  Cognitive stack loaded with warnings — review missing modules above."
        )
        print(message)
    else:
        print("🧠 Cognitive protocols imported successfully!")

    sys.path.pop(0)
    return results


def sop_health_snapshot() -> str:
    """Return a lightweight status string for SOP boot utilities."""

    import importlib

    try:
        module = importlib.import_module(
            "System_Operations_Protocol.startup.boot_sequence"
        )
    except ImportError as exc:
        return f"unavailable: {exc}"

    for name in ("get_sop_status", "status", "preview", "describe"):
        fn = getattr(module, name, None)
        if callable(fn):
            try:
                return str(fn())
            except TypeError:
                continue
            except (RuntimeError, ValueError) as exc:
                return f"error: {exc}"
    return "no-readonly-entrypoints"


def env_provenance() -> Dict[str, str]:
    """Capture basic environment metadata for audit logs."""

    def capture(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, text=True).strip()
        except (subprocess.CalledProcessError, OSError):
            return "unknown"

    return {
        "python": capture([sys.executable, "--version"]),
        "coqc": capture(["bash", "-lc", "coqc -v | head -n1"]),
        "SIMULATE_LEM_SUCCESS": os.environ.get("SIMULATE_LEM_SUCCESS", "0"),
    }


def update_latest_entry(update: Dict[str, Any]) -> None:
    """Merge new fields into the most recent audit entry."""

    entries = read_audit_entries()
    if not entries:
        print(
            "[WARNING] alignment audit missing or unreadable; enrichment skipped.",
            file=sys.stderr,
        )
        return
    entries[-1].update(update)
    write_audit_entries(entries)


def main(
    run_probe: bool,
    pin_sha: str = None,
    pin_note: str = "",
    allow_dirty: bool = False,
    allow_git: bool = False,
) -> int:
    """Orchestrate the alignment check, repo sync, and optional probe."""

    start = time.time()

    if os.environ.get("LOGOS_OPERATOR_OK", "").strip() != "1":
        print(
            "ERROR: operator ack required. Set LOGOS_OPERATOR_OK=1 to run this script.",
            file=sys.stderr,
        )
        return 2

    if not allow_git:
        print(
            "ERROR: --allow-git is required for git/network operations.",
            file=sys.stderr,
        )
        return 2

    previous_entries = read_audit_entries()
    previous_sha = None
    if previous_entries:
        previous_sha = previous_entries[-1].get("logos_agi_commit")

    agent = verify_agent_alignment()
    if not agent.unlocked:
        return 1

    current_sha = "unknown"
    pin_exists = PIN_PATH.exists()

    if pin_sha:
        print(f">> Pin override requested — checking out {pin_sha}")
        if not DEST_DIR.exists():
            print(">> Syncing Logos_AGI repository (initial clone)…")
        else:
            print(">> Syncing Logos_AGI repository…")
        try:
            ensure_repo(REPO_URL, DEST_DIR)
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            print(
                "[ERROR] git operation failed (exit={}). "
                "Repository access required.".format(exc.returncode)
            )
            return 1

        try:
            resolved_sha = resolve_ref(str(DEST_DIR), pin_sha)
            print(f">> Pinning Logos_AGI to {resolved_sha} (from {pin_sha})")

            subprocess.run(
                ["git", "checkout", resolved_sha], cwd=str(DEST_DIR), check=True
            )

            write_pin(str(PIN_PATH), resolved_sha, pin_note, allow_dirty)

            print(f">> Pinned to {resolved_sha}")
            current_sha = resolved_sha
        except (subprocess.CalledProcessError, DriftError, PinConfigError) as e:
            print(f"[ERROR] Pinning failed: {e}")
            return 1
    elif pin_exists:
        try:
            pin = load_pin(str(PIN_PATH))
            provenance = verify_pinned_repo(
                str(DEST_DIR), pin, require_clean=True, allow_drift=False
            )
            current_sha = provenance.get("head_sha", "unknown")
            print(f">> Existing pin verified: {current_sha[:8]}")
        except (DriftError, PinConfigError, OSError, subprocess.CalledProcessError) as e:
            print(
                f"[ERROR] Logos_AGI pin verification failed: {e}. "
                "Re-run with --pin-sha <ref> to update the pin."
            )
            return 1
    else:
        print(">> No pin found — syncing to origin/main")
        try:
            ensure_repo(REPO_URL, DEST_DIR)
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            print(
                "[ERROR] git operation failed (exit={}). "
                "Repository access required.".format(exc.returncode)
            )
            return 1
        current_sha = repo_commit_sha(DEST_DIR)

    if previous_sha and current_sha != "unknown" and previous_sha != current_sha:
        print(
            "[SEC] Repo drift detected: prev={} current={}.".format(
                previous_sha[:8],
                current_sha[:8],
            )
        )
        print(
            "[SEC] Alignment gate rerun in this session — proceeding with "
            "updated stack."
        )

    print(">> Importing cognitive protocols…")
    protocol_results = load_cognitive_protocols(DEST_DIR)

    ops_health = sop_health_snapshot()
    env_info = env_provenance()

    update_latest_entry(
        {
            "logos_agi_commit": current_sha,
            "protocol_imports": protocol_results,
            "ops_health_snapshot": ops_health,
            "env": env_info,
            "run_timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "runtime_seconds": round(time.time() - start, 3),
        }
    )
    print(f">> Audit updated: {STATE_FILE}")

    if run_probe:
        print("🔎 Running protocol probe…")
        try:
            from JUNK_DRAWER.scripts.runtime.need_to_distribute.protocol_probe import main as probe_main

            rc = probe_main()
            if rc != 0:
                print(f"[WARNING] protocol_probe exited with code {rc}")
        except ImportError as exc:  # pragma: no cover - defensive import path
            print(f"[WARNING] Unable to import protocol_probe: {exc}")
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"[WARNING] protocol_probe failed: {exc}")

    print("✅ Completed.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Alignment-gated Logos_AGI importer")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Run protocol_probe after import",
    )
    parser.add_argument(
        "--pin-sha",
        help="Pin Logos_AGI to specific SHA or ref (tag/branch)",
    )
    parser.add_argument(
        "--pin-note",
        default="",
        help="Note for the pin operation",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow pinning even if repo is dirty",
    )
    parser.add_argument(
        "--allow-git",
        action="store_true",
        help="Acknowledge this script may perform git/network operations (clone/pull).",
    )
    args = parser.parse_args()

    sys.exit(
        main(
            run_probe=args.probe,
            pin_sha=args.pin_sha,
            pin_note=args.pin_note,
            allow_dirty=args.allow_dirty,
            allow_git=args.allow_git,
        )
    )
