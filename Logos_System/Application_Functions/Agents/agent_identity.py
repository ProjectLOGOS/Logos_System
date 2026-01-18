# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""
Persistent Agent Identity (PAI)
===============================

Versioned, auditable, cross-context continuity layer for LOGOS AGI.
Provides stable identity binding across proof-gate, capabilities, mission, and continuity.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import subprocess

from Logos_System.System_Stack.Logos_Protocol.Unified_Working_Memory.World_Modeling.commitment_ledger import (
    DEFAULT_LEDGER_PATH as DEFAULT_COMMITMENT_LEDGER_PATH,
    LEDGER_VERSION as COMMITMENT_LEDGER_VERSION,
    compute_ledger_hash as compute_commitment_ledger_hash,
    validate_ledger as validate_commitment_ledger,
)


class PersistentAgentIdentity:
    """
    Manages the persistent agent identity record with validation and continuity.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent

        self.base_dir = base_dir
        self.identity_path = base_dir / "state" / "agent_identity.json"
        self.identity_path.parent.mkdir(parents=True, exist_ok=True)

        # Stable agent ID - deterministic for this workspace
        self.agent_id = self._generate_stable_agent_id()

    def _generate_stable_agent_id(self) -> str:
        """Generate a stable, deterministic agent ID for this workspace."""
        workspace_path = str(self.base_dir.resolve())
        stable_seed = f"LOGOS-AGI-{workspace_path}"
        return f"LOGOS-{str(uuid.uuid5(uuid.NAMESPACE_DNS, stable_seed))[:8].upper()}"

    def load_or_create_identity(self, theory_hash: str) -> Dict[str, Any]:
        """
        Load existing identity or create new one if it doesn't exist.
        """
        if self.identity_path.exists():
            try:
                with open(self.identity_path, 'r') as f:
                    identity = json.load(f)
                self._validate_identity_structure(identity)
                return identity
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Invalid identity file, recreating: {e}")
                # Fall through to create new

        # Create new identity
        identity = self._create_new_identity(theory_hash)
        self._save_identity(identity)
        return identity

    def _create_new_identity(self, theory_hash: str) -> Dict[str, Any]:
        """Create a new identity record."""
        now = datetime.now(timezone.utc).isoformat()

        # Get git info if available
        git_commit, git_branch = self._get_git_info()

        return {
            "agent_id": self.agent_id,
            "identity_version": 1,
            "created_utc": now,
            "updated_utc": now,
            "repo": {
                "root": str(self.base_dir),
                "git_commit": git_commit,
                "branch": git_branch
            },
            "proof_gate": {
                "theory_hash": theory_hash,
                "coq_build_hash": None,
                "axiom_policy_profile": "PXL_TRUSTED_CORE_V1"
            },
            "capabilities": {
                "catalog_path": "training_data/index/catalog.jsonl",
                "catalog_tail_hash": None,
                "deployed_set_hash": None,
                "last_entry_id": None
            },
            "mission": {
                "mission_label": "DEMO_STABLE",
                "mission_profile_hash": None,
                "allow_enhancements": False
            },
            "continuity": {
                "last_cycle_utc": None,
                "last_planner_digest": None,
                "last_run_id": None,
                "prev_identity_hash": None
            },
            "world_model": {
                "snapshot_path": "state/world_model_snapshot.json",
                "snapshot_hash": None,
                "world_model_version": 1
            },
            "commitments": {
                "ledger_path": DEFAULT_COMMITMENT_LEDGER_PATH.as_posix(),
                "ledger_hash": None,
                "ledger_version": COMMITMENT_LEDGER_VERSION,
            },
        }

    def _get_git_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Get current git commit and branch."""
        try:
            # Get commit hash
            commit_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=5
            )
            commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None

            # Get branch name
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=5
            )
            branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None

            return commit, branch
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None, None

    def validate_identity(self, identity: Dict[str, Any], mission_profile_path: Optional[Path] = None,
                         catalog_path: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Validate identity against current system state.
        Returns (is_valid, reason)
        """
        try:
            warnings: List[str] = []
            self._validate_identity_structure(identity)
            # Check catalog exists and is readable
            if catalog_path is None:
                catalog_path = self.base_dir / identity["capabilities"]["catalog_path"]

            if not catalog_path.exists():
                return False, f"Catalog file does not exist: {catalog_path}"

            if not catalog_path.is_file():
                return False, f"Catalog path is not a file: {catalog_path}"

            try:
                with open(catalog_path, 'r') as f:
                    f.read(1024)  # Just check we can read
            except Exception as e:
                return False, f"Cannot read catalog file: {e}"

            # Check mission profile hash if mission profile exists
            if mission_profile_path and mission_profile_path.exists():
                expected_hash = identity["mission"]["mission_profile_hash"]
                if expected_hash:
                    actual_hash = self._compute_file_hash(mission_profile_path)
                    if actual_hash != expected_hash:
                        return False, f"Mission profile hash mismatch: expected {expected_hash}, got {actual_hash}"

            # Validate world model snapshot if recorded
            world_model = identity.get("world_model")
            if world_model:
                snapshot_path_val = world_model.get("snapshot_path")
                snapshot_hash = world_model.get("snapshot_hash")
                if snapshot_path_val and snapshot_hash:
                    snapshot_path = Path(snapshot_path_val)
                    if not snapshot_path.is_absolute():
                        snapshot_path = self.base_dir / snapshot_path
                    if not snapshot_path.exists():
                        return False, f"World model snapshot missing: {snapshot_path}"
                    try:
                        # Import lazily to avoid hard dependency if module unavailable
                        from Logos_Protocol.logos_core.world_model import uwm as world_model_module  # type: ignore
                    except ImportError as exc:
                        return False, f"World model validation unavailable: {exc}"

                    snapshot_data = world_model_module.load_snapshot(snapshot_path)
                    if not snapshot_data:
                        return False, "World model snapshot unreadable"

                    ok, reasons = world_model_module.validate_snapshot(
                        snapshot_data,
                        identity,
                        self.base_dir,
                    )
                    if not ok:
                        joined = "; ".join(reasons)
                        return False, f"World model validation failed: {joined}"

                    stored_hash = world_model.get("snapshot_hash")
                    actual_hash = snapshot_data.get("integrity", {}).get("snapshot_hash")
                    if stored_hash and actual_hash and stored_hash != actual_hash:
                        return False, "World model hash mismatch"

            commitments = identity.get("commitments") or {}
            ledger_path_value = commitments.get("ledger_path") or DEFAULT_COMMITMENT_LEDGER_PATH.as_posix()
            ledger_path = Path(ledger_path_value)
            if not ledger_path.is_absolute():
                ledger_path = self.base_dir / ledger_path
            if not ledger_path.exists():
                warnings.append(f"Commitment ledger missing: {ledger_path}")
            else:
                try:
                    with open(ledger_path, "r", encoding="utf-8") as handle:
                        ledger_data = json.load(handle)
                    if not isinstance(ledger_data, dict):
                        raise ValueError("Commitment ledger root must be an object")
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    return False, f"Commitment ledger unreadable: {exc}"

                ok, ledger_reasons, ledger_warnings = validate_commitment_ledger(ledger_data, identity)
                if not ok:
                    reason = "; ".join(ledger_reasons) if ledger_reasons else "validation failed"
                    return False, f"Commitment ledger invalid: {reason}"
                if ledger_warnings:
                    warnings.extend(str(entry) for entry in ledger_warnings if entry)

                identity_hash = commitments.get("ledger_hash")
                computed_hash = compute_commitment_ledger_hash(ledger_data)
                if identity_hash and identity_hash != computed_hash:
                    return False, (
                        "Commitment ledger hash mismatch: "
                        f"identity={identity_hash} computed={computed_hash}"
                    )
                commitments.setdefault("ledger_version", COMMITMENT_LEDGER_VERSION)
                commitments.setdefault("ledger_path", DEFAULT_COMMITMENT_LEDGER_PATH.as_posix())

            message = "Identity validation successful"
            if warnings:
                joined = "; ".join(warnings)
                message = f"{message} (warnings: {joined})"
            return True, message

        except Exception as e:
            return False, f"Identity validation error: {e}"

    def update_identity(self, identity: Dict[str, Any], mission_profile_path: Optional[Path] = None,
                       catalog_path: Optional[Path] = None, last_entry_id: Optional[str] = None,
                       planner_digest_path: Optional[str] = None, last_run_id: Optional[str] = None,
                       world_model_snapshot_path: Optional[str] = None,
                       world_model_snapshot_hash: Optional[str] = None,
                       world_model_version: Optional[int] = None,
                       commitment_ledger_path: Optional[str] = None,
                       commitment_ledger_hash: Optional[str] = None,
                       commitment_ledger_version: Optional[int] = None) -> Dict[str, Any]:
        """
        Update identity with current system state.
        """
        # Compute previous identity hash before updating
        prev_hash = self.identity_hash(identity)

        # Update timestamps
        now = datetime.now(timezone.utc).isoformat()
        identity["updated_utc"] = now
        identity["continuity"]["last_cycle_utc"] = now

        # Update continuity
        identity["continuity"]["prev_identity_hash"] = prev_hash
        if planner_digest_path:
            identity["continuity"]["last_planner_digest"] = str(planner_digest_path)
        if last_run_id:
            identity["continuity"]["last_run_id"] = last_run_id

        # Update capabilities
        if catalog_path is None:
            catalog_path = self.base_dir / identity["capabilities"]["catalog_path"]

        if catalog_path.exists():
            # Update catalog tail hash
            identity["capabilities"]["catalog_tail_hash"] = self._compute_catalog_tail_hash(catalog_path)

            # Update deployed set hash
            identity["capabilities"]["deployed_set_hash"] = self._compute_deployed_set_hash(catalog_path)

        if last_entry_id:
            identity["capabilities"]["last_entry_id"] = last_entry_id

        # Update mission profile hash
        if mission_profile_path and mission_profile_path.exists():
            identity["mission"]["mission_profile_hash"] = self._compute_file_hash(mission_profile_path)

        # Update world model binding
        world_model = identity.setdefault(
            "world_model",
            {
                "snapshot_path": "state/world_model_snapshot.json",
                "snapshot_hash": None,
                "world_model_version": 1,
            },
        )
        if world_model_snapshot_path:
            world_model["snapshot_path"] = world_model_snapshot_path
        if world_model_snapshot_hash:
            world_model["snapshot_hash"] = world_model_snapshot_hash
        if world_model_version is not None:
            world_model["world_model_version"] = world_model_version

        commitments = identity.setdefault(
            "commitments",
            {
                "ledger_path": DEFAULT_COMMITMENT_LEDGER_PATH.as_posix(),
                "ledger_hash": None,
                "ledger_version": COMMITMENT_LEDGER_VERSION,
            },
        )
        if commitment_ledger_path:
            commitments["ledger_path"] = commitment_ledger_path
        if commitment_ledger_hash:
            commitments["ledger_hash"] = commitment_ledger_hash
        if commitment_ledger_version is not None:
            commitments["ledger_version"] = commitment_ledger_version
        else:
            commitments.setdefault("ledger_version", COMMITMENT_LEDGER_VERSION)

        return identity

    def identity_hash(self, identity: Dict[str, Any]) -> str:
        """Compute canonical hash of identity record."""
        # Create canonical JSON representation
        canonical_json = json.dumps(identity, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_json.encode()).hexdigest()

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file contents."""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _compute_catalog_tail_hash(self, catalog_path: Path, tail_lines: int = 200) -> str:
        """Compute hash of last N lines of catalog."""
        with open(catalog_path, 'r') as f:
            lines = f.readlines()

        # Take last N lines or all if fewer
        tail_content = ''.join(lines[-tail_lines:] if len(lines) > tail_lines else lines)
        return hashlib.sha256(tail_content.encode()).hexdigest()

    def _compute_deployed_set_hash(self, catalog_path: Path) -> str:
        """Compute hash of all deployed artifacts."""
        deployed_entries = []

        with open(catalog_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        if entry.get("deployed", False):
                            # Create stable representation: entry_id + deployment_path + code_hash
                            deployed_entries.append(
                                f"{entry['entry_id']}|{entry.get('deployment_path', '')}|{entry.get('code_hash', '')}"
                            )
                    except json.JSONDecodeError:
                        continue

        # Sort for deterministic ordering
        deployed_entries.sort()

        # Concatenate and hash
        deployed_content = '|'.join(deployed_entries)
        return hashlib.sha256(deployed_content.encode()).hexdigest()

    def _validate_identity_structure(self, identity: Dict[str, Any]):
        """Validate that identity has required structure."""
        if "world_model" not in identity or not isinstance(identity.get("world_model"), dict):
            identity["world_model"] = {
                "snapshot_path": "state/world_model_snapshot.json",
                "snapshot_hash": None,
                "world_model_version": 1,
            }
        else:
            world_model = identity["world_model"]
            world_model.setdefault("snapshot_path", "state/world_model_snapshot.json")
            world_model.setdefault("snapshot_hash", None)
            world_model.setdefault("world_model_version", 1)

        if "commitments" not in identity or not isinstance(identity.get("commitments"), dict):
            identity["commitments"] = {
                "ledger_path": DEFAULT_COMMITMENT_LEDGER_PATH.as_posix(),
                "ledger_hash": None,
                "ledger_version": COMMITMENT_LEDGER_VERSION,
            }
        else:
            commitments = identity["commitments"]
            commitments.setdefault("ledger_path", DEFAULT_COMMITMENT_LEDGER_PATH.as_posix())
            commitments.setdefault("ledger_hash", None)
            commitments.setdefault("ledger_version", COMMITMENT_LEDGER_VERSION)

        required_fields = [
            "agent_id", "identity_version", "created_utc", "updated_utc",
            "repo", "proof_gate", "capabilities", "mission", "continuity", "world_model", "commitments"
        ]

        for field in required_fields:
            if field not in identity:
                raise KeyError(f"Missing required field: {field}")

        # Validate nested structures
        if "root" not in identity["repo"]:
            raise KeyError("Missing repo.root")

        if "theory_hash" not in identity["proof_gate"]:
            raise KeyError("Missing proof_gate.theory_hash")

        if "catalog_path" not in identity["capabilities"]:
            raise KeyError("Missing capabilities.catalog_path")

        if "mission_label" not in identity["mission"]:
            raise KeyError("Missing mission.mission_label")

        # Fields ensured above; nothing further required here

    def _save_identity(self, identity: Dict[str, Any]):
        """Save identity to disk atomically."""
        # Write to temporary file first
        temp_path = self.identity_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(identity, f, indent=2)

        # Atomic rename
        temp_path.replace(self.identity_path)


# Convenience functions
def load_or_create_identity(theory_hash: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load or create agent identity."""
    pai = PersistentAgentIdentity(base_dir)
    return pai.load_or_create_identity(theory_hash)


def validate_identity(identity: Dict[str, Any], mission_profile_path: Optional[Path] = None,
                     catalog_path: Optional[Path] = None, base_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """Validate agent identity."""
    pai = PersistentAgentIdentity(base_dir)
    return pai.validate_identity(identity, mission_profile_path, catalog_path)


def update_identity(identity: Dict[str, Any], mission_profile_path: Optional[Path] = None,
                   catalog_path: Optional[Path] = None, last_entry_id: Optional[str] = None,
                   planner_digest_path: Optional[Path] = None, last_run_id: Optional[str] = None,
                   world_model_snapshot_path: Optional[str] = None,
                   world_model_snapshot_hash: Optional[str] = None,
                   world_model_version: Optional[int] = None,
                   commitment_ledger_path: Optional[str] = None,
                   commitment_ledger_hash: Optional[str] = None,
                   commitment_ledger_version: Optional[int] = None,
                   base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Update agent identity."""
    pai = PersistentAgentIdentity(base_dir)
    updated = pai.update_identity(
        identity,
        mission_profile_path,
        catalog_path,
        last_entry_id,
        planner_digest_path,
        last_run_id,
        world_model_snapshot_path,
        world_model_snapshot_hash,
        world_model_version,
        commitment_ledger_path,
        commitment_ledger_hash,
        commitment_ledger_version,
    )
    pai._save_identity(updated)
    return updated


def identity_hash(identity: Dict[str, Any], base_dir: Optional[Path] = None) -> str:
    """Compute identity hash."""
    pai = PersistentAgentIdentity(base_dir)
    return pai.identity_hash(identity)


def check_identity(identity_path: Optional[Path] = None, catalog_path: Optional[Path] = None,
                  mission_path: Optional[Path] = None, base_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Check identity validation for CLI use.
    Returns (is_valid, status_message)
    """
    if base_dir is None:
        # Try to find repo root by looking for common files
        cwd = Path.cwd()
        # Check if we're in the main repo (has state/, training_data/, etc.)
        if (cwd / "state").exists() and (cwd / "training_data").exists():
            base_dir = cwd
        else:
            # Try parent directories
            for parent in cwd.parents:
                if (parent / "state").exists() and (parent / "training_data").exists():
                    base_dir = parent
                    break
            else:
                # Fallback to cwd
                base_dir = cwd

    if identity_path is None:
        identity_path = base_dir / "state" / "agent_identity.json"

    if catalog_path is None:
        catalog_path = base_dir / "training_data" / "index" / "catalog.jsonl"

    if mission_path is None:
        mission_path = base_dir / "state" / "mission_profile.json"

    try:
        # Check if identity file exists
        if not identity_path.exists():
            return False, f"Identity file does not exist: {identity_path}"

        # Load identity
        with open(identity_path, 'r') as f:
            identity = json.load(f)

        # Validate identity structure first
        pai = PersistentAgentIdentity(base_dir)
        pai._validate_identity_structure(identity)

        # Check identity hash consistency (tamper detection)
        expected_agent_id = pai.agent_id
        actual_agent_id = identity.get('agent_id')
        if actual_agent_id != expected_agent_id:
            return False, f"Identity tampering detected: expected agent_id={expected_agent_id}, got {actual_agent_id}"

        # Validate against system state
        is_valid, reason = pai.validate_identity(identity, mission_path, catalog_path)

        if is_valid:
            agent_id = identity.get('agent_id', 'unknown')
            updated_utc = identity.get('updated_utc', 'unknown')
            return True, f"PASS: agent_id={agent_id} updated_utc={updated_utc}"
        else:
            return False, f"FAIL: {reason}"

    except json.JSONDecodeError as e:
        return False, f"FAIL: Invalid JSON in identity file: {e}"
    except Exception as e:
        return False, f"FAIL: Unexpected error: {e}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LOGOS Agent Identity Management")
    parser.add_argument(
        "--check",
        action="store_true",
        default=True,
        help="Validate existing identity (default action)"
    )
    parser.add_argument(
        "--identity-path",
        type=Path,
        default=None,
        help="Path to agent identity file (default: state/agent_identity.json)"
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=None,
        help="Path to capability catalog (default: training_data/index/catalog.jsonl)"
    )
    parser.add_argument(
        "--mission-path",
        type=Path,
        default=None,
        help="Path to mission profile (default: state/mission_profile.json)"
    )

    args = parser.parse_args()

    if args.check:
        is_valid, message = check_identity(
            identity_path=args.identity_path,
            catalog_path=args.catalog_path,
            mission_path=args.mission_path
        )
        print(message)
        exit(0 if is_valid else 2)
