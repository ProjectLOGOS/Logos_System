# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
"""
Knowledge Catalog
=================

Authoritative on-disk artifact catalog for autonomous gap-filling system.
Provides atomic writes, deterministic IDs, and comprehensive metadata tracking.
"""

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


class KnowledgeCatalog:
    """
    Knowledge catalog for tracking generated artifacts with atomic writes
    and comprehensive metadata.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            # Default to workspace root
            base_dir = Path(__file__).resolve().parent.parent.parent.parent

        self.base_dir = base_dir
        self.index_path = base_dir / "training_data" / "index" / "catalog.jsonl"
        self.artifacts_dir = base_dir / "training_data" / "artifacts"

        # Ensure directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def persist_artifact(self, request: Dict[str, Any], code: str,
                        stage_result: Dict[str, Any], policy_result: Dict[str, Any],
                        generator_method: str) -> str:
        """
        Persist a complete artifact with all metadata.

        Returns the deterministic entry_id.
        """
        # Generate deterministic entry_id
        entry_id = self._generate_entry_id(request, code)

        # Create artifact directory
        artifact_dir = self.artifacts_dir / entry_id
        artifact_dir.mkdir(exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # Prepare metadata
        artifact_data = {
            "entry_id": entry_id,
            "improvement_id": request.get("improvement_id"),
            "description": request.get("description"),
            "target_module": request.get("target_module"),
            "improvement_type": request.get("improvement_type"),
            "gap_category": request.get("requirements", {}).get("gap_category"),
            "code_hash": hashlib.sha256(code.encode()).hexdigest(),
            "timestamp": timestamp
        }

        stage_data = {
            "entry_id": entry_id,
            "compile_ok": stage_result.get("compile_ok", False),
            "import_ok": stage_result.get("import_ok", False),
            "smoke_test_ok": stage_result.get("smoke_test_ok", False),
            "stage_ok": stage_result.get("stage_ok", False),
            "errors": stage_result.get("errors", []),
            "timestamp": timestamp
        }

        policy_data = {
            "entry_id": entry_id,
            "policy_class": policy_result.get("policy_class"),
            "deploy_allowed": policy_result.get("deploy_allowed"),
            "reasoning": policy_result.get("reasoning", ""),
            "timestamp": timestamp
        }

        provenance_data = {
            "entry_id": entry_id,
            "generator_method": generator_method,
            "repo_relative_paths": {
                "artifact_dir": f"training_data/artifacts/{entry_id}",
                "catalog_index": "training_data/index/catalog.jsonl"
            },
            "workspace_root": str(self.base_dir),
            "timestamp": timestamp
        }

        # Atomic writes
        self._write_json_atomic(artifact_dir / "artifact.json", artifact_data)
        self._write_json_atomic(artifact_dir / "stage.json", stage_data)
        self._write_json_atomic(artifact_dir / "policy.json", policy_data)
        self._write_json_atomic(artifact_dir / "provenance.json", provenance_data)
        self._write_code_atomic(artifact_dir / "code.py", code)

        return entry_id

    def update_deployment_result(self, entry_id: str, deployment_result: Dict[str, Any]):
        """Update catalog entry with deployment result."""
        artifact_dir = self.artifacts_dir / entry_id
        deploy_data = {
            "entry_id": entry_id,
            "deployed": deployment_result.get("deployed", False),
            "deployment_path": deployment_result.get("deployment_path"),
            "error": deployment_result.get("error"),
            "checksum": deployment_result.get("checksum"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        self._write_json_atomic(artifact_dir / "deploy.json", deploy_data)

        # Update catalog index
        self._update_catalog_index(entry_id, deploy_data)

    def _generate_entry_id(self, request: Dict[str, Any], code: str) -> str:
        """Generate deterministic entry ID."""
        key_components = [
            request.get("target_module", ""),
            request.get("improvement_type", ""),
            request.get("description", ""),
            hashlib.sha256(code.encode()).hexdigest()
        ]

        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def _write_json_atomic(self, path: Path, data: Dict[str, Any]):
        """Write JSON file atomically."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                       dir=path.parent, delete=False) as f:
            json.dump(data, f, indent=2)
            temp_path = Path(f.name)

        temp_path.replace(path)

    def _write_code_atomic(self, path: Path, code: str):
        """Write code file atomically."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                       dir=path.parent, delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)

        temp_path.replace(path)

    def _update_catalog_index(self, entry_id: str, deploy_data: Dict[str, Any]):
        """Update the catalog index with deployment result."""
        # Read existing artifact data
        artifact_dir = self.artifacts_dir / entry_id
        with open(artifact_dir / "artifact.json", 'r') as f:
            artifact_data = json.load(f)

        with open(artifact_dir / "stage.json", 'r') as f:
            stage_data = json.load(f)

        with open(artifact_dir / "policy.json", 'r') as f:
            policy_data = json.load(f)

        # Create index record
        index_record = {
            "entry_id": entry_id,
            "timestamp": artifact_data["timestamp"],
            "target_module": artifact_data["target_module"],
            "improvement_type": artifact_data["improvement_type"],
            "policy_class": policy_data["policy_class"],
            "stage_ok": stage_data["stage_ok"],
            "deployed": deploy_data["deployed"],
            "artifact_path": f"training_data/artifacts/{entry_id}",
            "deployment_path": deploy_data.get("deployment_path")
        }

        # Append to catalog (atomic write)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                       dir=self.index_path.parent, delete=False) as f:
            # Write existing content
            if self.index_path.exists():
                with open(self.index_path, 'r') as existing:
                    f.write(existing.read())

            # Append new record
            f.write(json.dumps(index_record) + '\n')
            temp_path = Path(f.name)

        temp_path.replace(self.index_path)

    def get_artifact_info(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get complete artifact information."""
        artifact_dir = self.artifacts_dir / entry_id
        if not artifact_dir.exists():
            return None

        info = {}

        # Load all metadata files
        for filename in ["artifact.json", "stage.json", "policy.json",
                        "provenance.json", "deploy.json"]:
            filepath = artifact_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    info[filename.replace('.json', '')] = json.load(f)

        return info

    def list_artifacts(self, filter_criteria: Optional[Dict[str, Any]] = None) -> list:
        """List all artifacts, optionally filtered."""
        artifacts = []

        if not self.index_path.exists():
            return artifacts

        with open(self.index_path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if self._matches_filter(record, filter_criteria):
                        artifacts.append(record)

        return artifacts

    def _matches_filter(self, record: Dict[str, Any], filter_criteria: Optional[Dict[str, Any]]) -> bool:
        """Check if record matches filter criteria."""
        if not filter_criteria:
            return True

        for key, value in filter_criteria.items():
            if record.get(key) != value:
                return False

        return True