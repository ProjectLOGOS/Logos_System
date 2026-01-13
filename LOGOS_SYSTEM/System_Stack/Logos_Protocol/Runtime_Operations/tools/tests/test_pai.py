#!/usr/bin/env python3
"""
Test script for Persistent Agent Identity (PAI) implementation.
"""

import sys
import json
from pathlib import Path

# Add paths
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "Logos_AGI"))
from external.Logos_AGI.identity_paths import CANONICAL_IDENTITY_PATH


def test_pai():
    """Test PAI functionality."""
    print("=== Testing Persistent Agent Identity ===\n")

    # Test 1: Create agent identity module
    print("1. Testing AgentIdentity module...")
    try:
        from logos_core.governance.agent_identity import (
            load_or_create_identity,
            validate_identity,
            update_identity,
            identity_hash,
            PersistentAgentIdentity,
        )

        print("✓ AgentIdentity module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AgentIdentity: {e}")
        return

    # Test 2: Load or create identity
    print("\n2. Testing identity creation...")
    try:
        theory_hash = "test_theory_hash_123"
        identity = load_or_create_identity(theory_hash, REPO_ROOT)
        print(f"✓ Identity created/loaded: {identity['agent_id']}")
        print(f"  - Version: {identity['identity_version']}")
        print(f"  - Theory hash: {identity['proof_gate']['theory_hash']}")
    except Exception as e:
        print(f"✗ Failed to create identity: {e}")
        return

    # Test 3: Identity validation
    print("\n3. Testing identity validation...")
    try:
        mission_file = REPO_ROOT / "state" / "mission_profile.json"
        catalog_file = REPO_ROOT / "training_data" / "index" / "catalog.jsonl"
        is_valid, reason = validate_identity(
            identity, mission_file, catalog_file, REPO_ROOT
        )
        print(f"✓ Identity validation: {'PASS' if is_valid else 'FAIL'} - {reason}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")

    # Test 4: Identity hash computation
    print("\n4. Testing identity hash...")
    try:
        hash1 = identity_hash(identity, REPO_ROOT)
        hash2 = identity_hash(identity, REPO_ROOT)
        print(f"✓ Identity hash: {hash1[:16]}...")
        print(f"✓ Hash consistency: {'PASS' if hash1 == hash2 else 'FAIL'}")
    except Exception as e:
        print(f"✗ Hash computation failed: {e}")

    # Test 5: Identity update
    print("\n5. Testing identity update...")
    try:
        # Create a test catalog entry
        catalog_file = REPO_ROOT / "training_data" / "index" / "catalog.jsonl"
        catalog_file.parent.mkdir(parents=True, exist_ok=True)

        test_entry = {
            "entry_id": "test_entry_123",
            "timestamp": "2025-12-27T01:00:00Z",
            "target_module": "test.module",
            "improvement_type": "function",
            "policy_class": "repair",
            "stage_ok": True,
            "deployed": True,
            "deployment_path": "test/path.py",
            "code_hash": "abcd1234",
        }

        with open(catalog_file, "w") as f:
            f.write(json.dumps(test_entry) + "\n")

        updated = update_identity(
            identity,
            mission_file,
            catalog_file,
            "test_entry_123",
            None,
            "test_run_456",
            REPO_ROOT,
        )

        print("✓ Identity updated successfully")
        print(f"  - Last entry ID: {updated['capabilities']['last_entry_id']}")
        print(f"  - Last cycle: {updated['continuity']['last_cycle_utc']}")

        # Verify hash chain
        old_hash = identity["continuity"]["prev_identity_hash"]
        new_hash = identity_hash(updated, REPO_ROOT)
        print(
            f"✓ Hash chain: prev={old_hash[:16] if old_hash else 'None'}..., current={new_hash[:16]}..."
        )

    except Exception as e:
        print(f"✗ Update failed: {e}")

    # Test 6: Check identity file persistence
    print("\n6. Testing identity persistence...")
    try:
        identity_file = CANONICAL_IDENTITY_PATH
        if identity_file.exists():
            with open(identity_file, "r") as f:
                persisted = json.load(f)
            print("✓ Identity file exists and is valid JSON")
            print(f"  - Agent ID: {persisted['agent_id']}")
            print(f"  - Last updated: {persisted['updated_utc']}")
        else:
            print("✗ Identity file not found")
    except Exception as e:
        print(f"✗ Persistence check failed: {e}")

    print("\n=== PAI Tests Complete ===")


if __name__ == "__main__":
    test_pai()
