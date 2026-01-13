#!/usr/bin/env python3
"""Test suite for Logos_AGI commit pinning and drift detection.

Tests cover:
- Missing pin file handling
- Pin match scenarios
- Drift detection and override
- Provenance verification edge cases

All tests use stdlib only and mock git operations where needed.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from JUNK_DRAWER.scripts.runtime.need_to_distribute.provenance import (
    DriftError,
    PinConfigError,
    load_pin,
    verify_pinned_repo,
    write_pin,
    resolve_ref,
    run_git,
    is_git_dirty,
    get_git_head_sha,
)


class TestLogosAgiPinDrift(unittest.TestCase):
    """Test Logos_AGI pinning and drift detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repo_dir = self.temp_dir / "repo"
        self.repo_dir.mkdir()

        # Initialize a git repo
        run_git(["init"], cwd=self.repo_dir)
        run_git(["config", "user.email", "test@example.com"], cwd=self.repo_dir)
        run_git(["config", "user.name", "Test User"], cwd=self.repo_dir)

        # Create initial commit
        (self.repo_dir / "file.txt").write_text("initial")
        run_git(["add", "file.txt"], cwd=self.repo_dir)
        run_git(["commit", "-m", "initial commit"], cwd=self.repo_dir)
        self.initial_sha = get_git_head_sha(str(self.repo_dir))

        # Create second commit
        (self.repo_dir / "file.txt").write_text("modified")
        run_git(["add", "file.txt"], cwd=self.repo_dir)
        run_git(["commit", "-m", "second commit"], cwd=self.repo_dir)
        self.second_sha = get_git_head_sha(str(self.repo_dir))

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_missing_pin_file(self):
        """Test behavior when no pin file exists."""
        pin_path = self.temp_dir / "pin.json"
        with self.assertRaises(PinConfigError):
            load_pin(str(pin_path))

    def test_load_pin_valid(self):
        """Test loading a valid pin file."""
        pin_path = self.temp_dir / "pin.json"
        pin_data = {
            "repo": "ProjectLOGOS/Logos_AGI",
            "pinned_sha": self.initial_sha,
            "pinned_at": "2023-01-01T00:00:00Z",
            "pinned_by": "test",
            "note": "test pin",
            "allow_dirty": False,
        }
        pin_path.write_text(json.dumps(pin_data))

        loaded = load_pin(str(pin_path))
        self.assertEqual(loaded["pinned_sha"], self.initial_sha)
        self.assertEqual(loaded["note"], "test pin")

    def test_load_pin_invalid_schema(self):
        """Test loading pin file with invalid schema."""
        pin_path = self.temp_dir / "pin.json"
        pin_path.write_text(json.dumps({"invalid": "data"}))

        with self.assertRaises(PinConfigError):
            load_pin(str(pin_path))

    def test_write_pin(self):
        """Test writing a pin file."""
        pin_path = self.temp_dir / "pin.json"
        write_pin(str(pin_path), self.initial_sha, "test note")

        loaded = load_pin(str(pin_path))
        self.assertEqual(loaded["pinned_sha"], self.initial_sha)
        self.assertEqual(loaded["note"], "test note")
        self.assertIn("pinned_at", loaded)

    def test_resolve_ref_head(self):
        """Test resolving HEAD ref."""
        sha = resolve_ref(str(self.repo_dir), "HEAD")
        self.assertEqual(sha, self.second_sha)

    def test_resolve_ref_sha(self):
        """Test resolving a SHA ref."""
        sha = resolve_ref(str(self.repo_dir), self.initial_sha)
        self.assertEqual(sha, self.initial_sha)

    def test_resolve_ref_invalid(self):
        """Test resolving invalid ref."""
        with self.assertRaises(subprocess.CalledProcessError):
            resolve_ref(str(self.repo_dir), "invalid-ref")

    def test_is_git_dirty_clean(self):
        """Test dirty check on clean repo."""
        self.assertFalse(is_git_dirty(str(self.repo_dir)))

    def test_is_git_dirty_modified(self):
        """Test dirty check on modified repo."""
        (self.repo_dir / "file.txt").write_text("modified again")
        self.assertTrue(is_git_dirty(str(self.repo_dir)))

    def test_verify_pinned_repo_match(self):
        """Test verification when pin matches HEAD."""
        pin = {"pinned_sha": self.second_sha}
        result = verify_pinned_repo(str(self.repo_dir), pin, require_clean=True)

        self.assertEqual(result["pinned_sha"], self.second_sha)
        self.assertEqual(result["head_sha"], self.second_sha)
        self.assertFalse(result["dirty"])
        self.assertTrue(result["match"])

    def test_verify_pinned_repo_drift(self):
        """Test verification when pin does not match HEAD."""
        pin = {"pinned_sha": self.initial_sha}
        with self.assertRaises(DriftError) as cm:
            verify_pinned_repo(
                str(self.repo_dir), pin, require_clean=True, allow_drift=False
            )

        self.assertIn("SHA mismatch", str(cm.exception))

    def test_verify_pinned_repo_drift_override(self):
        """Test verification with drift but override allowed."""
        pin = {"pinned_sha": self.initial_sha}
        result = verify_pinned_repo(
            str(self.repo_dir), pin, require_clean=False, allow_drift=True
        )

        self.assertEqual(result["pinned_sha"], self.initial_sha)
        self.assertEqual(result["head_sha"], self.second_sha)
        self.assertFalse(result["dirty"])
        self.assertFalse(result["match"])

    def test_verify_pinned_repo_dirty_no_override(self):
        """Test verification with dirty repo and no override."""
        (self.repo_dir / "file.txt").write_text("dirty")
        pin = {"pinned_sha": self.second_sha}

        with self.assertRaises(DriftError) as cm:
            verify_pinned_repo(
                str(self.repo_dir), pin, require_clean=True, allow_drift=False
            )

        self.assertIn("Repository is dirty", str(cm.exception))

    def test_verify_pinned_repo_dirty_override(self):
        """Test verification with dirty repo but override allowed."""
        (self.repo_dir / "file.txt").write_text("dirty")
        pin = {"pinned_sha": self.second_sha}

        result = verify_pinned_repo(str(self.repo_dir), pin, require_clean=False)

        self.assertEqual(result["pinned_sha"], self.second_sha)
        self.assertEqual(result["head_sha"], self.second_sha)
        self.assertTrue(result["dirty"])
        self.assertTrue(result["match"])


if __name__ == "__main__":
    unittest.main()
