"""Tests for Logos_AGI adapter persistence and integration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

from scripts.logos_agi_adapter import LogosAgiNexus


class LogosAgiAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.state_dir = Path(self.temp_dir.name) / "state"
        self.state_dir.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_persistence_verification(self) -> None:
        """Test that persistence creates measurable state changes."""
        # Create adapter with persistence enabled
        adapter = LogosAgiNexus(
            enable=True, state_dir=self.state_dir, repo_sha="test_sha"
        )

        # Bootstrap (may fail if Logos_AGI not available, but that's ok for this test)
        adapter.bootstrap()

        # Add some observations
        initial_obs = [
            {"type": "tool_execution", "tool": "mission.status", "result": "active"},
            {"type": "attestation", "status": "passed", "proof": "lem_discharge"},
        ]
        for obs in initial_obs:
            adapter.observe(obs)

        # Generate a proposal
        proposal = adapter.propose("Check system status", {})
        self.assertIsInstance(proposal, dict)
        self.assertIn("proposals", proposal)

        # Persist state
        adapter.persist()

        # Verify persistence file was created
        persisted_path = self.state_dir / "logos_agi_scp_state.json"
        self.assertTrue(persisted_path.exists(), "Persistence file should be created")

        # Load and verify contents
        with open(persisted_path) as f:
            state = json.load(f)

        # Verify structure
        self.assertIn("timestamp", state)
        self.assertIn("observations", state)
        self.assertIn("repo_sha", state)
        self.assertEqual(state["repo_sha"], "test_sha")

        # Verify observations were persisted
        self.assertEqual(len(state["observations"]), len(initial_obs))
        for i, obs in enumerate(initial_obs):
            self.assertEqual(state["observations"][i], obs)

        # Verify cognitive status is included (may be error if SCP not available)
        self.assertIn("cognitive_status", state)

        # Modify observations and persist again
        adapter.observe(
            {"type": "tool_execution", "tool": "probe.last", "result": "completed"}
        )
        adapter.persist()

        # Reload and verify change
        with open(persisted_path) as f:
            new_state = json.load(f)

        self.assertEqual(len(new_state["observations"]), len(initial_obs) + 1)
        self.assertGreater(new_state["timestamp"], state["timestamp"])

    def test_adapter_health_reporting(self) -> None:
        """Test health status reporting."""
        adapter = LogosAgiNexus(
            enable=True, state_dir=self.state_dir, repo_sha="test_sha"
        )
        adapter.bootstrap()

        health = adapter.health()
        self.assertIsInstance(health, dict)
        self.assertIn("available", health)
        self.assertIn("last_error", health)
        self.assertIn("observations_count", health)
        self.assertIn("persisted_path", health)

    @patch("scripts.logos_agi_adapter.SCPNexus")
    @patch("scripts.logos_agi_adapter.AdvancedReasoner")
    def test_real_api_integration(self, mock_arp, mock_scp) -> None:
        """Test integration with mocked real APIs."""
        # Mock SCP
        mock_scp_instance = MagicMock()
        mock_scp_instance.initialize = AsyncMock(return_value=True)
        mock_scp_instance.process_agent_request = AsyncMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.data = {
            "proposals": [{"tool": "mission.status", "rationale": "test"}]
        }
        mock_scp_instance.process_agent_request.return_value = mock_response
        mock_scp.return_value = mock_scp_instance

        # Mock ARP
        mock_arp_instance = MagicMock()
        mock_arp_instance.start = MagicMock()
        mock_arp_instance.status = MagicMock(return_value={"online": True})
        mock_arp.return_value = mock_arp_instance

        adapter = LogosAgiNexus(
            enable=True, state_dir=self.state_dir, repo_sha="test_sha"
        )
        adapter.bootstrap()

        # Verify bootstrap attempted real imports
        self.assertTrue(adapter.available)
        mock_scp.assert_called_once()
        mock_arp.assert_called_once_with(agent_identity="test_sha")

        # Test proposal generation
        proposal = adapter.propose("test objective", {})
        self.assertIn("proposals", proposal)
        mock_scp_instance.process_agent_request.assert_called()


if __name__ == "__main__":
    unittest.main()
