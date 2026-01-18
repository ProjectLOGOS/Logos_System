# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

import asyncio
import unittest
from typing import Any, Dict, Tuple

from external.Logos_AGI.System_Operations_Protocol.infrastructure.agent_system.base_nexus import (
    AgentRequest,
    AgentType,
    BaseNexus,
    ProtocolType,
)


class DummyNexus(BaseNexus):
    def __init__(self, allowed_ops, validators):
        super().__init__(
            ProtocolType.UIP,
            "DummyNexus",
            allowed_operations=list(allowed_ops),
            payload_validators=validators,
        )
        self.core_called = False

    async def _protocol_specific_initialization(self) -> bool:
        return True

    async def _protocol_specific_security_validation(self, request: AgentRequest) -> Dict[str, Any]:
        # Accept only system agents for parity with real nexuses.
        if request.agent_type != AgentType.SYSTEM_AGENT:
            return {"valid": False, "reason": "SYSTEM_AGENT required"}
        return {"valid": True}

    async def _protocol_specific_activation(self) -> None:
        return None

    async def _protocol_specific_deactivation(self) -> None:
        return None

    async def _route_to_protocol_core(self, request: AgentRequest) -> Dict[str, Any]:
        self.core_called = True
        return {"success": True, "data": {"echo": request.payload}}

    async def _protocol_specific_smoke_test(self) -> Dict[str, Any]:
        return {"passed": True}


def _require_key_payload(required_key: str):
    def _validator(payload: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(payload, dict):
            return False, "payload must be a dict"
        if required_key not in payload:
            return False, f"missing key: {required_key}"
        return True, ""

    return _validator


class NexusCapabilityGovernanceTest(unittest.TestCase):
    def setUp(self) -> None:
        allowed = {"allowed_op"}
        validators = {"allowed_op": _require_key_payload("ok")}
        self.nexus = DummyNexus(allowed, validators)

    def _run(self, request: AgentRequest):
        return asyncio.run(self.nexus.process_agent_request(request))

    def test_disallowed_operation_rejected(self):
        request = AgentRequest(
            agent_id="SYSTEM_AGENT_TEST",
            operation="blocked_op",
            payload={"ok": True},
        )
        response = self._run(request)
        self.assertFalse(response.success)
        self.assertIn("not allowed", response.error)
        self.assertFalse(self.nexus.core_called)

    def test_invalid_payload_rejected(self):
        request = AgentRequest(
            agent_id="SYSTEM_AGENT_TEST",
            operation="allowed_op",
            payload={},
        )
        response = self._run(request)
        self.assertFalse(response.success)
        self.assertIn("Payload validation failed", response.error)
        self.assertFalse(self.nexus.core_called)

    def test_allowed_operation_with_valid_payload_routes(self):
        request = AgentRequest(
            agent_id="SYSTEM_AGENT_TEST",
            operation="allowed_op",
            payload={"ok": True},
        )
        response = self._run(request)
        self.assertTrue(response.success)
        self.assertTrue(self.nexus.core_called)
        self.assertEqual(response.data.get("echo"), {"ok": True})


if __name__ == "__main__":
    unittest.main()
