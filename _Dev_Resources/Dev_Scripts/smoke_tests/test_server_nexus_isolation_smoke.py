#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_server_nexus_isolation_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke test ensuring server nexus isolation per session."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, Tuple

import JUNK_DRAWER.scripts.need_to_distribute._bootstrap as _bootstrap  # noqa: F401

from JUNK_DRAWER.scripts.runtime.could_be_dev.start_agent import STATE_DIR


REPO_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    return int(port)


def _tail(text: bytes, lines: int = 80) -> str:
    decoded = text.decode("utf-8", errors="replace") if text else ""
    return "\n".join(decoded.strip().splitlines()[-lines:])


def _wait_health(base_url: str, timeout: float = 12.0) -> None:
    start = time.time()
    last_exc: Exception | None = None
    while time.time() - start < timeout:
        try:
            from urllib import request as urllib_request

            with urllib_request.urlopen(f"{base_url}/health", timeout=2) as resp:  # nosec B310
                if resp.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - best effort
            last_exc = exc
            time.sleep(0.25)
    raise RuntimeError(f"server did not become ready: {last_exc}")


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str] | None = None) -> Tuple[int, Dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    from urllib import request as urllib_request

    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)
    req = urllib_request.Request(url, data=body, headers=merged_headers, method="POST")
    with urllib_request.urlopen(req, timeout=6) as resp:  # nosec B310
        data = resp.read().decode("utf-8")
        return resp.status, json.loads(data)


def _get_json(url: str) -> Tuple[int, Dict[str, Any]]:
    from urllib import request as urllib_request

    with urllib_request.urlopen(url, timeout=6) as resp:  # nosec B310
        data = resp.read().decode("utf-8")
        return resp.status, json.loads(data)


class ServerNexusIsolationSmoke(unittest.TestCase):
    def _start_server(self) -> tuple[int, subprocess.Popen]:
        port = _free_port()
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(REPO_ROOT))
        env.setdefault("LOGOS_DEV_BYPASS_OK", "1")
        env.setdefault("LOGOS_LLM_PROVIDER", "stub")
        env.setdefault("LOGOS_LLM_MODEL", "stub")
        env.setdefault("LOGOS_AGI_MODE", "stub")
        env.setdefault("TEST_MODE", "1")

        state_path = STATE_DIR / "scp_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        if state_path.exists():
            state_path.unlink()

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "scripts.llm_interface_suite.logos_gpt_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            _wait_health(f"http://127.0.0.1:{port}", timeout=12.0)
            return port, proc
        except Exception:
            proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate(timeout=5)
            print("SERVER START STDOUT TAIL:\n" + _tail(stdout))
            print("SERVER START STDERR TAIL:\n" + _tail(stderr), file=sys.stderr)
            raise

    def test_sessions_isolated_and_refresh_local(self) -> None:
        port, proc = self._start_server()
        base_url = f"http://127.0.0.1:{port}"
        session_a = "sessA1234"
        session_b = "sessB1234"
        headers_a = {"X-Session-Id": session_a, "Cookie": f"session_id={session_a}"}
        headers_b = {"X-Session-Id": session_b, "Cookie": f"session_id={session_b}"}

        try:
            status_a, body_a = _post_json(
                f"{base_url}/api/chat",
                {"message": "hello", "read_only": True, "session_id": session_a},
                headers=headers_a,
            )
            status_b, body_b = _post_json(
                f"{base_url}/api/chat",
                {"message": "hola", "read_only": True, "session_id": session_b},
                headers=headers_b,
            )
            self.assertEqual(status_a, 200)
            self.assertEqual(status_b, 200)

            self.assertFalse(body_a.get("pending_approvals"))
            self.assertFalse(body_b.get("pending_approvals"))

            status_debug, debug = _get_json(f"{base_url}/api/debug/nexus")
            self.assertEqual(status_debug, 200)
            self.assertEqual(debug.get("count"), 2)
            self.assertIn(session_a, debug.get("sessions", []))
            self.assertIn(session_b, debug.get("sessions", []))

            state_path = STATE_DIR / "scp_state.json"
            self.assertTrue(state_path.exists())
            state = json.loads(state_path.read_text())
            wm_items = state.get("working_memory", {}).get("long_term", []) + state.get(
                "working_memory", {}
            ).get("short_term", [])
            tags = [tag for item in wm_items for tag in item.get("objective_tags", [])]
            self.assertTrue(any(f"SESSION:{session_a[:8]}" == t for t in tags))
            self.assertTrue(any(f"SESSION:{session_b[:8]}" == t for t in tags))
            self.assertTrue(any(t == "CHAT" for t in tags))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
