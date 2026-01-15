#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_web_grounding_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Stub-friendly grounding smoke test for LOGOS-GPT server."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    return int(port)


def _wait(url: str, timeout: float = 10.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            from urllib import request as urllib_request

            with urllib_request.urlopen(url, timeout=2) as resp:  # nosec B310
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("server did not become ready")


def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    from urllib import request as urllib_request

    req = urllib_request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib_request.urlopen(req, timeout=5) as resp:  # nosec B310
        body = resp.read().decode("utf-8")
        return json.loads(body)


def main() -> int:
    env = os.environ.copy()
    env.setdefault("LOGOS_DEV_BYPASS_OK", "1")
    env.setdefault("LOGOS_LLM_PROVIDER", "stub")

    port = _free_port()
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
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        _wait(f"http://127.0.0.1:{port}/health")
        chat_url = f"http://127.0.0.1:{port}/api/chat"
        payload = {"message": "grounding check", "provider": "stub"}
        data = _post(chat_url, payload)
        claims = data.get("claims", []) if isinstance(data, dict) else []
        retrieval = data.get("retrieval", {}) if isinstance(data, dict) else {}
        if not isinstance(claims, list):
            print("FAIL: claims missing or not list")
            return 1
        for claim in claims:
            truth = claim.get("truth") if isinstance(claim, dict) else None
            if truth == "PROVED":
                print("FAIL: PROVED claim present in stub response")
                return 1
            if truth == "VERIFIED" and not claim.get("evidence_refs"):
                print("FAIL: VERIFIED claim missing evidence_refs")
                return 1
        ledger_path = data.get("ledger_path")
        grounding = {}
        if ledger_path:
            try:
                ledger = json.loads(Path(ledger_path).read_text())
                grounding = ledger.get("grounding", {}) if isinstance(ledger, dict) else {}
            except Exception:
                grounding = {}
        retrieval_seen = bool(retrieval) or bool(grounding.get("retrieval"))
        if not retrieval_seen:
            print("FAIL: retrieval metadata missing")
            return 1
        executed = data.get("executed_results", [])
        for item in executed:
            tool = item.get("tool") if isinstance(item, dict) else None
            if tool in {"tool_proposal_pipeline", "start_agent"}:
                print("FAIL: high-impact tool executed without approval")
                return 1
        print("PASS: web grounding smoke")
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"FAIL: {exc}")
        if proc.stdout:
            try:
                print(proc.stdout.read().decode("utf-8"))
            except Exception:
                pass
        if proc.stderr:
            try:
                print(proc.stderr.read().decode("utf-8"), file=sys.stderr)
            except Exception:
                pass
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
