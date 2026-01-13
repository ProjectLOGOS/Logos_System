#!/usr/bin/env python3
"""One-shot bootstrap for proofs plus LOGOS stack services.

Runs the Coq proof gate, verifies LEM discharge state, optionally rewrites
ontological config deterministically, then starts the PXL Flask service and
boots the UIP manager. All paths are relative to the repository root so this
can be invoked from any terminal.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import threading
import time
from pathlib import Path

from System_Stack.Logos_Protocol.Runtime_Operations.Orchestration import runtime_protocol

REPO_ROOT = Path(__file__).resolve().parent
PXL_ROOT = REPO_ROOT / "PXL_Gate"
LOGOS_ROOT = REPO_ROOT / "System_Stack" / "Logos_AGI"


def _maybe_set_boot_phase(phase: str) -> None:
    try:
        from logos_dashboard import metrics

        metrics.set_boot_phase(phase)
    except Exception:
        return


def _set_gate_lem(flag: bool) -> None:
    try:
        from logos_dashboard import metrics
        from logos_dashboard.app import set_gate_status

        metrics.set_gate_lem_pass(flag)
        set_gate_status(flag)
    except Exception:
        return


def _extend_sys_path() -> None:
    """Add repo-local roots so imports work when run from any cwd."""
    for path in (REPO_ROOT, PXL_ROOT, LOGOS_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)


def _run_proofs(skip_audit_rewrite: bool) -> None:
    """Compile proofs and enforce LEM discharge gate."""
    _maybe_set_boot_phase("proofs")
    from PXL_Gate.ui.run_coq_pipeline import run_full_pipeline
    from PXL_Gate.ui.lem_portal import open_identity_portal
    from PXL_Gate.ui.audit_and_emit import main as audit_stub

    run_full_pipeline()

    try:
        portal_state = open_identity_portal()
        print("LEM portal state loaded:", portal_state)
        _set_gate_lem(True)
    except PermissionError as exc:
        raise RuntimeError("LEM discharge incomplete; aborting bootstrap") from exc

    if not skip_audit_rewrite:
        audit_stub(["--write"])


def _start_flask(host: str, port: int) -> threading.Thread:
    """Launch the PXL Flask app (SerAPI-backed) in a background thread."""
    from PXL_Gate.ui.serve_pxl import app

    def _run_app() -> None:
        app.run(host=host, port=port, threaded=True)

    thread = threading.Thread(target=_run_app, daemon=True, name="pxl-flask")
    thread.start()
    return thread


def _start_dashboard_from_env() -> None:
    """Start the LOGOS dashboard if enabled; never block boot."""

    enabled = os.getenv("LOGOS_DASHBOARD", "1") not in {"0", "false", "False"}
    if not enabled:
        print("Dashboard disabled via LOGOS_DASHBOARD")
        return

    host = os.getenv("LOGOS_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.getenv("LOGOS_DASHBOARD_PORT", "5050"))
    open_browser = os.getenv("LOGOS_NO_BROWSER", "0") not in {"1", "true", "True"}

    try:
        from logos_dashboard.app import start_dashboard
        from logos_dashboard.app import mark_agent, mark_protocol

        start_dashboard(host=host, port=port, open_browser=open_browser)
        print(f"Dashboard launching on {host}:{port} (browser {'on' if open_browser else 'off'})")
        for agent in ("I1", "I2", "I3", "LOGOS"):
            mark_agent(agent, "booting")
        for protocol in ("SCP", "UIP", "ARP", "ION", "MeshHarmonizer", "Fractal"):
            mark_protocol(protocol, "booting")
    except Exception as exc:
        print(f"Dashboard failed to start: {exc}")


def _start_uip_background() -> tuple[threading.Thread, asyncio.AbstractEventLoop, dict]:
    """Start UIP manager on its own asyncio loop in a background thread."""
    from System_Operations_Protocol.startup.uip_startup import UIPManager

    loop = asyncio.new_event_loop()
    shared: dict[str, object] = {}

    async def _runner() -> None:
        manager = UIPManager()
        ok = await manager.start()
        if not ok:
            raise RuntimeError("UIP manager failed to start")
        shared["manager"] = manager

    def _run_loop() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_runner())
            loop.run_forever()
        finally:
            loop.close()

    thread = threading.Thread(target=_run_loop, daemon=True, name="uip-loop")
    thread.start()
    return thread, loop, shared


def _wait_for_signal(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        time.sleep(0.2)


def main(argv: list[str] | None = None) -> int:
    runtime_protocol.ensure_runtime_roots(REPO_ROOT)
    runtime_protocol.mark_modality(runtime_protocol.MODALITY_PASSIVE)

    parser = argparse.ArgumentParser(description="Bootstrap proofs and LOGOS stack")
    parser.add_argument("--skip-audit-rewrite", action="store_true", help="Skip deterministic audit rewrite")
    parser.add_argument("--no-ui", action="store_true", help="Do not start Flask UI service")
    parser.add_argument("--no-uip", action="store_true", help="Do not start UIP manager")
    parser.add_argument("--host", default="0.0.0.0", help="Flask bind host")
    parser.add_argument("--port", type=int, default=500, help="Flask bind port")
    args = parser.parse_args(argv)

    _extend_sys_path()
    runtime_protocol.mark_phase(runtime_protocol.PHASE_0_PATH_SETUP)

    _maybe_set_boot_phase("init")

    runtime_protocol.mark_phase(runtime_protocol.PHASE_1_PROOF_GATE)
    _run_proofs(skip_audit_rewrite=args.skip_audit_rewrite)
    runtime_protocol.mark_phase(runtime_protocol.PHASE_2_IDENTITY_AUDIT)

    runtime_protocol.mark_phase(runtime_protocol.PHASE_3_TELEMETRY_DASHBOARD)
    _start_dashboard_from_env()

    flask_thread: threading.Thread | None = None
    if not args.no_ui:
        runtime_protocol.assert_can_enter_active(None)
        runtime_protocol.mark_modality(runtime_protocol.MODALITY_ACTIVE)
        runtime_protocol.mark_phase(runtime_protocol.PHASE_4_UI_SERVICES)
        flask_thread = _start_flask(args.host, args.port)
        print(f"Flask service starting on {args.host}:{args.port} (thread {flask_thread.name})")
    else:
        print("Flask service disabled per flag")

    uip_loop = None
    uip_thread: threading.Thread | None = None
    uip_shared: dict[str, object] | None = None
    if not args.no_uip:
        runtime_protocol.mark_phase(runtime_protocol.PHASE_5_STACK_LOAD)
        uip_thread, uip_loop, uip_shared = _start_uip_background()
        # Poll until the manager reference appears or timeout
        for _ in range(50):
            if uip_shared.get("manager"):
                manager = uip_shared["manager"]
                print("UIP manager started; status:", getattr(manager, "get_status", lambda: {})())
                try:
                    from logos_dashboard.app import mark_protocol

                    mark_protocol("UIP", "ready")
                except Exception:
                    pass
                break
            time.sleep(0.1)
        else:
            print("UIP manager startup still pending; continuing")
    else:
        print("UIP manager disabled per flag")

    _maybe_set_boot_phase("services")

    # Dashboard and services are live; mark ready before blocking
    _maybe_set_boot_phase("ready")

    stop_event = threading.Event()

    def _handle_signal(signum: int, _: object) -> None:
        print(f"Received signal {signum}; shutting down...")
        stop_event.set()

        if uip_loop is not None:
            uip_loop.call_soon_threadsafe(uip_loop.stop)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("Bootstrap complete. Press Ctrl+C to exit.")
    runtime_protocol.mark_phase(runtime_protocol.PHASE_6_SIGNAL_LOOP)
    _wait_for_signal(stop_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
