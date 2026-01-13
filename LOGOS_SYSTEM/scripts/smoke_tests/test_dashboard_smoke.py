"""Smoke test for the LOGOS dashboard service."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from logos_dashboard import metrics, smp_bus
from logos_dashboard.app import app, publish_smp, socketio


def _reset_state_dir(tmp: Path) -> None:
    smp_bus.STATE_DIR = tmp
    smp_bus.STREAM_PATH = tmp / "smp_stream.jsonl"


def test_health_and_metrics() -> None:
    client = app.test_client()
    assert client.get("/healthz").status_code == 200
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"logos_cpu_percent" in resp.data


def test_socketio_broadcast() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_state_dir(Path(tmpdir))
        client = socketio.test_client(app)
        event = publish_smp({"source": "TEST", "msg": "SMP::Emit(test)"})
        received = client.get_received()
        assert any(pkt.get("name") == "smp_event" for pkt in received)
        assert (smp_bus.STREAM_PATH).exists()
        assert metrics.logos_smp_events_total._value.get() >= 1  # type: ignore[attr-defined]


if __name__ == "__main__":
    test_health_and_metrics()
    test_socketio_broadcast()
    print("dashboard smoke tests passed")
