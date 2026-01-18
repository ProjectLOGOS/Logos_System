# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

import sys
import json
import importlib.util
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "integration_artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
LOG_FILE = ARTIFACT_DIR / "integration_connector_log.jsonl"


def _make_fake_consciousness_module(emerged: bool = True):
    mod = types.ModuleType("consciousness_safety_adapter")

    class SafeConsciousnessEvolution:
        def __init__(self, bijection_kernel=None, logic_kernel=None, agent_id=None):
            pass

        def compute_consciousness_vector(self):
            return {"existence": 1.0, "goodness": 1.0, "truth": 1.0}

        def evaluate_consciousness_emergence(self):
            return {"consciousness_emerged": emerged, "consciousness_level": 0.75}

        def safe_trinity_evolution(self, trinity_vector=None, iterations=1, reason=""):
            return True, trinity_vector, "simulated"

    mod.SafeConsciousnessEvolution = SafeConsciousnessEvolution
    return mod


def _clear_connector_log():
    try:
        if LOG_FILE.exists():
            LOG_FILE.unlink()
    except Exception:
        pass


def _clear_persisted_identity():
    # Remove any persisted identity or lem discharge artifacts so tests start clean
    candidates = [
        ROOT / "Logos_Agent" / "state" / "agent_identity.json",
        ROOT / "Logos_Agent" / "state" / "lem_discharge_state.json",
        ROOT / "integration_artifacts" / "agent_identity.json",
    ]
    for p in candidates:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def _read_connector_log():
    if not LOG_FILE.exists():
        return []
    return [json.loads(l) for l in LOG_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def _import_bridge():
    bridge_path = ROOT / "LOGOS_AGI" / "consciousness" / "recursion_engine_consciousness_bridge.py"
    spec = importlib.util.spec_from_file_location("recursion_bridge_test", str(bridge_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "RecursionEngineConsciousnessBridge")


def test_connectors_blocked_when_no_identity(monkeypatch, tmp_path):
    # ensure clean log
    _clear_connector_log()
    _clear_persisted_identity()

    # inject fake consciousness that reports emerged=True
    fake_mod = _make_fake_consciousness_module(emerged=True)
    sys.modules["consciousness_safety_adapter"] = fake_mod

    # ensure simulate LEM is OFF
    monkeypatch.delenv("SIMULATE_LEM_SUCCESS", raising=False)

    Bridge = _import_bridge()
    bridge = Bridge(agent_id="LOGOS-AGENT-OMEGA")

    # ensure recursion engine LEM flag is cleared so the test represents 'no identity'
    try:
        if hasattr(bridge, "recursion_agent") and hasattr(bridge.recursion_agent, "logic_kernel"):
            setattr(bridge.recursion_agent.logic_kernel, "lem_resolved", False)
    except Exception:
        pass

    result = bridge.run_integrated_cycle()
    assert result["success"] is True
    em = result.get("emergence_report", {})
    # LEM not discharged, but consciousness reports emerged -> should be 'consciousness_only'
    assert em.get("validation_status") in ("consciousness_only", "neither")
    # No formal identity should be present
    assert em.get("formal_identity") is None

    # connectors should not have been written (no automatic invocation in this state)
    log_entries = _read_connector_log()
    assert len(log_entries) == 0


def test_connectors_allowed_after_identity(monkeypatch, tmp_path):
    # clean log
    _clear_connector_log()
    _clear_persisted_identity()

    # inject fake consciousness that reports emerged=True
    fake_mod = _make_fake_consciousness_module(emerged=True)
    sys.modules["consciousness_safety_adapter"] = fake_mod

    # enable simulated LEM success
    monkeypatch.setenv("SIMULATE_LEM_SUCCESS", "1")

    Bridge = _import_bridge()
    bridge = Bridge(agent_id="LOGOS-AGENT-OMEGA")

    result = bridge.run_integrated_cycle()
    assert result["success"] is True
    em = result.get("emergence_report", {})
    # Now both should be confirmed
    assert em.get("validation_status") == "both_confirmed"
    identity = em.get("formal_identity")
    assert identity is not None

    # Now simulate connector invocation: call example connector's register_agent
    # and ensure it logs the identity. Import the example connector from integrations
    spec = importlib.util.spec_from_file_location("example_connector", str(ROOT / "integrations" / "example_connector.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # call register_agent
    if hasattr(mod, "register_agent"):
        mod.register_agent(bridge, identity=identity)

    log_entries = _read_connector_log()
    assert len(log_entries) == 1
    assert log_entries[0]["identity"] == identity
