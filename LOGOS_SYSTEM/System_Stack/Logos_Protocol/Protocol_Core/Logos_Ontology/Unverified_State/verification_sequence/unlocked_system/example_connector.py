#!/usr/bin/env python3
"""
Simple example connector for testing automatic integration.
Exports: register_agent(agent, identity=None)
When called, appends a JSON line to integration_artifacts/integration_connector_log.jsonl
"""
import json
import os
from datetime import datetime

os.makedirs("integration_artifacts", exist_ok=True)

def register_agent(agent, identity=None):
    out = {
        "called_at": datetime.utcnow().isoformat() + "Z",
        "identity": identity,
        "agent_repr": repr(agent)[:200]
    }
    with open("integration_artifacts/integration_connector_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out) + "\n")
    return True

# Provide alias names often used by connectors
connect_agent = register_agent
connect_to_agent = register_agent
