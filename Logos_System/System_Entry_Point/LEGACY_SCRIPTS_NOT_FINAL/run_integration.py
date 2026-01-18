# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

import os
import sys
import json
import logging
import importlib.util
import glob
import traceback

logging.basicConfig(level=logging.INFO)
ROOT = os.path.abspath(os.getcwd())
# ensure workspace root is on PYTHONPATH so imports inside modules work
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

# Try common locations then fallback to a workspace-wide search
candidates = [
	os.path.join(ROOT, "LOGOS_AGI", "consciousness", "recursion_engine_consciousness_bridge.py"),
	os.path.join(ROOT, "consciousness", "recursion_engine_consciousness_bridge.py"),
	os.path.join(ROOT, "LOGOS_AGI", "consciousness", "recursion_engine_consciousness_bridge.py"),
]
bridge_path = None
for p in candidates:
	if os.path.isfile(p):
		bridge_path = p
		break

if not bridge_path:
	matches = glob.glob(os.path.join(ROOT, "**", "recursion_engine_consciousness_bridge.py"), recursive=True)
	if matches:
		bridge_path = matches[0]

if not bridge_path:
	raise FileNotFoundError("Could not find recursion_engine_consciousness_bridge.py in the workspace")

BRIDGE_PATH = bridge_path

try:
	spec = importlib.util.spec_from_file_location("recursion_bridge", BRIDGE_PATH)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

	Bridge = getattr(module, "RecursionEngineConsciousnessBridge")
	bridge = Bridge(agent_id="LOGOS-AGENT-OMEGA")

	print("integration status:")
	print(json.dumps(bridge.get_integration_status(), indent=2))

	result = bridge.run_integrated_cycle()
	print("cycle result:")
	print(json.dumps(result, indent=2))
	# If the bridge exposed a global_commutator with an integrate_with_agent method,
	# call it so any workspace connectors (register_agent / connect_agent) are invoked.
	try:
		gc = getattr(bridge, "global_commutator", None)
		if gc and hasattr(gc, "integrate_with_agent"):
			try:
				connected = gc.integrate_with_agent(bridge)
				print("connector integration invoked, success:", bool(connected))
			except Exception:
				print("connector integration attempt failed:")
				traceback.print_exc()
	except Exception:
		# best-effort: don't crash the runner on connector issues
		pass
except Exception:
	print("An error occurred while running the integration script:")
	traceback.print_exc()
	sys.exit(1)
