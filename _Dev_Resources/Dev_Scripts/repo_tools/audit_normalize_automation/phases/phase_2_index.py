from pathlib import Path
import json
from datetime import datetime


PHASE_ID = "phase_2"
PHASE_NAME = "Index"
TIMEOUT_SECONDS = 3600  # 1 hour


def run(reports_root: Path) -> dict:
	"""Build canonical Audit_Input_Index.json from Phase 1 outputs."""
	ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
	phase1_root = reports_root / "Phase_1"
	out_dir = reports_root / "Phase_2" / ts
	out_dir.mkdir(parents=True, exist_ok=True)

	result = {
		"phase": PHASE_ID,
		"timestamp": ts,
		"status": "UNKNOWN",
		"index_path": None,
		"error": None,
	}

	try:
		if not phase1_root.exists():
			result["status"] = "FAIL"
			result["error"] = "Missing Phase 1 outputs"
			return result

		index = {}

		for script_dir in phase1_root.iterdir():
			if not script_dir.is_dir():
				continue

			runs = sorted(script_dir.iterdir())
			if not runs:
				continue

			latest = runs[-1]
			if not (latest / "DONE.marker").exists():
				continue

			index[script_dir.name] = {
				"phase_1_path": str(latest),
				"artifacts": [
					str(p)
					for p in latest.iterdir()
					if p.is_file() and p.name != "DONE.marker"
				],
			}

		if not index:
			result["status"] = "FAIL"
			result["error"] = "No valid Phase 1 runs found"
			return result

		index_path = out_dir / "Audit_Input_Index.json"
		index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

		result["status"] = "PASS"
		result["index_path"] = str(index_path)
		return result

	except Exception as e:
		result["status"] = "FAIL"
		result["error"] = str(e)
		return result
