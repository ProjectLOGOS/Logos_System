from pathlib import Path
import json
from datetime import datetime


PHASE_ID = "phase_3"
PHASE_NAME = "Planning"
TIMEOUT_SECONDS = 3600  # 1 hour


def run(reports_root: Path) -> dict:
	"""Consume Phase 2 Audit_Input_Index.json and emit planning artifacts."""
	ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
	phase2_root = reports_root / "Phase_2"
	out_dir = reports_root / "Phase_3" / ts
	out_dir.mkdir(parents=True, exist_ok=True)

	result = {
		"phase": PHASE_ID,
		"timestamp": ts,
		"status": "UNKNOWN",
		"plans": [],
		"error": None,
	}

	try:
		index_files = list(phase2_root.rglob("Audit_Input_Index.json"))
		if not index_files:
			result["status"] = "FAIL"
			result["error"] = "Missing Audit_Input_Index.json"
			return result

		index_path = index_files[-1]
		index = json.loads(index_path.read_text(encoding="utf-8"))

		plan = {
			"source_index": str(index_path),
			"script_count": len(index),
			"actions": {
				"rename_candidates": [],
				"move_candidates": [],
				"merge_candidates": [],
				"archive_candidates": [],
			},
		}

		plan_path = out_dir / "Resolution_Plan.json"
		plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

		(out_dir / "DONE.marker").touch()

		result["status"] = "PASS"
		result["plans"].append(str(plan_path))
		return result

	except Exception as e:
		result["status"] = "FAIL"
		result["error"] = str(e)
		return result
