from pathlib import Path
import json


def validate_phase_1(result: dict) -> bool:
	"""Validate Phase 1 result structure and artifacts."""
	if result.get("status") != "PASS":
		return False

	outputs = result.get("outputs", [])
	if not outputs:
		return False

	for p in outputs:
		if not Path(p).exists():
			return False

	out_dir = Path(outputs[0]).parent
	if not (out_dir / "DONE.marker").exists():
		return False

	return True


def validate_phase_2(result: dict) -> bool:
	"""Validate Phase 2 index output."""
	if result.get("status") != "PASS":
		return False

	index_path = result.get("index_path")
	if not index_path:
		return False

	p = Path(index_path)
	if not p.exists():
		return False

	try:
		data = json.loads(p.read_text(encoding="utf-8"))
	except Exception:
		return False

	return bool(data)


def validate_phase_3(result: dict) -> bool:
	"""Validate Phase 3 planning outputs."""
	if result.get("status") != "PASS":
		return False

	plans = result.get("plans", [])
	if not plans:
		return False

	for p in plans:
		if not Path(p).exists():
			return False

	out_dir = Path(plans[0]).parent
	if not (out_dir / "DONE.marker").exists():
		return False

	return True
