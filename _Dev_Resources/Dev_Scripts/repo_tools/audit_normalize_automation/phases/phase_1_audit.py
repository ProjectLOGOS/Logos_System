from pathlib import Path
import subprocess
from datetime import datetime

PHASE_ID = "phase_1"
TIMEOUT_SECONDS = 3600  # 1 hour

SCAN_ROOT = Path("_Dev_Resources/Dev_Scripts/repo_tools/system_audit")
SCAN_TOOLS = [
	"scan_imports.py",
	"scan_symbols.py",
	"scan_tree.py",
	"scan_naming.py",
]


def run(script_path: Path, reports_root: Path) -> dict:
	ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
	out_dir = reports_root / "Phase_1" / script_path.stem / ts
	out_dir.mkdir(parents=True, exist_ok=True)

	result = {
		"phase": PHASE_ID,
		"script": str(script_path),
		"timestamp": ts,
		"status": "UNKNOWN",
		"logs": [],
		"outputs": [],  # artifact paths for downstream validation
		"error": None,
	}

	for tool in SCAN_TOOLS:
		tool_path = SCAN_ROOT / tool
		log_path = out_dir / f"{tool}.log"
		result["logs"].append(str(log_path))
		result["outputs"].append(str(log_path))

		if not tool_path.exists():
			log_path.write_text(f"Missing tool: {tool_path}\n", encoding="utf-8")
			result["status"] = "FAIL"
			result["error"] = f"Missing tool: {tool}"
			return result

		try:
			with log_path.open("w", encoding="utf-8") as f:
				proc = subprocess.run(
					["python3", str(tool_path), str(script_path)],
					stdout=f,
					stderr=subprocess.STDOUT,
					timeout=TIMEOUT_SECONDS,
				)
		except subprocess.TimeoutExpired:
			log_path.write_text("TIMEOUT\n", encoding="utf-8")
			result["status"] = "TIMEOUT"
			result["error"] = f"Timeout in {tool}"
			return result

		if proc.returncode != 0:
			result["status"] = "FAIL"
			result["error"] = f"Tool failed: {tool}"
			return result

	(out_dir / "DONE.marker").touch()
	result["status"] = "PASS"
	result["outputs"].append(str(out_dir / "DONE.marker"))
	return result
