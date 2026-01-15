import json
from pathlib import Path


def write_manifest(path: Path, data: dict):
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(data, indent=2), encoding="utf-8")
