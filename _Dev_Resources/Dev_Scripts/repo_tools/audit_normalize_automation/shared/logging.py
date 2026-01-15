from pathlib import Path


def log(path: Path, message: str):
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as f:
		f.write(message + "\n")
