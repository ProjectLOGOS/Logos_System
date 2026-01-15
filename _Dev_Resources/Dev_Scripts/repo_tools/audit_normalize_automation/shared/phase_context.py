from pathlib import Path
from datetime import datetime


def reports_root() -> Path:
	return Path("_Reports/Audit_Normalize")


def timestamp() -> str:
	return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
