from pathlib import Path
import sys

root = Path("/workspaces/Logos_System")

WHITELIST_PATHS = [
    Path("/workspaces/Logos_System/START_LOGOS.py").resolve(),
    Path("/workspaces/Logos_System/LOGOS_SYSTEM.py").resolve(),
]
violations = []

for p in root.rglob("*.py"):
    try:
        text = p.read_text(errors="ignore")
    except Exception:
        continue

    if any(m in text for m in ["Phase-F", "GOVERNED", "NON-BYPASSABLE"]):
        if p.resolve() in WHITELIST_PATHS:
            continue
        if "Logos_System_Rebuild" not in p.parts:
            violations.append(str(p))

if violations:
    print("FILESYSTEM GOVERNANCE VIOLATION:")
    for v in violations:
        print(" ", v)
    sys.exit(1)

print("Filesystem governance check: PASS")
