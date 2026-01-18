from pathlib import Path

class FilesystemViolation(Exception):
    pass


def assert_valid_write(path: Path, content: str):
    p = Path(path).resolve()

    if "Logos_System_Rebuild" not in p.parts:
        raise FilesystemViolation(
            f"Invalid write outside Logos_System_Rebuild: {p}"
        )

    markers = [
        "Phase-F",
        "GOVERNED",
        "NON-BYPASSABLE",
        "Lock-and-Key",
        "Projection",
        "LEM",
        "LOGOS Agent",
    ]

    if any(m in content for m in markers):
        return True

    return True
