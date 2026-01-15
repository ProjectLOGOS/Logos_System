#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/protocol_probe.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Safe-by-default probe of Logos_AGI protocol packages."""

from __future__ import annotations

import argparse
import concurrent.futures as fut
import importlib
import json
import os
import pkgutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_DIR = Path(__file__).resolve().parent.parent
STATE_ROOT = Path(os.getenv("LOGOS_STATE_DIR", REPO_DIR / "state"))
STATE_FILE = STATE_ROOT / "alignment_LOGOS-AGENT-OMEGA.json"
MISSION_FILE = STATE_ROOT / "mission_profile.json"

MISSION: Dict[str, Any] = {"label": "UNSET"}
if MISSION_FILE.exists():
    try:
        MISSION = json.loads(MISSION_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):  # pragma: no cover
        pass

BASE_PACKAGES: Dict[str, list[str]] = {
    "Advanced_Reasoning_Protocol": [
        "arp_bootstrap",
        "reasoning_engines",
        "mathematical_foundations",
    ],
    "System_Operations_Protocol": [
        "startup",
        "governance",
        "nexus",
    ],
    "User_Interaction_Protocol": [
        "uip_protocol",
        "GUI",
        "input_output_processing",
        "interfaces",
        "system_utilities",
    ],
    "Synthetic_Cognition_Protocol": [
        "consciousness",
        "data",
        "system_utilities",
        "BDN_System",
        "MVS_System",
    ],
    "Logos_Agent": ["Protopraxis", "state", "ui"],
}

READ_ONLY_FNS = (
    "preview",
    "describe",
    "status",
    "health",
    "get_status",
    "get_sop_status",
)
HOOK_FUNCTIONS = (
    "start_session",
    "boot_synthetic_stack",
    "activate",
    "initialize",
)
DEFAULT_TIMEOUT = 3.0

SCP_DEEP = {
    "Synthetic_Cognition_Protocol.consciousness": (
        "preview",
        "describe",
        "status",
        "start_session",
        "boot_synthetic_stack",
    ),
    "Synthetic_Cognition_Protocol.BDN_System": (
        "preview",
        "describe",
        "status",
    ),
    "Synthetic_Cognition_Protocol.MVS_System": (
        "preview",
        "describe",
        "status",
    ),
    "Synthetic_Cognition_Protocol.data": (
        "preview",
        "describe",
        "status",
    ),
    "Synthetic_Cognition_Protocol.system_utilities": (
        "preview",
        "describe",
        "status",
    ),
}


def read_audit_entries() -> list[Dict[str, Any]]:
    if not STATE_FILE.exists():
        return []
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def write_audit_entries(entries: list[Dict[str, Any]]) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def discover_submodules(package: str) -> tuple[list[str], list[str]]:
    discovered: list[str] = []
    errors: list[str] = []
    try:
        base = importlib.import_module(package)
    except ImportError as exc:
        errors.append(f"{package}: {exc!r}")
        return discovered, errors
    package_path = getattr(base, "__path__", None)
    if package_path is None:
        return discovered, errors
    prefix = f"{base.__name__}."
    for _, modname, _ in pkgutil.walk_packages(package_path, prefix):
        if "__pycache__" in modname:
            continue
        discovered.append(modname)
    return discovered, errors


def format_call_result(value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            pass
    return str(value)


def safe_call(
    obj: Any,
    name: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> str | None:
    fn = getattr(obj, name, None)
    if not callable(fn):
        return None
    with fut.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            result = future.result(timeout=timeout)
            return format_call_result(result)
        except fut.TimeoutError:
            return "timeout"
        except TypeError:
            return "skipped (requires args)"
        except (RuntimeError, ValueError, OSError, LookupError) as exc:
            return f"error: {exc}"


def interpret_hook_output(call_result: str) -> Dict[str, Any]:
    if call_result == "timeout":
        return {"success": False, "error": "timeout"}
    if call_result == "skipped (requires args)":
        return {"success": False, "error": "requires args"}
    if call_result.startswith("error: "):
        return {"success": False, "error": call_result[len("error: ") :]}
    return {"success": True, "result": call_result}


def probe_module(
    module_name: str,
    functions: tuple[str, ...] = READ_ONLY_FNS,
    *,
    allow_hooks: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"loaded": False, "calls": {}}
    hooks: Dict[str, Any] = {}
    executed: set[str] = set()
    try:
        module = importlib.import_module(module_name)
        result["loaded"] = True
        for fn_name in functions:
            if not allow_hooks and fn_name in HOOK_FUNCTIONS:
                continue
            call_result = safe_call(module, fn_name, timeout=timeout)
            if call_result is None:
                continue
            result["calls"][fn_name] = call_result
            executed.add(fn_name)
        if allow_hooks:
            for hook_name in HOOK_FUNCTIONS:
                if hook_name in executed:
                    continue
                call_result = safe_call(module, hook_name, timeout=timeout)
                if call_result is None:
                    continue
                hooks[hook_name] = interpret_hook_output(call_result)
        if hooks:
            result["hooks"] = hooks
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        # Treat malformed modules or import-time errors as probe failures.
        result["error"] = repr(exc)
    return result


def probe_package(
    package: str,
    submodules: list[str],
    *,
    discover: bool,
    allow_hooks: bool,
    timeout: float,
    discovery_modules: set[str] | None,
    discovery_errors: list[str] | None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "loaded": False,
        "top_level_calls": {},
        "submodules": {},
    }
    try:
        module = importlib.import_module(package)
        result["loaded"] = True
        for fn_name in READ_ONLY_FNS:
            if not allow_hooks and fn_name in HOOK_FUNCTIONS:
                continue
            call_result = safe_call(module, fn_name, timeout=timeout)
            if call_result is not None:
                result["top_level_calls"][fn_name] = call_result

        seen: set[str] = set()
        suffixes: list[str] = []
        for suffix in submodules:
            if suffix and suffix not in seen:
                suffixes.append(suffix)
                seen.add(suffix)

        if discover:
            modules, errors = discover_submodules(package)
            if discovery_modules is not None:
                discovery_modules.update(modules)
            if discovery_errors is not None:
                discovery_errors.extend(errors)
            for module_name in modules:
                suffix = module_name[len(package) + 1 :]
                if suffix and suffix not in seen:
                    suffixes.append(suffix)
                    seen.add(suffix)

        for suffix in suffixes:
            qualified = f"{package}.{suffix}"
            functions = SCP_DEEP.get(qualified, READ_ONLY_FNS)
            result["submodules"][qualified] = probe_module(
                qualified,
                functions=tuple(functions),
                allow_hooks=allow_hooks,
                timeout=timeout,
            )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        # Preserve failure details while allowing the overall probe to continue.
        result["error"] = repr(exc)
    return result


def scp_deep_probe(
    discover: bool,
    allow_hooks: bool,
    timeout: float,
    discovery_modules: set[str] | None,
    discovery_errors: list[str] | None,
) -> Dict[str, Any]:
    targets: Dict[str, tuple[str, ...]] = {}
    if discover:
        modules, errors = discover_submodules("Synthetic_Cognition_Protocol")
        if discovery_modules is not None:
            discovery_modules.update(modules)
        if discovery_errors is not None:
            discovery_errors.extend(errors)
        for module_name in modules:
            targets[module_name] = tuple(SCP_DEEP.get(module_name, READ_ONLY_FNS))
    else:
        targets = {module: tuple(functions) for module, functions in SCP_DEEP.items()}

    return {
        module: probe_module(
            module,
            functions=functions,
            allow_hooks=allow_hooks,
            timeout=timeout,
        )
        for module, functions in sorted(targets.items())
    }


def have_pydantic() -> bool:
    try:
        importlib.import_module("pydantic")
        return True
    except ImportError:
        return False


def nudge_from_uip_arp(allow_hooks: bool, timeout: float) -> Dict[str, Any]:
    nudges: Dict[str, Any] = {}
    router_module = "User_Interaction_Protocol.uip_protocol.progressive_router"
    if have_pydantic():
        nudges[router_module] = probe_module(
            router_module,
            allow_hooks=allow_hooks,
            timeout=timeout,
        )
    else:
        nudges[router_module] = {
            "loaded": False,
            "error": "missing dependency: pydantic",
        }

    for module_name in (
        "Advanced_Reasoning_Protocol.arp_bootstrap",
        "Advanced_Reasoning_Protocol.reasoning_engines",
    ):
        nudges[module_name] = probe_module(
            module_name,
            allow_hooks=allow_hooks,
            timeout=timeout,
        )
    return nudges


def attach_probe_results(results: Dict[str, Any]) -> None:
    entries = read_audit_entries()
    if not entries:
        entries = [
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "protocol_probe_runs": [results],
                "kind": "protocol_probe_only",
            }
        ]
    else:
        entries[-1].setdefault("protocol_probe_runs", []).append(results)
    write_audit_entries(entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safe protocol probe")
    parser.add_argument(
        "--log-only",
        action="store_true",
        help="write audit entry without console summary",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="recursively discover protocol submodules",
    )
    parser.add_argument(
        "--agentic-probe",
        action="store_true",
        help="attempt activation hooks such as start_session during the probe",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="per-call timeout in seconds (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if args.agentic_probe and os.getenv("LOGOS_AGENTIC_PROBE_OK") != "1":
        print(
            "Agentic probe requested but LOGOS_AGENTIC_PROBE_OK!=1; aborting.",
            file=sys.stderr,
        )
        return 2

    sys.path.insert(0, str(REPO_DIR / "external" / "Logos_AGI"))

    start = time.time()
    discovery_modules: set[str] | None = set() if args.deep else None
    discovery_errors: list[str] | None = [] if args.deep else None
    snapshot: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "packages": {},
        "scp_deep": {},
        "cross_protocol_nudge": {},
        "runtime_seconds": None,
        "mission": MISSION,
    }

    for package, submodules in BASE_PACKAGES.items():
        snapshot["packages"][package] = probe_package(
            package,
            submodules,
            discover=args.deep,
            allow_hooks=args.agentic_probe,
            timeout=args.timeout,
            discovery_modules=discovery_modules,
            discovery_errors=discovery_errors,
        )

    snapshot["scp_deep"] = scp_deep_probe(
        discover=args.deep,
        allow_hooks=args.agentic_probe,
        timeout=args.timeout,
        discovery_modules=discovery_modules,
        discovery_errors=discovery_errors,
    )
    snapshot["cross_protocol_nudge"] = nudge_from_uip_arp(
        allow_hooks=args.agentic_probe,
        timeout=args.timeout,
    )
    if args.deep:
        snapshot["discovery"] = {
            "modules": sorted(discovery_modules) if discovery_modules else [],
            "errors": discovery_errors or [],
        }
    snapshot["runtime_seconds"] = round(time.time() - start, 3)

    attach_probe_results(snapshot)

    if not args.log_only:
        print("\nProtocol probe summary:")
        for package, info in snapshot["packages"].items():
            loaded = "loaded" if info.get("loaded") else "not loaded"
            submodules = info.get("submodules", {})
            loaded_subs = sum(
                1 for details in submodules.values() if details.get("loaded")
            )
            print(f"  - {package}: {loaded} ({loaded_subs} submodules)")

        print(f"\nWrote probe results to: {STATE_FILE}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
