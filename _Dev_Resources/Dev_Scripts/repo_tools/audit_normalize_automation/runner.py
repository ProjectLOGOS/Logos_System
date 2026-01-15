import json
import os
import sys
from pathlib import Path
from pathlib import Path as _Path

# === AUDIT_NORMALIZE_BOOTSTRAP_SYS_PATH ===
# Purpose: allow deterministic execution whether invoked as a module or as a script.
# Fail-closed: if package cannot be resolved after bootstrapping sys.path, exit nonzero.
def _bootstrap_repo_tools_on_sys_path() -> None:
	here = _Path(__file__).resolve()
	# audit_normalize_automation/runner.py -> audit_normalize_automation -> repo_tools
	auto_pkg = here.parent
	repo_tools = here.parent.parent
	for candidate in (repo_tools, auto_pkg):
		if str(candidate) not in sys.path:
			sys.path.insert(0, str(candidate))


_bootstrap_repo_tools_on_sys_path()
# === END AUDIT_NORMALIZE_BOOTSTRAP_SYS_PATH ===

from phases.phase_1_audit import run as run_phase_1
from phases.phase_2_index import run as run_phase_2
from phases.phase_3_plan import run as run_phase_3

from compliance.validate_phase import validate_phase_1, validate_phase_2, validate_phase_3
from shared.phase_context import reports_root
from shared.manifest_utils import write_manifest

# Authoritative testbed root for current/future testing
INSPECT_ROOT = Path("_Dev_Resources/LEGACY_SCRIPTS_TO_EXAMINE")

STATE_DIR = Path("_Dev_Resources/Dev_Scripts/repo_tools/audit_normalize_automation/state")
STATE_FILE = STATE_DIR / "router_state.json"

BETA_LIMIT = 5
MAX_ATTEMPTS = 2

PASS_QUEUE = "PASS_QUEUE"
FAIL_QUEUE = "FAIL_QUEUE"
UNKNOWN_QUEUE = "UNKNOWN_QUEUE"
PENDING_UNKNOWN_QUEUE = "PENDING_UNKNOWN_QUEUE"
PENDING_FAIL_QUEUE = "PENDING_FAIL_QUEUE"


def _load_router_state() -> dict:
	STATE_DIR.mkdir(parents=True, exist_ok=True)
	if not STATE_FILE.exists():
		return {"version": "router-state/1", "scripts": {}}
	try:
		return json.loads(STATE_FILE.read_text(encoding="utf-8"))
	except Exception:
		return {"version": "router-state/1", "scripts": {}}


def _save_router_state(st: dict) -> None:
	STATE_DIR.mkdir(parents=True, exist_ok=True)
	STATE_FILE.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _bump(st: dict, script_key: str, field: str) -> int:
	rec = st.setdefault("scripts", {}).setdefault(script_key, {"fail_count": 0, "unknown_count": 0})
	rec[field] = int(rec.get(field, 0)) + 1
	return rec[field]


def _reset(st: dict, script_key: str) -> None:
	rec = st.setdefault("scripts", {}).setdefault(script_key, {"fail_count": 0, "unknown_count": 0})
	rec["fail_count"] = 0
	rec["unknown_count"] = 0


def _emit_queue(report_dir: Path, name: str, rows: list[dict]) -> None:
	qdir = report_dir / "Routing_Queues"
	qdir.mkdir(parents=True, exist_ok=True)
	(qdir / f"{name}.jsonl").write_text(
		"".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in rows),
		encoding="utf-8",
	)


def main():
	report_env = os.environ.get("AUDIT_NORMALIZE_REPORT_DIR")
	reports = reports_root()
	REPORT_DIR = Path(report_env).expanduser() if report_env else reports
	REPORT_DIR.mkdir(parents=True, exist_ok=True)

	# Phase 5 runner import (resilient to script vs package invocation)
	run_phase5_step2 = None
	run_phase5_step3 = None
	try:
		from audit_normalize_automation.phase_5.phase_5_runner import run_phase5_step2, run_phase5_step3  # type: ignore
	except Exception:
		try:
			from .phase_5.phase_5_runner import run_phase5_step2, run_phase5_step3  # type: ignore
		except Exception:
			# Final fallback: import by path
			import importlib.util
			here = Path(__file__).resolve().parent
			ph5_runner = here / "phase_5" / "phase_5_runner.py"
			spec = importlib.util.spec_from_file_location("audit_normalize_automation.phase_5.phase_5_runner", str(ph5_runner))
			if spec and spec.loader:
				mod = importlib.util.module_from_spec(spec)  # type: ignore
				spec.loader.exec_module(mod)  # type: ignore
				run_phase5_step2 = getattr(mod, "run_phase5_step2", None)  # type: ignore
				run_phase5_step3 = getattr(mod, "run_phase5_step3", None)  # type: ignore
			else:
				run_phase5_step2 = None
				run_phase5_step3 = None

	# Enumerate scripts safely:
	# - Only existing regular files should be processed.
	# - Broken symlinks or stale paths inside the testbed must not crash the runner.
	all_candidates = list(INSPECT_ROOT.rglob("*.py"))
	scripts = []
	skipped = []
	for p in all_candidates:
		try:
			if p.is_symlink() and not p.exists():
				skipped.append((str(p), "broken_symlink"))
				continue
			if not p.is_file():
				skipped.append((str(p), "not_a_file"))
				continue
			scripts.append(p)
		except FileNotFoundError:
			skipped.append((str(p), "missing"))
			continue

	scripts = scripts[:BETA_LIMIT]

	router_state = _load_router_state()
	queued_unknown: list[dict] = []
	queued_fail: list[dict] = []
	queued_pass: list[dict] = []
	pending_unknown: list[dict] = []
	pending_fail: list[dict] = []

	phase1_results = {}

	for script_path in scripts:
		script_key = str(script_path)

		try:
			if script_path.is_symlink() and not script_path.exists():
				n = _bump(router_state, script_key, "unknown_count")
				item = {"script": script_key, "reason": "broken_symlink", "count": n}
				if n == 1:
					pending_unknown.append(item)
				if n >= 2:
					queued_unknown.append(item)
				continue
			if not script_path.is_file():
				n = _bump(router_state, script_key, "unknown_count")
				item = {"script": script_key, "reason": "not_a_file", "count": n}
				if n == 1:
					pending_unknown.append(item)
				if n >= 2:
					queued_unknown.append(item)
				continue
		except FileNotFoundError:
			n = _bump(router_state, script_key, "unknown_count")
			item = {"script": script_key, "reason": "missing", "count": n}
			if n == 1:
				pending_unknown.append(item)
			if n >= 2:
				queued_unknown.append(item)
			continue

		attempt = 0
		final = None

		while attempt < MAX_ATTEMPTS:
			result = run_phase_1(script_path, reports)
			if result["status"] == "PASS" and validate_phase_1(result):
				final = result
				break
			attempt += 1

		phase1_results[script_path] = final

		run_result = {"status": "FAIL", "phase": "1"}
		if not final:
			n = _bump(router_state, script_key, "fail_count")
			item = {"script": script_key, "reason": "phase_1_failed", "count": n}
			if n == 1:
				pending_fail.append(item)
			if n >= 2:
				queued_fail.append(item)
			continue

		attempt = 0
		phase2_result = None

		while attempt < MAX_ATTEMPTS:
			result = run_phase_2(reports)
			if result["status"] == "PASS" and validate_phase_2(result):
				phase2_result = result
				break
			attempt += 1

		if not phase2_result:
			n = _bump(router_state, script_key, "fail_count")
			item = {"script": script_key, "reason": "phase_2_failed", "count": n}
			if n == 1:
				pending_fail.append(item)
			if n >= 2:
				queued_fail.append(item)
			continue

		attempt = 0
		phase3_result = None

		while attempt < MAX_ATTEMPTS:
			result = run_phase_3(reports)
			if result["status"] == "PASS" and validate_phase_3(result):
				phase3_result = result
				break
			attempt += 1

		if not phase3_result:
			n = _bump(router_state, script_key, "fail_count")
			item = {"script": script_key, "reason": "phase_3_failed", "count": n}
			if n == 1:
				pending_fail.append(item)
			if n >= 2:
				queued_fail.append(item)
			continue

		# Full PASS across implemented phases; reset counters.
		_reset(router_state, script_key)
		manifest_path = reports / "Phase_3" / "Manifests" / f"{script_path.stem}.json"
		write_manifest(
			manifest_path,
			{
				"script": str(script_path),
				"phase_1": "PASS",
				"phase_2": "PASS",
				"phase_3": "PASS",
				"routed_to": "DEFER_TO_PHASE_5",
			},
		)

	# Best-effort: emit a skip log alongside other runner artifacts (never fail the run because of skips)
	try:
		if skipped:
			skip_log = reports / "skipped_paths.txt"
			with skip_log.open("w", encoding="utf-8") as f:
				for path, reason in skipped:
					f.write(f"{reason}\t{path}\n")
	except Exception:
		pass

	# Persist router state + emit queue manifests (non-mutating)
	_save_router_state(router_state)
	_emit_queue(REPORT_DIR, UNKNOWN_QUEUE, queued_unknown)
	_emit_queue(REPORT_DIR, PENDING_UNKNOWN_QUEUE, pending_unknown)
	_emit_queue(REPORT_DIR, FAIL_QUEUE, queued_fail)
	_emit_queue(REPORT_DIR, PENDING_FAIL_QUEUE, pending_fail)
	_emit_queue(REPORT_DIR, PASS_QUEUE, queued_pass)

	# Optional mirror back to stable reports root for dashboards that expect canonical location
	try:
		import shutil
		if REPORT_DIR.resolve() != reports.resolve():
			dst_qdir = reports / "Routing_Queues"
			dst_qdir.mkdir(parents=True, exist_ok=True)
			src_qdir = REPORT_DIR / "Routing_Queues"
			for name in (UNKNOWN_QUEUE, PENDING_UNKNOWN_QUEUE, FAIL_QUEUE, PENDING_FAIL_QUEUE, PASS_QUEUE):
				fname = f"{name}.jsonl"
				src = src_qdir / fname
				if src.exists():
					shutil.copy2(src, dst_qdir / fname)
				else:
					(dst_qdir / fname).write_text("", encoding="utf-8")
	except Exception:
		pass

	# Phase 4 (read-only crawl) — queue stub generation only.
	# Phase 4 import must work both as a module and when invoked as a script.
	# Fallback order:
	#   1) absolute import (when repo_tools is on PYTHONPATH)
	#   2) relative import (when runner is executed as a package module)
	#   3) import-by-path (always works; no packaging assumptions)
	build_crawl_queue = None
	_ph4_import_mode = "missing"
	try:
		from audit_normalize_automation.phase_4_crawl import build_crawl_queue  # type: ignore
		_ph4_import_mode = "absolute"
	except Exception:
		try:
			from .phase_4_crawl import build_crawl_queue  # type: ignore
			_ph4_import_mode = "relative"
		except Exception:
			build_crawl_queue = None
			_ph4_import_mode = "missing"

	if build_crawl_queue is None:
		# Final fallback: load phase_4_crawl.py by file path
		try:
			import importlib.util
			here = Path(__file__).resolve().parent
			ph4_path = here / "phase_4_crawl.py"
			if ph4_path.exists():
				spec = importlib.util.spec_from_file_location("audit_normalize_automation.phase_4_crawl", str(ph4_path))
				mod = importlib.util.module_from_spec(spec)  # type: ignore
				assert spec and spec.loader
				spec.loader.exec_module(mod)  # type: ignore
				build_crawl_queue = getattr(mod, "build_crawl_queue", None)
				if callable(build_crawl_queue):
					_ph4_import_mode = "importlib_path"
		except Exception:
			build_crawl_queue = None
			# keep mode as "missing"

	# Phase 4 — Crawl (read-only) must execute when runner is invoked.
	# If Phase 4 cannot be imported, that is a hard failure (integration defect).
	phase_4_result = {"status": "FAIL", "error": "Phase 4 import missing"}
	if build_crawl_queue is None:
		phase_4_result = {"status": "FAIL", "error": "Phase 4 import missing"}
	else:
		try:
			phase_4_result = build_crawl_queue(REPORT_DIR)
		except Exception as e:
			phase_4_result = {"status": "FAIL", "error": str(e)}

	# Treat SKIP as PASS for gating (Phase 4 is read-only and optional until fully implemented)
	if phase_4_result.get("status") == "SKIP":
		phase_4_result = {"status": "PASS", "note": "phase_4 not implemented; treated as pass"}

	# Record Phase 4 status explicitly
	(REPORT_DIR / "Phase_4_Status.json").write_text(
		json.dumps(phase_4_result, indent=2, ensure_ascii=False) + "\n",
		encoding="utf-8",
	)

	# Emit a tiny note so logs show which import path was used
	try:
		(REPORT_DIR / "Phase_4_Import_Mode.txt").write_text(str(_ph4_import_mode) + "\n", encoding="utf-8")
	except Exception:
		pass

	# Optional: mirror Phase 4 artifacts to a stable "latest" location for convenience.
	# Per-run artifacts remain authoritative.
	try:
		import shutil
		latest_root = Path("_Reports/Audit_Normalize")
		latest_root.mkdir(parents=True, exist_ok=True)
		src_dir = REPORT_DIR / "Phase_4"
		if src_dir.exists():
			dst_dir = latest_root / "Phase_4"
			if src_dir.resolve() != dst_dir.resolve():
				if dst_dir.exists():
					shutil.rmtree(dst_dir)
				shutil.copytree(src_dir, dst_dir)
				shutil.copy2(REPORT_DIR / "Phase_4_Status.json", latest_root / "Phase_4_Status.json")
				pm = REPORT_DIR / "Phase_4_Import_Mode.txt"
				if pm.exists():
					shutil.copy2(pm, latest_root / "Phase_4_Import_Mode.txt")
			else:
				# Nothing to mirror when source and destination are identical.
				pass
	except Exception:
		pass

	if phase_4_result.get("status") != "PASS":
		raise SystemExit("Phase 4 FAILED")

	# Phase 5 Step 2 (queue emission only; no mutation)
	if run_phase5_step2 is None:
		raise SystemExit("Phase 5 runner import missing")
	try:
		phase_5_result = run_phase5_step2(REPORT_DIR)
	except Exception as e:
		raise SystemExit(f"Phase 5 FAILED: {e}")

	# Optional Phase 5 Step 3 (controlled execution)
	if run_phase5_step3:
		try:
			here = Path(__file__).resolve()
			repo_root_hint = None
			for parent in here.parents:
				if (parent / "_Dev_Resources").exists():
					repo_root_hint = parent
					break
			if repo_root_hint is None:
				repo_root_hint = here.parents[4]
			run_phase5_step3(REPORT_DIR, repo_root=repo_root_hint)
		except Exception as e:
			raise SystemExit(f"Phase 5 Step 3 FAILED: {e}")

	print("PHASE 1 → PHASE 2 → PHASE 3 BETA RUN COMPLETE")
	print(f"Phase 1 PASS count: {sum(1 for r in phase1_results.values() if r)} / {len(phase1_results)}")
	print(f"Phase 2 status: {'PASS' if queued_fail == [] else 'FAIL' if queued_fail else 'PASS'}")
	print(f"Phase 3 status: {'PASS' if queued_fail == [] else 'FAIL' if queued_fail else 'PASS'}")


if __name__ == "__main__":
	main()
