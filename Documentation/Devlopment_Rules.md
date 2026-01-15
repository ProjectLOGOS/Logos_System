# Codespace Inbound Rules

These rules are the source of truth for this Logos_System codespace: routing, naming, and guardrails.

## Routing
- All reports and similar inbound artifacts go to `_Reports/`; do not create any `_reports` directories.
- All dev/support inbound items go to `_Dev_Resources/` in the appropriate subdirectory.
- If no suitable subdirectory exists under `_Reports/` or `_Dev_Resources/`, create a new one at that root and place the inbound item there.
- System documentation meant to ship with the codebase (readmes, tech sheets, ops/function docs) goes to `Documentation/`.
- Hazardous or in-transition items (task-specific moves, temporary holding) go to `_Dev_Resources/QUARANTINE/`.

## Naming
- Use Title_Case_Underscore for any new files or directories created under these locations.
- For scripts, enforce Title_Case_Underscore in tandem with the script header rule: whenever a script is written or edited, ensure the filename matches the header `Name` field. Renames are allowed but require explicit approval and must keep content aligned with the header. Maintain a log mapping script filenames to their headers for quick lookup and purpose briefing.

## Notes
- Keep using the existing `_Reports/` and `_Dev_Resources/` roots; do not add new top-level variants.
- When adding new subdirectories for sorting, prefer concise names that reflect the content type.

## Coq Stack Guardrails (Lock & Key)
- Default posture: Coq-related paths are locked read-only via `_Dev_Resources/Dev_Scripts/repo_tools/lock_coq_stacks.sh lock`.
- Guarded paths: `/workspaces/Logos_System/PXL_Gate` and `/workspaces/Logos_System/Logos_System/System_Entry_Point/Runtime_Compiler`.
- Status: `_Dev_Resources/Dev_Scripts/repo_tools/lock_coq_stacks.sh status`.
- Temporary unlock requires the exact override token: `_Dev_Resources/Dev_Scripts/repo_tools/lock_coq_stacks.sh unlock "edit coq stacks"`.
- Relock after any edits: rerun the script with `lock` to restore read-only permissions.

## Audit and Normalize Workflow
- Each audit/normalize task corresponds to one root subdirectory under `SYSTEM_AUDIT` (11 total tasks).
- When a task is finished, commit and sync (push) before starting the next task.
- After completing an entire category across those tasks, commit, push, and sync to git as well.

## Legacy Audit Moves (Orphan/Unreachable)
- Use `_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE` as the destination for legacy/orphaned files when running consolidation scripts.
- Place all run artifacts under `_Reports/Legacy_Move_<timestamp>` (Title_Case_Underscore; do not create `_reports`). Artifacts include `audit_sources.txt`, `move_manifest.jsonl`, `moved_files.txt`, and `skipped_files.txt`.
- Require `orphaned_files.json` and `unreachable_modules.json` to be present before running; fail closed if missing.
- For each file moved: copy with metadata tagging (origin root/subsystem, relative path, UTC move time), write sidecar metadata for unsupported formats, then remove the original. Skip referenced files, directories, missing files, or existing destinations.
- After the move, ensure sources are deleted, keep the manifest, and run a best-effort `python3 -m compileall -q .` check.

## Doc Visibility in Chat
- Whenever a new file is created or a doc is appended, include its link in this chat thread to keep a visible trail.

## Legacy Script Normalization Tagging
- Destination for normalization sweeps: `_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/`.
- Preserve the full relative path inside the destination (e.g., `System_Stack/SCP/logic/old_script.py` → `.../INSPECT_DECIDE/System_Stack/SCP/logic/old_script.py`).
- Every moved file must carry origin metadata: root protocol category (e.g., `System_Entry_Point`, `System_Stack`), immediate subdirectory (e.g., `UIP`, `SCP`, `ARP`), and full repo-relative path.
- For files supporting inline comments, prepend an origin header with that metadata. For files without inline comments (e.g., `.json`), create an adjacent `.origin.json` sidecar containing the same metadata.

## Script Header Normalization (Deferred Until Script_Crawl)
- When the Script_Crawl phase begins, normalize headers for all new and touched scripts to the standard template below; until then, do not modify headers for this rule.
- Target scope: scripts under `System_Entry_Point`, `System_Stack`, `Dev_Scripts`, `Reports`, and `Tools` layers.
- Standard header (ASCII, keep blank lines as shown):

	# =============================================================================
	# File: <relative/path/from/repo/root>
	# Name: <Title_Case_With_Underscores.py>
	# Status: <CANONICAL | LEGACY | DEV_TOOLING | EXPERIMENTAL | DEPRECATED>
	# Protocol: <SOP | SCP | ARP | MTP | Logos_Protocol | PXL_Gate | Other>
	# Layer: <System_Entry_Point | System_Stack | Dev_Scripts | Reports | Tools>
	#
	# Purpose (plain language):
	#   <1–3 sentences describing what this file does and why it exists.>
	#
	# Execution Paths:
	#   - Direct: <how to run, if applicable> (e.g., python -m ... / python path/to/file.py)
	#   - Imported by: <primary importers / callers, if known>
	#   - CLI/Server: <CLI entrypoints / server ports / routes, if applicable>
	#
	# Inputs / Outputs:
	#   - Inputs: <args/env/files/network>
	#   - Outputs: <files written, network calls, stdout, state changes>
	#
	# Side Effects / Safety:
	#   - Filesystem writes: <yes/no + locations>
	#   - Network access: <yes/no + destinations>
	#   - Must fail closed: <yes/no + what “fail closed” means here>
	#
	# Dependencies:
	#   - Internal: <key internal modules>
	#   - External: <key packages>
	#
	# Change Log:
	#   - <YYYY-MM-DD>: <what changed> (<who/why> optional)
	# =============================================================================

## Script Rewrite Rule (Audit & Normalize)
- Any script touched for promotion or reintegration during Audit & Normalize must be fully rewritten (no incremental patches). Requirements: new file body, correct Title_Case_With_Underscores naming, standardized header, explicit imports (no wildcards), lint/syntax clean, clear runtime classification (dev tooling, test harness, runtime module, adapter/interface), and logged as a rewrite event with source reference.
- Scripts only archived or quarantined do not need rewriting; they remain frozen artifacts.

## Configuration and Proof Sidecars
- Config files (`.json`, `.yaml`, `.yml`, `.toml`, `.ini`): do not embed headers/comments beyond what the format allows. Create a sidecar `*.header.md` with the same logical fields as the Python header (`File`, `Status`, `Protocol`, `Layer`, `Purpose`, `Inputs / Outputs`, `Side Effects / Safety`, `Dependencies`, `Change Log`). Example: `My_Config.json` → `My_Config.json.header.md`.
- Coq proof files (`.v`) are immutable: never change headers, comments, formatting, or whitespace. Documentation lives in a sidecar `Some_Proof_Module.v.header.md` only, using the template:

	=============================================================================
	File: <relative/path/to/Some_Proof_Module.v>
	Sidecar: <relative/path/to/Some_Proof_Module.v.header.md>
	Status: <CANONICAL | LEGACY | EXPERIMENTAL | DEPRECATED>
	Domain: <PXL_Gate | Runtime_Compiler | Other>

	Purpose (plain language):
		<What this proof/module establishes and why it exists.>

	Build / Proof Gate Paths:
		- Compiled by: <make target / dune / opam invocation>
		- Imported by: <key modules>
		- Key theorems: <if applicable>

	Constraints:
		- Admits: <expected none / allowed with justification>
		- Fail-closed relevance: <yes/no>

	Change Log (sidecar only):
		- <YYYY-MM-DD>: <documentation update>
	=============================================================================

- Legacy protocol names in existing headers/comments (e.g., Archon, Telos, Tetragnos, Thonoc, Kyrex) are documentation-only. When editing a file for other reasons, replace or remove legacy references in favor of canonical protocol naming. No mass rewrites outside approved refactor phases.
- Enforcement: headers/sidecars are mandatory for new files and files touched during refactors; script headers remain deferred until Script_Crawl. `.v` files stay untouched; use sidecars for documentation.
