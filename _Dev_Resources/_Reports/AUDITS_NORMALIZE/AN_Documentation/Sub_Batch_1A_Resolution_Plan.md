# Sub-Batch 1A — Import Resolution Plan

## 1. Legacy Entry
- JUNK_DRAWER.scripts.runtime.could_be_dev.start_agent
- Disposition: ARCHIVE (remain under _Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/)
- Runtime imports must be removed or redirected.

## 2. Agent System Import
- Logos_AGI.System_Operations_Protocol.infrastructure.agent_system
- Canonical targets:
  - initialize_agent_system.py
  - logos_agent_system.py
- Action: rewrite imports to canonical paths.

## 3. Tool Registry Loader
- logos.tool_registry_loader
- Status: No canonical module exists.
- Required action:
  - designate or create a single canonical tool registry module
  - all LLM / SOP tooling imports must converge here
- No code changes until module location is finalized.

## 4. Execution Gate
- No import rewrites occur until this plan is approved.
