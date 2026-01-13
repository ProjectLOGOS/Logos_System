# JUNK_DRAWER/scripts Catalog

Every file under JUNK_DRAWER/scripts with a brief purpose note. Shell helpers are listed alongside Python entrypoints and state artifacts.

| File | Description |
| ---- | ----------- |
| [JUNK_DRAWER/scripts/README.md](JUNK_DRAWER/scripts/README.md) | High-level overview of the scripts collection, alignment flow, and operator guardrails. |
| [JUNK_DRAWER/scripts/__init__.py](JUNK_DRAWER/scripts/__init__.py) | Marks the scripts directory as a Python package for relative imports. |
| [JUNK_DRAWER/scripts/_bootstrap.py](JUNK_DRAWER/scripts/_bootstrap.py) | Shared bootstrap that resolves repo root and sets import paths for scripts. |
| [JUNK_DRAWER/scripts/aligned_agent_import.py](JUNK_DRAWER/scripts/aligned_agent_import.py) | Alignment-gated importer that refreshes external Logos_AGI code with drift checks and optional protocol probing. |
| [JUNK_DRAWER/scripts/archive_planner_digests.py](JUNK_DRAWER/scripts/archive_planner_digests.py) | Compresses planner digest logs into archives for retention. |
| [JUNK_DRAWER/scripts/attestation.py](JUNK_DRAWER/scripts/attestation.py) | Shared attestation validator used by proof-gated agent runtime. |
| [JUNK_DRAWER/scripts/boot_aligned_agent.py](JUNK_DRAWER/scripts/boot_aligned_agent.py) | Boots a sandboxed Logos agent only after constructive LEM is discharged; records audit entries. |
| [JUNK_DRAWER/scripts/build_coq_theorem_index.py](JUNK_DRAWER/scripts/build_coq_theorem_index.py) | Builds an index of Coq theorems from existing proof artifacts. |
| [JUNK_DRAWER/scripts/capture_arp_traces_and_backfill.py](JUNK_DRAWER/scripts/capture_arp_traces_and_backfill.py) | Captures deterministic ARP traces and backfills fixture hashes for audits. |
| [JUNK_DRAWER/scripts/check_run_cycle_prereqs.py](JUNK_DRAWER/scripts/check_run_cycle_prereqs.py) | Quick diagnostic ensuring dependencies for tools/run_cycle.sh are present. |
| [JUNK_DRAWER/scripts/cleanup_extensions.sh](JUNK_DRAWER/scripts/cleanup_extensions.sh) | Removes non-essential VS Code extensions to reduce memory usage. |
| [JUNK_DRAWER/scripts/compute_identity_hash.py](JUNK_DRAWER/scripts/compute_identity_hash.py) | Computes or verifies a canonical identity hash over gated artifacts. |
| [JUNK_DRAWER/scripts/cycle_ledger.py](JUNK_DRAWER/scripts/cycle_ledger.py) | Ledger utilities for supervised promotion cycles. |
| [JUNK_DRAWER/scripts/demo.sh](JUNK_DRAWER/scripts/demo.sh) | One-command LOGOS demo launcher with health checks and modes (quick, full, web). |
| [JUNK_DRAWER/scripts/emergency_recovery.sh](JUNK_DRAWER/scripts/emergency_recovery.sh) | Codespace recovery helper that kills hung Coq processes and cleans VS Code caches. |
| [JUNK_DRAWER/scripts/evidence.py](JUNK_DRAWER/scripts/evidence.py) | Helpers for handling grounded evidence references in responses. |
| [JUNK_DRAWER/scripts/export_tool_registry.py](JUNK_DRAWER/scripts/export_tool_registry.py) | Exports the native tool catalog from start_agent into JSON and Markdown registry snapshots. |
| [JUNK_DRAWER/scripts/golden_run.sh](JUNK_DRAWER/scripts/golden_run.sh) | Rebuilds Coq artifacts, runs proof gate, computes identity hash, and records run fingerprints. |
| [JUNK_DRAWER/scripts/investor_dashboard.py](JUNK_DRAWER/scripts/investor_dashboard.py) | Investor verification dashboard renderer for demo/reporting. |
| [JUNK_DRAWER/scripts/investor_demo.py](JUNK_DRAWER/scripts/investor_demo.py) | Investor-facing narrated demo script emphasizing proof-before-autonomy. |
| [JUNK_DRAWER/scripts/logos_agi_adapter.py](JUNK_DRAWER/scripts/logos_agi_adapter.py) | Adapter wiring Logos_AGI ARP/SCP integration into the supervised entrypoint loop. |
| [JUNK_DRAWER/scripts/logos_chat.py](JUNK_DRAWER/scripts/logos_chat.py) | Interactive LOGOS chat interface with advisory gating. |
| [JUNK_DRAWER/scripts/logos_help.sh](JUNK_DRAWER/scripts/logos_help.sh) | CLI quick reference for chat vs protocol modes and setup steps. |
| [JUNK_DRAWER/scripts/logos_interface.py](JUNK_DRAWER/scripts/logos_interface.py) | Protocol-grounded interface routing natural language to LOGOS backends with minimal hallucination risk. |
| [JUNK_DRAWER/scripts/make_release_bundle.sh](JUNK_DRAWER/scripts/make_release_bundle.sh) | Creates an audit tarball containing proofs, fingerprints, and manifests for release. |
| [JUNK_DRAWER/scripts/nexus_manager.py](JUNK_DRAWER/scripts/nexus_manager.py) | Bounded in-memory manager for LogosAgiNexus instances. |
| [JUNK_DRAWER/scripts/protocol_probe.py](JUNK_DRAWER/scripts/protocol_probe.py) | Safe-by-default probe of Logos_AGI protocol packages with telemetry capture. |
| [JUNK_DRAWER/scripts/prototype_grounding_workflow.py](JUNK_DRAWER/scripts/prototype_grounding_workflow.py) | Prototype runner that feeds ingestion outputs into the planner for grounding. |
| [JUNK_DRAWER/scripts/provenance.py](JUNK_DRAWER/scripts/provenance.py) | Provenance utilities for Logos_AGI pinning and drift detection. |
| [JUNK_DRAWER/scripts/run_cycle_loop.py](JUNK_DRAWER/scripts/run_cycle_loop.py) | Orchestrates repeated tools/run_cycle.sh runs with health checks, logging, and backoff. |
| [JUNK_DRAWER/scripts/run_mvp_acceptance.py](JUNK_DRAWER/scripts/run_mvp_acceptance.py) | Executes the frozen MVP acceptance contract and reports results. |
| [JUNK_DRAWER/scripts/run_recursive_immersion_cycle.py](JUNK_DRAWER/scripts/run_recursive_immersion_cycle.py) | Runs IMMERSION CYCLE probes in read-only sandbox mode, logging structured reflections and commitment hashes. |
| [JUNK_DRAWER/scripts/run_simulation_stub.py](JUNK_DRAWER/scripts/run_simulation_stub.py) | Generates example simulation events for sandbox ingestion. |
| [JUNK_DRAWER/scripts/scan_repo.py](JUNK_DRAWER/scripts/scan_repo.py) | Scans the repository to build/update the LOGOS knowledge base. |
| [JUNK_DRAWER/scripts/schemas.py](JUNK_DRAWER/scripts/schemas.py) | Schema definitions and validators for Logos_AGI persistence. |
| [JUNK_DRAWER/scripts/scp_local_diagnostic.py](JUNK_DRAWER/scripts/scp_local_diagnostic.py) | Deterministic SCP stress diagnostic without the full stress runtime. |
| [JUNK_DRAWER/scripts/security/session.py](JUNK_DRAWER/scripts/security/session.py) | Session ID utilities and cookie helpers for the LOGOS-GPT FastAPI surface. |
| [JUNK_DRAWER/scripts/start_agent.py](JUNK_DRAWER/scripts/start_agent.py) | Bounded supervised agent loop wiring tools under mission profile constraints. |
| [JUNK_DRAWER/scripts/state/alignment_LOGOS-AGENT-OMEGA.json](JUNK_DRAWER/scripts/state/alignment_LOGOS-AGENT-OMEGA.json) | Audit log entry capturing alignment verification metadata (agent ID, hash, proof status). |
| [JUNK_DRAWER/scripts/state/mission_profile.json](JUNK_DRAWER/scripts/state/mission_profile.json) | Persisted mission profile settings controlling autonomy, logging, and guardrails. |
| [JUNK_DRAWER/scripts/state_policy.md](JUNK_DRAWER/scripts/state_policy.md) | Notes on state handling and persistence policy for scripts outputs. |
| [JUNK_DRAWER/scripts/stress_sop_runtime.py](JUNK_DRAWER/scripts/stress_sop_runtime.py) | Stress test for SOP runtime scheduler and telemetry pipeline. |
| [JUNK_DRAWER/scripts/system_mode_initializer.py](JUNK_DRAWER/scripts/system_mode_initializer.py) | Initializes and writes mission profiles to state with sandbox write caps. |
| [JUNK_DRAWER/scripts/test_alignment_gate_smoke.py](JUNK_DRAWER/scripts/test_alignment_gate_smoke.py) | Smoke test for alignment gate enforcement. |
| [JUNK_DRAWER/scripts/test_belief_consolidation_smoke.py](JUNK_DRAWER/scripts/test_belief_consolidation_smoke.py) | Smoke test for belief consolidation. |
| [JUNK_DRAWER/scripts/test_belief_quarantine_persistence.py](JUNK_DRAWER/scripts/test_belief_quarantine_persistence.py) | Ensures belief quarantine persists across consolidation cycles. |
| [JUNK_DRAWER/scripts/test_belief_tool_policy.py](JUNK_DRAWER/scripts/test_belief_tool_policy.py) | Tests belief-informed tool selection policy. |
| [JUNK_DRAWER/scripts/test_evaluator_learning_smoke.py](JUNK_DRAWER/scripts/test_evaluator_learning_smoke.py) | Smoke test demonstrating deterministic evaluator learning improvement. |
| [JUNK_DRAWER/scripts/test_goal_proposal_smoke.py](JUNK_DRAWER/scripts/test_goal_proposal_smoke.py) | Smoke test for goal proposal generation. |
| [JUNK_DRAWER/scripts/test_grounded_reply_enforcement.py](JUNK_DRAWER/scripts/test_grounded_reply_enforcement.py) | Ensures grounding sanitizer downgrades unsupported claims. |
| [JUNK_DRAWER/scripts/test_lem_discharge.py](JUNK_DRAWER/scripts/test_lem_discharge.py) | End-to-end proof gate demonstrating Coq build and proof footprint. |
| [JUNK_DRAWER/scripts/test_llm_advisor_smoke.py](JUNK_DRAWER/scripts/test_llm_advisor_smoke.py) | Smoke test ensuring advisor can propose safe low-impact tool in stub mode. |
| [JUNK_DRAWER/scripts/test_llm_bypass_smoke.py](JUNK_DRAWER/scripts/test_llm_bypass_smoke.py) | Smoke test verifying the advisor cannot bypass execution gates. |
| [JUNK_DRAWER/scripts/test_llm_real_provider_smoke.py](JUNK_DRAWER/scripts/test_llm_real_provider_smoke.py) | Smoke tests for real LLM providers (advisor-only). |
| [JUNK_DRAWER/scripts/test_llm_streaming_smoke.py](JUNK_DRAWER/scripts/test_llm_streaming_smoke.py) | Smoke test for streaming chat path. |
| [JUNK_DRAWER/scripts/test_logos_agi_bootstrap_modes.py](JUNK_DRAWER/scripts/test_logos_agi_bootstrap_modes.py) | Tests Logos_AGI bootstrap modes. |
| [JUNK_DRAWER/scripts/test_logos_agi_integration_smoke.py](JUNK_DRAWER/scripts/test_logos_agi_integration_smoke.py) | Smoke test for Logos_AGI integration. |
| [JUNK_DRAWER/scripts/test_logos_agi_persistence_smoke.py](JUNK_DRAWER/scripts/test_logos_agi_persistence_smoke.py) | Smoke test for Logos_AGI persistence across cycles. |
| [JUNK_DRAWER/scripts/test_logos_agi_pin_drift.py](JUNK_DRAWER/scripts/test_logos_agi_pin_drift.py) | Tests Logos_AGI commit pinning and drift detection. |
| [JUNK_DRAWER/scripts/test_logos_agi_replay_proposals.py](JUNK_DRAWER/scripts/test_logos_agi_replay_proposals.py) | Regression test for Logos_AGI proposal replay behavior. |
| [JUNK_DRAWER/scripts/test_logos_gpt_chat_smoke.py](JUNK_DRAWER/scripts/test_logos_gpt_chat_smoke.py) | Smoke test ensuring LOGOS-GPT chat loop remains advisory and gated. |
| [JUNK_DRAWER/scripts/test_logos_gpt_web_smoke.py](JUNK_DRAWER/scripts/test_logos_gpt_web_smoke.py) | Stub-friendly grounding smoke test for the LOGOS-GPT web FastAPI surface. |
| [JUNK_DRAWER/scripts/test_logos_runtime_phase2_smoke.py](JUNK_DRAWER/scripts/test_logos_runtime_phase2_smoke.py) | Phase 2 runtime smoke test coverage. |
| [JUNK_DRAWER/scripts/test_logos_runtime_smoke.py](JUNK_DRAWER/scripts/test_logos_runtime_smoke.py) | Baseline runtime smoke test for LOGOS pipelines. |
| [JUNK_DRAWER/scripts/test_plan_checkpoint_smoke.py](JUNK_DRAWER/scripts/test_plan_checkpoint_smoke.py) | Smoke test for plan checkpoint execution. |
| [JUNK_DRAWER/scripts/test_plan_history_inprocess_refresh.py](JUNK_DRAWER/scripts/test_plan_history_inprocess_refresh.py) | Verifies in-process plan history refresh reorders selection. |
| [JUNK_DRAWER/scripts/test_plan_history_update_smoke.py](JUNK_DRAWER/scripts/test_plan_history_update_smoke.py) | Smoke tests for plan history persistence and chaining. |
| [JUNK_DRAWER/scripts/test_plan_revision_on_contradiction.py](JUNK_DRAWER/scripts/test_plan_revision_on_contradiction.py) | Tests plan revision when beliefs contradict steps. |
| [JUNK_DRAWER/scripts/test_plan_scoring_smoke.py](JUNK_DRAWER/scripts/test_plan_scoring_smoke.py) | Smoke tests for deterministic plan scoring. |
| [JUNK_DRAWER/scripts/test_plan_validation_smoke.py](JUNK_DRAWER/scripts/test_plan_validation_smoke.py) | Smoke tests for deterministic plan validation invariants. |
| [JUNK_DRAWER/scripts/test_proved_grounding.py](JUNK_DRAWER/scripts/test_proved_grounding.py) | Tests PROVED truth grounding against theorem index. |
| [JUNK_DRAWER/scripts/test_retrieval_local_smoke.py](JUNK_DRAWER/scripts/test_retrieval_local_smoke.py) | Smoke test for retrieve.local tool over docs/README.md. |
| [JUNK_DRAWER/scripts/test_run_ledger_smoke.py](JUNK_DRAWER/scripts/test_run_ledger_smoke.py) | Smoke test for run ledger generation. |
| [JUNK_DRAWER/scripts/test_scp_recovery_mode_gate.py](JUNK_DRAWER/scripts/test_scp_recovery_mode_gate.py) | Tests SCP recovery mode gating. |
| [JUNK_DRAWER/scripts/test_server_nexus_isolation_smoke.py](JUNK_DRAWER/scripts/test_server_nexus_isolation_smoke.py) | Smoke test ensuring server nexus isolation per session. |
| [JUNK_DRAWER/scripts/test_stub_beliefs_never_verified.py](JUNK_DRAWER/scripts/test_stub_beliefs_never_verified.py) | Ensures stub-mode synthesized beliefs never escalate truth tiers. |
| [JUNK_DRAWER/scripts/test_tool_approval_tamper_smoke.py](JUNK_DRAWER/scripts/test_tool_approval_tamper_smoke.py) | Tests tamper detection for approved tools. |
| [JUNK_DRAWER/scripts/test_tool_fallback_proposal.py](JUNK_DRAWER/scripts/test_tool_fallback_proposal.py) | Ensures failed tools surface fallback proposals without execution. |
| [JUNK_DRAWER/scripts/test_tool_improvement_intent_smoke.py](JUNK_DRAWER/scripts/test_tool_improvement_intent_smoke.py) | Smoke test for tool improvement intent flow. |
| [JUNK_DRAWER/scripts/test_tool_introspection_smoke.py](JUNK_DRAWER/scripts/test_tool_introspection_smoke.py) | Smoke test for deterministic tool introspection. |
| [JUNK_DRAWER/scripts/test_tool_pipeline_smoke.py](JUNK_DRAWER/scripts/test_tool_pipeline_smoke.py) | Smoke test for governed tool proposal pipeline. |
| [JUNK_DRAWER/scripts/test_tool_playbook_load.py](JUNK_DRAWER/scripts/test_tool_playbook_load.py) | Verifies tool playbooks cover all registered tools. |
| [JUNK_DRAWER/scripts/test_tool_repair_proposal_smoke.py](JUNK_DRAWER/scripts/test_tool_repair_proposal_smoke.py) | Smoke test for UIP-gated tool repair proposal generation. |
| [JUNK_DRAWER/scripts/test_tool_validation_smoke.py](JUNK_DRAWER/scripts/test_tool_validation_smoke.py) | Smoke tests for deterministic tool validation pipeline. |
| [JUNK_DRAWER/scripts/test_truth_categories_smoke.py](JUNK_DRAWER/scripts/test_truth_categories_smoke.py) | Smoke test for truth category emission and persistence. |
| [JUNK_DRAWER/scripts/test_truth_evaluator_ranking.py](JUNK_DRAWER/scripts/test_truth_evaluator_ranking.py) | Tests evaluator ranking using truth categories. |
| [JUNK_DRAWER/scripts/test_uwm_smoke.py](JUNK_DRAWER/scripts/test_uwm_smoke.py) | Smoke test for unified working memory integration. |
| [JUNK_DRAWER/scripts/test_web_grounding_smoke.py](JUNK_DRAWER/scripts/test_web_grounding_smoke.py) | Grounding smoke test for LOGOS-GPT server in stub mode. |
| [JUNK_DRAWER/scripts/tool_introspection.py](JUNK_DRAWER/scripts/tool_introspection.py) | Deterministic tool capability introspection (read-only, audited). |
| [JUNK_DRAWER/scripts/tool_proposal_pipeline.py](JUNK_DRAWER/scripts/tool_proposal_pipeline.py) | Governed pipeline to generate, validate, approve, and register tools. |
| [JUNK_DRAWER/scripts/tool_repair_proposal.py](JUNK_DRAWER/scripts/tool_repair_proposal.py) | Deterministic UIP-gated tool repair proposal generator. |
| [JUNK_DRAWER/scripts/vscode_health_check.sh](JUNK_DRAWER/scripts/vscode_health_check.sh) | VS Code health check and optional cleanup for Codespaces. |
