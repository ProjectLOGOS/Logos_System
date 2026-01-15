# Architecture & Grounding Extension Plan

## Objectives
- Broaden the LOGOS demo beyond proof-gated verification by outlining a transferable cognitive architecture.
- Introduce pathways for multimodal grounding (text, symbolic datasets, and optional sensory feeds) without compromising existing proof guarantees.
- Preserve the safety gates enforced by scripts such as scripts/boot_aligned_agent.py while layering new cognitive capabilities.

## Current Baseline
- Runtime orchestration centers on proof verification and alignment gating (see README.md).
- No reusable planning layer or task decomposition framework is shipped; agent autonomy remains locked behind LEM verification.
- Repository bundles no datasets or environment shims, so demonstrations lack external grounding inputs.

## Extension Pillars

### 1. Cognitive Module Stack
- **Perception Ingestors**: Define Python interfaces under `plugins/` for text corpora, symbolic knowledge (e.g., ontologies), and optional sensor simulators. Each ingestor emits structured observations consumed by the agent once alignment is confirmed.
- **Deliberation Core**: Introduce a planning service (e.g., goal stack, agenda manager) in `Protopraxis/agent_boot.py` that can run after `SandboxedLogosAgent` unlock. Maintain guardrails by verifying that all actions route through proof-gated policies.
- **Action Library**: Catalog safe effectors in `sandbox/` with metadata describing proof requirements, ensuring new actions cannot bypass Coq validation.

### Prototype Artifacts (December 2025)
- `plugins/perception_ingestors.py` now loads symbolic concepts from `data/symbolic_concepts.json`, operator-provided corpora within `data/training/`, and runtime telemetry summaries harvested from SOP resource logs once safe-interface restrictions lift.
- `Protopraxis/agent_planner.py` offers an `AlignmentAwarePlanner` that now derives a mission agenda stack, transforms observations into safe action suggestions, including training-brief synthesis and runtime telemetry reviews once alignment unlocks planning.
- `demos/perception_planner_demo.py` and `scripts/prototype_grounding_workflow.py` surface `ObservationBroker.health_report()` snapshots so operators can verify ingest readiness before the planner activates.
- `scripts/prototype_grounding_workflow.py` demonstrates the ingestion → planner loop while respecting alignment gating.

### 2. Multimodal Grounding Assets
- **Symbolic Dataset**: Package an optional curated CSV or JSON bundle (e.g., metaphysical taxonomy, scenario annotations) under `data/` with a loader utility in `scripts/data_providers.py`.
- **Training Dropzone**: `data/training/` holds operator-supplied corpora. Ingestion modules must record provenance hashes before use.
- **Telemetry Summaries**: SOP runtime events in `external/Logos_AGI/.../resource_events.jsonl` feed aggregate statistics (peak CPU, memory, last event timestamps) into the planner, ensuring resource anomalies surface in safe actions.
- **Textual Corpus**: Provide instructions for mounting public domain corpora via a documented download script. Enforce read-only mounts to stay within audit guardrails.
- **Simulated Environment Hooks**: `sandbox/simulations/gridworld.py` implements a lightweight environment that logs JSONL traces, with `SimulationEventIngestor` summarising those events into the new `simulation` modality for planner awareness.

### 3. Workflow Integration
- Update `scripts/start_agent.py` to include a guarded configuration flag enabling the extended modules only after alignment proof success.
- Extend `scripts/system_mode_initializer.py` mission profiles to reference available datasets and simulations, ensuring audits capture which modules were enabled per run.
- Capture richer telemetry from `scripts/stress_sop_runtime.py` once CPU access returns, including dataset usage metrics and module activation traces.

## Implementation Phases
1. **Design Specs (Week 1)**
   - Draft module interface schemas (perception, deliberation, actions).
   - Select initial datasets and simulation targets.
2. **Prototype Modules (Week 2-3)**
   - Implement ingestion stubs that read sample data and log outputs post-alignment.
   - Build a minimal planner invoking existing safe actions.
3. **Grounding Integration (Week 4)**
   - Wire datasets into mission profiles and update audit logs to record source hashes.
   - Provide CLI flags to toggle grounding modules during demos.
4. **Validation & Documentation (Week 5)**
   - Extend `demos/` with a new scenario showcasing perception → deliberation → action flow.
   - Publish a runbook documenting data ingestion, module activation, and safety checks.

## Guardrail Considerations
- All new modules must respect `require_safe_interfaces` and `restrict_writes_to` constraints defined in `scripts/start_agent.py`.
- Audit logs under `state/` remain append-only; new modules append structured entries instead of overwriting existing logs.
- Any external datasets must ship with SHA-256 manifests to ensure deterministic provenance.
- Simulated environments must emit telemetry compatible with existing SOP scheduler logging for traceability.

## Next Actions
- Review and approve the interface designs for cognitive modules.
- Stand up a prototype ingestion pipeline using a small symbolic dataset.
- Coordinate with compute substrate rollout to ensure new modules have adequate resources.
