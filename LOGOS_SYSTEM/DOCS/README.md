# LOGOS Proof-Oriented Demo

This repository contains the proof artifacts and Python instrumentation used to
showcase the LOGOS constructive Law of Excluded Middle (LEM) discharge and the
aligned agent boot sequence. All demonstrations execute the real Coq kernel and
report on its assumption footprint in real time—there are no pre-recorded
outputs.

## Key Capabilities

- **8-Axiom Minimal Kernel**: Reduced from 20 axioms to 8 irreducible metaphysical axioms (Phase 2/3 complete)
  - **60% reduction** achieved through systematic elimination:
    - **3 modal axioms** → Semantic frame conditions (Kripke S5)
    - **5 structural axioms** → Definitional (Ident := Leibniz equality)
    - **4 bridge axioms** → Definitional (PImp := →, MEquiv := ↔)
  - **Final kernel**: 2 core metaphysical axioms (A2_noncontradiction, A7_triune_necessity) + 6 bridging principles (entailment/grounding semantics)
- **Constructive LEM**: Law of Excluded Middle proven as theorem from triune grounding (not assumed)
- **Privative Negation**: Resolves classical paradoxes (Liar, Russell's) via absence-based negation
- **Semantic Modal Logic**: S5 Kripke semantics with frame conditions (Box/Dia operators)
- **Proof-Gated Alignment**: Agent actions blocked unless safety theorems verify with zero extra assumptions- **Truth-Category Sorting**: Proposals annotated with truth tiers (PROVED/VERIFIED/INFERRED/HEURISTIC/UNVERIFIED/CONTRADICTED) for deterministic ranking and lightweight policy constraints- Rebuilds the Protopraxis baseline kernel from `_CoqProject` on every run
- Confirms that `pxl_excluded_middle` is proved constructively (no extra axioms, no `Admitted.` stubs)
- Boots a sandboxed LOGOS agent that unlocks only after the constructive LEM verification succeeds
- Provides investor-facing demos translating technical outputs into an executive summary
- Emits tamper-evident audit logs containing the agent identity hash and run timestamp

## Practical Demonstrations

**NEW**: Interactive demonstrations showcasing PXL's unique capabilities:

```bash
# Run individual demos
python3 demos/alignment_demo.py          # Proof-gated agent safety
python3 demos/lem_demo.py                # Constructive LEM walkthrough
python3 demos/privative_negation_demo.py # Paradox resolution

# Run all demos with summary report
python3 demos/run_all_demos.py
```

## Prerequisites

| Dependency | Notes |
|------------|-------|
| Python 3.11+ | Used for the orchestration scripts. |
| Coq 8.18+   | Required to rebuild the PXL kernel. Available via `apt-get install coq`. |
| GNU Make    | Invoked by the scripts through the `coq_makefile` tooling. |

The scripts rely only on the Python standard library; see `requirements.txt`
for optional utilities used during development.

## Truth Categories

Proposals are annotated with truth tiers for deterministic ranking and policy constraints:

- **PROVED**: Backed by verified Coq theorem reference matching the theorem index (file + theorem name + statement hash)
- **VERIFIED**: Deterministic validator output or downgraded PROVED claims
- **INFERRED**: Heuristic inference from available evidence
- **HEURISTIC**: Best-effort approximation without formal verification
- **UNVERIFIED**: Insufficient evidence for reliable assessment
- **CONTRADICTED**: Evidence indicates logical inconsistency

Example PROVED evidence:
```json
{
  "truth": "PROVED",
  "evidence": {
    "type": "coq",
    "ref": {
      "theorem": "pxl_excluded_middle",
      "file": "Protopraxis/formal_verification/coq/baseline/PXL_Internal_LEM.v",
      "statement_hash": "abc123...",
      "index_hash": "def456..."
    }
  }
}
```

## Quick Start

```bash
# Install system dependencies (Ubuntu example)
sudo apt-get update && sudo apt-get install -y coq make

# Clone the repository
git clone https://github.com/ProjectLOGOS/pxl_demo_wcoq_proofs.git
cd pxl_demo_wcoq_proofs

# (Optional) create a virtual environment if you plan to install extra tools
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Developer Environment Helpers

- `_bootstrap.py` centralizes repository path setup. Import `REPO_ROOT` from
  this module whenever a script needs repo-relative access instead of mutating
  `sys.path` inline.
- `.vscode/settings.json` contains the authoritative spell-check dictionary.
  Add new project terminology there so editor warnings align with lint
  baselines; direct `cspell` CLI runs require an explicit config.

These helpers keep scripts and documentation consistent with the alignment
workflow and should be reused when adding new entry points.

## Core Workflows

### 1. Proof Regeneration Harness

```bash
python3 test_lem_discharge.py
```

This command regenerates the Coq makefile, performs a clean rebuild of the
kernel, prints the assumption footprint for the trinitarian optimization and
internal LEM proofs, and flags any remaining `Admitted.` stubs.

### 2. Sandboxed Agent Boot

```bash
python3 scripts/boot_aligned_agent.py
```

The script rebuilds the kernel, verifies the constructive LEM proof, and only
then unlocks the agent. Each run appends a JSON audit record under
`state/alignment_LOGOS-AGENT-OMEGA.json` including the agent hash and the exact
verification timestamp.

### 3. Investor-Facing Demonstrations

- **Narrated demo:** `python3 scripts/investor_demo.py`
- **Executive dashboard:** `python3 scripts/investor_dashboard.py`

Both workflows run the full verification pipeline. The demo script renders a
step-by-step translation for live presentations, while the dashboard provides a
concise go/no-go report.

### 4. Alignment-Gated Repository Import

```bash
python3 scripts/aligned_agent_import.py --probe
```

This utility repeats the constructive LEM verification and, once the agent is
confirmed aligned, clones `ProjectLOGOS/Logos_AGI` into `external/Logos_AGI`
before loading the ARP/SOP/UIP/SCP packages. It performs a smoke test of key
submodules, records the imported commit SHA, and appends both to the alignment
audit log. Passing `--probe` automatically runs `protocol_probe` afterwards to
exercise read-only APIs on each module and persist the findings under
`protocol_probe_runs` in the audit log.

### 5. Logos_AGI Commit Pinning and Drift Detection

To prevent silent changes in the external Logos_AGI dependency, the system
supports commit pinning with runtime drift detection:

```bash
# Pin the current HEAD
python3 scripts/aligned_agent_import.py --pin-sha HEAD --pin-note "Phase 4.5 implementation"

# Pin a specific commit or tag
python3 scripts/aligned_agent_import.py --pin-sha v1.2.3 --pin-note "Stable release"

# Allow dirty working directory during pinning
python3 scripts/aligned_agent_import.py --pin-sha HEAD --allow-dirty --pin-note "Development snapshot"
```

Once pinned, the agent enforces the pin at runtime:

```bash
# Normal operation (requires exact SHA match)
python3 scripts/start_agent.py --enable-logos-agi --objective "status"

# Allow drift in development (requires LOGOS_DEV_BYPASS_OK=1)
LOGOS_DEV_BYPASS_OK=1 python3 scripts/start_agent.py --enable-logos-agi --allow-logos-agi-drift --objective "status"
```

### 6. Governed Tool Proposal Pipeline

The system implements a governed pipeline for tool invention and optimization, ensuring all proposed tools undergo validation, operator approval, and auditable registration before execution.

```bash
# Generate tool proposals (writes to sandbox/tool_proposals/)
python3 scripts/tool_proposal_pipeline.py generate --objective "tool_invention"

# Validate proposal in sandbox
python3 scripts/tool_proposal_pipeline.py validate sandbox/tool_proposals/<proposal_id>.json

# Approve validated proposal (requires operator review)
python3 scripts/tool_proposal_pipeline.py approve sandbox/tool_proposals/<proposal_id>.json --operator "your_name"

# Test the pipeline end-to-end
python3 scripts/test_tool_pipeline_smoke.py
```

**Security Model:**
- Proposals are generated but not executed until approved
- Validation runs in isolated subprocess with timeouts
- All approvals are logged to `audit/tool_approvals.jsonl`
- Only approved tools are loaded into the agent registry
- Execution requires attestation and goes through `dispatch_tool()`

**Artifacts:**
- Proposals: `sandbox/tool_proposals/<id>.json`
- Validation reports: `sandbox/tool_proposals/<id>.validation.json`
- Approved tools: `tools/approved/<name>/`
- Audit log: `audit/tool_approvals.jsonl`

The pin file is stored at `state/logos_agi_pin.json` and contains the pinned SHA,
timestamp, and metadata. Drift detection checks both SHA mismatches and dirty
working directories. Provenance information is logged in audit events and
alignment records.

## Operator Guardrails

High-risk entrypoints require explicit operator acknowledgment:

- Set env: `LOGOS_OPERATOR_OK=1`
- Provide the matching flag when applicable:
  - `--allow-training-index-write` for start_agent runs that may touch `training_data/index` catalogs
  - `--allow-state-test-write` for state/audit persistence checks (e.g., PAI verification or Logos_AGI persistence smoke)
  - `--demo-ok` for demos that start servers or emit artifacts (e.g., interactive web server, run_all_demos)

### 6. Proposal Evaluation and Learning

The system includes a bounded proposal evaluator that improves action selection
deterministically based on historical outcomes, without expanding authority or
enabling self-modification:

- **Metrics State**: Stored at `state/proposal_metrics.json`, hash-chained like SCP state
- **Learning Mechanism**: Scores tools by success rate, penalizing recent failures
- **Deterministic Ranking**: Reorders proposals by score; evaluator cannot execute tools
- **Security**: Only ranks/filter proposals; `dispatch_tool()` remains the executor

Metrics are updated after each tool execution with outcomes (SUCCESS/DENY/ERROR).
The evaluator applies extra penalties for tools that failed in the last run under
the same objective class, demonstrating bounded learning.

To reset metrics in development:

```bash
rm state/proposal_metrics.json
```

## Continuous Integration

The GitHub Actions workflow at `.github/workflows/alignment-check.yml`
installs Coq and runs `scripts/boot_aligned_agent.py` on every push and pull
request. Any regression in the constructive LEM proof will fail the build.

The repository also includes `.github/workflows/integration-test.yml`, which
executes the full-stack pipeline (proof rebuild, runtime alignment, demos, and
regression tests) on schedule, on demand, and for pull requests into `main`.

Local smoke tests for ingestion health and the simulation CLI can be run with:

```bash
python3 tools/run_ci_checks.py
```

This wrapper executes `tests.test_perception_ingestors` and
`tests.test_simulation_cli`, mirroring the CI gating added for dataset
traceability and telemetry validation.

### Alignment Gate Smoke Test

To verify the proof-attestation gate is functioning:

```bash
python3 scripts/test_alignment_gate_smoke.py
```

This test checks that `start_agent.py` refuses to run without valid attestation,
accepts valid attestation, and blocks unauthorized bypass attempts.

## Repository Layout

```text
README.md                     Project overview (this file)
scripts/boot_aligned_agent.py         Sandboxed agent gate tied to constructive LEM
scripts/investor_demo.py              Investor narrative demo
scripts/investor_dashboard.py         Executive summary dashboard
test_lem_discharge.py         Proof regeneration harness
state/                        Persistent audit logs and agent state
Protopraxis/formal_verification/coq/baseline/  Baseline Coq sources
.github/workflows/            CI definitions
```

Additional archives and experimental front-end assets live under `archive_*`
and do not participate in the primary demo flow.

## Branch Strategy & Development Workflow

This repository uses a **4-branch structure** with clear divisions of concern:

### 1. `main` (Protected, Verified Releases Only)
- **Purpose**: Stable, production-ready code
- **Merge Requirements**:
  - All CI checks must pass (Coq proofs, runtime alignment, presentation validation)
  - Integration tests must pass
  - Requires code review
- **Direct commits**: ❌ Forbidden
- **Deploy target**: Investor demos, public releases

### 2. `coq-proofs` (Coq Development Branch)
- **Purpose**: All Coq proof development and axiom reduction work
- **Scope**:
  - `Protopraxis/formal_verification/coq/**`
  - `tools/axiom_*.py` (proof tooling)
  - `_CoqProject`, `CoqMakefile`
- **CI Gates** (`.github/workflows/coq-proofs-ci.yml`):
  - ✅ All Coq files compile
  - ✅ Axiom budget: 8/8 (enforced by `axiom_gate.py`)
  - ✅ Zero `Admitted.` stubs
  - ✅ `test_lem_discharge.py` passes
- **Merge to main**: Must pass all proof verification gates

### 3. `runtime-dev` (Agent Stack Branch)
- **Purpose**: LOGOS agent runtime, guardrails, and orchestration
- **Scope**:
  - `scripts/boot_aligned_agent.py`, `scripts/start_agent.py`
  - `scripts/system_mode_initializer.py`, `Protopraxis/agent_boot.py`
  - `plugins/*`, `sandbox/*`
  - `scripts/aligned_agent_import.py`, `scripts/protocol_probe.py`
- **CI Gates** (`.github/workflows/runtime-dev-ci.yml`):
  - ✅ Rebuilds Coq proofs (runtime depends on verified state)
  - ✅ `scripts/boot_aligned_agent.py` shows "ALIGNED" status
  - ✅ Mission profile initialization succeeds
  - ✅ Guardrails import successfully
- **Merge to main**: Must rebase on latest `main` to ensure proof compatibility
- **Critical**: Runtime cannot execute without verified Coq proofs

### 4. `presentation-dev` (Demos & Investor Materials)
- **Purpose**: Investor-facing dashboards, demos, and documentation
- **Scope**:
  - `scripts/investor_dashboard.py`, `scripts/investor_demo.py`
  - `pxl_logos-agi-safety-demonstration/*` (React demo)
  - `docs/investor_narrative_full.md` and other documentation
- **CI Gates** (`.github/workflows/presentation-ci.yml`):
  - ✅ Python syntax validation
  - ✅ Markdown formatting checks
  - ✅ Documentation completeness
- **Merge to main**: No blocking requirements (can always merge)

### Integration Testing
- **Workflow**: `.github/workflows/integration-test.yml`
- **Runs on**: All PRs to `main`, daily schedule, manual trigger
- **Full stack test**:
  1. Build Coq proofs
  2. Verify runtime alignment
  3. Initialize agent system
  4. Run agent smoke test
  5. Validate investor dashboard
  6. Execute full test suite
- **Purpose**: Ensures all three domains (Coq, Runtime, Presentation) work together

### Development Workflow

**Starting new work:**
```bash
# For Coq proof development
git checkout coq-proofs
git pull origin coq-proofs

# For runtime/agent work
git checkout runtime-dev
git pull origin runtime-dev

# For presentation/docs
git checkout presentation-dev
git pull origin presentation-dev
```

**Merging to main:**
```bash
# 1. Ensure your branch is up to date
git checkout coq-proofs  # or runtime-dev, presentation-dev
git pull origin coq-proofs

# 2. Push and create PR
git push origin coq-proofs
# Create PR via GitHub UI

# 3. Wait for CI checks
# - coq-proofs: Must pass proof verification
# - runtime-dev: Must pass alignment checks (and rebase on main if proofs changed)
# - presentation-dev: Must pass formatting checks

# 4. After approval and green CI, merge to main
```

**Cross-branch dependencies:**
- If `coq-proofs` changes axiom counts → `runtime-dev` must update SHA-256 guards
- If `runtime-dev` changes audit log format → `presentation-dev` dashboards may need updates
- Use integration tests to catch these dependencies

## Truth Categories

The LOGOS agent incorporates truth-category annotations on all proposals and observations to enable deterministic ranking improvements without expanding tool authority. Truth categories are advisory metadata that influence evaluator scoring and apply lightweight policy constraints, but they do **not** replace Coq proof gating or attestation requirements.

### Truth Tiers

- **PROVED**: Directly backed by Coq theorem/proof artifact
- **VERIFIED**: Checked by deterministic validator (hash/provenance/schema)
- **INFERRED**: Derived by ARP reasoning from verified inputs
- **HEURISTIC**: Weak reasoning/pattern match/best guess
- **UNVERIFIED**: No backing
- **CONTRADICTED**: Conflicts with known facts/invariants

### How They Work

1. **Emission**: LogosAgiNexus emits truth annotations for each proposal rationale based on derivation method (e.g., VERIFIED for replayed state rules, HEURISTIC for keyword patterns).

2. **Persistence**: Truth-annotated events are persisted in SCP state (`state/scp_state.json`) with bounded history for audit and replay.

3. **Ranking**: Evaluator scoring incorporates truth bonuses/penalties:
   - PROVED: +0.20
   - VERIFIED: +0.10
   - INFERRED: +0.00
   - HEURISTIC: -0.10
   - UNVERIFIED: -0.20
   - CONTRADICTED: Forces score to -1.0

4. **Constraints**: For high-impact tools (e.g., tool proposal approval), HEURISTIC/UNVERIFIED proposals require explicit mission profile permission (`allow_heuristic_high_impact: true`). Read-only tools are unaffected.

Truth categories enhance decision quality by preferring higher-trust proposals when outcomes are otherwise similar, while maintaining strict safety boundaries.

## Belief→Tool Policy

The LOGOS agent incorporates a centralized belief-informed policy module for deterministic tool selection that operates before evaluator ranking. This replaces ad-hoc belief logic scattered across codepaths, ensuring auditable and deterministic proposal filtering/boosting without expanding tool authority.

### Policy Flow

Beliefs → Policy → Evaluator → Constraints → dispatch_tool

1. **Belief Sources**: Beliefs are persisted in SCP state (`state/scp_state.json`) with truth + confidence + status (ACTIVE/QUARANTINED) + content (preferred_tool, contradicted_tools)

2. **Policy Application**: `logos/policy.py` applies deterministic rules:
   - **Boost**: ACTIVE belief with truth ≥ VERIFIED and confidence ≥ 0.80 → +0.15 adjustment to preferred_tool
   - **Filter**: QUARANTINED belief contradicts tool → remove from proposals
   - **Override**: PROVED beliefs override VERIFIED conflicts (filter takes precedence)
   - **Exclusion**: HEURISTIC-only beliefs never boost

3. **Evaluator Integration**: `logos/evaluator.py` incorporates policy_adjustment into final score, records policy contribution in evaluator_reason

4. **Auditable Metadata**: Proposals include `policy_adjustment`, `policy_reason`, `policy_belief_id` for transparency

5. **Plan Integration**: Plan generation applies policy before selecting tools for steps, annotates plan notes with belief_id used

The policy ensures PROVED beliefs dominate VERIFIED, which dominate HEURISTIC, while maintaining separation of concerns (policy filters/boosts, evaluator scores).

## Unified Working Memory (UWM)

The LOGOS agent includes a Unified Working Memory system that provides deterministic, bounded cognitive persistence for improved proposal generation. UWM maintains short-term and long-term memory partitions with salience-based decay and promotion, enabling objective-scoped recall and truth-aware retrieval without expanding tool authority.

### Memory Partitions

- **Short-Term Memory**: Bounded to 30 items, subject to per-run decay (default 15% salience reduction)
- **Long-Term Memory**: Bounded to 200 items, slower decay (optional 2%), curated through promotion
- **Deterministic Growth**: Hard caps prevent unbounded expansion; oldest/lowest-salience items are evicted

### Salience and Decay

- **Initial Salience**: Base truth weight (PROVED=1.00, VERIFIED=0.85, etc.) + success bonus for TOOL outcomes
- **Decay**: Short-term items decay per run; long-term items may decay slowly or not at all
- **Promotion**: Items promote to long-term if salience ≥0.70 + access_count ≥2, or PROVED/VERIFIED + access_count ≥1

### Objective-Scoped Recall

- **Tagging**: Each memory item includes objective_tags derived from normalized objective classes (e.g., "status" → "STATUS")
- **Recall**: Retrieves top-K relevant items for a given objective, preferring items with matching objective_tags
- **Integration**: Recalled summaries are included in proposal rationales to provide context without authority expansion

### Truth-Aware Retrieval

- **Ranking Score**: 0.60*salience + 0.25*truth_rank + 0.15*objective_match
- **Truth Ranking**: Normalized 0..1 mapping from truth tiers (PROVED highest, CONTRADICTED lowest)
- **Exclusion**: CONTRADICTED items excluded unless explicitly requested (debug mode)
- **Ordering**: Higher-truth items preferred when salience/relevance comparable

### Persistence and Boundedness

- **Storage**: All memory resides in SCP state (`state/scp_state.json`) under `working_memory` key
- **Validation**: Schema-validated with version control and hash chaining
- **No Parallel Stores**: Everything goes through existing SCP persistence; no new files/databases

UWM enhances proposal quality by incorporating relevant historical context while remaining advisory—execution still requires `dispatch_tool()` and proof gating.

## Plan Execution

The LOGOS agent supports deterministic multi-step plan execution for complex objectives, decomposing objectives into checkpointed sequences of tool calls with runtime authorization and truth-aware validation.

### Plan Objects

- **Schema**: Plans stored in SCP state under `plans.active` and `plans.history` with schema validation
- **Structure**: Each plan contains steps with tool/args, truth annotations, evaluator scores, and execution checkpoints
- **Bounded Storage**: Max 3 active plans, max 10 historical plans; automatic cleanup prevents unbounded growth

### Execution Flow

- **Generation**: For STATUS objectives, Logos_AGI generates 2-step plans (mission.status + probe.last) using UWM recall
- **Authorization**: Each step requires runtime consent via `ask_user()` unless `--assume-yes` flag used
- **Checkpoints**: Step results recorded with truth updates, timestamps, and evaluator outcomes
- **Resumption**: Active plans persist across runs; agent resumes from last checkpoint on restart

### Truth Integration

- **Step Validation**: Each step includes truth_annotation (truth level, evidence type, evaluator score)
- **Outcome Recording**: Success/error outcomes update truth categories and feed back to UWM
- **Policy Constraints**: High-impact tools blocked for HEURISTIC/UNVERIFIED truth unless mission allows

### Persistence and Safety

- **State Storage**: Plans reside in `state/scp_state.json` alongside UWM; no new persistence layers
- **Advisory Only**: Plans are suggestions; execution requires `dispatch_tool()` with full safety gates
- **Deterministic Bounds**: Fixed step counts and bounded history prevent runaway execution

Plan execution enables complex multi-step workflows while maintaining the agent's safety-first architecture and proof-gated authorization.

## Belief Consolidation & Revision

The LOGOS agent implements deterministic belief consolidation as a curated layer derived from working memory, plan checkpoints, and truth events, enabling belief-aware proposal generation and contradiction-driven plan revision.

### Belief Derivation

- **Evidence Sources**: Tool outcomes from working memory, plan step results, and validated truth events (PROVED/VERIFIED/INFERRED only)
- **Normalization**: Stable belief IDs created by hashing objective class + claim type + normalized subject/predicate
- **Claim Types**: tool_outcome (e.g., mission.status success), plan_step_outcome (e.g., plan execution results)

### Confidence & Truth Updates

- **Initial Confidence**: Mapped from truth tiers (PROVED=1.00, VERIFIED=0.85, INFERRED=0.65, etc.)
- **Promotion Rules**: Confidence increases for repeated VERIFIED support (≥2 distinct timestamps) or PROVED evidence
- **Contradiction Handling**: VERIFIED contradictions downgrade truth to CONTRADICTED, reduce confidence, quarantine belief
- **Staleness Decay**: Beliefs without new evidence decay confidence (0.95 multiplier per run without updates)

### Quarantine & Revision

- **Quarantine**: Beliefs with CONTRADICTED truth and verified contradictions become QUARANTINED
- **Plan Revision**: Active plans skip steps contradicted by QUARANTINED beliefs with matching objectives
- **Checkpoint Annotation**: Skipped steps record contradiction reason and belief reference

### Proposal Integration

- **Belief Recall**: Proposals include summaries of high-confidence (≥0.80) ACTIVE beliefs for objective context
- **Advisory Influence**: Beliefs inform rationale but do not override evaluator scoring or safety gates
- **Deterministic Ranking**: Beliefs ranked by confidence, preferring VERIFIED/PROVED over raw memory

### Persistence & Boundedness

- **Storage**: Beliefs in `state/scp_state.json` under `beliefs` key with schema validation and hash chaining
- **Bounds**: Max 100 beliefs, sorted by confidence/recency, deterministic eviction of lowest-ranked
- **No Expansion**: Fully contained within existing SCP state; no new authority or parallel stores

Belief consolidation provides truth-aware cognitive persistence while maintaining deterministic bounds and safety-first constraints.

## Epistemic Ledger

The LOGOS agent generates a deterministic per-run ledger summarizing epistemic activity for audit and replayability. The ledger is stored at `audit/run_ledgers/run_<timestamp>_<repo_sha>.json` and contains:

- **Hashes**: Attestation, Coq index, SCP state, beliefs, and metrics state hashes for deterministic replay
- **Truth Summary**: Counts of truth tiers across proposals generated/executed and plan steps
- **Belief Usage**: Referenced belief IDs and confidence metrics from policy decisions
- **Policy Interventions**: Boosted/filtered tools with associated belief IDs
- **Execution Trace**: Ordered list of tool executions with outcomes, truth tiers, and policy metadata
- **Governance Flags**: Logos_AGI mode, state validation status, and override flags

The ledger enables complete epistemic accounting without duplicating audit logs, providing a compact summary derived from captured data.

## Contributing

See `CONTRIBUTING.md` for guidance on verifying changes, running tests, and
raising pull requests.

## Support

For questions about the LOGOS proof pipeline or integration guidance, open an
issue on the repository or contact the Project LOGOS engineering team.
