# LOGOS Agentic Stack -- Demo Script Addendum

## Summary

This repository demonstrates a reflective, safety-gated cognitive architecture designed to support agentic emergence through controlled tool invocation, structured self-evaluation, and deterministic planning loops.

While not yet autonomously recursive, the system fully implements:

- Human-in-the-loop goal execution
- Reflective evaluation (`novelty_score`, `plan_quality`)
- Deterministic stub interfaces for integration hooks
- A scriptable supervision cycle via `run_cycle.sh`

This forms the core of an agent cognition framework, ready to scale toward autonomous improvement in future phases.

## Demo Goals (For Review or Replay)

1. Initialize the agent in experimental mode

   ```bash
   ./.venv/bin/python scripts/system_mode_initializer.py --mode experimental
   ```

2. Run a clean system probe

   ```bash
   timeout 120s ./.venv/bin/python scripts/protocol_probe.py --log-only --deep --agentic-probe
   ```

3. Execute a supervised reflective loop (produces recap and measurable plan from the last artifact)

   ```bash
   tools/run_cycle.sh experimental
   ```

   Output artifacts appear in `sandbox/`:

   - `plan.md` -- structured execution goal
   - `_latest_reflection.json` -- reflection metrics
   - `artifact_*.txt|py` -- generated outputs
   - `agent_state.json` -- run history with metrics

## Safety and Cognition Architecture

| Layer | Description |
| --- | --- |
| Consent-Gated Execution | All tool calls are supervised; no autonomous recursion permitted unless explicitly toggled. |
| Metric-Driven Reflection | Each agent cycle evaluates `novelty_score` and `plan_quality_0_3` from its own output. |
| Controlled Write Access | All generated files are sandboxed (`sandbox/`) with write caps and runtime limits. |
| Stubbed Hooks | `LOGOS_V2`, `TFAT`, `iel_registry`, and others are stubbed to suppress probe errors and support future activation. |
| Auditability | All cycles persist state and reflection data in JSON for introspection and log replay. |

## Future Direction (Autonomy Path)

The system is scaffolded for reflexive agent behavior. To activate full agentic emergence, the following will be toggled in future phases:

- Autonomous plan extension via self-queuing goals
- Auto-consent mode for toolchain whitelists
- Active hook integration, including:
  - `LOGOS_V2.intelligence`
  - `TFAT.security_reference_monitor`
  - `System_Operations_Protocol.code_generator.coherence`
- Rolling reflection windows to compare multi-cycle behavior and goal drift

## Legacy Notes

The terms `tetranose`, `telose`, `thonoc`, `arcon`, and `kyrix` are deprecated. These were bespoke naming conventions from earlier prototypes and no longer apply to agent logic, hooks, or outputs. They should be treated as legacy tokens and excluded from reflection relevance.
