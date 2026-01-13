# Runtime Entry Points

| Root | Entry | Invocation | Role summary |
| --- | --- | --- | --- |
| START_LOGOS | [START_LOGOS.py](START_LOGOS.py#L144-L210) | `python START_LOGOS.py [--skip-audit-rewrite] [--no-ui] [--no-uip] [--host HOST] [--port PORT]` | Parses bootstrap flags, extends `sys.path`, runs the PXL proof gate, starts the dashboard + Flask server, and spawns the UIP manager before blocking on signal wait. |
| System_Stack | [System_Operations_Protocol/deployment/configuration/entry.py](System_Stack/System_Operations_Protocol/deployment/configuration/entry.py#L641-L679) | `python entry.py [--status|--test-modal φ|--test-iel ψ|--shutdown|--emergency-halt REASON]` | Initializes the LOGOS core, integrity safeguards, IEL domains, and exposes CLI toggles for diagnostics and emergency halts. |
| System_Stack | [System_Operations_Protocol/deployment/configuration/LOGOS.py](System_Stack/System_Operations_Protocol/deployment/configuration/LOGOS.py#L1-L215) | `python LOGOS.py` | Multi-phase launcher that imports the core entry module, runs health/coherence/tool-readiness checks, then spawns the GUI interface. |
| PXL_Gate | [ui/run_coq_pipeline.py](PXL_Gate/ui/run_coq_pipeline.py#L1-L91) | `python run_coq_pipeline.py` | Compiles baseline + meta Coq packets and enforces the single LEM `Admitted.` gate; raises on any proof drift. |
| PXL_Gate | [ui/serve_pxl.py](PXL_Gate/ui/serve_pxl.py#L1-L220) | `python serve_pxl.py` | Starts the Flask proof server, boots SerAPI (`sertop`), hashes compiled kernels, and exposes `/health`, `/prove`, `/countermodel` endpoints. |

## Additional CLI / test harnesses
- Alignment tests and adapters across [System_Stack/Logos_Protocol/Runtime_Operations/tools/implementations/tool_proposal_pipeline.py](System_Stack/Logos_Protocol/Runtime_Operations/tools/implementations/tool_proposal_pipeline.py#L357-L372) and siblings expose `__main__` smoke-test hooks for the runtime toolchain.
- GUI + dashboard layers such as [System_Stack/Logos_Protocol/GUI/logos_dashboard/app.py](System_Stack/Logos_Protocol/GUI/logos_dashboard/app.py#L189-L210) run standalone monitoring experiences.
- Numerous protocol-specific sandboxes (e.g., [System_Stack/User_Interaction_Protocol/system_utilities/lem_portal_cli.py](System_Stack/User_Interaction_Protocol/system_utilities/lem_portal_cli.py#L1-L60), [System_Stack/Advanced_Reasoning_Protocol/arp_operations.py](System_Stack/Advanced_Reasoning_Protocol/arp_operations.py#L1100-L1135)) surfaced in the `rg "__main__"` sweep remain available but are secondary to the primary runtime contract.
