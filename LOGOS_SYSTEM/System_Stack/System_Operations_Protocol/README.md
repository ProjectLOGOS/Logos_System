# System Operations Protocol (SOP)

The System Operations Protocol governs infrastructure, governance, and lifecycle coordination for the LOGOS stack. SOP remains continuously active, issuing tokens, detecting capability gaps, and routing maintenance workflows across ARP, SCP, and UIP.

## Core Responsibilities
- Maintain the SOP nexus as the central hub for inter-protocol coordination
- Issue, validate, and audit operation, TODO, integration, and test tokens
- Detect documentation, code, and infrastructure gaps and convert them into actionable tasks
- Manage startup sequences for SCP, UIP, and SOP components via sandbox-aware loaders
- Provide self-improvement hooks, code generation scaffolds, and backup tooling

## Key Entry Points
- `nexus/sop_nexus.py`: Nexus hub implementing token lifecycle management, gap analysis, and cross-protocol routing
- `nexus/sop_operations.py`: CLI automation mirroring `PROTOCOL_OPERATIONS.txt` for daily operations
- `startup/`: Async bootstrappers (`sop_startup.py`, `scp_startup.py`, `uip_startup.py`) wired into LOGOS agent launch flows
- `governance/`, `compliance/`, `operations/`, `infrastructure/`: Specialized modules covering policy enforcement, data handling, system maintenance, and runtime observability
- `deployment/` and `file_management/`: Tooling for sandbox writes, backups, and staged updates

## Operational Workflow
1. Startup manager brings SOP online, sets component readiness, and exposes status via `get_sop_status`
2. SOP nexus stays resident, authenticating system-agent requests and issuing tokens for ARP/SCP/UIP tasks
3. Gap detectors and TODO generators log deficiencies (including missing documentation) and coordinate remediation tickets
4. Optional self-improvement engines under `operations/code_generator/` enable guarded code generation in sandboxes

## Logging & Persistence
- Token registries and gap reports persist under `state/` and `logs/` (see nexus configuration)
- Startup scripts log activation timelines for governance audits
- Backups and payload snapshots land in `file_management/` per protocol guardrails

## Local Usage
```bash
cd external/Logos_AGI/System_Operations_Protocol
python3 nexus/sop_operations.py --help
python3 startup/sop_startup.py  # Uses asyncio; wrap in event loop for scripted runs
```
SOP should be active before invoking ARP, SCP, or UIP workflows to guarantee token issuance and compliance checks.
