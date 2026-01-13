# Compute Substrate Expansion Plan

## Objectives
- Provide sufficient compute capacity for repeated Coq kernel rebuilds, alignment-gated agent boots, and future stress harness loads.
- Preserve proof-gated guarantees enforced by scripts/boot_aligned_agent.py (guarded SHA-256, audit logging) across any remote infrastructure.
- Deliver a reproducible provisioning and monitoring runbook that scales beyond the current single host.

## Workload Profile
- **Kernel rebuild loop**: `python3 test_lem_discharge.py` runs full `coq_makefile` clean + rebuild (~20 OCaml compilation units, peaks on CPU).
- **Aligned agent boot**: `python3 scripts/boot_aligned_agent.py` repeats the rebuild, validates `pxl_excluded_middle`, and writes alignment audit entries.
- **SOP stress harness (future)**: `python3 scripts/stress_sop_runtime.py` currently stubbed; when hardware returns it may schedule concurrent meta-cycles via the SOP scheduler.
- **CI-style orchestrations**: `.github/workflows/*` assume serial execution; scaling requires deterministic concurrency limits.

Estimated resources per active run (baseline measurements):
- CPU: 4 cores sustained, 8 cores peak during Coq compilation.
- RAM: 4 GB typical, 8 GB recommended headroom.
- Disk: 2 GB working set for `_CoqProject`, logs, and build artifacts.

## Capacity Targets
1. **Pilot**: Single 8 vCPU / 16 GB RAM node dedicated to LOGOS workloads.
2. **Scaling**: Pool of 3 nodes with identical specs to allow parallel CI, stress tests, and on-demand demo rebuilds.
3. **Burst allowance**: Optional autoscale to 6 nodes when stress harness or investor demos require additional headroom.

## Deployment Options

### Option A: Self-Hosted Bare Metal
- Provision physical servers on-premises with Ubuntu 24.04.
- Pros: Maximum control, no cloud egress, reuse existing security perimeter.
- Cons: Higher upfront cost, manual failover, slower scaling.
- Integration steps:
  1. Install Coq, make, Python 3.12 per README prerequisites.
  2. Mirror repository via Git, enforce signed commits.
  3. Configure `scripts/boot_aligned_agent.py` as a systemd service triggered by queue submissions.

### Option B: Cloud VM Pool (AWS/GCP/Azure)
- Launch medium compute instances (e.g., c7i.xlarge, n2-standard-8, or Standard_D8s_v5).
- Pros: Elastic scaling, managed networking, snapshot-based recovery.
- Cons: Requires cloud security posture, potential data residency concerns.
- Integration steps:
  1. Bake a VM image with Coq 8.18, Python 3.12, GNU Make, repo checkout.
  2. Tag instances with `logos-proof-gated` and restrict ingress to VPN or bastion hosts.
  3. Use instance metadata to register hosts with the SOP scheduler via `scripts/system_mode_initializer.py`.
  4. Route all agent boots through `scripts/start_agent.py` using `restrict_writes_to` for attached storage.

### Option C: Managed Batch / CI Service
- Employ services like GitHub Actions self-hosted runners, AWS Batch, or GCP Cloud Build.
- Pros: Minimal infrastructure management, simplified job queueing.
- Cons: Less control over persistent state; must ensure audit logs remain append-only.
- Integration steps:
  1. Define job templates that mount a persistent volume for `state/` to keep alignment logs.
  2. Wrap `test_lem_discharge.py` and `boot_aligned_agent.py` as CI steps with explicit artifact uploads.
  3. Enforce the guarded agent hash precondition by verifying the SHA-256 inside each job before execution.

## Recommended Path
- Adopt **Option B** with three medium-sized VMs to balance elasticity and control.
- Maintain one standby VM for failover; use infrastructure-as-code (Terraform or similar) stored under `infrastructure/` (future work).
- Centralize logs via secure object storage; sync `state/alignment_LOGOS-AGENT-OMEGA.json` after each run.

## Implementation Phases
1. **Design & Approval (Week 1)**
   - Finalize instance types, network boundaries, and storage encryption requirements.
   - Document SLA expectations for rebuild turnaround (<15 minutes) and availability (99%).
2. **Provisioning (Week 2)**
   - Deploy pilot VM, configure OS baseline, install Coq/Python toolchain.
   - Clone repository and validate `python3 test_lem_discharge.py` and `python3 scripts/boot_aligned_agent.py` end-to-end.
3. **Scheduler Integration (Week 3)**
   - Register compute nodes with SOP scheduler; confirm CPU slot reporting is accurate.
   - Update `scripts/system_mode_initializer.py` profiles to include remote node endpoints.
4. **Monitoring & Guardrails (Week 4)**
   - Emit metrics for CPU, memory, disk, and job status; aggregate via existing logging.
   - Verify SHA-256 guard and audit logging remain intact after remote execution.
5. **Scaling & Documentation (Week 5)**
   - Expand to full node pool, enable autoscale triggers for stress harness loads.
   - Publish runbook covering provisioning, failover, and teardown.

## Guardrail Considerations
- Ensure all remote runs append to `state/alignment_LOGOS-AGENT-OMEGA.json` without truncation.
- Mirror the guarded agent hash in `scripts/boot_aligned_agent.py`; abort any run where the binary hash diverges.
- Maintain Coq build determinism by pinning `_CoqProject` dependencies; avoid installing non-standard packages.
- Require SSH bastion or zero-trust tunnel for operator access; disable direct internet exposure on compute nodes.

## Next Actions
- Approve the recommended Option B approach and instance sizing.
- Stand up the pilot VM and validate the proof-gated workflow remotely.
- Follow the provisioning steps captured in docs/compute_substrate_runbook.md and iterate as the pool scales.
