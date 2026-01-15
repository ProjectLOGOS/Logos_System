# Compute Substrate Runbook

## Purpose
- Stand up and manage the elastic node pool that powers proof-gated Coq rebuilds, alignment agent boots, and future SOP stress cycles.
- Preserve guarded alignment guarantees (agent hash, audit append-only logs) across remote infrastructure.
- Provide reproducible procedures for provisioning, scaling, monitoring, and teardown.

## 1. Baseline Image Preparation
1. Start from Ubuntu 24.04 LTS (HWE kernel recommended for newest hardware).
2. Install system dependencies:
   ```bash
   sudo apt update
   sudo apt install -y build-essential opam m4 unzip rsync python3-venv python3-pip
   ```
3. Install Coq 8.18: `opam switch create logos-coq 4.14.2 && opam install coq.8.18.1`.
4. Create `/opt/logos` and pre-clone the repository (read-only deploy key).
5. Bake image artifacts (AMI, custom GCP image, Azure managed image) tagged `logos-proof-gated`.

## 2. Node Provisioning Workflow
1. Launch VM using the baseline image with minimum specs 8 vCPU / 16 GB RAM / 50 GB SSD.
2. Attach a dedicated data disk (100 GB) mounted at `/var/logos-cache` for build artifacts and sandbox snapshots.
3. Configure systemd units:
   - `logos-agent.service`: runs `python3 scripts/boot_aligned_agent.py` on demand.
   - `logos-rebuild.service`: wraps `python3 test_lem_discharge.py` for scheduled integrity checks.
4. Register node metadata with the SOP scheduler via `scripts/system_mode_initializer.py --mode demo`.
5. Harden access: disable password SSH, require bastion or VPN, enforce host-based firewall allowing SOP control plane only.

## 3. Artifact Cache Strategy
- **Location**: `/var/logos-cache/coq` shared via NFS or object storage (rclone mount) to hold `_CoqProject` build outputs.
- **Sync policy**:
  1. After each successful `test_lem_discharge.py`, rsync `_build` and `state/` deltas to the cache target.
  2. On job start, prefetch latest cache snapshot to reduce rebuild time.
- **Integrity**: store SHA-256 manifest alongside cache; fail fast if mismatch.

## 4. Sandbox Replay Support
1. Persist `sandbox/` directory snapshots in `/var/logos-cache/sandbox/<timestamp>` after each aligned agent run.
2. Generate replay metadata JSON including mission profile, commit hash, alignment audit entry ID.
3. Provide `scripts/replay_sandbox.py` (future) to restore snapshots onto staging nodes for debugging.
4. Retain last 20 snapshots; archive older ones to cold storage (S3 Glacier / GCS Coldline).

## 5. Telemetry & Monitoring
- **Metrics collection**:
  - Install Node Exporter or lightweight Telegraf agent to emit CPU, memory, disk, and network metrics.
  - Extend `scripts/stress_sop_runtime.py` to log `peak_cpu_slots`, `peak_memory_mb`, `alignment_audit_id` once hardware access returns.
- **Log shipping**:
  - Ship `/var/logos-cache/logs/*.log` and `state/*.json` to centralized storage every 5 minutes using `systemd` timer + `rclone`.
  - Preserve append-only semantics by writing new audit entries locally before sync.
- **Alerting**: trigger alerts when Coq rebuild exceeds 15 minutes, CPU saturation >90% for 30m, or audit sync fails twice.

## 6. Operational Playbooks
| Scenario | Action |
| --- | --- |
| Node joins pool | Run provisioning workflow, execute smoke test `python3 test_lem_discharge.py`, confirm audit entry appended. |
| Node unhealthy | Drain from SOP scheduler, capture logs, snapshot `/var/logos-cache`, destroy VM, respawn from image. |
| Scaling event | Use IaC tool (Terraform) to adjust desired node count; ensure cache mounts attach to new nodes. |
| Security rotation | Rotate deploy key, refresh image, redeploy nodes; verify EXPECTED_AGENT_HASH in `boot_aligned_agent.py` matches. |

## 7. Teardown Checklist
1. Drain node from SOP scheduler.
2. Sync final cache and audit logs to long-term storage.
3. Destroy VM and detach persistent disks (retain caches if needed).
4. Revoke SSH credentials and access policies.

## 8. Future Enhancements
- Automate provisioning via Terraform module stored under `infrastructure/`.
- Implement `scripts/stress_sop_runtime.py` integration tests once compute quota restored.
- Add automated cache warmup and eviction policies per mission profile.
- Explore GPU-enabled nodes if proof search experiments require ML acceleration, keeping guardrails intact.
