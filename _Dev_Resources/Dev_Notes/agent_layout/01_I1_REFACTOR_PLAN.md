# I1 Agent Refactor Plan

## Current Tree Snapshot (depth ≤ 3)
```
System_Stack/Logos_Agents_Protocol/Logos_Agents/I1_Agent
├── _core
├── config
├── connections
├── diagnostics
└── protocol_operations
    ├── scp_analysis
    ├── scp_bdn_adapter
    ├── scp_cycle
    │   ├── scp_cycle
    │   ├── scp_integrations
    │   ├── scp_mvs_adapter
    │   ├── scp_runtime
    │   └── scp_tests
    ├── scp_integrations
    │   ├── scp_pipeline
    │   └── scp_predict
    ├── scp_predict
    └── scp_runtime
```

## Target Tree (canonical template)
```
I1_Agent/
├── _core/
├── config/
├── connections/
├── diagnostics/
└── protocol_operations/
    └── scp/
        ├── adapters/
        ├── analysis/
        ├── runtime/
        ├── transforms/
        ├── cycle/
        ├── integrations/
        └── tests/
```

## Move Table
| From | To |
| --- | --- |
| protocol_operations/scp_bdn_adapter/bdn_adapter.py | protocol_operations/scp/adapters/bdn_adapter.py |
| protocol_operations/scp_bdn_adapter/bdn_types.py | protocol_operations/scp/adapters/bdn_types.py |
| protocol_operations/scp_cycle/scp_mvs_adapter/mvs_adapter.py | protocol_operations/scp/adapters/mvs_adapter.py |
| protocol_operations/scp_cycle/scp_mvs_adapter/mvs_types.py | protocol_operations/scp/adapters/mvs_types.py |
| protocol_operations/scp_predict/risk_estimator.py | protocol_operations/scp/analysis/risk_estimator.py |
| protocol_operations/scp_predict/trajectory_types.py | protocol_operations/scp/analysis/trajectory_types.py |
| protocol_operations/scp_integrations/scp_predict/risk_estimator.py | (delete duplicate) |
| protocol_operations/scp_integrations/scp_predict/trajectory_types.py | (delete duplicate) |
| protocol_operations/scp_runtime/result_packet.py | protocol_operations/scp/runtime/result_packet.py |
| protocol_operations/scp_runtime/smp_intake.py | protocol_operations/scp/runtime/smp_intake.py |
| protocol_operations/scp_runtime/work_order.py | protocol_operations/scp/runtime/work_order.py |
| protocol_operations/scp_runtime/README.md | protocol_operations/scp/runtime/README.md |
| protocol_operations/scp_cycle/scp_runtime/result_packet.py | (delete duplicate) |
| protocol_operations/scp_cycle/scp_runtime/smp_intake.py | (delete duplicate) |
| protocol_operations/scp_cycle/scp_runtime/work_order.py | (delete duplicate) |
| protocol_operations/scp_cycle/scp_transform/iterative_loop.py | protocol_operations/scp/transforms/iterative_loop.py |
| protocol_operations/scp_cycle/scp_transform/transform_registry.py | protocol_operations/scp/transforms/transform_registry.py |
| protocol_operations/scp_cycle/scp_transform/transform_types.py | protocol_operations/scp/transforms/transform_types.py |
| protocol_operations/scp_integrations/iterative_loop.py | (delete duplicate) |
| protocol_operations/scp_integrations/transform_registry.py | (delete duplicate) |
| protocol_operations/scp_integrations/transform_types.py | (delete duplicate) |
| protocol_operations/scp_cycle/cycle_runner.py | protocol_operations/scp/cycle/cycle_runner.py |
| protocol_operations/scp_cycle/policy.py | protocol_operations/scp/cycle/policy.py |
| protocol_operations/scp_cycle/scp_cycle/policy.py | (delete duplicate) |
| protocol_operations/scp_integrations/analysis_runner.py | protocol_operations/scp/integrations/analysis_runner.py |
| protocol_operations/scp_integrations/predict_integration.py | protocol_operations/scp/integrations/predict_integration.py |
| protocol_operations/scp_integrations/scp_pipeline/pipeline_runner.py | protocol_operations/scp/integrations/scp_pipeline/pipeline_runner.py |
| protocol_operations/scp_cycle/scp_integrations/predict_integration.py | (delete duplicate) |
| protocol_operations/scp_cycle/scp_tests/run_pipeline_smoke.py | protocol_operations/scp/tests/run_pipeline_smoke.py |
| protocol_operations/scp_cycle/scp_tests/sample_smp.py | protocol_operations/scp/tests/sample_smp.py |
