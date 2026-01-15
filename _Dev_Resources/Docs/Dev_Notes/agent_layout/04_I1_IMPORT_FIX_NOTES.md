# I1 Import Fix Notes

| File | Before | After |
| --- | --- | --- |
| connections/id_handler.py | `from .config.hashing import safe_hash` | `from ..config.hashing import safe_hash` |
| connections/router.py | `from .config.constants import PRIORITY_HIGH, PRIORITY_NORMAL` | `from ..config.constants import PRIORITY_HIGH, PRIORITY_NORMAL` |
| protocol_operations/scp/integrations/predict_integration.py | `from ..scp_predict.risk_estimator import estimate_trajectory` | `from ..analysis.risk_estimator import estimate_trajectory` |
| protocol_operations/scp/integrations/scp_pipeline/pipeline_runner.py | `from ..scp_runtime.*`, `from ..scp_analysis.*`, `from ..scp_predict.*`, `from ..scp_transform.*`, `from ..scp_cycle.*` | Updated to `..runtime.*`, `..integrations.analysis_runner`, `..analysis.risk_estimator`, `..transforms.iterative_loop`, `..cycle.policy` |
| protocol_operations/scp/cycle/cycle_runner.py | `from ..scp_runtime.*`, `from ..scp_transform.iterative_loop` | `from ..runtime.*`, `from ..transforms.iterative_loop` |
| protocol_operations/scp/integrations/analysis_runner.py | `from ..scp_mvs_adapter.*`, `from ..scp_bdn_adapter.*` | `from ..adapters.mvs_*`, `from ..adapters.bdn_*` |
| protocol_operations/scp/tests/run_pipeline_smoke.py | `from ..scp_pipeline.pipeline_runner import run_scp_pipeline` | `from ..integrations.scp_pipeline.pipeline_runner import run_scp_pipeline` |
| protocol_operations/scp/runtime/result_packet.py | `from ..I1_core.hashing import safe_hash` | `from ...._core.hashing import safe_hash` |
| protocol_operations/scp/runtime/smp_intake.py | `from ..I1_core.(hashing|schema_utils|errors)` | `from ...._core.(hashing|schema_utils|errors)` |
