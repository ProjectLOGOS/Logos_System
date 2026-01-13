System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_integrations/pipeline_runner.py:5:from ..scp_runtime.smp_intake import load_smp
System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_integrations/pipeline_runner.py:6:from ..scp_runtime.work_order import build_work_order
System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_integrations/pipeline_runner.py:7:from ..scp_runtime.result_packet import emit_result_packet
System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_integrations/pipeline_runner.py:10:from .scp_predict.risk_estimator import estimate_trajectory
System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_integrations/pipeline_runner.py:15:def run_scp_pipeline(*, smp: Dict[str, Any], payload_ref: Any = None) -> Any:
System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_cycle/cycle_runner.py:5:from .scp_runtime.smp_intake import load_smp
System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_cycle/cycle_runner.py:6:from .scp_runtime.work_order import build_work_order
System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_cycle/cycle_runner.py:7:from .scp_runtime.result_packet import emit_result_packet
System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_runtime/result_packet.py:7:from ..I1_core.hashing import safe_hash
System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_runtime/smp_intake.py:6:from ..I1_core.hashing import safe_hash
System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_runtime/smp_intake.py:7:from ..I1_core.schema_utils import require_dict, get_dict, get_list, get_str
System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_runtime/smp_intake.py:8:from ..I1_core.errors import SchemaError
System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_tests/run_pipeline_smoke.py:8:from ..scp_pipeline.pipeline_runner import run_scp_pipeline
System_Stack/Logos_Protocol/Logos_Agents/I1_Agent/protocol_operations/scp_tests/run_pipeline_smoke.py:13:    result = run_scp_pipeline(smp=smp, payload_ref={"opaque": True, "input_hash": smp["input_reference"]["input_hash"]})
