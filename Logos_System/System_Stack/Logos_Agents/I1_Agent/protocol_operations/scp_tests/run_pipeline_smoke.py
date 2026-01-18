# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

from __future__ import annotations

import json
import sys
from typing import Any, Dict

from .sample_smp import make_sample_smp
from I1_Agent.protocol_operations.scp_integrations.pipeline_runner import run_scp_pipeline


def main() -> int:
    smp = make_sample_smp()
    result = run_scp_pipeline(smp=smp, payload_ref={"opaque": True, "input_hash": smp["input_reference"]["input_hash"]})

    out = result.to_dict() if hasattr(result, "to_dict") else result
    print(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
