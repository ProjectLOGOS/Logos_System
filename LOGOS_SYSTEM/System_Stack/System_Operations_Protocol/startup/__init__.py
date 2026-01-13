"""
Protocol startup managers initialization
"""

from .sop_startup import (
    start_sop_system,
    get_sop_status,
    shutdown_sop_system,
)

from .uip_startup import (
    start_uip_system,
    get_uip_status,
    process_user_request,
    shutdown_uip_system,
)

from .scp_startup import (
    start_scp_system,
    get_scp_status,
    execute_scp_reasoning,
    shutdown_scp_system,
)

__all__ = [
    'start_sop_system',
    'get_sop_status',
    'shutdown_sop_system',
    'start_uip_system',
    'get_uip_status',
    'process_user_request',
    'shutdown_uip_system',
    'start_scp_system',
    'get_scp_status',
    'execute_scp_reasoning',
    'shutdown_scp_system',
]