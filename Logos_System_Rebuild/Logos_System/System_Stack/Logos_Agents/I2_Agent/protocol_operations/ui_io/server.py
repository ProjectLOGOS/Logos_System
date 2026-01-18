# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from I2_Agent.protocol_operations.ui_io.adapter import handle_inbound

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InboundPacket(BaseModel):
    payload: str

@app.post("/ingest")
async def ingest(packet: InboundPacket):
    response = handle_inbound(inbound=packet.payload, default_route="LOGOS")
    return {
        "route": response.route,
        "priority": response.priority,
        "reason": response.reason,
        "payload": response.payload,
    }
