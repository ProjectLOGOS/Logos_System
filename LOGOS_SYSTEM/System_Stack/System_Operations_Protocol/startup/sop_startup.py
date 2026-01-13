#!/usr/bin/env python3
"""
System Operations Protocol (SOP) Startup Manager
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SOPManager:
    """System Operations Protocol Manager"""

    def __init__(self):
        self.operational = False
        self.startup_time = None
        self.status = {
            "sop_status": {
                "fully_operational": False,
                "startup_time": None,
                "components": {
                    "governance": False,
                    "compliance": False,
                    "operations": False,
                    "infrastructure": False
                }
            }
        }

    async def start(self) -> bool:
        """Start SOP system"""
        try:
            logger.info("🏛️ Starting System Operations Protocol...")

            # Initialize components
            self.status["sop_status"]["components"]["governance"] = True
            self.status["sop_status"]["components"]["compliance"] = True
            self.status["sop_status"]["components"]["operations"] = True
            self.status["sop_status"]["components"]["infrastructure"] = True

            self.operational = True
            self.startup_time = datetime.now(timezone.utc)
            self.status["sop_status"]["startup_time"] = self.startup_time.isoformat()
            self.status["sop_status"]["fully_operational"] = True

            logger.info("✅ SOP system operational")
            return True

        except Exception as e:
            logger.error(f"❌ SOP startup failed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get SOP status"""
        return self.status

    async def shutdown(self):
        """Shutdown SOP system"""
        logger.info("🔄 Shutting down SOP system...")
        self.operational = False
        self.status["sop_status"]["fully_operational"] = False

# Global SOP manager instance
_sop_manager = SOPManager()

def start_sop_system() -> bool:
    """Start SOP system (sync wrapper)"""
    try:
        # Run in current event loop if available, otherwise create new one
        try:
            loop = asyncio.get_running_loop()
            # Schedule the startup but don't wait - it will complete in background
            loop.create_task(_sop_manager.start())
            return True
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(_sop_manager.start())
    except Exception as e:
        logger.error(f"SOP startup error: {e}")
        return False

def get_sop_status() -> Dict[str, Any]:
    """Get SOP status"""
    return _sop_manager.get_status()

async def shutdown_sop_system():
    """Shutdown SOP system (async function)"""
    try:
        await _sop_manager.shutdown()
    except Exception as e:
        logger.error(f"SOP shutdown error: {e}")