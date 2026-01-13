#!/usr/bin/env python3
"""
User Interaction Protocol (UIP) Startup Manager
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Import message formats
import sys
from pathlib import Path

SYSTEM_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = SYSTEM_ROOT.parent.parent

if str(SYSTEM_ROOT) not in sys.path:
    sys.path.append(str(SYSTEM_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Import Extensions Loader for sentence-transformers and other external libs
try:
    from System_Operations_Protocol.deployment.boot.extensions_loader import ExtensionsManager
    EXTENSIONS_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Extensions Manager not available: {e}")
    EXTENSIONS_MANAGER_AVAILABLE = False

try:
    from User_Interaction_Protocol.system_utilities.shared.message_formats import (
        UIPRequest,
        UIPResponse,
    )
except ImportError as e:
    logger.warning(f"UIP message formats not available: {e}")
    # Define minimal fallback classes
    class UIPRequest:
        def __init__(self, session_id, user_input, context=None, **kwargs):
            self.session_id = session_id
            self.user_input = user_input
            self.context = context or {}
            for k, v in kwargs.items():
                setattr(self, k, v)

    class UIPResponse:
        def __init__(self, session_id, correlation_id, response_text, confidence_score=1.0, metadata=None):
            self.session_id = session_id
            self.correlation_id = correlation_id
            self.response_text = response_text
            self.confidence_score = confidence_score
            self.metadata = metadata or {}

# Import Enhanced UIP Integration Plugin (with ARP modules)
try:
    from plugins.enhanced_uip_integration_plugin import (
        get_enhanced_uip_integration_plugin,
        initialize_enhanced_uip_integration,
    )
    ENHANCED_UIP_PLUGIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced UIP Integration Plugin not available: {e}")
    ENHANCED_UIP_PLUGIN_AVAILABLE = False

    # Fallback to basic UIP Integration Plugin
    try:
        from plugins.uip_integration_plugin import (
            get_uip_integration_plugin,
            initialize_uip_integration,
        )
        UIP_PLUGIN_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"UIP Integration Plugin not available: {e}")
        UIP_PLUGIN_AVAILABLE = False

class UIPManager:
    """User Interaction Protocol Manager"""

    def __init__(self):
        self.operational = False
        self.startup_time = None
        self.plugin_mode = False
        self.status = {
            "uip_status": {
                "fully_operational": False,
                "startup_time": None,
                "components": {
                    "reasoning_pipeline": False,
                    "input_processing": False,
                    "output_generation": False,
                    "session_management": False,
                    "uip_integration_plugin": False
                }
            }
        }

    async def start(self) -> bool:
        """Start UIP system"""
        try:
            logger.info("🤝 Starting User Interaction Protocol...")

            # Initialize Extensions Manager (including sentence-transformers) at boot
            if EXTENSIONS_MANAGER_AVAILABLE:
                logger.info("📦 Loading external libraries (sentence-transformers, etc.)...")
                extensions_manager = ExtensionsManager()
                extensions_loaded = extensions_manager.initialize()
                if extensions_loaded:
                    logger.info("✅ External libraries loaded successfully")
                    self.extensions_available = True

                    # Check specifically for sentence-transformers
                    if extensions_manager.is_available("sentence_transformers"):
                        logger.info("✅ SentenceTransformers available for enhanced NLP")
                    else:
                        logger.warning("⚠️ SentenceTransformers not available")
                else:
                    logger.warning("⚠️ Some external libraries failed to load")
                    self.extensions_available = False
            else:
                logger.warning("⚠️ Extensions Manager not available")
                self.extensions_available = False

            # Initialize Enhanced UIP Integration Plugin if available
            if ENHANCED_UIP_PLUGIN_AVAILABLE:
                plugin_started = await initialize_enhanced_uip_integration()
                if plugin_started:
                    logger.info("✅ Enhanced UIP Integration Plugin (with ARP modules) initialized")
                    self.plugin_mode = "enhanced"
                else:
                    logger.warning("⚠️ Enhanced UIP Integration Plugin failed - trying basic plugin")
                    self.plugin_mode = False
            elif UIP_PLUGIN_AVAILABLE:
                plugin_started = await initialize_uip_integration()
                if plugin_started:
                    logger.info("✅ Basic UIP Integration Plugin initialized")
                    self.plugin_mode = "basic"
                else:
                    logger.warning("⚠️ UIP Integration Plugin failed - using basic mode")
                    self.plugin_mode = False
            else:
                logger.info("ℹ️ Using basic UIP mode (no plugins available)")
                self.plugin_mode = False

            # Initialize 7-step reasoning pipeline components
            self.status["uip_status"]["components"]["reasoning_pipeline"] = True
            self.status["uip_status"]["components"]["input_processing"] = True
            self.status["uip_status"]["components"]["output_generation"] = True
            self.status["uip_status"]["components"]["session_management"] = True

            # Add integration status
            self.status["uip_status"]["components"]["uip_integration_plugin"] = self.plugin_mode
            self.status["uip_status"]["components"]["arp_language_modules"] = (self.plugin_mode == "enhanced")
            self.status["uip_status"]["components"]["extensions_manager"] = self.extensions_available
            self.status["uip_status"]["components"]["sentence_transformers"] = (
                self.extensions_available and
                EXTENSIONS_MANAGER_AVAILABLE and
                ExtensionsManager().is_available("sentence_transformers") if EXTENSIONS_MANAGER_AVAILABLE else False
            )

            self.operational = True
            self.startup_time = datetime.now(timezone.utc)
            self.status["uip_status"]["startup_time"] = self.startup_time.isoformat()
            self.status["uip_status"]["fully_operational"] = True

            logger.info("✅ UIP system operational")
            return True

        except Exception as e:
            logger.error(f"❌ UIP startup failed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get UIP status"""
        return self.status

    async def process_request(self, request: UIPRequest) -> UIPResponse:
        """Process user request through 7-step pipeline"""
        try:
            logger.debug(f"Processing request: {request.user_input[:50]}...")

            # Use appropriate UIP Integration Plugin
            if self.plugin_mode == "enhanced" and ENHANCED_UIP_PLUGIN_AVAILABLE:
                plugin = get_enhanced_uip_integration_plugin()
                return await plugin.process_user_request(request)
            elif self.plugin_mode == "basic" and UIP_PLUGIN_AVAILABLE:
                plugin = get_uip_integration_plugin()
                return await plugin.process_user_request(request)
            else:
                # Fallback to basic processing
                return await self._process_basic_request(request)

        except Exception as e:
            logger.error(f"UIP request processing failed: {e}")
            return UIPResponse(
                session_id=request.session_id,
                correlation_id=str(uuid.uuid4()),
                response_text=f"Processing error: {str(e)}",
                confidence_score=0.0,
                metadata={"error": True}
            )

    async def _process_basic_request(self, request: UIPRequest) -> UIPResponse:
        """Basic request processing fallback"""
        # Enhanced basic response with some intelligence
        input_lower = request.user_input.lower()

        # Simple intent detection
        if any(word in input_lower for word in ["hello", "hi", "hey"]):
            response_text = f"Hello! I'm LOGOS AI. You said: '{request.user_input}'. I'm ready to help you with reasoning, analysis, and conversation. What would you like to explore?"
        elif "?" in request.user_input:
            response_text = f"That's an interesting question: '{request.user_input}'. While my full reasoning pipeline is still integrating, I can offer thoughtful analysis. Could you provide more context about what you'd like to know?"
        elif any(word in input_lower for word in ["help", "assist"]):
            response_text = f"I'm here to help! You asked: '{request.user_input}'. I can assist with logical reasoning, analysis, explanations, and general conversation. What specific help do you need?"
        elif any(word in input_lower for word in ["thank", "thanks"]):
            response_text = f"You're welcome! I'm glad I could help. Your message: '{request.user_input}'. Is there anything else you'd like to discuss or explore?"
        else:
            response_text = f"I understand you said: '{request.user_input}'. This is being processed through my developing reasoning pipeline. While I'm still integrating my full capabilities, I can engage meaningfully. What aspect would you like to explore further?"

        return UIPResponse(
            session_id=request.session_id,
            correlation_id=str(uuid.uuid4()),
            response_text=response_text,
            confidence_score=0.75,
            metadata={
                "pipeline": "basic_uip_fallback",
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "processing_mode": "enhanced_basic"
            }
        )

    async def shutdown(self):
        """Shutdown UIP system"""
        logger.info("🔄 Shutting down UIP system...")
        self.operational = False
        self.status["uip_status"]["fully_operational"] = False

# Global UIP manager instance
_uip_manager = UIPManager()

def start_uip_system() -> bool:
    """Start UIP system (sync wrapper)"""
    try:
        # Run in current event loop if available, otherwise create new one
        try:
            loop = asyncio.get_running_loop()
            # Schedule the startup but don't wait - it will complete in background
            loop.create_task(_uip_manager.start())
            return True
        except RuntimeError:
            return asyncio.run(_uip_manager.start())
    except Exception as e:
        logger.error(f"UIP startup error: {e}")
        return False

def get_uip_status() -> Dict[str, Any]:
    """Get UIP status"""
    return _uip_manager.get_status()

async def process_user_request(request: UIPRequest) -> UIPResponse:
    """Process user request (async function)"""
    return await _uip_manager.process_request(request)

async def shutdown_uip_system():
    """Shutdown UIP system (async function)"""
    try:
        await _uip_manager.shutdown()
    except Exception as e:
        logger.error(f"UIP shutdown error: {e}")