"""
System C: Orchestration Layer
LOGOS Passive Runtime - A/B Coordination & Enhancement Framework

This orchestrator coordinates System A (Tetrahedral Evaluator) and 
System B (Constraint Pipeline) with pluggable enhancements that can be
dynamically enabled/disabled at runtime.

ARCHITECTURE:
- Core orchestrator: ~300 lines (lightweight coordination)
- Enhancements: Pluggable modules loaded on demand
- Zero overhead when enhancements disabled
- Pay-as-you-go complexity

USAGE:
    # Minimal mode
    orchestrator = SystemOrchestrator()
    
    # With enhancements
    orchestrator = SystemOrchestrator()
    orchestrator.enable_enhancement('adaptive_control')
    orchestrator.enable_enhancement('quality_classifier')
    
    # Full intelligence mode
    orchestrator = SystemOrchestrator(enhancements=['all'])
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# Import System A and B (assuming they're available)
# from logos.evaluation.system_a import TetrahedralEvaluator, RefinedSMP, SMP
# from logos.evaluation.system_b import ConstraintPipeline, ConstraintResult, TLMToken

# For prototype, use minimal type definitions
from dataclasses import dataclass as minimal_dataclass

# ============================================================================
# ENUMS & TYPES
# ============================================================================

class SMPQuality(Enum):
    """Quality classification for refined SMPs"""
    EXCELLENT = "excellent"
    HIGH = "high"
    GOOD = "good"
    MARGINAL = "marginal"
    POOR = "poor"
    UNSTABLE = "unstable"


class FailureMode(Enum):
    """Failure classification"""
    LOGIC_FAILURE = "logic_failure"
    DOMAIN_FAILURE = "domain_failure"
    MATH_FAILURE = "math_failure"
    RESISTANCE_FAILURE = "resistance_failure"
    CONVERGENCE_FAILURE = "convergence_failure"
    MESH_FAILURE = "mesh_failure"
    CONSTRAINT_FAILURE = "constraint_failure"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Recovery actions for failures"""
    RETRY_SIMPLIFIED = "retry_simplified"
    RETRY_CONSERVATIVE = "retry_conservative"
    DOWNGRADE_TIER = "downgrade_tier"
    QUEUE_MANUAL_REVIEW = "queue_manual_review"
    REJECT_PERMANENTLY = "reject_permanently"
    PARTIAL_ACCEPT = "partial_accept"


# ============================================================================
# ORCHESTRATION RESULT TYPES
# ============================================================================

@dataclass
class CommitResult:
    """Final result from orchestrated evaluation"""
    smp_id: str
    committed: bool
    uwm_ref: Optional[str] = None
    tlm_token: Optional[str] = None
    quality: Optional[SMPQuality] = None
    
    # Audit trail
    system_a_iterations: int = 0
    system_a_stability: float = 0.0
    system_b_constraints_passed: Dict[str, bool] = field(default_factory=dict)
    
    # Timing
    total_time_ms: float = 0.0
    system_a_time_ms: float = 0.0
    system_b_time_ms: float = 0.0
    orchestration_overhead_ms: float = 0.0
    
    # Failure info (if not committed)
    failure_mode: Optional[FailureMode] = None
    failure_reason: Optional[str] = None
    recovery_action: Optional[RecoveryAction] = None
    
    # Enhancement data
    enhancement_metadata: Dict[str, Any] = field(default_factory=dict)
    
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class OrchestrationMetrics:
    """Performance metrics for orchestrator"""
    total_smps_processed: int = 0
    total_committed: int = 0
    total_rejected: int = 0
    
    # Success rates
    system_a_success_rate: float = 0.0
    system_b_success_rate: float = 0.0
    end_to_end_success_rate: float = 0.0
    
    # Timing
    avg_total_time_ms: float = 0.0
    avg_system_a_time_ms: float = 0.0
    avg_system_b_time_ms: float = 0.0
    
    # Quality distribution
    quality_distribution: Dict[SMPQuality, int] = field(default_factory=dict)
    
    # Failure modes
    failure_mode_counts: Dict[FailureMode, int] = field(default_factory=dict)
    
    # Enhancement usage
    enhancements_enabled: List[str] = field(default_factory=list)


# ============================================================================
# ENHANCEMENT BASE CLASS
# ============================================================================

class Enhancement(ABC):
    """
    Base class for pluggable enhancements.
    
    Enhancements can hook into various stages of the orchestration pipeline:
    - pre_process: Before System A
    - post_system_a: After System A, before System B
    - post_system_b: After System B
    - post_process: After everything
    - on_failure: When failures occur
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"enhancement.{self.name}")
        self.enabled = True
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Enhancement name"""
        pass
    
    async def pre_process(self, smp: Any) -> Any:
        """Hook before System A evaluation"""
        return smp
    
    async def post_system_a(self, smp: Any, refined: Any) -> Tuple[Any, Any]:
        """Hook after System A, before System B"""
        return smp, refined
    
    async def post_system_b(self, refined: Any, result: Any) -> Tuple[Any, Any]:
        """Hook after System B"""
        return refined, result
    
    async def post_process(self, commit_result: CommitResult) -> CommitResult:
        """Hook after everything"""
        return commit_result
    
    async def on_failure(
        self, 
        smp: Any, 
        failure_mode: FailureMode, 
        error: Exception
    ) -> Optional[RecoveryAction]:
        """Hook when failures occur"""
        return None


# ============================================================================
# SYSTEM ORCHESTRATOR - CORE
# ============================================================================

class SystemOrchestrator:
    """
    Core orchestrator coordinating System A and System B.
    
    Responsibilities:
    - Route SMPs through A → B pipeline
    - Handle failures and retries
    - Aggregate audit trails
    - Manage enhancement lifecycle
    - Track performance metrics
    """
    
    def __init__(
        self, 
        system_a = None,
        system_b = None,
        config: Optional[Dict[str, Any]] = None,
        enhancements: Optional[List[str]] = None
    ):
        self.config = config or {}
        self.logger = logging.getLogger("orchestrator")
        
        # Core systems (injected or created)
        self.system_a = system_a  # TetrahedralEvaluator instance
        self.system_b = system_b  # ConstraintPipeline instance
        
        # Enhancements
        self.enhancements: List[Enhancement] = []
        self.enhancement_registry: Dict[str, type] = {}
        
        # Metrics
        self.metrics = OrchestrationMetrics()
        
        # Configuration
        self.max_retries = self.config.get('max_retries', 2)
        self.enable_recovery = self.config.get('enable_recovery', True)
        self.enable_feedback_loops = self.config.get('enable_feedback_loops', True)
        
        # Load enhancements
        if enhancements:
            if 'all' in enhancements:
                self._load_all_enhancements()
            else:
                for enh_name in enhancements:
                    self.enable_enhancement(enh_name)
    
    def register_enhancement(self, name: str, enhancement_class: type):
        """Register an enhancement class"""
        self.enhancement_registry[name] = enhancement_class
        self.logger.info(f"Registered enhancement: {name}")
    
    def enable_enhancement(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Dynamically enable an enhancement"""
        if name not in self.enhancement_registry:
            self.logger.warning(f"Unknown enhancement: {name}")
            return
        
        enhancement_class = self.enhancement_registry[name]
        enhancement = enhancement_class(config)
        self.enhancements.append(enhancement)
        self.metrics.enhancements_enabled.append(name)
        
        self.logger.info(f"Enabled enhancement: {name}")
    
    def disable_enhancement(self, name: str):
        """Disable an enhancement"""
        self.enhancements = [e for e in self.enhancements if e.name != name]
        self.metrics.enhancements_enabled = [
            n for n in self.metrics.enhancements_enabled if n != name
        ]
        self.logger.info(f"Disabled enhancement: {name}")
    
    async def evaluate_and_authorize(
        self, 
        smp: Any,  # SMP type from System A
        retry_count: int = 0
    ) -> CommitResult:
        """
        Main entry point: Orchestrate SMP through A → B pipeline.
        
        Flow:
        1. Pre-processing (enhancements)
        2. System A evaluation
        3. Post-A processing (enhancements)
        4. System B validation
        5. Post-B processing (enhancements)
        6. Audit trail aggregation
        7. Post-processing (enhancements)
        """
        start_time = time.time()
        orchestration_start = start_time
        
        smp_id = getattr(smp, 'id', 'unknown')
        self.logger.info(f"Orchestrating SMP {smp_id} (retry={retry_count})")
        
        try:
            # ================================================================
            # STAGE 1: Pre-processing
            # ================================================================
            for enhancement in self.enhancements:
                if enhancement.enabled:
                    smp = await enhancement.pre_process(smp)
            
            orchestration_overhead = (time.time() - orchestration_start) * 1000
            
            # ================================================================
            # STAGE 2: System A Evaluation
            # ================================================================
            system_a_start = time.time()
            
            try:
                if self.system_a is None:
                    # Stub: Simulate System A
                    refined = self._stub_system_a(smp)
                else:
                    refined = await self.system_a.evaluate(smp)
                
                system_a_time = (time.time() - system_a_start) * 1000
                
            except Exception as e:
                system_a_time = (time.time() - system_a_start) * 1000
                return await self._handle_system_a_failure(smp, e, retry_count)
            
            # ================================================================
            # STAGE 3: Post-System A Processing
            # ================================================================
            orchestration_start = time.time()
            
            for enhancement in self.enhancements:
                if enhancement.enabled:
                    smp, refined = await enhancement.post_system_a(smp, refined)
            
            orchestration_overhead += (time.time() - orchestration_start) * 1000
            
            # ================================================================
            # STAGE 4: System B Validation
            # ================================================================
            system_b_start = time.time()
            
            try:
                if self.system_b is None:
                    # Stub: Simulate System B
                    result = self._stub_system_b(refined)
                else:
                    result = await self.system_b.evaluate(refined)
                
                system_b_time = (time.time() - system_b_start) * 1000
                
            except Exception as e:
                system_b_time = (time.time() - system_b_start) * 1000
                return await self._handle_system_b_failure(smp, refined, e, retry_count)
            
            # ================================================================
            # STAGE 5: Post-System B Processing
            # ================================================================
            orchestration_start = time.time()
            
            for enhancement in self.enhancements:
                if enhancement.enabled:
                    refined, result = await enhancement.post_system_b(refined, result)
            
            orchestration_overhead += (time.time() - orchestration_start) * 1000
            
            # ================================================================
            # STAGE 6: Build Commit Result
            # ================================================================
            total_time = (time.time() - start_time) * 1000
            
            commit_result = self._build_commit_result(
                smp=smp,
                refined=refined,
                result=result,
                system_a_time_ms=system_a_time,
                system_b_time_ms=system_b_time,
                total_time_ms=total_time,
                orchestration_overhead_ms=orchestration_overhead
            )
            
            # ================================================================
            # STAGE 7: Post-processing
            # ================================================================
            for enhancement in self.enhancements:
                if enhancement.enabled:
                    commit_result = await enhancement.post_process(commit_result)
            
            # ================================================================
            # STAGE 8: Update Metrics
            # ================================================================
            self._update_metrics(commit_result)
            
            # ================================================================
            # STAGE 9: Feedback Loops (if enabled and failed)
            # ================================================================
            if not commit_result.committed and self.enable_feedback_loops:
                await self._apply_feedback_loops(commit_result)
            
            return commit_result
            
        except Exception as e:
            self.logger.error(f"Orchestration error for SMP {smp_id}: {e}", exc_info=True)
            return self._build_error_result(smp, e)
    
    async def _handle_system_a_failure(
        self, 
        smp: Any, 
        error: Exception,
        retry_count: int
    ) -> CommitResult:
        """Handle System A failures with recovery logic"""
        self.logger.warning(f"System A failed for SMP {smp.id}: {error}")
        
        failure_mode = self._classify_failure(error, "system_a")
        
        # Ask enhancements for recovery suggestions
        recovery_action = None
        if self.enable_recovery:
            for enhancement in self.enhancements:
                if enhancement.enabled:
                    action = await enhancement.on_failure(smp, failure_mode, error)
                    if action:
                        recovery_action = action
                        break
        
        # Default recovery logic
        if recovery_action is None:
            if retry_count < self.max_retries:
                if failure_mode == FailureMode.CONVERGENCE_FAILURE:
                    recovery_action = RecoveryAction.RETRY_SIMPLIFIED
                elif failure_mode == FailureMode.TIMEOUT:
                    recovery_action = RecoveryAction.RETRY_CONSERVATIVE
                else:
                    recovery_action = RecoveryAction.QUEUE_MANUAL_REVIEW
            else:
                recovery_action = RecoveryAction.REJECT_PERMANENTLY
        
        # Execute recovery
        if recovery_action == RecoveryAction.RETRY_SIMPLIFIED:
            # STUB: Simplify SMP and retry
            self.logger.info(f"Retrying SMP {smp.id} with simplified version")
            # simplified_smp = self._simplify_smp(smp)
            # return await self.evaluate_and_authorize(simplified_smp, retry_count + 1)
        
        elif recovery_action == RecoveryAction.RETRY_CONSERVATIVE:
            # STUB: Retry with conservative settings
            self.logger.info(f"Retrying SMP {smp.id} with conservative settings")
            # return await self.evaluate_and_authorize(smp, retry_count + 1)
        
        # Build failure result
        return CommitResult(
            smp_id=smp.id,
            committed=False,
            failure_mode=failure_mode,
            failure_reason=str(error),
            recovery_action=recovery_action
        )
    
    async def _handle_system_b_failure(
        self,
        smp: Any,
        refined: Any,
        error: Exception,
        retry_count: int
    ) -> CommitResult:
        """Handle System B failures with recovery logic"""
        self.logger.warning(f"System B failed for SMP {smp.id}: {error}")
        
        failure_mode = self._classify_failure(error, "system_b")
        
        # Ask enhancements for recovery suggestions
        recovery_action = None
        if self.enable_recovery:
            for enhancement in self.enhancements:
                if enhancement.enabled:
                    action = await enhancement.on_failure(smp, failure_mode, error)
                    if action:
                        recovery_action = action
                        break
        
        # Default recovery logic
        if recovery_action is None:
            if retry_count < self.max_retries:
                if failure_mode == FailureMode.MESH_FAILURE:
                    recovery_action = RecoveryAction.RETRY_CONSERVATIVE
                elif failure_mode == FailureMode.CONSTRAINT_FAILURE:
                    recovery_action = RecoveryAction.DOWNGRADE_TIER
                else:
                    recovery_action = RecoveryAction.QUEUE_MANUAL_REVIEW
            else:
                recovery_action = RecoveryAction.REJECT_PERMANENTLY
        
        # Execute recovery
        if recovery_action == RecoveryAction.DOWNGRADE_TIER:
            # STUB: Accept as lower tier
            self.logger.info(f"Downgrading SMP {smp.id} to Tier 3 provisional")
            # return self._commit_as_provisional(refined)
        
        # Build failure result
        return CommitResult(
            smp_id=smp.id,
            committed=False,
            failure_mode=failure_mode,
            failure_reason=str(error),
            recovery_action=recovery_action
        )
    
    def _classify_failure(self, error: Exception, system: str) -> FailureMode:
        """Classify failure based on error type and system"""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return FailureMode.TIMEOUT
        elif "convergence" in error_str:
            return FailureMode.CONVERGENCE_FAILURE
        elif "logic" in error_str or "pxl" in error_str:
            return FailureMode.LOGIC_FAILURE
        elif "domain" in error_str or "iel" in error_str:
            return FailureMode.DOMAIN_FAILURE
        elif "math" in error_str or "arp" in error_str:
            return FailureMode.MATH_FAILURE
        elif "mesh" in error_str:
            return FailureMode.MESH_FAILURE
        elif "constraint" in error_str:
            return FailureMode.CONSTRAINT_FAILURE
        else:
            return FailureMode.UNKNOWN
    
    async def _apply_feedback_loops(self, commit_result: CommitResult):
        """Apply feedback from System B to System A configuration"""
        if commit_result.failure_mode == FailureMode.MESH_FAILURE:
            self.logger.info("Applying feedback: Mesh failure → conservative synthesis")
            # STUB: Update System A config
            # self.system_a.update_config({'synthesis_strategy': 'conservative'})
        
        elif commit_result.failure_mode == FailureMode.DOMAIN_FAILURE:
            self.logger.info("Applying feedback: Domain failure → increase IEL weight")
            # STUB: Update System A config
            # self.system_a.update_config({'iel_weight': 1.5})
    
    def _build_commit_result(
        self,
        smp: Any,
        refined: Any,
        result: Any,
        system_a_time_ms: float,
        system_b_time_ms: float,
        total_time_ms: float,
        orchestration_overhead_ms: float
    ) -> CommitResult:
        """Build unified commit result from A/B outputs"""
        
        # Extract data from result (System B output)
        authorized = getattr(result, 'authorized', False)
        tlm_token = getattr(result, 'tlm_token', None)
        audit_trail = getattr(result, 'audit_trail', {})
        
        # Extract data from refined (System A output)
        iterations = getattr(refined, 'iterations', 0)
        stability = getattr(refined, 'stability_score', 0.0)
        quality = getattr(refined, 'quality', None)
        
        # Build constraints dict
        constraints_passed = {
            name: report.passed 
            for name, report in audit_trail.items()
        } if audit_trail else {}
        
        return CommitResult(
            smp_id=smp.id,
            committed=authorized,
            uwm_ref=f"uwm:pending:{smp.id}" if authorized else None,
            tlm_token=tlm_token.token_id if tlm_token else None,
            quality=quality,
            system_a_iterations=iterations,
            system_a_stability=stability,
            system_b_constraints_passed=constraints_passed,
            total_time_ms=total_time_ms,
            system_a_time_ms=system_a_time_ms,
            system_b_time_ms=system_b_time_ms,
            orchestration_overhead_ms=orchestration_overhead_ms,
            failure_mode=None if authorized else FailureMode.CONSTRAINT_FAILURE,
            failure_reason=None if authorized else getattr(result, 'reason', 'Unknown')
        )
    
    def _build_error_result(self, smp: Any, error: Exception) -> CommitResult:
        """Build error result for unhandled exceptions"""
        return CommitResult(
            smp_id=getattr(smp, 'id', 'unknown'),
            committed=False,
            failure_mode=FailureMode.UNKNOWN,
            failure_reason=str(error),
            recovery_action=RecoveryAction.QUEUE_MANUAL_REVIEW
        )
    
    def _update_metrics(self, commit_result: CommitResult):
        """Update orchestrator metrics"""
        self.metrics.total_smps_processed += 1
        
        if commit_result.committed:
            self.metrics.total_committed += 1
        else:
            self.metrics.total_rejected += 1
        
        # Update success rates
        if self.metrics.total_smps_processed > 0:
            self.metrics.end_to_end_success_rate = (
                self.metrics.total_committed / self.metrics.total_smps_processed
            )
        
        # Update timing averages
        n = self.metrics.total_smps_processed
        self.metrics.avg_total_time_ms = (
            (self.metrics.avg_total_time_ms * (n - 1) + commit_result.total_time_ms) / n
        )
        self.metrics.avg_system_a_time_ms = (
            (self.metrics.avg_system_a_time_ms * (n - 1) + commit_result.system_a_time_ms) / n
        )
        self.metrics.avg_system_b_time_ms = (
            (self.metrics.avg_system_b_time_ms * (n - 1) + commit_result.system_b_time_ms) / n
        )
        
        # Update quality distribution
        if commit_result.quality:
            self.metrics.quality_distribution[commit_result.quality] = (
                self.metrics.quality_distribution.get(commit_result.quality, 0) + 1
            )
        
        # Update failure modes
        if commit_result.failure_mode:
            self.metrics.failure_mode_counts[commit_result.failure_mode] = (
                self.metrics.failure_mode_counts.get(commit_result.failure_mode, 0) + 1
            )
    
    def get_metrics(self) -> OrchestrationMetrics:
        """Get current metrics"""
        return self.metrics
    
    def reset_metrics(self):
        """Reset metrics"""
        self.metrics = OrchestrationMetrics(
            enhancements_enabled=list(self.metrics.enhancements_enabled)
        )
    
    # ========================================================================
    # STUBS FOR TESTING WITHOUT SYSTEMS A/B
    # ========================================================================
    
    def _stub_system_a(self, smp: Any) -> Any:
        """Stub System A for testing"""
        @minimal_dataclass
        class StubRefined:
            original_smp: Any
            refined_content: Dict[str, Any]
            iterations: int = 4
            stability_score: float = 0.92
            quality: SMPQuality = SMPQuality.HIGH
        
        return StubRefined(
            original_smp=smp,
            refined_content={"stub": "system_a_output", "smp_id": smp.id}
        )
    
    def _stub_system_b(self, refined: Any) -> Any:
        """Stub System B for testing"""
        @minimal_dataclass
        class StubResult:
            authorized: bool = True
            tlm_token: Any = None
            audit_trail: Dict = field(default_factory=dict)
            reason: Optional[str] = None
        
        @minimal_dataclass
        class StubToken:
            token_id: str = "stub_token_123"
        
        return StubResult(
            authorized=True,
            tlm_token=StubToken(),
            audit_trail={
                "sign": type('obj', (object,), {'passed': True})(),
                "mind": type('obj', (object,), {'passed': True})(),
                "bridge": type('obj', (object,), {'passed': True})(),
            }
        )
    
    def _load_all_enhancements(self):
        """Load all registered enhancements"""
        for name in self.enhancement_registry.keys():
            self.enable_enhancement(name)


# ============================================================================
# ENHANCEMENT IMPLEMENTATIONS (STUBS)
# ============================================================================

class AdaptiveControlEnhancement(Enhancement):
    """
    ENHANCEMENT: Adaptive Control
    
    Adjusts iteration budgets and thresholds based on SMP characteristics.
    
    STUB FUNCTIONALITY:
    - Sets max_iterations based on SMP priority
    - Adjusts stability threshold based on SMP tier
    - Provides early termination hints
    """
    
    @property
    def name(self) -> str:
        return "adaptive_control"
    
    async def pre_process(self, smp: Any) -> Any:
        """Adjust SMP metadata based on priority/tier"""
        priority = getattr(smp, 'priority', 5)
        tier = getattr(smp, 'tier', 3)
        
        # STUB: Set iteration budget
        if priority <= 3:
            budget = 3  # Fast-track
        elif priority >= 8:
            budget = 15  # Deep analysis
        else:
            budget = 8  # Normal
        
        # STUB: Adjust stability threshold
        if tier == 1:
            threshold = 0.98  # Tier 1: Maximum rigor
        elif tier == 2:
            threshold = 0.95  # Tier 2: High standards
        else:
            threshold = 0.90  # Tier 3: Relaxed
        
        # Store in metadata (would actually configure System A)
        if not hasattr(smp, 'metadata'):
            smp.metadata = {}
        smp.metadata['_adaptive_budget'] = budget
        smp.metadata['_adaptive_threshold'] = threshold
        
        self.logger.debug(
            f"Adaptive control: SMP {smp.id} → "
            f"budget={budget}, threshold={threshold:.2f}"
        )
        
        return smp


class QualityClassifierEnhancement(Enhancement):
    """
    ENHANCEMENT: Quality Classifier
    
    Classifies refined SMPs into quality tiers.
    
    STUB FUNCTIONALITY:
    - Scores based on stability and engine reports
    - Classifies as EXCELLENT/HIGH/GOOD/MARGINAL/POOR/UNSTABLE
    - Flags poor quality for review
    """
    
    @property
    def name(self) -> str:
        return "quality_classifier"
    
    async def post_system_a(self, smp: Any, refined: Any) -> Tuple[Any, Any]:
        """Classify quality after System A"""
        stability = getattr(refined, 'stability_score', 0.0)
        iterations = getattr(refined, 'iterations', 0)
        
        # STUB: Simple quality classification
        if stability >= 0.98 and iterations <= 5:
            quality = SMPQuality.EXCELLENT
        elif stability >= 0.95:
            quality = SMPQuality.HIGH
        elif stability >= 0.90:
            quality = SMPQuality.GOOD
        elif stability >= 0.80:
            quality = SMPQuality.MARGINAL
        elif stability >= 0.70:
            quality = SMPQuality.POOR
        else:
            quality = SMPQuality.UNSTABLE
        
        # Attach quality to refined SMP
        refined.quality = quality
        
        self.logger.info(
            f"Quality: SMP {smp.id} → {quality.value} "
            f"(stability={stability:.3f}, iter={iterations})"
        )
        
        return smp, refined


class PerformanceMonitorEnhancement(Enhancement):
    """
    ENHANCEMENT: Performance Monitor
    
    Tracks and reports performance metrics.
    
    STUB FUNCTIONALITY:
    - Logs timing for each stage
    - Detects slow evaluations
    - Warns on timeouts or bottlenecks
    """
    
    @property
    def name(self) -> str:
        return "performance_monitor"
    
    async def post_process(self, commit_result: CommitResult) -> CommitResult:
        """Report performance after everything"""
        total = commit_result.total_time_ms
        sys_a = commit_result.system_a_time_ms
        sys_b = commit_result.system_b_time_ms
        overhead = commit_result.orchestration_overhead_ms
        
        # Log performance
        self.logger.info(
            f"Performance: SMP {commit_result.smp_id} → "
            f"total={total:.1f}ms (A={sys_a:.1f}ms, B={sys_b:.1f}ms, "
            f"overhead={overhead:.1f}ms)"
        )
        
        # STUB: Detect slow evaluations
        if total > 5000:  # > 5 seconds
            self.logger.warning(f"Slow evaluation: {total:.1f}ms for SMP {commit_result.smp_id}")
        
        # Store in enhancement metadata
        commit_result.enhancement_metadata['performance'] = {
            'total_ms': total,
            'system_a_ms': sys_a,
            'system_b_ms': sys_b,
            'overhead_ms': overhead,
            'overhead_percentage': (overhead / total * 100) if total > 0 else 0
        }
        
        return commit_result


class FailureRecoveryEnhancement(Enhancement):
    """
    ENHANCEMENT: Failure Recovery
    
    Implements intelligent failure recovery strategies.
    
    STUB FUNCTIONALITY:
    - Suggests recovery actions based on failure mode
    - Implements retry logic with backoff
    - Downgrades tier for partial acceptance
    """
    
    @property
    def name(self) -> str:
        return "failure_recovery"
    
    async def on_failure(
        self, 
        smp: Any, 
        failure_mode: FailureMode, 
        error: Exception
    ) -> Optional[RecoveryAction]:
        """Suggest recovery action based on failure mode"""
        
        # STUB: Simple recovery strategy
        if failure_mode == FailureMode.CONVERGENCE_FAILURE:
            self.logger.info(f"Recovery: {smp.id} → retry_simplified")
            return RecoveryAction.RETRY_SIMPLIFIED
        
        elif failure_mode == FailureMode.MESH_FAILURE:
            self.logger.info(f"Recovery: {smp.id} → retry_conservative")
            return RecoveryAction.RETRY_CONSERVATIVE
        
        elif failure_mode == FailureMode.DOMAIN_FAILURE:
            tier = getattr(smp, 'tier', 3)
            if tier >= 2:
                self.logger.info(f"Recovery: {smp.id} → downgrade_tier")
                return RecoveryAction.DOWNGRADE_TIER
        
        elif failure_mode == FailureMode.TIMEOUT:
            self.logger.info(f"Recovery: {smp.id} → retry_simplified")
            return RecoveryAction.RETRY_SIMPLIFIED
        
        # Default: manual review
        return RecoveryAction.QUEUE_MANUAL_REVIEW


class AuditAggregatorEnhancement(Enhancement):
    """
    ENHANCEMENT: Audit Aggregator
    
    Aggregates detailed audit trails from Systems A and B.
    
    STUB FUNCTIONALITY:
    - Combines A/B audit data
    - Adds traceability metadata
    - Generates unified audit report
    """
    
    @property
    def name(self) -> str:
        return "audit_aggregator"
    
    async def post_process(self, commit_result: CommitResult) -> CommitResult:
        """Aggregate audit trail"""
        
        # STUB: Build comprehensive audit trail
        audit_data = {
            'smp_id': commit_result.smp_id,
            'committed': commit_result.committed,
            'timestamp': commit_result.timestamp_utc,
            'system_a': {
                'iterations': commit_result.system_a_iterations,
                'stability': commit_result.system_a_stability,
                'time_ms': commit_result.system_a_time_ms,
            },
            'system_b': {
                'constraints': commit_result.system_b_constraints_passed,
                'time_ms': commit_result.system_b_time_ms,
            },
            'quality': commit_result.quality.value if commit_result.quality else None,
            'tlm_token': commit_result.tlm_token,
        }
        
        if not commit_result.committed:
            audit_data['failure'] = {
                'mode': commit_result.failure_mode.value if commit_result.failure_mode else None,
                'reason': commit_result.failure_reason,
                'recovery_action': commit_result.recovery_action.value if commit_result.recovery_action else None,
            }
        
        commit_result.enhancement_metadata['audit_trail'] = audit_data
        
        self.logger.debug(f"Audit: Aggregated trail for SMP {commit_result.smp_id}")
        
        return commit_result


class PatternLearnerEnhancement(Enhancement):
    """
    ENHANCEMENT: Pattern Learner
    
    Learns from historical evaluation patterns.
    
    STUB FUNCTIONALITY:
    - Tracks common failure patterns
    - Detects systematic issues
    - Suggests configuration adjustments
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.failure_patterns: Dict[str, int] = {}
        self.success_patterns: Dict[str, int] = {}
    
    @property
    def name(self) -> str:
        return "pattern_learner"
    
    async def post_process(self, commit_result: CommitResult) -> CommitResult:
        """Learn from evaluation result"""
        
        # STUB: Track patterns
        pattern_key = f"{commit_result.quality.value if commit_result.quality else 'unknown'}"
        
        if commit_result.committed:
            self.success_patterns[pattern_key] = self.success_patterns.get(pattern_key, 0) + 1
        else:
            failure_key = f"{commit_result.failure_mode.value if commit_result.failure_mode else 'unknown'}"
            self.failure_patterns[failure_key] = self.failure_patterns.get(failure_key, 0) + 1
        
        # STUB: Detect systematic issues (e.g., all math SMPs failing)
        if self.failure_patterns.get(FailureMode.MATH_FAILURE.value, 0) > 5:
            self.logger.warning(
                "Pattern detected: Multiple math failures. "
                "Consider adjusting ARP weight or thresholds."
            )
        
        commit_result.enhancement_metadata['learned_patterns'] = {
            'success_patterns': dict(self.success_patterns),
            'failure_patterns': dict(self.failure_patterns)
        }
        
        return commit_result


# ============================================================================
# ORCHESTRATOR FACTORY & REGISTRATION
# ============================================================================

def create_orchestrator(
    system_a = None,
    system_b = None,
    config: Optional[Dict[str, Any]] = None,
    enhancements: Optional[List[str]] = None
) -> SystemOrchestrator:
    """
    Factory function to create orchestrator with enhancements.
    
    Usage:
        # Minimal
        orch = create_orchestrator()
        
        # With specific enhancements
        orch = create_orchestrator(enhancements=['adaptive_control', 'quality_classifier'])
        
        # Full intelligence
        orch = create_orchestrator(enhancements=['all'])
    """
    orchestrator = SystemOrchestrator(system_a, system_b, config, enhancements=[])
    
    # Register all available enhancements
    orchestrator.register_enhancement('adaptive_control', AdaptiveControlEnhancement)
    orchestrator.register_enhancement('quality_classifier', QualityClassifierEnhancement)
    orchestrator.register_enhancement('performance_monitor', PerformanceMonitorEnhancement)
    orchestrator.register_enhancement('failure_recovery', FailureRecoveryEnhancement)
    orchestrator.register_enhancement('audit_aggregator', AuditAggregatorEnhancement)
    orchestrator.register_enhancement('pattern_learner', PatternLearnerEnhancement)
    
    # Enable requested enhancements
    if enhancements:
        if 'all' in enhancements:
            orchestrator._load_all_enhancements()
        else:
            for enh_name in enhancements:
                orchestrator.enable_enhancement(enh_name)
    
    return orchestrator


# ============================================================================
# TESTING / DEMONSTRATION
# ============================================================================

async def _test_orchestrator():
    """Test harness for orchestrator"""
    logging.basicConfig(level=logging.INFO)
    
    # Create test SMP
    @minimal_dataclass
    class TestSMP:
        id: str = "test_smp_001"
        content: Dict[str, Any] = field(default_factory=lambda: {
            "proposition": "Test proposition",
            "confidence": 0.95
        })
        priority: int = 7
        tier: int = 2
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    test_smp = TestSMP()
    
    # Test 1: Minimal orchestrator (no enhancements)
    print("\n" + "="*80)
    print("TEST 1: Minimal Orchestrator (No Enhancements)")
    print("="*80)
    
    orch_minimal = create_orchestrator()
    result1 = await orch_minimal.evaluate_and_authorize(test_smp)
    
    print(f"Committed: {result1.committed}")
    print(f"Quality: {result1.quality}")
    print(f"Total Time: {result1.total_time_ms:.1f}ms")
    print(f"Overhead: {result1.orchestration_overhead_ms:.1f}ms")
    print(f"Enhancements: {orch_minimal.metrics.enhancements_enabled}")
    
    # Test 2: With enhancements
    print("\n" + "="*80)
    print("TEST 2: Orchestrator with Enhancements")
    print("="*80)
    
    orch_enhanced = create_orchestrator(enhancements=[
        'adaptive_control',
        'quality_classifier',
        'performance_monitor',
        'audit_aggregator'
    ])
    
    result2 = await orch_enhanced.evaluate_and_authorize(test_smp)
    
    print(f"Committed: {result2.committed}")
    print(f"Quality: {result2.quality}")
    print(f"Total Time: {result2.total_time_ms:.1f}ms")
    print(f"Overhead: {result2.orchestration_overhead_ms:.1f}ms")
    print(f"Enhancements: {orch_enhanced.metrics.enhancements_enabled}")
    
    if 'performance' in result2.enhancement_metadata:
        perf = result2.enhancement_metadata['performance']
        print(f"Overhead %: {perf['overhead_percentage']:.1f}%")
    
    if 'audit_trail' in result2.enhancement_metadata:
        print("\nAudit Trail:")
        print(json.dumps(result2.enhancement_metadata['audit_trail'], indent=2))
    
    # Test 3: Full intelligence mode
    print("\n" + "="*80)
    print("TEST 3: Full Intelligence Mode")
    print("="*80)
    
    orch_full = create_orchestrator(enhancements=['all'])
    result3 = await orch_full.evaluate_and_authorize(test_smp)
    
    print(f"Committed: {result3.committed}")
    print(f"Quality: {result3.quality}")
    print(f"Total Time: {result3.total_time_ms:.1f}ms")
    print(f"Overhead: {result3.orchestration_overhead_ms:.1f}ms")
    print(f"Enhancements: {orch_full.metrics.enhancements_enabled}")
    
    # Metrics summary
    print("\n" + "="*80)
    print("ORCHESTRATOR METRICS")
    print("="*80)
    
    metrics = orch_full.get_metrics()
    print(f"Total Processed: {metrics.total_smps_processed}")
    print(f"Committed: {metrics.total_committed}")
    print(f"Rejected: {metrics.total_rejected}")
    print(f"Success Rate: {metrics.end_to_end_success_rate:.1%}")
    print(f"Avg Total Time: {metrics.avg_total_time_ms:.1f}ms")
    print(f"Avg System A Time: {metrics.avg_system_a_time_ms:.1f}ms")
    print(f"Avg System B Time: {metrics.avg_system_b_time_ms:.1f}ms")


if __name__ == "__main__":
    asyncio.run(_test_orchestrator())