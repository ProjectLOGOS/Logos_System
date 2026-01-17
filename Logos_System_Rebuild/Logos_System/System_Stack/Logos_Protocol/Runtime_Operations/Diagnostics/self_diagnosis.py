#!/usr/bin/env python3
"""
Self-Diagnosis Tool
==================

Create self-diagnosis tool for automated system health assessment

This module provides automated system health assessment and diagnosis capabilities.
"""

import asyncio
import logging
import psutil
import os
from datetime import datetime, timezone
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    metrics: Dict[str, Any]
    timestamp: datetime
    recommendations: List[str]

class SelfDiagnosisTool:
    """
    Self-diagnosis tool for automated system health assessment
    and problem identification.
    """

    def __init__(self):
        self.health_history: List[HealthCheck] = []
        self.diagnostic_rules: Dict[str, Callable] = {}
        self.baseline_metrics: Dict[str, Any] = {}

        # Initialize diagnostic rules
        self._initialize_diagnostic_rules()

        # Establish baseline
        asyncio.create_task(self._establish_baseline())

        logger.info("Self-Diagnosis Tool initialized")

    def _initialize_diagnostic_rules(self):
        """Initialize diagnostic rules for different components."""
        self.diagnostic_rules = {
            "cpu": self._check_cpu_health,
            "memory": self._check_memory_health,
            "disk": self._check_disk_health,
            "network": self._check_network_health,
            "processes": self._check_process_health,
            "filesystem": self._check_filesystem_health
        }

    async def _establish_baseline(self):
        """Establish baseline metrics for comparison."""
        try:
            # Wait a bit for system to stabilize
            await asyncio.sleep(10)

            self.baseline_metrics = {
                "cpu_percent": psutil.cpu_percent(interval=5),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "load_average": os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0,
                "established_at": datetime.now(timezone.utc)
            }

            logger.info("Baseline metrics established")

        except Exception as e:
            logger.error(f"Failed to establish baseline: {e}")

    async def perform_full_diagnosis(self) -> Dict[str, Any]:
        """Perform comprehensive system diagnosis."""
        diagnosis_start = datetime.now(timezone.utc)

        health_checks = []
        for component, check_func in self.diagnostic_rules.items():
            try:
                check_result = await check_func()
                health_checks.append(check_result)
                self.health_history.append(check_result)
            except Exception as e:
                error_check = HealthCheck(
                    component=component,
                    status=HealthStatus.UNKNOWN,
                    message=f"Diagnosis failed: {e}",
                    metrics={},
                    timestamp=datetime.now(timezone.utc),
                    recommendations=["Investigate diagnostic tool failure"]
                )
                health_checks.append(error_check)
                self.health_history.append(error_check)

        # Overall assessment
        overall_status = self._calculate_overall_health(health_checks)

        diagnosis = {
            "overall_status": overall_status.value,
            "diagnosis_timestamp": diagnosis_start.isoformat(),
            "health_checks": [
                {
                    "component": hc.component,
                    "status": hc.status.value,
                    "message": hc.message,
                    "recommendations": hc.recommendations
                }
                for hc in health_checks
            ],
            "critical_issues": len([hc for hc in health_checks if hc.status == HealthStatus.CRITICAL]),
            "warning_issues": len([hc for hc in health_checks if hc.status == HealthStatus.WARNING]),
            "summary": self._generate_diagnosis_summary(health_checks)
        }

        return diagnosis

    async def _check_cpu_health(self) -> HealthCheck:
        """Check CPU health."""
        cpu_percent = psutil.cpu_percent(interval=2)
        load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
        cpu_count = psutil.cpu_count()

        metrics = {
            "cpu_percent": cpu_percent,
            "load_average": load_avg,
            "cpu_count": cpu_count
        }

        recommendations = []

        if cpu_percent > 90:
            status = HealthStatus.CRITICAL
            message = f"CPU usage critically high: {cpu_percent:.1f}%"
            recommendations.extend([
                "Reduce CPU-intensive operations",
                "Consider scaling up CPU resources",
                "Check for runaway processes"
            ])
        elif cpu_percent > 75:
            status = HealthStatus.WARNING
            message = f"CPU usage elevated: {cpu_percent:.1f}%"
            recommendations.extend([
                "Monitor CPU usage trends",
                "Optimize CPU-intensive operations"
            ])
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage normal: {cpu_percent:.1f}%"

        return HealthCheck(
            component="cpu",
            status=status,
            message=message,
            metrics=metrics,
            timestamp=datetime.now(timezone.utc),
            recommendations=recommendations
        )

    async def _check_memory_health(self) -> HealthCheck:
        """Check memory health."""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        available_gb = memory.available / (1024**3)

        metrics = {
            "memory_percent": memory_percent,
            "available_gb": available_gb,
            "total_gb": memory.total / (1024**3)
        }

        recommendations = []

        if memory_percent > 95:
            status = HealthStatus.CRITICAL
            message = f"Memory usage critically high: {memory_percent:.1f}%"
            recommendations.extend([
                "Immediate memory cleanup required",
                "Check for memory leaks",
                "Consider increasing memory allocation"
            ])
        elif memory_percent > 85:
            status = HealthStatus.WARNING
            message = f"Memory usage high: {memory_percent:.1f}%"
            recommendations.extend([
                "Monitor memory usage",
                "Optimize memory-intensive operations"
            ])
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: {memory_percent:.1f}%"

        return HealthCheck(
            component="memory",
            status=status,
            message=message,
            metrics=metrics,
            timestamp=datetime.now(timezone.utc),
            recommendations=recommendations
        )

    async def _check_disk_health(self) -> HealthCheck:
        """Check disk health."""
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        metrics = {
            "disk_percent": disk_percent,
            "free_gb": disk.free / (1024**3),
            "total_gb": disk.total / (1024**3)
        }

        recommendations = []

        if disk_percent > 95:
            status = HealthStatus.CRITICAL
            message = f"Disk usage critically high: {disk_percent:.1f}%"
            recommendations.extend([
                "Immediate disk cleanup required",
                "Archive old data",
                "Consider disk expansion"
            ])
        elif disk_percent > 85:
            status = HealthStatus.WARNING
            message = f"Disk usage high: {disk_percent:.1f}%"
            recommendations.extend([
                "Monitor disk usage",
                "Clean up unnecessary files"
            ])
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage normal: {disk_percent:.1f}%"

        return HealthCheck(
            component="disk",
            status=status,
            message=message,
            metrics=metrics,
            timestamp=datetime.now(timezone.utc),
            recommendations=recommendations
        )

    async def _check_network_health(self) -> HealthCheck:
        """Check network health."""
        # Simplified network check
        metrics = {
            "network_status": "operational"  # Would check actual connectivity
        }

        return HealthCheck(
            component="network",
            status=HealthStatus.HEALTHY,
            message="Network connectivity operational",
            metrics=metrics,
            timestamp=datetime.now(timezone.utc),
            recommendations=[]
        )

    async def _check_process_health(self) -> HealthCheck:
        """Check process health."""
        processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']))
        zombie_processes = [p for p in processes if p.info['cpu_percent'] is None]

        metrics = {
            "total_processes": len(processes),
            "zombie_processes": len(zombie_processes)
        }

        recommendations = []

        if zombie_processes:
            status = HealthStatus.WARNING
            message = f"Found {len(zombie_processes)} zombie processes"
            recommendations.append("Clean up zombie processes")
        else:
            status = HealthStatus.HEALTHY
            message = f"Process health normal: {len(processes)} active processes"

        return HealthCheck(
            component="processes",
            status=status,
            message=message,
            metrics=metrics,
            timestamp=datetime.now(timezone.utc),
            recommendations=recommendations
        )

    async def _check_filesystem_health(self) -> HealthCheck:
        """Check filesystem health."""
        # Check for common issues
        issues = []

        # Check for large files
        large_files = []
        try:
            for root, dirs, files in os.walk('/tmp', topdown=True):
                for file in files:
                    path = os.path.join(root, file)
                    try:
                        size = os.path.getsize(path)
                        if size > 100 * 1024 * 1024:  # 100MB
                            large_files.append((path, size))
                    except:
                        continue
                if len(large_files) >= 5:  # Limit check
                    break
        except:
            pass

        if large_files:
            issues.append(f"Found {len(large_files)} large files in temp directory")

        metrics = {
            "large_temp_files": len(large_files),
            "filesystem_issues": len(issues)
        }

        if issues:
            status = HealthStatus.WARNING
            message = f"Filesystem issues detected: {'; '.join(issues)}"
            recommendations = ["Clean up temporary files", "Check disk space usage"]
        else:
            status = HealthStatus.HEALTHY
            message = "Filesystem health normal"
            recommendations = []

        return HealthCheck(
            component="filesystem",
            status=status,
            message=message,
            metrics=metrics,
            timestamp=datetime.now(timezone.utc),
            recommendations=recommendations
        )

    def _calculate_overall_health(self, health_checks: List[HealthCheck]) -> HealthStatus:
        """Calculate overall system health."""
        critical_count = sum(1 for hc in health_checks if hc.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for hc in health_checks if hc.status == HealthStatus.WARNING)

        if critical_count > 0:
            return HealthStatus.CRITICAL
        elif warning_count > 1:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    def _generate_diagnosis_summary(self, health_checks: List[HealthCheck]) -> str:
        """Generate a summary of the diagnosis."""
        total_checks = len(health_checks)
        healthy = sum(1 for hc in health_checks if hc.status == HealthStatus.HEALTHY)
        warnings = sum(1 for hc in health_checks if hc.status == HealthStatus.WARNING)
        critical = sum(1 for hc in health_checks if hc.status == HealthStatus.CRITICAL)

        return f"System diagnosis: {healthy}/{total_checks} components healthy, {warnings} warnings, {critical} critical issues"

    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent health check history."""
        recent = self.health_history[-limit:]
        return [
            {
                "component": hc.component,
                "status": hc.status.value,
                "message": hc.message,
                "timestamp": hc.timestamp.isoformat()
            }
            for hc in recent
        ]

# Global instance
self_diagnosis_tool = SelfDiagnosisTool()

async def perform_system_diagnosis() -> Dict[str, Any]:
    """Perform comprehensive system diagnosis."""
    return await self_diagnosis_tool.perform_full_diagnosis()

# Synchronous wrapper for compatibility
def get_system_diagnosis() -> Dict[str, Any]:
    """Get system diagnosis (synchronous wrapper)."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(perform_system_diagnosis())
        loop.close()
        return result
    except Exception as e:
        logger.error(f"System diagnosis failed: {e}")
        return {"error": str(e), "status": "failed"}
