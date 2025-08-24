#!/usr/bin/env python3
"""
Performance Monitoring and Profiling

This module provides comprehensive performance monitoring and profiling
capabilities for the OpenLLM project, including system resources,
model performance, and optimization recommendations.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import time
import psutil
import torch
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import json
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""

    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    network_io: Dict[str, float]
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    inference_time_ms: float
    tokens_per_second: float
    memory_usage_mb: float
    batch_size: int
    sequence_length: int
    model_parameters: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingMetrics:
    """Training performance metrics."""

    loss: float
    learning_rate: float
    gradient_norm: float
    training_time_ms: float
    samples_per_second: float
    memory_usage_mb: float
    epoch: int
    step: int
    timestamp: float = field(default_factory=time.time)


class PerformanceProfiler:
    """
    Performance profiler for monitoring and optimizing system performance.

    This profiler tracks system resources, model performance, and training metrics
    to provide insights and optimization recommendations.
    """

    def __init__(
        self,
        history_size: int = 1000,
        monitoring_interval: float = 1.0,
        enable_gpu_monitoring: bool = True,
    ):
        """
        Initialize performance profiler.

        Args:
            history_size: Number of metrics to keep in history
            monitoring_interval: Interval between system checks (seconds)
            enable_gpu_monitoring: Whether to monitor GPU usage
        """
        self.history_size = history_size
        self.monitoring_interval = monitoring_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring

        # Metrics storage
        self.system_metrics = deque(maxlen=history_size)
        self.model_metrics = deque(maxlen=history_size)
        self.training_metrics = deque(maxlen=history_size)

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None

        # Performance counters
        self.total_inference_requests = 0
        self.total_training_steps = 0
        self.start_time = time.time()

        # Optimization recommendations
        self.recommendations = []

        logger.info("PerformanceProfiler initialized")

    def start_monitoring(self):
        """Start continuous system monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop continuous system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("System monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)

                # Check for performance issues
                self._check_performance_issues(metrics)

                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)

        # Disk usage
        disk_usage = psutil.disk_usage("/")
        disk_usage_percent = disk_usage.percent

        # Network I/O
        network_io = psutil.net_io_counters()
        network_metrics = {
            "bytes_sent": network_io.bytes_sent,
            "bytes_recv": network_io.bytes_recv,
            "packets_sent": network_io.packets_sent,
            "packets_recv": network_io.packets_recv,
        }

        # GPU metrics (if available)
        gpu_utilization = None
        gpu_memory_percent = None

        if self.enable_gpu_monitoring and torch.cuda.is_available():
            try:
                gpu_utilization = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_stats()
                gpu_memory_percent = (
                    (
                        gpu_memory["allocated_bytes.all.current"]
                        / gpu_memory["reserved_bytes.all.current"]
                    )
                    * 100
                    if gpu_memory["reserved_bytes.all.current"] > 0
                    else 0
                )
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            disk_usage_percent=disk_usage_percent,
            network_io=network_metrics,
            gpu_utilization=gpu_utilization,
            gpu_memory_percent=gpu_memory_percent,
        )

    def _check_performance_issues(self, metrics: SystemMetrics):
        """Check for performance issues and generate recommendations."""
        recommendations = []

        # Memory usage check
        if metrics.memory_percent > 90:
            recommendations.append(
                {
                    "type": "memory_high",
                    "severity": "high",
                    "message": f"Memory usage is very high ({metrics.memory_percent:.1f}%)",
                    "suggestion": "Consider reducing batch size or using gradient checkpointing",
                }
            )
        elif metrics.memory_percent > 80:
            recommendations.append(
                {
                    "type": "memory_high",
                    "severity": "medium",
                    "message": f"Memory usage is high ({metrics.memory_percent:.1f}%)",
                    "suggestion": "Monitor memory usage and consider optimization",
                }
            )

        # CPU usage check
        if metrics.cpu_percent > 95:
            recommendations.append(
                {
                    "type": "cpu_high",
                    "severity": "high",
                    "message": f"CPU usage is very high ({metrics.cpu_percent:.1f}%)",
                    "suggestion": "Consider reducing number of workers or using GPU",
                }
            )

        # GPU usage check
        if metrics.gpu_utilization is not None:
            if metrics.gpu_utilization < 50:
                recommendations.append(
                    {
                        "type": "gpu_underutilized",
                        "severity": "low",
                        "message": f"GPU utilization is low ({metrics.gpu_utilization:.1f}%)",
                        "suggestion": "Consider increasing batch size or using mixed precision",
                    }
                )
            elif metrics.gpu_memory_percent and metrics.gpu_memory_percent > 90:
                recommendations.append(
                    {
                        "type": "gpu_memory_high",
                        "severity": "high",
                        "message": f"GPU memory usage is very high ({metrics.gpu_memory_percent:.1f}%)",
                        "suggestion": "Consider reducing batch size or using gradient checkpointing",
                    }
                )

        # Add recommendations to history
        for rec in recommendations:
            rec["timestamp"] = time.time()
            self.recommendations.append(rec)

        # Keep only recent recommendations
        if len(self.recommendations) > 100:
            self.recommendations = self.recommendations[-100:]

    def record_inference(
        self,
        inference_time_ms: float,
        tokens_generated: int,
        memory_usage_mb: float,
        batch_size: int,
        sequence_length: int,
        model_parameters: int,
    ):
        """Record inference performance metrics."""
        tokens_per_second = (
            (tokens_generated / (inference_time_ms / 1000)) if inference_time_ms > 0 else 0
        )

        metrics = ModelMetrics(
            inference_time_ms=inference_time_ms,
            tokens_per_second=tokens_per_second,
            memory_usage_mb=memory_usage_mb,
            batch_size=batch_size,
            sequence_length=sequence_length,
            model_parameters=model_parameters,
        )

        self.model_metrics.append(metrics)
        self.total_inference_requests += 1

    def record_training(
        self,
        loss: float,
        learning_rate: float,
        gradient_norm: float,
        training_time_ms: float,
        samples_processed: int,
        memory_usage_mb: float,
        epoch: int,
        step: int,
    ):
        """Record training performance metrics."""
        samples_per_second = (
            (samples_processed / (training_time_ms / 1000)) if training_time_ms > 0 else 0
        )

        metrics = TrainingMetrics(
            loss=loss,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            training_time_ms=training_time_ms,
            samples_per_second=samples_per_second,
            memory_usage_mb=memory_usage_mb,
            epoch=epoch,
            step=step,
        )

        self.training_metrics.append(metrics)
        self.total_training_steps += 1

    def get_system_summary(self) -> Dict[str, Any]:
        """Get system performance summary."""
        if not self.system_metrics:
            return {"error": "No system metrics available"}

        recent_metrics = list(self.system_metrics)[-100:]  # Last 100 measurements

        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]

        return {
            "cpu": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": np.mean(cpu_values) if cpu_values else 0,
                "max": np.max(cpu_values) if cpu_values else 0,
                "min": np.min(cpu_values) if cpu_values else 0,
            },
            "memory": {
                "current_percent": memory_values[-1] if memory_values else 0,
                "average_percent": np.mean(memory_values) if memory_values else 0,
                "available_gb": recent_metrics[-1].memory_available_gb if recent_metrics else 0,
            },
            "gpu": {
                "utilization": recent_metrics[-1].gpu_utilization if recent_metrics else None,
                "memory_percent": recent_metrics[-1].gpu_memory_percent if recent_metrics else None,
            },
            "uptime_hours": (time.time() - self.start_time) / 3600,
        }

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model performance summary."""
        if not self.model_metrics:
            return {"error": "No model metrics available"}

        recent_metrics = list(self.model_metrics)[-100:]  # Last 100 measurements

        inference_times = [m.inference_time_ms for m in recent_metrics]
        tokens_per_sec = [m.tokens_per_second for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]

        return {
            "inference": {
                "avg_time_ms": np.mean(inference_times) if inference_times else 0,
                "min_time_ms": np.min(inference_times) if inference_times else 0,
                "max_time_ms": np.max(inference_times) if inference_times else 0,
                "avg_tokens_per_second": np.mean(tokens_per_sec) if tokens_per_sec else 0,
            },
            "memory": {
                "avg_usage_mb": np.mean(memory_usage) if memory_usage else 0,
                "max_usage_mb": np.max(memory_usage) if memory_usage else 0,
            },
            "total_requests": self.total_inference_requests,
            "recent_requests": len(recent_metrics),
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training performance summary."""
        if not self.training_metrics:
            return {"error": "No training metrics available"}

        recent_metrics = list(self.training_metrics)[-100:]  # Last 100 measurements

        losses = [m.loss for m in recent_metrics]
        samples_per_sec = [m.samples_per_second for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]

        return {
            "loss": {
                "current": losses[-1] if losses else 0,
                "average": np.mean(losses) if losses else 0,
                "min": np.min(losses) if losses else 0,
                "trend": "decreasing"
                if len(losses) > 1 and losses[-1] < losses[0]
                else "increasing",
            },
            "performance": {
                "avg_samples_per_second": np.mean(samples_per_sec) if samples_per_sec else 0,
                "avg_memory_usage_mb": np.mean(memory_usage) if memory_usage else 0,
            },
            "total_steps": self.total_training_steps,
            "recent_steps": len(recent_metrics),
            "current_epoch": recent_metrics[-1].epoch if recent_metrics else 0,
        }

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations."""
        return self.recommendations[-10:]  # Return last 10 recommendations

    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        system_summary = self.get_system_summary()
        model_summary = self.get_model_summary()
        training_summary = self.get_training_summary()
        recommendations = self.get_recommendations()

        # Calculate overall performance score
        performance_score = self._calculate_performance_score(
            system_summary, model_summary, training_summary
        )

        return {
            "timestamp": time.time(),
            "performance_score": performance_score,
            "system_summary": system_summary,
            "model_summary": model_summary,
            "training_summary": training_summary,
            "recommendations": recommendations,
            "optimization_priority": self._get_optimization_priority(recommendations),
        }

    def _calculate_performance_score(
        self, system_summary: Dict, model_summary: Dict, training_summary: Dict
    ) -> float:
        """Calculate overall performance score (0-100)."""
        score = 100.0

        # Deduct points for system issues
        if "cpu" in system_summary:
            cpu_avg = system_summary["cpu"]["average"]
            if cpu_avg > 90:
                score -= 20
            elif cpu_avg > 80:
                score -= 10
            elif cpu_avg > 70:
                score -= 5

        if "memory" in system_summary:
            memory_avg = system_summary["memory"]["average_percent"]
            if memory_avg > 90:
                score -= 20
            elif memory_avg > 80:
                score -= 10
            elif memory_avg > 70:
                score -= 5

        # Deduct points for model performance issues
        if "inference" in model_summary:
            avg_time = model_summary["inference"]["avg_time_ms"]
            if avg_time > 1000:  # More than 1 second
                score -= 15
            elif avg_time > 500:  # More than 500ms
                score -= 10
            elif avg_time > 100:  # More than 100ms
                score -= 5

        return max(0, score)

    def _get_optimization_priority(self, recommendations: List[Dict]) -> str:
        """Get optimization priority based on recommendations."""
        high_priority = sum(1 for r in recommendations if r.get("severity") == "high")
        medium_priority = sum(1 for r in recommendations if r.get("severity") == "medium")

        if high_priority > 0:
            return "high"
        elif medium_priority > 2:
            return "medium"
        else:
            return "low"

    def save_metrics(self, filepath: str):
        """Save metrics to file."""
        try:
            data = {
                "system_metrics": [self._metric_to_dict(m) for m in self.system_metrics],
                "model_metrics": [self._metric_to_dict(m) for m in self.model_metrics],
                "training_metrics": [self._metric_to_dict(m) for m in self.training_metrics],
                "recommendations": self.recommendations,
                "summary": {
                    "total_inference_requests": self.total_inference_requests,
                    "total_training_steps": self.total_training_steps,
                    "uptime_hours": (time.time() - self.start_time) / 3600,
                },
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Metrics saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def _metric_to_dict(self, metric) -> Dict:
        """Convert metric object to dictionary."""
        return {k: v for k, v in metric.__dict__.items() if not k.startswith("_")}

    def load_metrics(self, filepath: str):
        """Load metrics from file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Reconstruct metrics objects
            self.system_metrics = deque(
                [SystemMetrics(**m) for m in data.get("system_metrics", [])],
                maxlen=self.history_size,
            )
            self.model_metrics = deque(
                [ModelMetrics(**m) for m in data.get("model_metrics", [])], maxlen=self.history_size
            )
            self.training_metrics = deque(
                [TrainingMetrics(**m) for m in data.get("training_metrics", [])],
                maxlen=self.history_size,
            )
            self.recommendations = data.get("recommendations", [])

            logger.info(f"Metrics loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def start_monitoring():
    """Start global performance monitoring."""
    profiler = get_profiler()
    profiler.start_monitoring()


def stop_monitoring():
    """Stop global performance monitoring."""
    profiler = get_profiler()
    profiler.stop_monitoring()


def record_inference(**kwargs):
    """Record inference metrics using global profiler."""
    profiler = get_profiler()
    profiler.record_inference(**kwargs)


def record_training(**kwargs):
    """Record training metrics using global profiler."""
    profiler = get_profiler()
    profiler.record_training(**kwargs)


def get_performance_report() -> Dict[str, Any]:
    """Get performance report using global profiler."""
    profiler = get_profiler()
    return profiler.generate_optimization_report()
