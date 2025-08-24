#!/usr/bin/env python3
"""
OpenLLM Performance Optimization Script

This script provides a comprehensive interface for optimizing OpenLLM performance,
integrating all optimization techniques including model architecture, training pipeline,
inference server, and system-level optimizations.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import argparse
import sys
import os
import time
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Add core/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core", "src"))

from model import GPTConfig, GPTModel
from mixed_precision import MixedPrecisionTrainer, get_optimal_dtype
from quantization import QuantizedModel, quantize_model_dynamic
from optimized_data_loader import OptimizedDataLoader, create_optimized_loader
from performance_monitor import PerformanceProfiler, get_profiler, start_monitoring, stop_monitoring
from optimized_inference_server import OptimizedInferenceEngine


class PerformanceOptimizer:
    """
    Comprehensive performance optimizer for OpenLLM.

    This class integrates all optimization techniques and provides
    a unified interface for performance optimization.
    """

    def __init__(self, model_path: str, device: str = "auto", enable_monitoring: bool = True):
        """
        Initialize performance optimizer.

        Args:
            model_path: Path to the model
            device: Device to use
            enable_monitoring: Whether to enable performance monitoring
        """
        self.model_path = model_path
        self.device = device
        self.enable_monitoring = enable_monitoring

        # Initialize components
        self.model = None
        self.mixed_precision_trainer = None
        self.quantized_model = None
        self.inference_engine = None
        self.profiler = None

        # Performance metrics
        self.optimization_results = {}

        print("🚀 OpenLLM Performance Optimizer Initialized")
        print(f"Model Path: {model_path}")
        print(f"Device: {device}")

    def load_model(self) -> bool:
        """Load and prepare the model for optimization."""
        try:
            print("📦 Loading model...")

            # Load model configuration
            config_path = Path(self.model_path) / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config_data = json.load(f)

                # Convert Hugging Face config to GPTConfig format
                config_dict = {
                    "vocab_size": config_data.get("vocab_size", 32000),
                    "n_layer": config_data.get("n_layer", 6),
                    "n_head": config_data.get("n_head", 8),
                    "n_embd": config_data.get("n_embd", 512),
                    "block_size": config_data.get("block_size", 1024),
                    "dropout": config_data.get("dropout", 0.1),
                    "bias": config_data.get("bias", True),
                    "model_name": f"gpt-{config_data.get('model_size', 'small')}",
                }
                config = GPTConfig(**config_dict)
            else:
                print("⚠️ No config.json found, using default small config")
                config = GPTConfig.small()

            # Create model with gradient checkpointing
            self.model = GPTModel(config, use_checkpoint=True)

            # Load model weights if available
            model_weights_path = Path(self.model_path) / "pytorch_model.bin"
            if model_weights_path.exists():
                try:
                    # Load weights to CPU first, then move to device
                    weights = torch.load(model_weights_path, map_location="cpu")
                    self.model.load_state_dict(weights)
                    print("✅ Model weights loaded")
                except Exception as e:
                    print(f"⚠️ Failed to load weights: {e}, using initialized weights")
            else:
                print("⚠️ No model weights found, using initialized weights")

            # Move to device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(self.device)

            self.model.to(self.device)
            self.model.eval()

            print(f"✅ Model loaded successfully")
            print(f"   Parameters: {self.model.get_num_params():,}")
            print(f"   Device: {self.device}")

            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    def apply_mixed_precision(self, optimizer=None) -> bool:
        """Apply mixed precision optimization."""
        try:
            print("🔧 Applying mixed precision optimization...")

            if optimizer is None:
                # Create a dummy optimizer for demonstration
                import torch.optim as optim

                optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)

            # Get optimal dtype
            dtype = get_optimal_dtype()

            # Create mixed precision trainer
            self.mixed_precision_trainer = MixedPrecisionTrainer(
                model=self.model, optimizer=optimizer, device=self.device, dtype=dtype, enabled=True
            )

            print(f"✅ Mixed precision enabled with dtype: {dtype}")
            self.optimization_results["mixed_precision"] = {
                "enabled": True,
                "dtype": str(dtype),
                "device": str(self.device),
            }

            return True

        except Exception as e:
            print(f"❌ Failed to apply mixed precision: {e}")
            return False

    def apply_quantization(self) -> bool:
        """Apply model quantization."""
        try:
            print("🔧 Applying model quantization...")

            # Create quantized model
            self.quantized_model = QuantizedModel(self.model)

            # Apply dynamic quantization
            self.quantized_model.quantize_dynamic()

            # Get memory usage comparison
            memory_usage = self.quantized_model.get_memory_usage()

            print(f"✅ Quantization completed")
            print(f"   Original size: {memory_usage['original_mb']:.1f} MB")
            print(f"   Quantized size: {memory_usage['quantized_mb']:.1f} MB")
            print(f"   Compression ratio: {memory_usage['compression_ratio']:.2f}x")

            self.optimization_results["quantization"] = {
                "enabled": True,
                "original_mb": memory_usage["original_mb"],
                "quantized_mb": memory_usage["quantized_mb"],
                "compression_ratio": memory_usage["compression_ratio"],
            }

            return True

        except Exception as e:
            print(f"❌ Failed to apply quantization: {e}")
            return False

    def create_optimized_inference_engine(self) -> bool:
        """Create optimized inference engine."""
        try:
            print("🔧 Creating optimized inference engine...")

            self.inference_engine = OptimizedInferenceEngine(
                model_path=self.model_path,
                device=self.device,
                use_quantization=True,
                cache_size=1000,
                max_batch_size=32,
                num_workers=4,
            )

            print("✅ Optimized inference engine created")
            print(f"   Cache size: 1000")
            print(f"   Max batch size: 32")
            print(f"   Workers: 4")

            self.optimization_results["inference_engine"] = {
                "enabled": True,
                "cache_size": 1000,
                "max_batch_size": 32,
                "num_workers": 4,
            }

            return True

        except Exception as e:
            print(f"❌ Failed to create inference engine: {e}")
            return False

    def start_performance_monitoring(self) -> bool:
        """Start performance monitoring."""
        try:
            if not self.enable_monitoring:
                print("⚠️ Performance monitoring disabled")
                return True

            print("📊 Starting performance monitoring...")

            # Initialize profiler
            self.profiler = PerformanceProfiler(
                history_size=1000, monitoring_interval=1.0, enable_gpu_monitoring=True
            )

            # Start monitoring
            self.profiler.start_monitoring()

            print("✅ Performance monitoring started")
            self.optimization_results["monitoring"] = {
                "enabled": True,
                "history_size": 1000,
                "monitoring_interval": 1.0,
            }

            return True

        except Exception as e:
            print(f"❌ Failed to start monitoring: {e}")
            return False

    def benchmark_performance(self, num_runs: int = 10) -> Dict[str, Any]:
        """Benchmark model performance."""
        try:
            print(f"🏃 Benchmarking performance ({num_runs} runs)...")

            if self.model is None:
                print("❌ No model loaded for benchmarking")
                return {}

            # Prepare test data
            import torch

            test_input = torch.randint(0, 1000, (1, 64)).to(self.device)

            # Benchmark original model
            self.model.eval()
            start_time = time.time()

            with torch.no_grad():
                for _ in range(num_runs):
                    _ = self.model(test_input)

            original_time = (time.time() - start_time) / num_runs * 1000  # Convert to ms

            # Benchmark quantized model if available
            quantized_time = None
            if self.quantized_model and self.quantized_model.is_quantized:
                start_time = time.time()

                with torch.no_grad():
                    for _ in range(num_runs):
                        _ = self.quantized_model.quantized_model(test_input)

                quantized_time = (time.time() - start_time) / num_runs * 1000

            # Calculate speedup
            speedup = original_time / quantized_time if quantized_time else 1.0

            results = {
                "original_time_ms": original_time,
                "quantized_time_ms": quantized_time,
                "speedup": speedup,
                "num_runs": num_runs,
            }

            print(f"✅ Benchmark completed")
            print(f"   Original time: {original_time:.2f} ms")
            if quantized_time:
                print(f"   Quantized time: {quantized_time:.2f} ms")
                print(f"   Speedup: {speedup:.2f}x")

            self.optimization_results["benchmark"] = results

            return results

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return {}

    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        print("📋 Generating optimization report...")

        # Get system information
        import psutil

        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "platform": sys.platform,
            "python_version": sys.version,
        }

        # Get performance metrics if monitoring is active
        performance_metrics = {}
        if self.profiler:
            performance_metrics = self.profiler.generate_optimization_report()

        # Compile final report
        report = {
            "timestamp": time.time(),
            "model_path": self.model_path,
            "device": self.device,
            "system_info": system_info,
            "optimization_results": self.optimization_results,
            "performance_metrics": performance_metrics,
            "recommendations": self._generate_recommendations(),
        }

        print("✅ Optimization report generated")
        return report

    def _generate_recommendations(self) -> list:
        """Generate optimization recommendations."""
        recommendations = []

        # Check if optimizations were applied
        if not self.optimization_results.get("mixed_precision", {}).get("enabled"):
            recommendations.append(
                {
                    "type": "mixed_precision",
                    "priority": "high",
                    "message": "Enable mixed precision training for better performance",
                    "impact": "2-3x training speed improvement",
                }
            )

        if not self.optimization_results.get("quantization", {}).get("enabled"):
            recommendations.append(
                {
                    "type": "quantization",
                    "priority": "medium",
                    "message": "Apply model quantization for reduced memory usage",
                    "impact": "30-50% memory reduction",
                }
            )

        if not self.optimization_results.get("inference_engine", {}).get("enabled"):
            recommendations.append(
                {
                    "type": "inference_engine",
                    "priority": "medium",
                    "message": "Use optimized inference engine for better throughput",
                    "impact": "3-5x inference speed improvement",
                }
            )

        # System-specific recommendations
        import psutil

        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            recommendations.append(
                {
                    "type": "system",
                    "priority": "high",
                    "message": f"Low memory system ({memory_gb:.1f}GB) - consider using smaller models",
                    "impact": "Prevent out-of-memory errors",
                }
            )

        return recommendations

    def save_report(self, filepath: str):
        """Save optimization report to file."""
        try:
            report = self.generate_optimization_report()

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)

            print(f"✅ Report saved to: {filepath}")

        except Exception as e:
            print(f"❌ Failed to save report: {e}")

    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up resources...")

        if self.profiler:
            self.profiler.stop_monitoring()

        if self.inference_engine:
            self.inference_engine.cleanup()

        print("✅ Cleanup completed")


def main():
    """Main function for performance optimization."""
    parser = argparse.ArgumentParser(description="OpenLLM Performance Optimizer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument(
        "--enable_monitoring", action="store_true", help="Enable performance monitoring"
    )
    parser.add_argument("--benchmark_runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--output_report", type=str, help="Output report file path")
    parser.add_argument("--apply_all", action="store_true", help="Apply all optimizations")

    args = parser.parse_args()

    # Create optimizer
    optimizer = PerformanceOptimizer(
        model_path=args.model_path, device=args.device, enable_monitoring=args.enable_monitoring
    )

    try:
        # Load model
        if not optimizer.load_model():
            return 1

        # Apply optimizations
        if args.apply_all:
            print("\n🔧 Applying all optimizations...")

            # Mixed precision
            optimizer.apply_mixed_precision()

            # Quantization
            optimizer.apply_quantization()

            # Inference engine
            optimizer.create_optimized_inference_engine()

            # Performance monitoring
            optimizer.start_performance_monitoring()

        # Benchmark performance
        optimizer.benchmark_performance(num_runs=args.benchmark_runs)

        # Generate and save report
        if args.output_report:
            optimizer.save_report(args.output_report)
        else:
            # Print summary
            report = optimizer.generate_optimization_report()
            print("\n📊 Optimization Summary:")
            print(f"   Model: {report['model_path']}")
            print(f"   Device: {report['device']}")
            print(
                f"   System: {report['system_info']['cpu_count']} CPUs, {report['system_info']['memory_gb']:.1f}GB RAM"
            )

            if "benchmark" in report["optimization_results"]:
                benchmark = report["optimization_results"]["benchmark"]
                print(f"   Performance: {benchmark['original_time_ms']:.2f}ms per inference")
                if benchmark["speedup"] > 1.0:
                    print(f"   Speedup: {benchmark['speedup']:.2f}x")

            if report["recommendations"]:
                print(f"\n💡 Recommendations:")
                for rec in report["recommendations"]:
                    print(f"   - {rec['message']} ({rec['impact']})")

        print("\n✅ Performance optimization completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        return 1
    finally:
        optimizer.cleanup()


if __name__ == "__main__":
    sys.exit(main())
