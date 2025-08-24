#!/usr/bin/env python3
"""
Model Quantization Utilities

This module provides utilities for model quantization to reduce memory usage
and improve inference speed while maintaining reasonable accuracy.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
from typing import Optional, Dict, Any
import copy


class QuantizedModel:
    """
    Wrapper for quantized models with easy conversion and inference.

    This class provides utilities for converting models to quantized versions
    and performing efficient inference with reduced memory usage.
    """

    def __init__(self, model: nn.Module, quantized_model: Optional[nn.Module] = None):
        """
        Initialize quantized model wrapper.

        Args:
            model: Original model
            quantized_model: Pre-quantized model (optional)
        """
        self.original_model = model
        self.quantized_model = quantized_model
        self.is_quantized = quantized_model is not None

    def quantize_dynamic(
        self, qconfig_spec: Optional[Dict] = None, dtype: torch.dtype = torch.qint8
    ) -> "QuantizedModel":
        """
        Perform dynamic quantization on the model.

        Args:
            qconfig_spec: Quantization configuration
            dtype: Quantization dtype (qint8, quint8)

        Returns:
            QuantizedModel: Self with quantized model
        """
        if qconfig_spec is None:
            qconfig_spec = {
                nn.Linear: quantization.default_dynamic_qconfig,
                nn.LSTM: quantization.default_dynamic_qconfig,
                nn.LSTMCell: quantization.default_dynamic_qconfig,
                nn.RNNCell: quantization.default_dynamic_qconfig,
                nn.GRUCell: quantization.default_dynamic_qconfig,
            }

        # Create a copy of the model for quantization
        model_copy = copy.deepcopy(self.original_model)
        model_copy.eval()

        # Prepare model for quantization
        model_prepared = quantization.prepare_dynamic(model_copy, qconfig_spec)

        # Convert to quantized model
        self.quantized_model = quantization.convert(model_prepared)
        self.is_quantized = True

        print(f"Dynamic quantization completed with dtype: {dtype}")
        return self

    def quantize_static(
        self,
        calibration_data: torch.utils.data.DataLoader,
        qconfig: Optional[quantization.QConfig] = None,
    ) -> "QuantizedModel":
        """
        Perform static quantization on the model.

        Args:
            calibration_data: DataLoader for calibration
            qconfig: Quantization configuration

        Returns:
            QuantizedModel: Self with quantized model
        """
        if qconfig is None:
            qconfig = quantization.get_default_qconfig("fbgemm")

        # Create a copy of the model for quantization
        model_copy = copy.deepcopy(self.original_model)
        model_copy.eval()

        # Prepare model for quantization
        model_prepared = quantization.prepare(model_copy, qconfig)

        # Calibrate the model
        print("Calibrating model...")
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_data):
                if batch_idx >= 100:  # Limit calibration samples
                    break
                model_prepared(data)

        # Convert to quantized model
        self.quantized_model = quantization.convert(model_prepared)
        self.is_quantized = True

        print("Static quantization completed")
        return self

    def forward(self, *args, **kwargs):
        """Forward pass using quantized model if available."""
        if self.is_quantized and self.quantized_model is not None:
            return self.quantized_model(*args, **kwargs)
        else:
            return self.original_model(*args, **kwargs)

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage comparison between original and quantized models.

        Returns:
            dict: Memory usage in MB
        """

        def get_model_size(model):
            param_size = 0
            buffer_size = 0

            for param in model.parameters():
                param_size += param.nelement() * param.element_size()

            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            return (param_size + buffer_size) / (1024 * 1024)  # Convert to MB

        original_size = get_model_size(self.original_model)
        quantized_size = (
            get_model_size(self.quantized_model) if self.quantized_model else original_size
        )

        return {
            "original_mb": original_size,
            "quantized_mb": quantized_size,
            "compression_ratio": original_size / quantized_size if quantized_size > 0 else 1.0,
        }

    def save_quantized(self, path: str):
        """Save quantized model."""
        if self.quantized_model is not None:
            torch.save(self.quantized_model.state_dict(), path)
            print(f"Quantized model saved to: {path}")
        else:
            raise ValueError("No quantized model available")

    def load_quantized(self, path: str):
        """Load quantized model."""
        self.quantized_model.load_state_dict(torch.load(path))
        self.is_quantized = True
        print(f"Quantized model loaded from: {path}")


def quantize_model_dynamic(model: nn.Module, dtype: torch.dtype = torch.qint8) -> QuantizedModel:
    """
    Convenience function for dynamic quantization.

    Args:
        model: Model to quantize
        dtype: Quantization dtype

    Returns:
        QuantizedModel: Quantized model wrapper
    """
    quantized = QuantizedModel(model)
    return quantized.quantize_dynamic(dtype=dtype)


def quantize_model_static(
    model: nn.Module,
    calibration_data: torch.utils.data.DataLoader,
    qconfig: Optional[quantization.QConfig] = None,
) -> QuantizedModel:
    """
    Convenience function for static quantization.

    Args:
        model: Model to quantize
        calibration_data: Data for calibration
        qconfig: Quantization configuration

    Returns:
        QuantizedModel: Quantized model wrapper
    """
    quantized = QuantizedModel(model)
    return quantized.quantize_static(calibration_data, qconfig)


def create_quantization_config(
    backend: str = "fbgemm", dtype: torch.dtype = torch.qint8
) -> quantization.QConfig:
    """
    Create quantization configuration.

    Args:
        backend: Quantization backend ('fbgemm', 'qnnpack')
        dtype: Quantization dtype

    Returns:
        QConfig: Quantization configuration
    """
    if backend == "fbgemm":
        return quantization.QConfig(
            activation=quantization.default_observer,
            weight=quantization.default_per_channel_weight_observer,
        )
    elif backend == "qnnpack":
        return quantization.QConfig(
            activation=quantization.default_observer, weight=quantization.default_weight_observer
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def benchmark_quantization(
    original_model: nn.Module,
    quantized_model: QuantizedModel,
    test_data: torch.Tensor,
    num_runs: int = 100,
) -> Dict[str, float]:
    """
    Benchmark original vs quantized model performance.

    Args:
        original_model: Original model
        quantized_model: Quantized model
        test_data: Test data for benchmarking
        num_runs: Number of runs for averaging

    Returns:
        dict: Performance metrics
    """
    original_model.eval()
    quantized_model.quantized_model.eval()

    # Benchmark original model
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    if start_time:
        start_time.record()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = original_model(test_data)

    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        original_time = start_time.elapsed_time(end_time) / num_runs
    else:
        import time

        start = time.time()
        for _ in range(num_runs):
            _ = original_model(test_data)
        original_time = (time.time() - start) * 1000 / num_runs  # Convert to ms

    # Benchmark quantized model
    if start_time:
        start_time.record()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = quantized_model.quantized_model(test_data)

    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        quantized_time = start_time.elapsed_time(end_time) / num_runs
    else:
        start = time.time()
        for _ in range(num_runs):
            _ = quantized_model.quantized_model(test_data)
        quantized_time = (time.time() - start) * 1000 / num_runs  # Convert to ms

    return {
        "original_time_ms": original_time,
        "quantized_time_ms": quantized_time,
        "speedup": original_time / quantized_time if quantized_time > 0 else 1.0,
    }
