#!/usr/bin/env python3
"""
Mixed Precision Training Utilities

This module provides utilities for mixed precision training using PyTorch's
automatic mixed precision (AMP) to improve training speed and reduce memory usage.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Callable


class MixedPrecisionTrainer:
    """
    Mixed precision training wrapper for improved performance.

    This class provides automatic mixed precision training capabilities
    that can significantly improve training speed and reduce memory usage
    on compatible hardware (especially NVIDIA GPUs with Tensor Cores).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        enabled: bool = True,
    ):
        """
        Initialize mixed precision trainer.

        Args:
            model: The model to train
            optimizer: The optimizer to use
            device: Device to use ("auto", "cpu", "cuda")
            dtype: Precision dtype (float16, bfloat16)
            enabled: Whether to enable mixed precision
        """
        self.model = model
        self.optimizer = optimizer
        self.device = self._get_device(device)
        self.dtype = dtype
        self.enabled = enabled and self.device.type == "cuda"

        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if self.enabled else None

        # Move model to device
        self.model.to(self.device)

        print(f"Mixed Precision Training: {'Enabled' if self.enabled else 'Disabled'}")
        print(f"Device: {self.device}")
        print(f"Precision: {self.dtype}")

    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)

    def train_step(
        self, batch: torch.Tensor, targets: torch.Tensor, loss_fn: Optional[Callable] = None
    ) -> dict:
        """
        Perform a single training step with mixed precision.

        Args:
            batch: Input batch
            targets: Target batch
            loss_fn: Optional custom loss function

        Returns:
            dict: Training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move data to device
        batch = batch.to(self.device)
        targets = targets.to(self.device)

        if self.enabled:
            # Mixed precision forward pass
            with autocast(dtype=self.dtype):
                if loss_fn is None:
                    # Use model's built-in loss computation
                    logits, loss = self.model(batch, targets)
                else:
                    # Use custom loss function
                    logits = self.model(batch)
                    loss = loss_fn(logits, targets)

            # Scaled backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision training
            if loss_fn is None:
                logits, loss = self.model(batch, targets)
            else:
                logits = self.model(batch)
                loss = loss_fn(logits, targets)

            loss.backward()
            self.optimizer.step()

        return {
            "loss": loss.item(),
            "logits": logits,
            "scaler_scale": self.scaler.get_scale() if self.scaler else 1.0,
        }

    def eval_step(
        self, batch: torch.Tensor, targets: torch.Tensor, loss_fn: Optional[Callable] = None
    ) -> dict:
        """
        Perform a single evaluation step.

        Args:
            batch: Input batch
            targets: Target batch
            loss_fn: Optional custom loss function

        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()

        # Move data to device
        batch = batch.to(self.device)
        targets = targets.to(self.device)

        with torch.no_grad():
            if self.enabled:
                with autocast(dtype=self.dtype):
                    if loss_fn is None:
                        logits, loss = self.model(batch, targets)
                    else:
                        logits = self.model(batch)
                        loss = loss_fn(logits, targets)
            else:
                if loss_fn is None:
                    logits, loss = self.model(batch, targets)
                else:
                    logits = self.model(batch)
                    loss = loss_fn(logits, targets)

        return {"loss": loss.item(), "logits": logits}

    def save_checkpoint(self, path: str, **kwargs):
        """Save model checkpoint with mixed precision state."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "dtype": self.dtype,
            "enabled": self.enabled,
            **kwargs,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint with mixed precision state."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return checkpoint


def enable_mixed_precision(
    model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs
) -> MixedPrecisionTrainer:
    """
    Convenience function to enable mixed precision training.

    Args:
        model: The model to train
        optimizer: The optimizer to use
        **kwargs: Additional arguments for MixedPrecisionTrainer

    Returns:
        MixedPrecisionTrainer: Configured trainer
    """
    return MixedPrecisionTrainer(model, optimizer, **kwargs)


def get_optimal_dtype() -> torch.dtype:
    """
    Get the optimal dtype for mixed precision training.

    Returns:
        torch.dtype: Optimal dtype (bfloat16 for newer GPUs, float16 for older)
    """
    if torch.cuda.is_available():
        # Check if bfloat16 is supported
        if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
            return torch.bfloat16
        else:
            return torch.float16
    else:
        return torch.float32
