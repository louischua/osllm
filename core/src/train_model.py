#!/usr/bin/env python3
# Copyright (C) 2024 Louis Chua Bean Chong
#
# This file is part of OpenLLM.
#
# OpenLLM is dual-licensed:
# 1. For open source use: GNU General Public License v3.0
# 2. For commercial use: Commercial License (contact for details)
#
# See LICENSE and docs/LICENSES.md for full license information.

"""
Language Model Training Script

This script implements the complete training pipeline for GPT-style language models.
It includes optimization, checkpointing, progress monitoring, and CPU-optimized training
for limited hardware environments.

FEATURES:
- CPU-optimized training with memory management
- Gradient accumulation for effective large batch sizes
- Learning rate scheduling with warmup
- Model checkpointing and resume capability
- Real-time monitoring of loss, perplexity, and speed
- Memory usage tracking and optimization
- Automatic mixed precision (if available)

HARDWARE OPTIMIZATION:
- Designed for 8GB RAM systems
- Efficient CPU training with PyTorch optimizations
- Gradient accumulation to simulate larger batches
- Memory cleanup and garbage collection
- Progress saving for long training runs

Usage:
    python core/src/train_model.py \\
        --model-size small \\
        --data-file data/clean/training_data.txt \\
        --tokenizer-dir data/tokenizer/ \\
        --output-dir models/my-model/ \\
        --max-steps 10000

Requirements:
    - PyTorch
    - SentencePiece
    - Our model architecture and data loader

Author: Louis Chua Bean Chong
License: GPLv3
"""

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import our modules
try:
    from data_loader import TextDataLoader
    from model import GPTModel, create_model
except ImportError:
    import sys

    sys.path.append(os.path.dirname(__file__))
    from data_loader import TextDataLoader
    from model import GPTModel, create_model


class ModelTrainer:
    """
    Comprehensive trainer for GPT-style language models.

    Handles the complete training pipeline including data loading, optimization,
    checkpointing, and progress monitoring.
    """

    def __init__(
        self,
        model: GPTModel,
        data_loader: TextDataLoader,
        output_dir: str,
        device: str = "cpu",
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        gradient_accumulation_steps: int = 4,
        gradient_clipping: float = 1.0,
        save_every: int = 1000,
        eval_every: int = 500,
        log_every: int = 100,
    ):
        """
        Initialize the model trainer.

        Args:
            model: GPT model to train
            data_loader: Data loader for training data
            output_dir: Directory to save checkpoints and logs
            device: Training device ("cpu" or "cuda")
            learning_rate: Peak learning rate
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps for learning rate
            max_steps: Maximum training steps
            gradient_accumulation_steps: Steps to accumulate gradients
            gradient_clipping: Maximum gradient norm
            save_every: Save checkpoint every N steps
            eval_every: Evaluate model every N steps
            log_every: Log progress every N steps
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.device = device

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping

        # Logging and saving
        self.save_every = save_every
        self.eval_every = eval_every
        self.log_every = log_every

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float("in")
        self.training_log = []

        # Performance tracking
        self.start_time = None
        self.step_times = []

        print("🚀 ModelTrainer initialized")
        print(f"  Device: {device}")
        print(f"  Model parameters: {model.get_num_params():,}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Max steps: {max_steps:,}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Output directory: {output_dir}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Don't apply weight decay to biases and layer norm parameters
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use AdamW with lower memory usage for CPU
        optimizer = optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.95),  # Slightly different from default for LLM training
            eps=1e-8,
        )

        return optimizer

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with warmup and cosine decay."""
        if self.warmup_steps > 0:
            # Use a custom scheduler to avoid deprecation warnings
            # This implements warmup + cosine decay without SequentialLR
            class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
                def __init__(self, optimizer, warmup_steps, max_steps, min_lr_factor=0.1):
                    self.warmup_steps = warmup_steps
                    self.max_steps = max_steps
                    self.min_lr_factor = min_lr_factor
                    super().__init__(optimizer)

                def get_lr(self):
                    if self.last_epoch < self.warmup_steps:
                        # Linear warmup
                        factor = self.last_epoch / self.warmup_steps
                        return [base_lr * (0.01 + 0.99 * factor) for base_lr in self.base_lrs]
                    else:
                        # Cosine decay
                        progress = (self.last_epoch - self.warmup_steps) / (
                            self.max_steps - self.warmup_steps
                        )
                        progress = min(progress, 1.0)  # Clamp to 1.0
                        factor = 0.5 * (1 + math.cos(math.pi * progress))
                        factor = self.min_lr_factor + (1 - self.min_lr_factor) * factor
                        return [base_lr * factor for base_lr in self.base_lrs]

            scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_steps=self.warmup_steps,
                max_steps=self.max_steps,
                min_lr_factor=0.1,
            )
        else:
            # Just cosine decay - this should not trigger warnings
            scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.max_steps, eta_min=self.learning_rate * 0.1
            )

        return scheduler

    def _calculate_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate cross-entropy loss for autoregressive language modeling.

        This method computes the standard cross-entropy loss used in language model training.
        The loss measures how well the model predicts the next token in the sequence.

        Mathematical formulation:
            Loss = -∑ log(P(target_token | context))
            where P is the softmax probability distribution over vocabulary

        Implementation details:
            - Reshapes 3D tensors to 2D for efficient computation
            - Uses PyTorch's optimized cross_entropy function
            - Handles padding tokens by ignoring them in loss calculation
            - Computes mean loss across all valid positions

        Why cross-entropy for language modeling:
            - Natural choice for multi-class classification (next token prediction)
            - Provides strong gradient signal for correct token probabilities
            - Mathematically equivalent to minimizing negative log-likelihood
            - Well-studied optimization properties for neural language models

        Args:
            logits: Raw model predictions of shape (batch_size, seq_len, vocab_size)
                   Contains unnormalized scores for each token in vocabulary
                   These will be converted to probabilities via softmax internally
            targets: Ground truth next tokens of shape (batch_size, seq_len)
                    Contains token IDs representing the true next tokens
                    Should be input sequence shifted by one position

        Returns:
            torch.Tensor: Scalar loss value representing prediction error
                         Lower values indicate better next-token prediction accuracy
        """
        # Reshape tensors from 3D to 2D for efficient loss computation
        # This converts per-sequence per-position predictions to a flat structure
        # where each row represents one prediction over the entire vocabulary
        logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
        targets = targets.view(-1)  # (batch_size * seq_len,)

        # Calculate cross-entropy loss with proper handling of special tokens
        # ignore_index=-1 excludes padding tokens from loss calculation
        # This prevents the model from learning to predict padding, which would skew training
        # The function internally applies softmax to logits and computes negative log-likelihood
        loss = nn.functional.cross_entropy(logits, targets, ignore_index=-1)

        # Return scalar loss for backpropagation
        # This loss will be used to compute gradients via automatic differentiation
        return loss

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_stats = {}

        if torch.cuda.is_available() and self.device.startswith("cuda"):
            memory_stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
            memory_stats["gpu_cached_mb"] = torch.cuda.memory_reserved() / (1024**2)

        # Estimate CPU memory (approximate)
        import psutil

        process = psutil.Process()
        memory_stats["cpu_memory_mb"] = process.memory_info().rss / (1024**2)

        return memory_stats

    def _log_step(self, step: int, loss: float, lr: float, step_time: float) -> None:
        """Log training progress for a single step."""
        perplexity = math.exp(min(loss, 10))  # Cap at exp(10) to avoid overflow

        # Calculate tokens per second
        tokens_per_batch = self.data_loader.batch_size * self.data_loader.seq_len
        tokens_per_second = tokens_per_batch / step_time if step_time > 0 else 0

        # Get memory usage
        memory_stats = self._get_memory_usage()

        # Create log entry
        log_entry = {
            "step": step,
            "loss": loss,
            "perplexity": perplexity,
            "learning_rate": lr,
            "step_time": step_time,
            "tokens_per_second": tokens_per_second,
            "memory_mb": memory_stats.get("cpu_memory_mb", 0),
        }

        self.training_log.append(log_entry)

        # Print progress
        _ = time.time() - self.start_time if self.start_time else 0
        eta_seconds = (self.max_steps - step) * step_time if step_time > 0 else 0
        eta_hours = eta_seconds / 3600

        print(
            f"Step {step:,}/{self.max_steps:,} | "
            f"Loss: {loss:.4f} | "
            f"PPL: {perplexity:.2f} | "
            f"LR: {lr:.2e} | "
            f"Time: {step_time:.2f}s | "
            f"Tokens/s: {tokens_per_second:.1f} | "
            f"Memory: {memory_stats.get('cpu_memory_mb', 0):.0f}MB | "
            f"ETA: {eta_hours:.1f}h"
        )

    def _save_checkpoint(self, step: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "step": step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "training_log": self.training_log,
            "config": self.model.config.__dict__,
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved: {best_path}")

        # Save training log
        log_path = self.output_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)

        print(f"💾 Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint to resume training."""
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return

        print(f"📂 Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.training_log = checkpoint.get("training_log", [])

        print("✓ Checkpoint loaded successfully")
        print(f"  Resuming from step: {self.step:,}")
        print(f"  Best loss so far: {self.best_loss:.4f}")

    def train(self) -> None:
        """Main training loop."""
        print("\n🚀 Starting training...")
        print(f"  Model: {self.model.config.model_name}")
        print(f"  Parameters: {self.model.get_num_params():,}")
        print(f"  Device: {self.device}")
        print(f"  Max steps: {self.max_steps:,}")
        print("=" * 80)

        self.model.train()
        self.start_time = time.time()

        # Initialize gradient accumulation
        accumulated_loss = 0.0
        self.optimizer.zero_grad()

        for batch_idx, (input_ids, target_ids) in enumerate(self.data_loader):
            if self.step >= self.max_steps:
                break

            step_start_time = time.time()

            # Move batch to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward pass (model computes loss internally when targets provided)
            logits, loss = self.model(input_ids, target_ids)

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            accumulated_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

                # Update parameters
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Update step count
                self.step += 1
                step_time = time.time() - step_start_time
                self.step_times.append(step_time)

                # Get current learning rate
                current_lr = self.scheduler.get_last_lr()[0]

                # Log progress
                if self.step % self.log_every == 0:
                    avg_loss = accumulated_loss
                    self._log_step(self.step, avg_loss, current_lr, step_time)

                # Save checkpoint
                if self.step % self.save_every == 0:
                    is_best = accumulated_loss < self.best_loss
                    if is_best:
                        self.best_loss = accumulated_loss

                    self._save_checkpoint(self.step, is_best)

                # Clean up memory periodically
                if self.step % 100 == 0:
                    gc.collect()

                # Reset accumulated loss
                accumulated_loss = 0.0

                # Check if training complete
                if self.step >= self.max_steps:
                    break

        # Final checkpoint
        print("\n🎉 Training completed!")
        self._save_checkpoint(self.step, is_best=True)

        # Training summary
        total_time = time.time() - self.start_time
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0

        print("\n📊 Training Summary:")
        print(f"  Steps completed: {self.step:,}")
        print(f"  Total time: {total_time/3600:.2f} hours")
        print(f"  Average time per step: {avg_step_time:.2f}s")
        print(f"  Final loss: {self.best_loss:.4f}")
        print(f"  Final perplexity: {math.exp(min(self.best_loss, 10)):.2f}")
        print(f"  Model saved to: {self.output_dir}")


def main():
    """Main function to handle command line training."""
    parser = argparse.ArgumentParser(
        description="Train a GPT-style language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train small model for quick experimentation
  python core/src/train_model.py \\
    --model-size small \\
    --max-steps 5000 \\
    --output-dir models/test-small

  # Train medium model with custom settings
  python core/src/train_model.py \\
    --model-size medium \\
    --learning-rate 1e-4 \\
    --batch-size 2 \\
    --max-steps 50000 \\
    --output-dir models/my-medium-model
        """,
    )

    # Model and data arguments
    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "large"],
        default="small",
        help="Model size to train (default: small)",
    )

    parser.add_argument(
        "--data-file",
        default="data/clean/training_data.txt",
        help="Path to training text file (default: data/clean/training_data.txt)",
    )

    parser.add_argument(
        "--tokenizer-dir",
        default="data/tokenizer/",
        help="Path to tokenizer directory (default: data/tokenizer/)",
    )

    parser.add_argument(
        "--output-dir", required=True, help="Output directory for model checkpoints"
    )

    # Training hyperparameters
    parser.add_argument(
        "--seq-len", type=int, default=512, help="Sequence length for training (default: 512)"
    )

    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")

    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate (default: 3e-4)"
    )

    parser.add_argument(
        "--max-steps", type=int, default=10000, help="Maximum training steps (default: 10000)"
    )

    parser.add_argument(
        "--warmup-steps", type=int, default=1000, help="Warmup steps (default: 1000)"
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Training device (default: auto)",
    )

    parser.add_argument("--resume", help="Path to checkpoint to resume training from")

    parser.add_argument(
        "--save-every", type=int, default=1000, help="Save checkpoint every N steps (default: 1000)"
    )

    args = parser.parse_args()

    print("🚀 OpenLLM Model Training")
    print("=" * 60)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    try:
        # Create model
        print(f"\n🏗️  Creating {args.model_size} model...")
        model = create_model(args.model_size)

        # Create data loader
        print("\n📊 Setting up data loader...")
        tokenizer_path = os.path.join(args.tokenizer_dir, "tokenizer.model")

        data_loader = TextDataLoader(
            data_file=args.data_file,
            tokenizer_path=tokenizer_path,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            shuffle=True,
        )

        # Get data statistics
        _ = data_loader.get_data_stats()

        # Create trainer
        print("\n🎯 Setting up trainer...")
        trainer = ModelTrainer(
            model=model,
            data_loader=data_loader,
            output_dir=args.output_dir,
            device=device,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_every=args.save_every,
        )

        # Resume from checkpoint if specified
        if args.resume:
            trainer._load_checkpoint(args.resume)

        # Start training
        trainer.train()

        print("\n🎉 Training completed successfully!")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
