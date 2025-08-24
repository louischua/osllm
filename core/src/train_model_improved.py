#!/usr/bin/env python3
"""
Improved Language Model Training Script

This script ensures proper checkpoint saving with full metadata like the 9k model.
It includes best checkpoint saving, training logs, and proper model state management.

Key Improvements:
- Always saves full checkpoints with metadata
- Implements best checkpoint saving
- Includes training logs and monitoring
- Proper early stopping mechanism
- Validation monitoring (if validation data available)

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
from typing import Dict, Optional

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


class ImprovedTrainingConfig:
    """Enhanced configuration for model training parameters."""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        max_steps: int = 100000,
        warmup_steps: int = 10000,
        gradient_clipping: float = 1.0,
        weight_decay: float = 0.01,
        mixed_precision: bool = True,
        gradient_checkpointing: bool = True,
        save_every: int = 1000,
        eval_every: int = 500,
        early_stopping_patience: int = 5,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.gradient_clipping = gradient_clipping
        self.weight_decay = weight_decay
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.save_every = save_every
        self.eval_every = eval_every
        self.early_stopping_patience = early_stopping_patience


class ImprovedModelTrainer:
    """
    Enhanced trainer for GPT-style language models with proper checkpoint saving.

    Ensures all checkpoints include full metadata like the 9k model.
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
        early_stopping_patience: int = 5,
        validation_data_loader: Optional[TextDataLoader] = None,
    ):
        self.model = model
        self.data_loader = data_loader
        self.validation_data_loader = validation_data_loader
        self.output_dir = Path(output_dir)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        self.save_every = save_every
        self.eval_every = eval_every
        self.log_every = log_every
        self.early_stopping_patience = early_stopping_patience

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_validation_loss = float('inf')
        self.training_log = []
        self.validation_log = []
        self.step_times = []
        self.start_time = None
        self.no_improvement_count = 0

        # Move model to device
        self.model.to(device)

        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps,
            eta_min=learning_rate * 0.1
        )

        # Enable gradient checkpointing if requested
        if hasattr(self.model, 'use_checkpoint'):
            self.model.use_checkpoint = True

        print(f"✅ Trainer initialized with {self.model.get_num_params():,} parameters")

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if torch.cuda.is_available():
            return {
                "gpu_memory_mb": torch.cuda.memory_allocated() / 1024**2,
                "gpu_memory_cached_mb": torch.cuda.memory_reserved() / 1024**2,
            }
        else:
            import psutil
            process = psutil.Process()
            return {
                "cpu_memory_mb": process.memory_info().rss / 1024**2,
            }

    def _log_step(self, step: int, loss: float, lr: float, step_time: float, validation_loss: Optional[float] = None) -> None:
        """Log training step with comprehensive information."""
        # Calculate perplexity (cap at exp(10) to avoid overflow)
        perplexity = math.exp(min(loss, 10))

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

        if validation_loss is not None:
            log_entry["validation_loss"] = validation_loss
            log_entry["validation_perplexity"] = math.exp(min(validation_loss, 10))

        self.training_log.append(log_entry)

        # Print progress
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        eta_seconds = (self.max_steps - step) * step_time if step_time > 0 else 0
        eta_hours = eta_seconds / 3600

        print(
            f"Step {step:,}/{self.max_steps:,} | "
            f"Loss: {loss:.4f} | "
            f"PPL: {perplexity:.2f} | "
            f"LR: {lr:.2e} | "
            f"Time: {step_time:.2f}s | "
            f"ETA: {eta_hours:.1f}h"
        )

        if validation_loss is not None:
            print(f"  Val Loss: {validation_loss:.4f} | Val PPL: {math.exp(min(validation_loss, 10)):.2f}")

    def _evaluate_model(self) -> float:
        """Evaluate model on validation set if available."""
        if self.validation_data_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for input_ids, target_ids in self.validation_data_loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                logits, loss = self.model(input_ids, target_ids)
                total_loss += loss.item()
                num_batches += 1

                # Limit evaluation to avoid taking too long
                if num_batches >= 10:
                    break

        self.model.train()
        return total_loss / num_batches if num_batches > 0 else None

    def _save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False) -> None:
        """Save comprehensive model checkpoint with full metadata."""
        
        # Create comprehensive checkpoint with all metadata
        checkpoint = {
            "step": step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "best_validation_loss": self.best_validation_loss,
            "training_log": self.training_log,
            "validation_log": self.validation_log,
            "config": self.model.config.__dict__,
            "training_config": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_steps": self.warmup_steps,
                "max_steps": self.max_steps,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "gradient_clipping": self.gradient_clipping,
                "save_every": self.save_every,
                "eval_every": self.eval_every,
            },
            "model_info": {
                "model_name": self.model.config.model_name,
                "parameters": self.model.get_num_params(),
                "vocab_size": self.model.config.vocab_size,
                "n_layer": self.model.config.n_layer,
                "n_head": self.model.config.n_head,
                "n_embd": self.model.config.n_embd,
                "block_size": self.model.config.block_size,
            },
            "training_stats": {
                "total_time": time.time() - self.start_time if self.start_time else 0,
                "average_step_time": sum(self.step_times) / len(self.step_times) if self.step_times else 0,
                "no_improvement_count": self.no_improvement_count,
            }
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint if this is the best so far
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved: {best_path}")

        # Save final checkpoint if training is complete
        if is_final:
            final_path = self.output_dir / "final_model.pt"
            torch.save(checkpoint, final_path)
            print(f"💾 Final model saved: {final_path}")

        # Save training log as JSON
        log_path = self.output_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)

        # Save validation log if available
        if self.validation_log:
            val_log_path = self.output_dir / "validation_log.json"
            with open(val_log_path, "w") as f:
                json.dump(self.validation_log, f, indent=2)

        # Save training configuration
        config_path = self.output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(checkpoint["training_config"], f, indent=2)

        print(f"💾 Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint to resume training."""
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return

        print(f"📂 Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state (handle compatibility issues)
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("✅ Optimizer state loaded successfully")
        except (ValueError, KeyError) as e:
            print(f"⚠️  Could not load optimizer state: {e}")
            print("🔄 Starting with fresh optimizer state")
        
        # Load scheduler state (handle compatibility issues)
        try:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("✅ Scheduler state loaded successfully")
        except (ValueError, KeyError) as e:
            print(f"⚠️  Could not load scheduler state: {e}")
            print("🔄 Starting with fresh scheduler state")

        # Load training state
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.best_validation_loss = checkpoint.get("best_validation_loss", float('inf'))
        self.training_log = checkpoint.get("training_log", [])
        self.validation_log = checkpoint.get("validation_log", [])

        print("✅ Checkpoint loaded successfully")
        print(f"  Resuming from step: {self.step:,}")
        print(f"  Best loss so far: {self.best_loss:.4f}")
        if self.best_validation_loss != float('inf'):
            print(f"  Best validation loss: {self.best_validation_loss:.4f}")

    def train(self) -> None:
        """Main training loop with proper checkpoint saving."""
        print("\n🚀 Starting improved training...")
        print(f"  Model: {self.model.config.model_name}")
        print(f"  Parameters: {self.model.get_num_params():,}")
        print(f"  Device: {self.device}")
        print(f"  Max steps: {self.max_steps:,}")
        print(f"  Save every: {self.save_every} steps")
        print(f"  Early stopping patience: {self.early_stopping_patience}")
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

            # Forward pass
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

                # Update step counter
                self.step += 1

                # Calculate step time
                step_time = time.time() - step_start_time
                self.step_times.append(step_time)

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Evaluate on validation set if needed
                validation_loss = None
                if self.step % self.eval_every == 0:
                    validation_loss = self._evaluate_model()
                    if validation_loss is not None:
                        self.validation_log.append({
                            "step": self.step,
                            "validation_loss": validation_loss,
                            "validation_perplexity": math.exp(min(validation_loss, 10))
                        })

                        # Update best validation loss
                        if validation_loss < self.best_validation_loss:
                            self.best_validation_loss = validation_loss
                            self.no_improvement_count = 0
                        else:
                            self.no_improvement_count += 1

                # Log progress
                if self.step % self.log_every == 0:
                    self._log_step(self.step, accumulated_loss, current_lr, step_time, validation_loss)

                # Save checkpoint
                if self.step % self.save_every == 0:
                    is_best = accumulated_loss < self.best_loss
                    if is_best:
                        self.best_loss = accumulated_loss
                        self.no_improvement_count = 0
                    else:
                        self.no_improvement_count += 1

                    self._save_checkpoint(self.step, is_best=is_best)

                # Check for early stopping
                if self.no_improvement_count >= self.early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered after {self.early_stopping_patience} steps without improvement")
                    break

                # Reset accumulated loss
                accumulated_loss = 0.0

                # Memory cleanup
                if self.step % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Save final checkpoint
        self._save_checkpoint(self.step, is_best=False, is_final=True)

        # Training summary
        total_time = time.time() - self.start_time
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0

        print("\n📊 Training Summary:")
        print(f"  Steps completed: {self.step:,}")
        print(f"  Total time: {total_time/3600:.2f} hours")
        print(f"  Average time per step: {avg_step_time:.2f}s")
        print(f"  Final loss: {self.best_loss:.4f}")
        print(f"  Final perplexity: {math.exp(min(self.best_loss, 10)):.2f}")
        if self.best_validation_loss != float('inf'):
            print(f"  Best validation loss: {self.best_validation_loss:.4f}")
            print(f"  Best validation perplexity: {math.exp(min(self.best_validation_loss, 10)):.2f}")
        print(f"  Model saved to: {self.output_dir}")


def main():
    """Main function to handle command line training."""
    parser = argparse.ArgumentParser(
        description="Train a GPT-style language model with improved checkpoint saving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train small model with improved checkpointing
  python core/src/train_model_improved.py \\
    --model-size small \\
    --max-steps 10000 \\
    --output-dir models/improved-small \\
    --save-every 500 \\
    --early-stopping-patience 3

  # Train medium model with validation
  python core/src/train_model_improved.py \\
    --model-size medium \\
    --learning-rate 1e-4 \\
    --batch-size 2 \\
    --max-steps 50000 \\
    --output-dir models/improved-medium \\
    --validation-data data/clean/validation_data.txt
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
        "--validation-data",
        help="Path to validation text file (optional)",
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

    parser.add_argument(
        "--eval-every", type=int, default=500, help="Evaluate every N steps (default: 500)"
    )

    parser.add_argument(
        "--early-stopping-patience", type=int, default=5, 
        help="Early stopping patience (default: 5)"
    )

    args = parser.parse_args()

    print("🚀 OpenLLM Improved Model Training")
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

        # Create validation data loader if provided
        validation_data_loader = None
        if args.validation_data:
            print(f"\n📊 Setting up validation data loader...")
            validation_data_loader = TextDataLoader(
                data_file=args.validation_data,
                tokenizer_path=tokenizer_path,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                shuffle=False,
            )

        # Get data statistics
        _ = data_loader.get_data_stats()

        # Create trainer
        print("\n🎯 Setting up improved trainer...")
        trainer = ImprovedModelTrainer(
            model=model,
            data_loader=data_loader,
            validation_data_loader=validation_data_loader,
            output_dir=args.output_dir,
            device=device,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_every=args.save_every,
            eval_every=args.eval_every,
            early_stopping_patience=args.early_stopping_patience,
        )

        # Resume from checkpoint if specified
        if args.resume:
            trainer._load_checkpoint(args.resume)

        # Start training
        trainer.train()

        print("\n🎉 Improved training completed successfully!")
        print(f"📁 Model saved to: {args.output_dir}")
        print("📋 Checkpoints include full metadata like the 9k model")

        return True

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
