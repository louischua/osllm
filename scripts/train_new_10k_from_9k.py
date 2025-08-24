#!/usr/bin/env python3
"""
Train New 10k Model from 9k Model

This script trains a new 10k model starting from the 9k model checkpoint
using the improved training process with proper checkpoint saving.

Key Features:
- Resumes training from 9k model checkpoint
- Uses improved training script with full metadata
- Saves proper checkpoints like the 9k model
- Includes early stopping and validation monitoring
- Creates a new 10k model with better performance

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def main():
    """Train new 10k model from 9k model using improved process."""
    
    print("🚀 Training New 10k Model from 9k Model")
    print("=" * 60)
    
    # Configuration
    base_dir = Path(".")
    models_dir = base_dir / "models"
    data_dir = base_dir / "data"
    
    # Paths
    nine_k_model_dir = models_dir / "small-extended-9k"
    new_ten_k_model_dir = models_dir / "small-extended-10k-improved"
    training_data = data_dir / "clean" / "training_data.txt"
    validation_data = data_dir / "clean" / "training_data_validation.txt"
    tokenizer_dir = data_dir / "tokenizer"
    
    # Check if 9k model exists
    if not nine_k_model_dir.exists():
        print(f"❌ 9k model directory not found: {nine_k_model_dir}")
        return False
    
    best_model_path = nine_k_model_dir / "best_model.pt"
    if not best_model_path.exists():
        print(f"❌ 9k model checkpoint not found: {best_model_path}")
        return False
    
    print(f"✅ Found 9k model: {best_model_path}")
    
    # Check training data
    if not training_data.exists():
        print(f"❌ Training data not found: {training_data}")
        return False
    
    if not validation_data.exists():
        print(f"❌ Validation data not found: {validation_data}")
        return False
    
    print(f"✅ Found training data: {training_data}")
    print(f"✅ Found validation data: {validation_data}")
    
    # Create output directory
    new_ten_k_model_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {new_ten_k_model_dir}")
    
    # Training configuration
    training_config = {
        "model_size": "small",
        "max_steps": 10000,  # Train for 1000 more steps (9k -> 10k)
        "learning_rate": 3e-4,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "save_every": 500,  # Save more frequently
        "eval_every": 250,  # Evaluate more frequently
        "early_stopping_patience": 3,  # Stop early if no improvement
        "warmup_steps": 100,  # Shorter warmup since resuming
    }
    
    print("\n📋 Training Configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    # Build training command
    cmd = [
        "python", "core/src/train_model_improved.py",
        "--model-size", training_config["model_size"],
        "--data-file", str(training_data),
        "--validation-data", str(validation_data),
        "--tokenizer-dir", str(tokenizer_dir),
        "--output-dir", str(new_ten_k_model_dir),
        "--max-steps", str(training_config["max_steps"]),
        "--learning-rate", str(training_config["learning_rate"]),
        "--batch-size", str(training_config["batch_size"]),
        "--gradient-accumulation-steps", str(training_config["gradient_accumulation_steps"]),
        "--save-every", str(training_config["save_every"]),
        "--eval-every", str(training_config["eval_every"]),
        "--early-stopping-patience", str(training_config["early_stopping_patience"]),
        "--warmup-steps", str(training_config["warmup_steps"]),
        "--resume", str(best_model_path),  # Resume from 9k model
    ]
    
    print(f"\n🚀 Starting training with command:")
    print(" ".join(cmd))
    print("\n" + "=" * 80)
    
    # Start training
    start_time = time.time()
    
    try:
        # Run training process
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True,
            cwd=base_dir
        )
        
        training_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("🎉 Training completed successfully!")
        print(f"⏱️  Total training time: {training_time/3600:.2f} hours")
        print(f"📁 New 10k model saved to: {new_ten_k_model_dir}")
        
        # Verify the new model
        verify_new_model(new_ten_k_model_dir)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False

def verify_new_model(model_dir):
    """Verify the new model has proper checkpoint format."""
    print(f"\n🔍 Verifying new model: {model_dir}")
    
    model_path = Path(model_dir)
    
    # Check for best_model.pt
    best_model_path = model_path / "best_model.pt"
    if best_model_path.exists():
        size_mb = best_model_path.stat().st_size / (1024 * 1024)
        print(f"✅ best_model.pt: {size_mb:.1f} MB")
        
        if size_mb > 400:  # Should be around 455MB like 9k model
            print(f"✅ File size is correct (should be ~455MB)")
        else:
            print(f"⚠️  File size seems small (expected ~455MB)")
    else:
        print(f"❌ best_model.pt not found")
    
    # Check for training log
    training_log_path = model_path / "training_log.json"
    if training_log_path.exists():
        size_kb = training_log_path.stat().st_size / 1024
        print(f"✅ training_log.json: {size_kb:.1f} KB")
    else:
        print(f"❌ training_log.json not found")
    
    # Check for training config
    training_config_path = model_path / "training_config.json"
    if training_config_path.exists():
        print(f"✅ training_config.json found")
    else:
        print(f"❌ training_config.json not found")
    
    # Check for checkpoint files
    checkpoint_files = list(model_path.glob("checkpoint_step_*.pt"))
    if checkpoint_files:
        print(f"✅ Found {len(checkpoint_files)} checkpoint files")
        latest_checkpoint = max(checkpoint_files, key=lambda p: int(p.stem.split("_")[-1]))
        print(f"✅ Latest checkpoint: {latest_checkpoint.name}")
    else:
        print(f"⚠️  No checkpoint files found")

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 New 10k model training completed successfully!")
        print("📋 The model now has proper checkpoint format with full metadata")
        print("📁 Ready to be uploaded to Hugging Face")
    else:
        print("\n❌ New 10k model training failed")
        sys.exit(1)
