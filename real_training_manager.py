#!/usr/bin/env python3
"""
Real Training Manager for OpenLLM

This module provides comprehensive training functionality with:
- Real model training using existing ModelTrainer
- Checkpoint management and resuming
- Real-time progress monitoring
- Configurable training settings
- Hugging Face integration

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Add core/src to path
core_src_path = str(Path(__file__).parent / "core" / "src")
sys.path.insert(0, core_src_path)

from huggingface_hub import HfApi, login, whoami, create_repo
from train_model import ModelTrainer, TextDataLoader
from model import GPTConfig, GPTModel
from evaluate_model import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Comprehensive training configuration."""
    # Model settings
    model_size: str = "small"  # small, medium, large
    vocab_size: int = 10000
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    
    # Training settings
    training_steps: int = 8000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    
    # Data settings
    data_file: str = "data/clean/training_data.txt"
    validation_split: float = 0.1
    
    # Checkpoint settings
    save_every: int = 1000
    eval_every: int = 500
    checkpoint_dir: str = "checkpoints"
    
    # Device settings
    device: str = "auto"  # auto, cpu, cuda
    
    # Advanced settings
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    early_stopping_patience: int = 5
    max_grad_norm: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save config to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load config from file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class RealTrainingManager:
    """Comprehensive training manager with real training capabilities."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the training manager."""
        self.config = config
        self.setup_authentication()
        self.setup_device()
        self.setup_directories()
        self.training_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def setup_authentication(self):
        """Setup Hugging Face authentication."""
        print("🔐 Setting up Space authentication...")
        
        try:
            # Try Space's built-in authentication first
            user_info = whoami()
            self.username = user_info.get("name", "unknown")
            print(f"✅ Space built-in authentication successful!")
            print(f"👤 User: {self.username}")
            
        except Exception as e:
            print(f"❌ Space built-in authentication failed: {e}")
            print("🔄 Trying HF access token...")
            
            # Fallback to HF access token
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                try:
                    login(token=hf_token)
                    user_info = whoami()
                    self.username = user_info.get("name", "unknown")
                    print(f"✅ HF access token authentication successful!")
                    print(f"👤 User: {self.username}")
                except Exception as e2:
                    print(f"❌ HF access token authentication failed: {e2}")
                    raise
            else:
                print("❌ No authentication method available")
                raise
    
    def setup_device(self):
        """Setup training device."""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        print(f"🖥️ Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def setup_directories(self):
        """Setup necessary directories."""
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        print(f"📁 Checkpoint directory: {self.checkpoint_dir}")
        print(f"📁 Logs directory: {self.logs_dir}")
    
    def create_model_config(self) -> GPTConfig:
        """Create model configuration based on settings."""
        if self.config.model_size == "small":
            config = GPTConfig.small()
        elif self.config.model_size == "medium":
            config = GPTConfig.medium()
        elif self.config.model_size == "large":
            config = GPTConfig.large()
        else:
            config = GPTConfig.small()
        
        # Override with custom settings
        config.vocab_size = 32000  # Match tokenizer vocabulary size
        config.block_size = self.config.block_size
        config.n_layer = self.config.n_layer
        config.n_head = self.config.n_head
        config.n_embd = self.config.n_embd
        
        return config
    
    def load_or_create_model(self, checkpoint_path: Optional[str] = None) -> GPTModel:
        """Load existing model or create new one."""
        config = self.create_model_config()
        
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"📂 Loading model from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model = GPTModel(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded successfully from step {checkpoint.get('step', 'unknown')}")
            return model
        else:
            print(f"🆕 Creating new model with config: {self.config.model_size}")
            model = GPTModel(config)
            return model
    
    def create_data_loaders(self) -> tuple:
        """Create training and validation data loaders."""
        print(f"📊 Loading training data from: {self.config.data_file}")
        
        # Use a simple tokenizer path (will be created if needed)
        tokenizer_path = "data/tokenizer/tokenizer.model"
        
        # Create training data loader
        train_loader = TextDataLoader(
            data_file=self.config.data_file,
            tokenizer_path=tokenizer_path,
            seq_len=self.config.block_size,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Create validation data loader (same data but different shuffle)
        val_loader = TextDataLoader(
            data_file=self.config.data_file,
            tokenizer_path=tokenizer_path,
            seq_len=self.config.block_size,
            batch_size=self.config.batch_size,
            shuffle=False  # No shuffle for validation
        )
        
        print(f"✅ Training data loader created")
        print(f"✅ Validation data loader created")
        
        return train_loader, val_loader
    
    def save_checkpoint(self, model: GPTModel, optimizer, step: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config.to_dict(),
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best loss
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved with loss: {loss:.4f}")
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def evaluate_model(self, model: GPTModel, val_loader) -> float:
        """Evaluate model on validation set."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device) if len(batch) > 1 else None
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                logits, loss = model(inputs, targets)
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 10:  # Limit evaluation to 10 batches for speed
                    break
        
        avg_loss = total_loss / num_batches
        model.train()
        return avg_loss
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop with real training."""
        print("🚀 Starting Real OpenLLM Training")
        print("=" * 50)
        print(f"📊 Model Size: {self.config.model_size}")
        print(f"🔄 Training Steps: {self.config.training_steps}")
        print(f"👤 User: {self.username}")
        print(f"🖥️ Device: {self.device}")
        
        # Load or create model
        model = self.load_or_create_model(resume_from)
        model = model.to(self.device)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders()
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Load optimizer state if resuming
        if resume_from and Path(resume_from).exists():
            checkpoint = torch.load(resume_from, map_location=self.device)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_history = checkpoint.get('training_history', [])
                self.best_loss = checkpoint.get('loss', float('inf'))
                print(f"✅ Optimizer state loaded from checkpoint")
        
        # Setup mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision and self.device.type == 'cuda' else None
        
        # Training loop
        print(f"\n🔄 Starting training loop...")
        start_time = time.time()
        global_step = len(self.training_history)
        
        try:
            # Create iterator for training data
            train_iterator = iter(train_loader)
            
            for step in range(global_step, self.config.training_steps):
                # Get batch
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    # Restart data loader if exhausted
                    train_loader = TextDataLoader(
                        data_file=self.config.data_file,
                        tokenizer_path="data/tokenizer/tokenizer.model",
                        seq_len=self.config.block_size,
                        batch_size=self.config.batch_size,
                        shuffle=True
                    )
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)
                
                # Prepare inputs
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device) if len(batch) > 1 else None
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                # Forward pass
                if scaler:
                    with torch.cuda.amp.autocast():
                        logits, loss = model(inputs, targets)
                        loss = loss / self.config.gradient_accumulation_steps
                else:
                    logits, loss = model(inputs, targets)
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.gradient_clipping > 0:
                        if scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    
                    # Optimizer step
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                
                # Record training history
                self.training_history.append({
                    'step': step + 1,
                    'loss': loss.item() * self.config.gradient_accumulation_steps,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Progress reporting
                if (step + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = (step + 1) / elapsed
                    eta = (self.config.training_steps - step - 1) / steps_per_sec
                    
                    print(f"Step {step + 1}/{self.config.training_steps} | "
                          f"Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f} | "
                          f"Speed: {steps_per_sec:.1f} steps/s | "
                          f"ETA: {eta/60:.1f} min")
                
                # Evaluation
                if (step + 1) % self.config.eval_every == 0:
                    val_loss = self.evaluate_model(model, val_loader)
                    print(f"📊 Validation Loss: {val_loss:.4f}")
                    
                    # Check for best model
                    is_best = val_loss < self.best_loss
                    if is_best:
                        self.best_loss = val_loss
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                    
                    # Early stopping
                    if self.patience_counter >= self.config.early_stopping_patience:
                        print(f"🛑 Early stopping triggered after {self.config.early_stopping_patience} evaluations without improvement")
                        break
                
                # Save checkpoint
                if (step + 1) % self.config.save_every == 0:
                    current_loss = loss.item() * self.config.gradient_accumulation_steps
                    self.save_checkpoint(model, optimizer, step + 1, current_loss, is_best=False)
            
            # Save final checkpoint
            final_loss = loss.item() * self.config.gradient_accumulation_steps
            self.save_checkpoint(model, optimizer, self.config.training_steps, final_loss, is_best=True)
            
            print(f"\n🎉 Training completed successfully!")
            print(f"📊 Final Results:")
            print(f"   - Total Steps: {self.config.training_steps}")
            print(f"   - Final Loss: {final_loss:.4f}")
            print(f"   - Best Validation Loss: {self.best_loss:.4f}")
            print(f"   - Training Time: {(time.time() - start_time)/3600:.2f} hours")
            
            return model
            
        except KeyboardInterrupt:
            print(f"\n⚠️ Training interrupted by user")
            print(f"💾 Saving checkpoint before exit...")
            current_loss = loss.item() * self.config.gradient_accumulation_steps
            self.save_checkpoint(model, optimizer, step + 1, current_loss, is_best=False)
            return model
    
    def upload_model(self, model: GPTModel, model_dir: str = "./trained_model"):
        """Upload trained model to Hugging Face Hub."""
        print(f"📤 Uploading model to Hugging Face Hub...")
        
        # Create model directory
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        # Save model files
        torch.save(model.state_dict(), model_path / "pytorch_model.bin")
        
        # Save config
        config = self.create_model_config()
        config_dict = {
            "model_type": "openllm",
            "model_size": self.config.model_size,
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "training_config": self.config.to_dict(),
            "training_history": self.training_history
        }
        
        with open(model_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Create model card
        readme_content = f"""# OpenLLM {self.config.model_size.title()} Model

This is a real OpenLLM {self.config.model_size} model trained for {self.config.training_steps} steps.

## Model Details

- **Model Type**: OpenLLM
- **Size**: {self.config.model_size}
- **Training Steps**: {self.config.training_steps}
- **Final Loss**: {self.training_history[-1]['loss']:.4f} if self.training_history else 'N/A'
- **Framework**: PyTorch
- **License**: GPL-3.0

## Training Configuration

```json
{json.dumps(self.config.to_dict(), indent=2)}
```

## Training History

The model was trained with the following key metrics:
- Best validation loss: {self.best_loss:.4f}
- Total training time: {len(self.training_history)} steps
- Device used: {self.device}

## Usage

This model can be used for text generation and language modeling tasks.

## Author

Louis Chua Bean Chong

## License

GPL-3.0
"""
        
        with open(model_path / "README.md", 'w') as f:
            f.write(readme_content)
        
        # Upload to Hugging Face
        repo_name = f"openllm-{self.config.model_size}-{self.config.training_steps}steps-real"
        repo_id = f"{self.username}/{repo_name}"
        
        try:
            # Create repository
            create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
            
            # Upload files
            api = HfApi()
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add real OpenLLM {self.config.model_size} model ({self.config.training_steps} steps)"
            )
            
            print(f"✅ Model uploaded successfully!")
            print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
            return repo_id
            
        except Exception as e:
            print(f"❌ Model upload failed: {e}")
            return None


def create_training_config_from_dict(config_dict: Dict[str, Any]) -> TrainingConfig:
    """Create training config from dictionary with validation."""
    return TrainingConfig(**config_dict)


def get_default_configs() -> Dict[str, TrainingConfig]:
    """Get default training configurations for different model sizes."""
    return {
        "small": TrainingConfig(
            model_size="small",
            training_steps=8000,
            batch_size=32,
            learning_rate=3e-4,
            n_layer=6,
            n_head=6,
            n_embd=384
        ),
        "medium": TrainingConfig(
            model_size="medium", 
            training_steps=16000,
            batch_size=16,
            learning_rate=2e-4,
            n_layer=12,
            n_head=12,
            n_embd=768
        ),
        "large": TrainingConfig(
            model_size="large",
            training_steps=32000,
            batch_size=8,
            learning_rate=1e-4,
            n_layer=24,
            n_head=16,
            n_embd=1024
        )
    }


if __name__ == "__main__":
    # Example usage
    config = TrainingConfig(
        model_size="small",
        training_steps=1000,  # Short for testing
        batch_size=16,
        learning_rate=3e-4
    )
    
    manager = RealTrainingManager(config)
    model = manager.train()
    repo_id = manager.upload_model(model)
    
    if repo_id:
        print(f"🎉 Training and upload completed successfully!")
        print(f"🚀 Your model is ready at: https://huggingface.co/{repo_id}")
    else:
        print(f"⚠️ Training completed but upload failed")
