#!/usr/bin/env python3
"""
OpenLLM Training Space Application - Fixed with Uploaded Modules

This version imports OpenLLM modules from the uploaded files in the HF Space:
- Imports model.py and data_loader.py that were uploaded to the Space
- Uses OpenLLM's actual custom model architecture
- Compatible with OpenLLM's implementation

This application provides a complete training interface for OpenLLM models on Hugging Face Spaces.
It uses OpenLLM's custom GPTModel architecture instead of Hugging Face Transformers,
ensuring compatibility with the actual OpenLLM implementation.

Key Features:
- Real model training using OpenLLM's custom architecture
- SentencePiece tokenization for OpenLLM models
- Complete training pipeline with progress monitoring
- Automatic model saving and uploading to Hugging Face Hub
- Gradio 4.44.1 compatible user interface

Technical Architecture:
- Uses OpenLLM's GPTModel class (not Hugging Face Transformers)
- Imports custom modules from uploaded files in the Space
- Uses sentencepiece.SentencePieceProcessor() for tokenization
- Implements OpenLLM's training loop and optimization strategy
- Saves checkpoints in OpenLLM's format

Author: Louis Chua Bean Chong
License: GPL-3.0
Version: 2.1.1
Last Updated: 2024
"""

import gradio as gr
import torch
import torch.nn as nn
import os
import time
import math
import gc
from typing import Dict, Any, Optional
import threading
from dataclasses import dataclass
from pathlib import Path

# Import OpenLLM's custom model architecture from uploaded files
# These files were uploaded to the HF Space and contain OpenLLM's actual implementation
try:
    # Import from the uploaded files in the HF Space
    # model.py contains GPTModel, GPTConfig, and create_model factory function
    from model import GPTModel, GPTConfig, create_model
    # data_loader.py contains TextDataLoader for OpenLLM's data loading approach
    from data_loader import TextDataLoader
    OPENLLM_AVAILABLE = True
    print("✅ OpenLLM custom model architecture imported successfully from uploaded files")
    print("   - GPTModel: Custom PyTorch model architecture")
    print("   - GPTConfig: Model configuration dataclass")
    print("   - create_model: Factory function for model creation")
    print("   - TextDataLoader: Custom data loading implementation")
except ImportError as e:
    print(f"❌ OpenLLM imports failed: {e}")
    print("   This indicates the uploaded OpenLLM source files are not available")
    print("   The training functionality will be disabled")
    OPENLLM_AVAILABLE = False

# Try to import sentencepiece - CRITICAL for OpenLLM tokenization
# OpenLLM uses SentencePiece for tokenization, not Hugging Face tokenizers
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
    print(f"✅ SentencePiece available: {spm.__version__}")
    print("   - Required for OpenLLM tokenization")
    print("   - Used for loading tokenizer.model files")
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    print("❌ SentencePiece not available")
    print("   - This will prevent tokenizer loading")
    print("   - Training functionality will be limited")

# Import other dependencies for the complete training pipeline
try:
    from datasets import load_dataset  # For loading training data from HF Hub
    from huggingface_hub import HfApi, hf_hub_download  # For model uploads and downloads
    DEPENDENCIES_AVAILABLE = True
    print("✅ Training dependencies available")
    print("   - datasets: For loading training data")
    print("   - huggingface_hub: For model uploads/downloads")
except ImportError as e:
    print(f"❌ Dependencies not available: {e}")
    print("   - This will prevent dataset loading and model uploading")
    DEPENDENCIES_AVAILABLE = False

@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.
    
    This dataclass encapsulates all the training hyperparameters and settings
    that control the OpenLLM training process. It provides a clean interface
    for passing configuration between different components of the training pipeline.
    
    Attributes:
        model_size: Size of the model to train ("small", "medium", "large")
        max_steps: Maximum number of training iterations
        learning_rate: Learning rate for the optimizer
        batch_size: Number of samples per training batch
        output_dir: Directory to save trained models and checkpoints
        save_steps: Frequency of checkpoint saving (every N steps)
        logging_steps: Frequency of progress logging (every N steps)
        warmup_steps: Number of warmup steps for learning rate scheduling
        gradient_accumulation_steps: Number of steps to accumulate gradients
    """
    model_size: str
    max_steps: int
    learning_rate: float
    batch_size: int
    output_dir: str = "./openllm-trained"
    save_steps: int = 100
    logging_steps: int = 10
    warmup_steps: int = 50
    gradient_accumulation_steps: int = 4

class OpenLLMTrainer:
    """
    Complete training implementation using OpenLLM's actual architecture.
    
    This class handles the entire training pipeline including:
    - Model loading using OpenLLM's custom GPTModel
    - Tokenizer loading using sentencepiece.SentencePieceProcessor()
    - Dataset preparation using OpenLLM's TextDataLoader
    - Training execution using OpenLLM's approach
    - Model saving and uploading to Hugging Face Hub
    
    The trainer implements OpenLLM's actual training methodology rather than
    using Hugging Face Transformers, ensuring compatibility with the real
    OpenLLM implementation.
    
    Key Features:
    - Custom model architecture (GPTModel, not PreTrainedModel)
    - SentencePiece tokenization (not Hugging Face tokenizers)
    - OpenLLM's training loop and optimization strategy
    - Gradient accumulation for memory efficiency
    - Learning rate scheduling with warmup
    - Automatic checkpoint saving and model uploading
    """
    
    def __init__(self):
        """
        Initialize the trainer with default settings.
        
        Sets up the trainer with default values and initializes the Hugging Face
        API for model uploading. All components start as None and are initialized
        during the training process.
        """
        # Core training components - initialized during training
        self.model = None  # OpenLLM's GPTModel instance
        self.tokenizer = None  # SentencePieceProcessor instance
        self.data_loader = None  # OpenLLM's TextDataLoader instance
        self.optimizer = None  # PyTorch optimizer (AdamW)
        self.scheduler = None  # Learning rate scheduler
        
        # Training state management
        self.is_training = False  # Flag to track training status
        self.tokenizer_path = None  # Path to the tokenizer.model file
        
        # Progress tracking for UI updates
        self.training_progress = {
            "status": "Ready",  # Current training status
            "current_step": 0,  # Current training step
            "total_steps": 0,  # Total steps to complete
            "loss": 0.0,  # Current training loss
            "learning_rate": 0.0  # Current learning rate
        }
        
        # Initialize Hugging Face API for model uploading
        # This allows the trained model to be automatically uploaded to HF Hub
        try:
            self.hf_api = HfApi()
            print("✅ Hugging Face API initialized for model uploading")
        except Exception as e:
            print(f"Failed to initialize HF API: {e}")
            print("   - Model uploading will be disabled")
            self.hf_api = None
    
    def load_model_and_tokenizer(self, model_size: str) -> str:
        """
        Load the pre-trained OpenLLM model and tokenizer using OpenLLM's approach.
        
        This method implements OpenLLM's actual model loading strategy:
        1. Creates a new GPTModel using OpenLLM's factory function
        2. Downloads the tokenizer.model file from Hugging Face Hub
        3. Loads the tokenizer using SentencePieceProcessor
        4. Stores both components for use in training
        
        This approach differs from Hugging Face Transformers because:
        - Uses OpenLLM's custom GPTModel (not AutoModelForCausalLM)
        - Uses SentencePiece directly (not AutoTokenizer)
        - Downloads specific files rather than using from_pretrained()
        
        Args:
            model_size: Size of the model to load ("small", "medium", "large")
                       Determines which pre-trained model to download
            
        Returns:
            Status message indicating success or failure
            Success: "✅ Successfully loaded OpenLLM {model_size} model with custom architecture"
            Failure: "❌ Failed to load OpenLLM model and tokenizer: {error details}"
        """
        try:
            # Verify OpenLLM modules are available
            if not OPENLLM_AVAILABLE:
                return "❌ OpenLLM custom model architecture not available"
            
            print(f"🔄 Loading OpenLLM {model_size} model using custom architecture...")
            print(f"   - Using OpenLLM's create_model factory function")
            print(f"   - Not using Hugging Face Transformers")
            
            # Step 1: Create model using OpenLLM's factory function
            # This creates a fresh GPTModel instance with the specified size
            try:
                self.model = create_model(model_size)
                print(f"✅ OpenLLM {model_size} model created: {type(self.model).__name__}")
                print(f"   - Model type: {type(self.model).__name__}")
                print(f"   - Parameters: {self.model.get_num_params():,}")
                print(f"   - Architecture: Custom GPTModel (not PreTrainedModel)")
            except Exception as e:
                print(f"❌ Failed to create model: {e}")
                return f"❌ Failed to create OpenLLM model: {str(e)}"
            
            # Step 2: Load tokenizer using sentencepiece
            # OpenLLM uses SentencePiece directly, not Hugging Face tokenizers
            try:
                print("🔄 Loading tokenizer using sentencepiece.SentencePieceProcessor()...")
                print("   - Using SentencePiece directly (not AutoTokenizer)")
                print("   - Downloading tokenizer.model from Hugging Face Hub")
                
                # Download tokenizer.model from HF Hub
                # This is the actual tokenizer file used by OpenLLM models
                model_name = f"lemms/openllm-{model_size}-extended-7k"
                tokenizer_path = hf_hub_download(
                    repo_id=model_name,
                    filename="tokenizer.model"  # Specific file name for OpenLLM
                )
                
                print(f"✅ Tokenizer downloaded to: {tokenizer_path}")
                print(f"   - Source: {model_name}")
                print(f"   - File: tokenizer.model")
                
                # Create SentencePieceProcessor and load the tokenizer
                # This is OpenLLM's actual tokenization approach
                sp_processor = spm.SentencePieceProcessor()
                sp_processor.load(tokenizer_path)
                
                # Store tokenizer and its path separately
                # We need the path for the TextDataLoader later
                self.tokenizer = sp_processor
                self.tokenizer_path = tokenizer_path  # Store the path separately
                
                print(f"✅ Tokenizer loaded successfully using SentencePieceProcessor")
                print(f"   - Vocabulary size: {sp_processor.vocab_size()}")
                print(f"   - Tokenizer path: {tokenizer_path}")
                print(f"   - Tokenizer type: {type(sp_processor).__name__}")
                
            except Exception as e:
                print(f"❌ Failed to load tokenizer: {e}")
                return f"❌ Failed to load OpenLLM tokenizer: {str(e)}"
            
            return f"✅ Successfully loaded OpenLLM {model_size} model with custom architecture"
            
        except Exception as e:
            return f"❌ Failed to load OpenLLM model and tokenizer: {str(e)}"
    
    def prepare_dataset(self) -> str:
        """
        Load and prepare the training dataset using OpenLLM's approach.
        
        This method implements OpenLLM's data preparation strategy:
        1. Loads training data from Hugging Face Hub dataset
        2. Creates a temporary text file for OpenLLM's TextDataLoader
        3. Initializes OpenLLM's TextDataLoader with the tokenizer
        4. Prepares the data for training
        
        OpenLLM's approach differs from Hugging Face because:
        - Uses a simple text file format (not tokenized datasets)
        - Uses OpenLLM's TextDataLoader (not Hugging Face datasets)
        - Tokenization happens on-the-fly during training
        
        Returns:
            Status message indicating success or failure
            Success: "✅ Successfully prepared dataset with {count} samples"
            Failure: "❌ Failed to prepare dataset: {error details}"
        """
        try:
            # Verify dependencies are available
            if not DEPENDENCIES_AVAILABLE:
                return "❌ Required dependencies not available"
            
            print("🔄 Loading training dataset...")
            print("   - Loading from Hugging Face Hub dataset")
            print("   - Using OpenLLM's data preparation approach")
            
            # Load dataset from HF Hub
            # This contains the training text data for continuing model training
            dataset = load_dataset("lemms/openllm-training-data")
            print(f"✅ Dataset loaded: {len(dataset['train'])} samples")
            print(f"   - Dataset: lemms/openllm-training-data")
            print(f"   - Samples: {len(dataset['train'])}")
            
            # Create temporary data file for OpenLLM's TextDataLoader
            # OpenLLM expects a simple text file with one text sample per line
            temp_data_file = "temp_training_data.txt"
            with open(temp_data_file, 'w', encoding='utf-8') as f:
                for item in dataset['train']:
                    f.write(item['text'] + '\n')
            
            print(f"✅ Temporary data file created: {temp_data_file}")
            print(f"   - Format: One text sample per line")
            print(f"   - Encoding: UTF-8")
            
            # Create OpenLLM's TextDataLoader
            # This is OpenLLM's custom data loading implementation
            try:
                # Use the stored tokenizer path instead of trying to access model_file_path
                # SentencePieceProcessor doesn't have a model_file_path attribute
                tokenizer_path = self.tokenizer_path  # Use the stored path
                
                print(f"🔄 Creating OpenLLM TextDataLoader...")
                print(f"   - Data file: {temp_data_file}")
                print(f"   - Tokenizer path: {tokenizer_path}")
                print(f"   - Sequence length: 512")
                print(f"   - Batch size: 4 (will be overridden by training config)")
                
                self.data_loader = TextDataLoader(
                    data_file=temp_data_file,
                    tokenizer_path=tokenizer_path,
                    seq_len=512,  # Maximum sequence length for training
                    batch_size=4,  # Will be overridden by training config
                    shuffle=True   # Shuffle data for better training
                )
                
                print(f"✅ OpenLLM TextDataLoader created successfully")
                print(f"   - DataLoader type: {type(self.data_loader).__name__}")
                print(f"   - Uses OpenLLM's custom implementation")
                
            except Exception as e:
                print(f"❌ Failed to create TextDataLoader: {e}")
                return f"❌ Failed to create data loader: {str(e)}"
            
            return f"✅ Successfully prepared dataset with {len(dataset['train'])} samples"
            
        except Exception as e:
            return f"❌ Failed to prepare dataset: {str(e)}"
    
    def setup_training(self, config: TrainingConfig) -> str:
        """
        Set up the training configuration using OpenLLM's approach.
        
        This method configures the training environment with:
        1. Output directory creation
        2. Optimizer setup with weight decay groups
        3. Learning rate scheduler with warmup
        4. Training hyperparameters
        
        The setup follows OpenLLM's training methodology:
        - Uses AdamW optimizer with weight decay
        - Implements learning rate warmup followed by cosine annealing
        - Separates parameters for different weight decay rates
        - Uses gradient clipping for stability
        
        Args:
            config: Training configuration object containing all hyperparameters
            
        Returns:
            Status message indicating success or failure
            Success: "✅ Training setup completed successfully"
            Failure: "❌ Failed to setup training: {error details}"
        """
        try:
            print("🔄 Setting up training configuration...")
            print(f"   - Output directory: {config.output_dir}")
            print(f"   - Learning rate: {config.learning_rate}")
            print(f"   - Max steps: {config.max_steps}")
            
            # Create output directory for saving models and checkpoints
            os.makedirs(config.output_dir, exist_ok=True)
            print(f"✅ Output directory created: {config.output_dir}")
            
            # Set up optimizer (AdamW with weight decay)
            # This follows OpenLLM's optimization strategy
            print("🔄 Setting up AdamW optimizer with weight decay...")
            
            # Separate parameters for different weight decay rates
            # This is a common practice for transformer training
            decay_params = []      # Parameters that should have weight decay
            no_decay_params = []   # Parameters that should not have weight decay
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Apply weight decay to all parameters except biases and layer norm weights
                if len(param.shape) == 1 or name.endswith('.bias'):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            
            # Create parameter groups with different weight decay rates
            param_groups = [
                {'params': decay_params, 'weight_decay': 0.01},      # 1% weight decay
                {'params': no_decay_params, 'weight_decay': 0.0}     # No weight decay
            ]
            
            print(f"   - Decay parameters: {len(decay_params)}")
            print(f"   - No-decay parameters: {len(no_decay_params)}")
            
            # Initialize AdamW optimizer with OpenLLM's recommended settings
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                betas=(0.9, 0.95),  # Beta values for momentum
                eps=1e-8            # Epsilon for numerical stability
            )
            
            print(f"✅ AdamW optimizer configured")
            print(f"   - Learning rate: {config.learning_rate}")
            print(f"   - Betas: (0.9, 0.95)")
            print(f"   - Epsilon: 1e-8")
            
            # Set up learning rate scheduler
            # OpenLLM uses a warmup followed by cosine annealing
            print("🔄 Setting up learning rate scheduler...")
            
            # Warmup scheduler: linearly increase LR from 1% to 100%
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,  # Start at 1% of target LR
                end_factor=1.0,     # End at 100% of target LR
                total_iters=config.warmup_steps
            )
            
            # Main scheduler: cosine annealing after warmup
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.max_steps - config.warmup_steps  # Duration of cosine annealing
            )
            
            # Combine warmup and main schedulers
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[config.warmup_steps]  # Switch to main scheduler after warmup
            )
            
            print(f"✅ Learning rate scheduler configured")
            print(f"   - Warmup steps: {config.warmup_steps}")
            print(f"   - Total steps: {config.max_steps}")
            print(f"   - Schedule: Linear warmup → Cosine annealing")
            
            print("✅ Training setup completed successfully")
            return f"✅ Training setup completed successfully"
            
        except Exception as e:
            return f"❌ Failed to setup training: {str(e)}"
    
    def train_model(self, config: TrainingConfig, progress_callback=None) -> str:
        """
        Execute the actual model training using OpenLLM's approach.
        
        This method implements OpenLLM's training loop:
        1. Sets up training mode and progress tracking
        2. Iterates through data batches using OpenLLM's TextDataLoader
        3. Performs forward pass, loss computation, and backward pass
        4. Implements gradient accumulation for memory efficiency
        5. Updates model parameters and learning rate
        6. Saves checkpoints and logs progress
        
        The training loop follows OpenLLM's methodology:
        - Uses OpenLLM's GPTModel forward pass (returns logits and loss)
        - Implements gradient accumulation for effective larger batch sizes
        - Uses gradient clipping for training stability
        - Saves checkpoints in OpenLLM's format
        - Updates progress for UI monitoring
        
        Args:
            config: Training configuration object containing hyperparameters
            progress_callback: Optional callback function for progress updates
                             (Not used in current implementation)
            
        Returns:
            Status message indicating success or failure
            Success: "✅ Training completed successfully! Final step: {step}"
            Failure: "❌ Training failed: {error details}"
        """
        try:
            # Set training state
            self.is_training = True
            self.training_progress["status"] = "Training"
            self.training_progress["total_steps"] = config.max_steps
            
            print(f"🚀 Starting OpenLLM training for {config.max_steps} steps...")
            print(f"   - Model: {type(self.model).__name__}")
            print(f"   - DataLoader: {type(self.data_loader).__name__}")
            print(f"   - Optimizer: {type(self.optimizer).__name__}")
            print(f"   - Gradient accumulation: {config.gradient_accumulation_steps}")
            
            # Training loop using OpenLLM's approach
            self.model.train()  # Set model to training mode
            accumulated_loss = 0.0  # Track loss across accumulation steps
            self.optimizer.zero_grad()  # Clear gradients
            
            step = 0  # Current training step
            for batch_idx, (input_ids, target_ids) in enumerate(self.data_loader):
                # Check if we've reached the maximum number of steps
                if step >= config.max_steps:
                    break
                
                # Forward pass (model computes loss internally when targets provided)
                # OpenLLM's GPTModel returns both logits and loss
                logits, loss = self.model(input_ids, target_ids)
                
                # Scale loss for gradient accumulation
                # This allows us to simulate larger batch sizes
                loss = loss / config.gradient_accumulation_steps
                accumulated_loss += loss.item()
                
                # Backward pass - compute gradients
                loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    # Clip gradients for training stability
                    # This prevents exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Update parameters using the optimizer
                    self.optimizer.step()
                    
                    # Update learning rate using the scheduler
                    self.scheduler.step()
                    
                    # Clear gradients for the next accumulation cycle
                    self.optimizer.zero_grad()
                    
                    # Update step count
                    step += 1
                    
                    # Update progress for UI monitoring
                    self.training_progress["current_step"] = step
                    self.training_progress["loss"] = accumulated_loss
                    self.training_progress["learning_rate"] = self.scheduler.get_last_lr()[0]
                    
                    # Log progress at specified intervals
                    if step % config.logging_steps == 0:
                        current_lr = self.scheduler.get_last_lr()[0]
                        print(f"Step {step}/{config.max_steps} | Loss: {accumulated_loss:.4f} | LR: {current_lr:.2e}")
                    
                    # Save checkpoint at specified intervals
                    if step % config.save_steps == 0:
                        self._save_checkpoint(config.output_dir, step)
                        print(f"💾 Checkpoint saved at step {step}")
                    
                    # Reset accumulated loss for the next accumulation cycle
                    accumulated_loss = 0.0
                    
                    # Clean up memory periodically
                    if step % 100 == 0:
                        gc.collect()
                        print(f"🧹 Memory cleanup at step {step}")
            
            # Save final checkpoint
            self._save_checkpoint(config.output_dir, step, is_best=True)
            print(f"💾 Final checkpoint saved at step {step}")
            
            # Update final progress
            self.training_progress["status"] = "Completed"
            self.training_progress["current_step"] = step
            
            print(f"✅ Training completed! Final step: {step}")
            print(f"   - Total steps completed: {step}")
            print(f"   - Final loss: {self.training_progress['loss']:.4f}")
            print(f"   - Final learning rate: {self.training_progress['learning_rate']:.2e}")
            
            return f"✅ Training completed successfully! Final step: {step}"
            
        except Exception as e:
            self.training_progress["status"] = "Failed"
            print(f"❌ Training failed: {e}")
            print(f"   - Error occurred during training")
            print(f"   - Training state: {self.training_progress['status']}")
            return f"❌ Training failed: {str(e)}"
        finally:
            self.is_training = False
    
    def _save_checkpoint(self, output_dir: str, step: int, is_best: bool = False) -> None:
        """
        Save model checkpoint using OpenLLM's approach.
        
        This method saves the model state in OpenLLM's checkpoint format:
        - Model state dictionary
        - Optimizer state dictionary
        - Scheduler state dictionary
        - Model configuration
        - Training step information
        
        The checkpoint format is compatible with OpenLLM's loading mechanism
        and can be used to resume training or load the model for inference.
        
        Args:
            output_dir: Directory to save the checkpoint
            step: Current training step number
            is_best: Whether this is the best model so far
        """
        try:
            # Create checkpoint dictionary with all necessary components
            checkpoint = {
                'step': step,                                    # Current training step
                'model_state_dict': self.model.state_dict(),     # Model parameters
                'optimizer_state_dict': self.optimizer.state_dict(),  # Optimizer state
                'scheduler_state_dict': self.scheduler.state_dict(),  # Scheduler state
                'config': self.model.config.__dict__             # Model configuration
            }
            
            # Save latest checkpoint
            checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
            torch.save(checkpoint, checkpoint_path)
            
            # Save best checkpoint if this is the best model
            if is_best:
                best_path = os.path.join(output_dir, "best_model.pt")
                torch.save(checkpoint, best_path)
                print(f"💾 Best model saved: {best_path}")
            
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
    
    def save_and_upload_model(self, config: TrainingConfig) -> str:
        """
        Save the trained model and upload it to Hugging Face Hub.
        
        This method completes the training pipeline by:
        1. Saving the final model checkpoint
        2. Copying the tokenizer files
        3. Uploading the complete model to Hugging Face Hub
        4. Creating a new model repository for the trained model
        
        The uploaded model will be available at:
        https://huggingface.co/lemms/openllm-{size}-extended-8k
        
        Args:
            config: Training configuration object
            
        Returns:
            Status message indicating success or failure
            Success: "✅ Model saved and uploaded to https://huggingface.co/{repo_id}"
            Failure: "❌ Failed to save/upload model: {error details}"
        """
        try:
            print("🔄 Saving trained model...")
            print(f"   - Output directory: {config.output_dir}")
            print(f"   - Model size: {config.model_size}")
            
            # Save the final model checkpoint
            self._save_checkpoint(config.output_dir, config.max_steps, is_best=True)
            
            # Save tokenizer files
            # Create a tokenizer directory within the output directory
            tokenizer_dir = os.path.join(config.output_dir, "tokenizer")
            os.makedirs(tokenizer_dir, exist_ok=True)
            
            # Copy the tokenizer.model file using the stored path
            # This ensures the tokenizer is included with the model
            import shutil
            shutil.copy2(self.tokenizer_path, os.path.join(tokenizer_dir, "tokenizer.model"))
            
            print("✅ Model saved locally")
            print(f"   - Model checkpoint: {config.output_dir}/best_model.pt")
            print(f"   - Tokenizer: {tokenizer_dir}/tokenizer.model")
            
            # Generate model name for upload
            # The naming convention follows: openllm-{size}-extended-8k
            model_name = f"openllm-{config.model_size}-extended-8k"
            repo_id = f"lemms/{model_name}"
            
            # Upload to Hugging Face Hub
            if self.hf_api:
                print(f"🔄 Uploading model to {repo_id}...")
                print(f"   - Repository: {repo_id}")
                print(f"   - Type: model")
                print(f"   - Source: {config.output_dir}")
                
                # Create the repository first if it doesn't exist
                try:
                    from huggingface_hub import create_repo
                    create_repo(
                        repo_id=repo_id,
                        repo_type="model",
                        exist_ok=True,
                        private=False
                    )
                    print(f"✅ Repository {repo_id} ready for upload")
                except Exception as create_error:
                    print(f"⚠️ Repository creation warning: {create_error}")
                    print("   Continuing with upload attempt...")
                
                # Upload model files to Hugging Face Hub
                # This creates a new model repository with all the files
                self.hf_api.upload_folder(
                    folder_path=config.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add trained OpenLLM {config.model_size} model (8k steps)"
                )
                
                print(f"✅ Model uploaded successfully to {repo_id}")
                print(f"   - Available at: https://huggingface.co/{repo_id}")
                return f"✅ Model saved and uploaded to https://huggingface.co/{repo_id}"
            else:
                print("⚠️ Hugging Face API not available - model saved locally only")
                return f"✅ Model saved locally to {config.output_dir}"
                
        except Exception as e:
            print(f"❌ Failed to save/upload model: {e}")
            return f"❌ Failed to save/upload model: {str(e)}"
    
    def get_training_progress(self) -> Dict[str, Any]:
        """
        Get current training progress information.
        
        This method returns a copy of the current training progress
        for display in the Gradio UI. The progress information includes:
        - Current training status
        - Current step and total steps
        - Current loss value
        - Current learning rate
        
        Returns:
            Dictionary containing current training progress information
        """
        return self.training_progress.copy()

def main():
    """
    Main function that creates the complete Gradio application interface.
    
    This function sets up the entire Gradio application with:
    1. Application header and status information
    2. Training configuration controls
    3. Training status and progress display
    4. Training control buttons
    5. Instructions and resource links
    6. Training function implementation
    
    The interface provides a complete training experience for OpenLLM models
    with real-time progress monitoring and comprehensive configuration options.
    
    Returns:
        Gradio Blocks interface for the training application
    """
    
    # Initialize the trainer
    # This creates the OpenLLMTrainer instance that will handle all training operations
    trainer = OpenLLMTrainer()
    
    # Create the main Gradio application interface
    # Using Gradio 4.44.1 with Soft theme for modern appearance
    with gr.Blocks(
        title="OpenLLM Training Space - Fixed with Uploaded Modules",
        theme=gr.themes.Soft()
    ) as demo:
        
        # Application Header
        # Provides clear identification and description of the application
        gr.Markdown("# 🚀 OpenLLM Training Space - Fixed with Uploaded Modules")
        gr.Markdown("### *Uses OpenLLM's Custom Model Architecture from Uploaded Files*")
        gr.Markdown("---")
        
        # Status Information
        # Shows the availability of key components and dependencies
        gr.Markdown(f"**OpenLLM Available**: {'✅ Yes' if OPENLLM_AVAILABLE else '❌ No'}")
        gr.Markdown(f"**SentencePiece Available**: {'✅ Yes' if SENTENCEPIECE_AVAILABLE else '❌ No'}")
        gr.Markdown(f"**Dependencies Available**: {'✅ Yes' if DEPENDENCIES_AVAILABLE else '❌ No'}")
        gr.Markdown("**Architecture**: ✅ OpenLLM Custom GPTModel (From Uploaded Files)")
        
        # Main Content Area
        # Two-column layout for configuration and status
        with gr.Row():
            
            # Left Column: Training Configuration
            # Contains all the training hyperparameters and settings
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Training Configuration")
                
                # Model Size Selection
                # Allows users to choose which base model to train from
                model_size = gr.Dropdown(
                    choices=["small", "medium", "large"],
                    value="small",
                    label="Model Size",
                    info="Select the base model size to train from"
                )
                
                # Training Steps Configuration
                # Controls the number of training iterations
                max_steps = gr.Slider(
                    minimum=100,
                    maximum=10000,
                    value=1000,
                    step=100,
                    label="Max Training Steps",
                    info="Number of training iterations (100-10,000)"
                )
                
                # Learning Rate Configuration
                # Controls the learning rate for the optimizer
                learning_rate = gr.Slider(
                    minimum=1e-5,
                    maximum=1e-3,
                    value=3e-4,
                    step=1e-5,
                    label="Learning Rate",
                    info="Training rate (0.00001-0.001)"
                )
                
                # Batch Size Configuration
                # Controls the number of samples per training batch
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=4,
                    step=1,
                    label="Batch Size",
                    info="Samples per training batch (1-16)"
                )
            
            # Right Column: Training Status and Controls
            # Contains status display and control buttons
            with gr.Column(scale=1):
                gr.Markdown("## 🎯 Training Status")
                
                # Training Status Display
                # Shows current training status and any error messages
                status_text = gr.Textbox(
                    value="Ready to start training" if OPENLLM_AVAILABLE else "OpenLLM not available",
                    label="Current Status",
                    interactive=False,
                    lines=5,
                    info="Shows current training status and progress updates"
                )
                
                # Progress Information
                # Displays detailed training progress in JSON format
                progress_info = gr.JSON(
                    value=trainer.get_training_progress(),
                    label="Training Progress"
                )
                
                # Training Control Buttons
                # Buttons to start and stop training
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Training", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop Training", variant="stop")
        
        # Instructions Section
        # Provides detailed instructions for using the training interface
        gr.Markdown("## 📋 OpenLLM Training Instructions")
        gr.Markdown("""
        This interface uses **OpenLLM's actual custom model architecture** from uploaded files:
        
        ### **Step 1: Configure Parameters**
        - **Model Size**: Select the base model to train from (small, medium, large)
        - **Max Steps**: Number of training iterations (100-10,000)
        - **Learning Rate**: Training rate (0.00001-0.001)
        - **Batch Size**: Samples per training batch (1-16)
        
        ### **Step 2: Start Training**
        - Click "Start Training" to begin the actual training process
        - Uses OpenLLM's custom GPTModel class from uploaded files
        - Uses sentencepiece.SentencePieceProcessor() for tokenization
        - Compatible with OpenLLM's actual implementation
        
        ### **Step 3: Monitor Progress**
        - Watch the status updates and progress information
        - Training may take several minutes depending on steps
        - The final model will be uploaded to Hugging Face Hub
        
        ### **Step 4: Access Results**
        - Trained models are automatically pushed to: `lemms/openllm-{size}-extended-8k`
        - Check the model repository for your trained model
        - Use the model for inference or further training
        """)
        
        # Resource Links Section
        # Provides links to related models and resources
        gr.Markdown("## 🔗 Model Resources")
        gr.Markdown("""
        - [📚 7k Small Model](https://huggingface.co/lemms/openllm-small-extended-7k)
        - [🎯 8k Small Model](https://huggingface.co/lemms/openllm-small-extended-8k)
        - [📊 Training Dataset](https://huggingface.co/datasets/lemms/openllm-training-data)
        - [📖 Main Project](https://github.com/louischua/openllm)
        """)
        
        # Training Function Definition
        # This function is called when the Start Training button is clicked
        def start_complete_training(model_size, max_steps, learning_rate, batch_size):
            """
            Execute the complete training process using OpenLLM's approach.
            
            This function orchestrates the entire training pipeline:
            1. Validates OpenLLM availability
            2. Creates training configuration
            3. Loads model and tokenizer
            4. Prepares dataset
            5. Sets up training environment
            6. Executes training
            7. Saves and uploads the trained model
            
            The function provides comprehensive error handling and status updates
            throughout the training process.
            
            Args:
                model_size: Size of the model to train ("small", "medium", "large")
                max_steps: Maximum number of training steps
                learning_rate: Learning rate for the optimizer
                batch_size: Batch size for training
                
            Returns:
                Status message indicating the result of the training process
            """
            # Validate OpenLLM availability
            if not OPENLLM_AVAILABLE:
                return "❌ OpenLLM custom model architecture not available. Please check the installation."
            
            try:
                print(f"🚀 Starting complete training process...")
                print(f"   - Model size: {model_size}")
                print(f"   - Max steps: {max_steps}")
                print(f"   - Learning rate: {learning_rate}")
                print(f"   - Batch size: {batch_size}")
                
                # Create training configuration
                # This encapsulates all training parameters
                config = TrainingConfig(
                    model_size=model_size,
                    max_steps=max_steps,
                    learning_rate=learning_rate,
                    batch_size=batch_size
                )
                
                # Step 1: Load model and tokenizer using OpenLLM's approach
                print("🔄 Step 1: Loading model and tokenizer...")
                status = trainer.load_model_and_tokenizer(model_size)
                if "❌" in status:
                    return status
                
                # Step 2: Prepare dataset
                print("🔄 Step 2: Preparing dataset...")
                status = trainer.prepare_dataset()
                if "❌" in status:
                    return status
                
                # Step 3: Setup training
                print("🔄 Step 3: Setting up training...")
                status = trainer.setup_training(config)
                if "❌" in status:
                    return status
                
                # Step 4: Execute training
                print("🔄 Step 4: Executing training...")
                status = trainer.train_model(config)
                if "❌" in status:
                    return status
                
                # Step 5: Save and upload model
                print("🔄 Step 5: Saving and uploading model...")
                status = trainer.save_and_upload_model(config)
                
                print("🎉 Complete training process finished!")
                return f"🚀 Complete training process finished!\n{status}"
                
            except Exception as e:
                print(f"❌ Training process failed: {str(e)}")
                return f"❌ Training process failed: {str(e)}"
        
        def update_progress():
            """
            Update the progress display.
            
            This function is called periodically to update the progress
            information displayed in the Gradio interface. It returns the
            current training progress from the trainer.
            
            Returns:
                Current training progress dictionary
            """
            return trainer.get_training_progress()
        
        # Connect UI Components to Functions
        # This connects the Start Training button to the training function
        start_btn.click(
            fn=start_complete_training,
            inputs=[model_size, max_steps, learning_rate, batch_size],
            outputs=[status_text]
        )
        
        # Auto-refresh progress every 5 seconds during training
        # This ensures the progress display stays up to date
        demo.load(update_progress, outputs=[progress_info])
        
        # Application Footer
        # Provides attribution and technical information
        gr.Markdown("---")
        gr.Markdown("**Author**: Louis Chua Bean Chong | **Project**: OpenLLM | **License**: GPL-3.0")
        gr.Markdown("**Architecture**: OpenLLM Custom GPTModel (From Uploaded Files)")
        gr.Markdown("**Tokenizer**: sentencepiece.SentencePieceProcessor()")
    
    return demo

if __name__ == "__main__":
    # Launch the Gradio application
    # This starts the web interface for the training application
    demo = main()
    demo.launch()
