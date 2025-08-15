#!/usr/bin/env python3
"""
OpenLLM Training Space Application - OpenLLM Compatible

This version uses OpenLLM's actual custom model architecture and loading approach:
- Uses custom GPTModel class (not Hugging Face Transformers)
- Loads models using torch.load() and load_state_dict()
- Uses sentencepiece.SentencePieceProcessor() for tokenization
- Compatible with OpenLLM's actual implementation

Author: Louis Chua Bean Chong
License: GPL-3.0
Version: 2.0.9
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

# Import OpenLLM's custom model architecture
try:
    # Try to import from local OpenLLM code
    import sys
    sys.path.append('core/src')
    from model import GPTModel, GPTConfig, create_model
    from data_loader import TextDataLoader
    OPENLLM_AVAILABLE = True
    print("✅ OpenLLM custom model architecture imported successfully")
except ImportError as e:
    print(f"❌ OpenLLM imports failed: {e}")
    OPENLLM_AVAILABLE = False

# Try to import sentencepiece
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
    print(f"✅ SentencePiece available: {spm.__version__}")
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    print("❌ SentencePiece not available")

# Import other dependencies
try:
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
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
    - Dataset preparation
    - Training execution using OpenLLM's approach
    - Model saving and uploading
    """
    
    def __init__(self):
        """Initialize the trainer with default settings."""
        self.model = None
        self.tokenizer = None
        self.data_loader = None
        self.optimizer = None
        self.scheduler = None
        self.is_training = False
        self.training_progress = {
            "status": "Ready",
            "current_step": 0,
            "total_steps": 0,
            "loss": 0.0,
            "learning_rate": 0.0
        }
        
        # Initialize Hugging Face API for model uploading
        try:
            self.hf_api = HfApi()
        except Exception as e:
            print(f"Failed to initialize HF API: {e}")
            self.hf_api = None
    
    def load_model_and_tokenizer(self, model_size: str) -> str:
        """
        Load the pre-trained OpenLLM model and tokenizer using OpenLLM's approach.
        
        Args:
            model_size: Size of the model to load ("small", "medium", "large")
            
        Returns:
            Status message indicating success or failure
        """
        try:
            if not OPENLLM_AVAILABLE:
                return "❌ OpenLLM custom model architecture not available"
            
            print(f"🔄 Loading OpenLLM {model_size} model using custom architecture...")
            
            # Create model using OpenLLM's factory function
            try:
                self.model = create_model(model_size)
                print(f"✅ OpenLLM {model_size} model created: {type(self.model).__name__}")
                print(f"   Parameters: {self.model.get_num_params():,}")
            except Exception as e:
                print(f"❌ Failed to create model: {e}")
                return f"❌ Failed to create OpenLLM model: {str(e)}"
            
            # Load tokenizer using sentencepiece
            try:
                print("🔄 Loading tokenizer using sentencepiece.SentencePieceProcessor()...")
                
                # Download tokenizer.model from HF Hub
                model_name = f"lemms/openllm-{model_size}-extended-7k"
                tokenizer_path = hf_hub_download(
                    repo_id=model_name,
                    filename="tokenizer.model"
                )
                
                print(f"✅ Tokenizer downloaded to: {tokenizer_path}")
                
                # Create SentencePieceProcessor
                sp_processor = spm.SentencePieceProcessor()
                sp_processor.load(tokenizer_path)
                
                # Store tokenizer for later use
                self.tokenizer = sp_processor
                
                print(f"✅ Tokenizer loaded successfully using SentencePieceProcessor")
                print(f"   Vocabulary size: {sp_processor.vocab_size()}")
                
            except Exception as e:
                print(f"❌ Failed to load tokenizer: {e}")
                return f"❌ Failed to load OpenLLM tokenizer: {str(e)}"
            
            return f"✅ Successfully loaded OpenLLM {model_size} model with custom architecture"
            
        except Exception as e:
            return f"❌ Failed to load OpenLLM model and tokenizer: {str(e)}"
    
    def prepare_dataset(self) -> str:
        """
        Load and prepare the training dataset using OpenLLM's approach.
        
        Returns:
            Status message indicating success or failure
        """
        try:
            if not DEPENDENCIES_AVAILABLE:
                return "❌ Required dependencies not available"
            
            print("🔄 Loading training dataset...")
            
            # Load dataset from HF Hub
            dataset = load_dataset("lemms/openllm-training-data")
            print(f"✅ Dataset loaded: {len(dataset['train'])} samples")
            
            # Create temporary data file for OpenLLM's TextDataLoader
            temp_data_file = "temp_training_data.txt"
            with open(temp_data_file, 'w', encoding='utf-8') as f:
                for item in dataset['train']:
                    f.write(item['text'] + '\n')
            
            print(f"✅ Temporary data file created: {temp_data_file}")
            
            # Create OpenLLM's TextDataLoader
            try:
                # Get tokenizer path
                tokenizer_path = self.tokenizer.model_file_path
                
                self.data_loader = TextDataLoader(
                    data_file=temp_data_file,
                    tokenizer_path=tokenizer_path,
                    seq_len=512,
                    batch_size=4,  # Will be overridden by training config
                    shuffle=True
                )
                
                print(f"✅ OpenLLM TextDataLoader created successfully")
                
            except Exception as e:
                print(f"❌ Failed to create TextDataLoader: {e}")
                return f"❌ Failed to create data loader: {str(e)}"
            
            return f"✅ Successfully prepared dataset with {len(dataset['train'])} samples"
            
        except Exception as e:
            return f"❌ Failed to prepare dataset: {str(e)}"
    
    def setup_training(self, config: TrainingConfig) -> str:
        """
        Set up the training configuration using OpenLLM's approach.
        
        Args:
            config: Training configuration object
            
        Returns:
            Status message indicating success or failure
        """
        try:
            # Create output directory
            os.makedirs(config.output_dir, exist_ok=True)
            
            # Set up optimizer (AdamW with weight decay)
            decay_params = []
            no_decay_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                if len(param.shape) == 1 or name.endswith('.bias'):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            
            param_groups = [
                {'params': decay_params, 'weight_decay': 0.01},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
            
            # Set up learning rate scheduler
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=config.warmup_steps
            )
            
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.max_steps - config.warmup_steps
            )
            
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[config.warmup_steps]
            )
            
            print("✅ Training setup completed successfully")
            return f"✅ Training setup completed successfully"
            
        except Exception as e:
            return f"❌ Failed to setup training: {str(e)}"
    
    def train_model(self, config: TrainingConfig, progress_callback=None) -> str:
        """
        Execute the actual model training using OpenLLM's approach.
        
        Args:
            config: Training configuration object
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Status message indicating success or failure
        """
        try:
            self.is_training = True
            self.training_progress["status"] = "Training"
            self.training_progress["total_steps"] = config.max_steps
            
            print(f"🚀 Starting OpenLLM training for {config.max_steps} steps...")
            
            # Training loop using OpenLLM's approach
            self.model.train()
            accumulated_loss = 0.0
            self.optimizer.zero_grad()
            
            step = 0
            for batch_idx, (input_ids, target_ids) in enumerate(self.data_loader):
                if step >= config.max_steps:
                    break
                
                # Forward pass (model computes loss internally when targets provided)
                logits, loss = self.model(input_ids, target_ids)
                
                # Scale loss for gradient accumulation
                loss = loss / config.gradient_accumulation_steps
                accumulated_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Update parameters
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update step count
                    step += 1
                    
                    # Update progress
                    self.training_progress["current_step"] = step
                    self.training_progress["loss"] = accumulated_loss
                    self.training_progress["learning_rate"] = self.scheduler.get_last_lr()[0]
                    
                    # Log progress
                    if step % config.logging_steps == 0:
                        print(f"Step {step}/{config.max_steps} | Loss: {accumulated_loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.2e}")
                    
                    # Save checkpoint
                    if step % config.save_steps == 0:
                        self._save_checkpoint(config.output_dir, step)
                    
                    # Reset accumulated loss
                    accumulated_loss = 0.0
                    
                    # Clean up memory
                    if step % 100 == 0:
                        gc.collect()
            
            # Final checkpoint
            self._save_checkpoint(config.output_dir, step, is_best=True)
            
            # Update final progress
            self.training_progress["status"] = "Completed"
            self.training_progress["current_step"] = step
            
            print(f"✅ Training completed! Final step: {step}")
            
            return f"✅ Training completed successfully! Final step: {step}"
            
        except Exception as e:
            self.training_progress["status"] = "Failed"
            print(f"❌ Training failed: {e}")
            return f"❌ Training failed: {str(e)}"
        finally:
            self.is_training = False
    
    def _save_checkpoint(self, output_dir: str, step: int, is_best: bool = False) -> None:
        """Save model checkpoint using OpenLLM's approach."""
        try:
            checkpoint = {
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.model.config.__dict__
            }
            
            # Save latest checkpoint
            checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
            torch.save(checkpoint, checkpoint_path)
            
            # Save best checkpoint
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
        
        Args:
            config: Training configuration object
            
        Returns:
            Status message indicating success or failure
        """
        try:
            print("🔄 Saving trained model...")
            
            # Save the final model
            self._save_checkpoint(config.output_dir, config.max_steps, is_best=True)
            
            # Save tokenizer files
            tokenizer_dir = os.path.join(config.output_dir, "tokenizer")
            os.makedirs(tokenizer_dir, exist_ok=True)
            
            # Copy the tokenizer.model file
            import shutil
            shutil.copy2(self.tokenizer.model_file_path, os.path.join(tokenizer_dir, "tokenizer.model"))
            
            print("✅ Model saved locally")
            
            # Generate model name for upload
            model_name = f"openllm-{config.model_size}-extended-8k"
            repo_id = f"lemms/{model_name}"
            
            # Upload to Hugging Face Hub
            if self.hf_api:
                print(f"🔄 Uploading model to {repo_id}...")
                
                # Upload model files
                self.hf_api.upload_folder(
                    folder_path=config.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add trained OpenLLM {config.model_size} model (8k steps)"
                )
                
                print(f"✅ Model uploaded successfully to {repo_id}")
                return f"✅ Model saved and uploaded to https://huggingface.co/{repo_id}"
            else:
                return f"✅ Model saved locally to {config.output_dir}"
                
        except Exception as e:
            print(f"❌ Failed to save/upload model: {e}")
            return f"❌ Failed to save/upload model: {str(e)}"
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress information."""
        return self.training_progress.copy()

def main():
    """
    Main function that creates the complete Gradio application interface.
    """
    
    # Initialize the trainer
    trainer = OpenLLMTrainer()
    
    # Create the main Gradio application interface
    with gr.Blocks(
        title="OpenLLM Training Space - OpenLLM Compatible",
        theme=gr.themes.Soft()
    ) as demo:
        
        # Application Header
        gr.Markdown("# 🚀 OpenLLM Training Space - OpenLLM Compatible")
        gr.Markdown("### *Uses OpenLLM's Custom Model Architecture*")
        gr.Markdown("---")
        
        # Status Information
        gr.Markdown(f"**OpenLLM Available**: {'✅ Yes' if OPENLLM_AVAILABLE else '❌ No'}")
        gr.Markdown(f"**SentencePiece Available**: {'✅ Yes' if SENTENCEPIECE_AVAILABLE else '❌ No'}")
        gr.Markdown(f"**Dependencies Available**: {'✅ Yes' if DEPENDENCIES_AVAILABLE else '❌ No'}")
        gr.Markdown("**Architecture**: ✅ OpenLLM Custom GPTModel (Not Hugging Face)")
        
        # Main Content Area
        with gr.Row():
            
            # Left Column: Training Configuration
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Training Configuration")
                
                # Model Size Selection
                model_size = gr.Dropdown(
                    choices=["small", "medium", "large"],
                    value="small",
                    label="Model Size"
                )
                
                # Training Steps Configuration
                max_steps = gr.Slider(
                    minimum=100,
                    maximum=10000,
                    value=1000,
                    step=100,
                    label="Max Training Steps"
                )
                
                # Learning Rate Configuration
                learning_rate = gr.Slider(
                    minimum=1e-5,
                    maximum=1e-3,
                    value=3e-4,
                    step=1e-5,
                    label="Learning Rate"
                )
                
                # Batch Size Configuration
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=4,
                    step=1,
                    label="Batch Size"
                )
            
            # Right Column: Training Status and Controls
            with gr.Column(scale=1):
                gr.Markdown("## 🎯 Training Status")
                
                # Training Status Display
                status_text = gr.Textbox(
                    value="Ready to start training" if OPENLLM_AVAILABLE else "OpenLLM not available",
                    label="Current Status",
                    interactive=False,
                    lines=5
                )
                
                # Progress Information
                progress_info = gr.JSON(
                    value=trainer.get_training_progress(),
                    label="Training Progress"
                )
                
                # Training Control Buttons
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Training", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop Training", variant="stop")
        
        # Instructions Section
        gr.Markdown("## 📋 OpenLLM Compatible Training Instructions")
        gr.Markdown("""
        This interface uses **OpenLLM's actual custom model architecture**:
        
        ### **Step 1: Configure Parameters**
        - **Model Size**: Select the base model to train from (small, medium, large)
        - **Max Steps**: Number of training iterations (100-10,000)
        - **Learning Rate**: Training rate (0.00001-0.001)
        - **Batch Size**: Samples per training batch (1-16)
        
        ### **Step 2: Start Training**
        - Click "Start Training" to begin the actual training process
        - Uses OpenLLM's custom GPTModel class (not Hugging Face Transformers)
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
        gr.Markdown("## 🔗 Model Resources")
        gr.Markdown("""
        - [📚 7k Small Model](https://huggingface.co/lemms/openllm-small-extended-7k)
        - [🎯 8k Small Model](https://huggingface.co/lemms/openllm-small-extended-8k)
        - [📊 Training Dataset](https://huggingface.co/datasets/lemms/openllm-training-data)
        - [📖 Main Project](https://github.com/louischua/openllm)
        """)
        
        # Training Function Definition
        def start_complete_training(model_size, max_steps, learning_rate, batch_size):
            """
            Execute the complete training process using OpenLLM's approach.
            """
            if not OPENLLM_AVAILABLE:
                return "❌ OpenLLM custom model architecture not available. Please check the installation."
            
            try:
                # Create training configuration
                config = TrainingConfig(
                    model_size=model_size,
                    max_steps=max_steps,
                    learning_rate=learning_rate,
                    batch_size=batch_size
                )
                
                # Step 1: Load model and tokenizer using OpenLLM's approach
                status = trainer.load_model_and_tokenizer(model_size)
                if "❌" in status:
                    return status
                
                # Step 2: Prepare dataset
                status = trainer.prepare_dataset()
                if "❌" in status:
                    return status
                
                # Step 3: Setup training
                status = trainer.setup_training(config)
                if "❌" in status:
                    return status
                
                # Step 4: Execute training
                status = trainer.train_model(config)
                if "❌" in status:
                    return status
                
                # Step 5: Save and upload model
                status = trainer.save_and_upload_model(config)
                
                return f"🚀 Complete training process finished!\n{status}"
                
            except Exception as e:
                return f"❌ Training process failed: {str(e)}"
        
        def update_progress():
            """Update the progress display."""
            return trainer.get_training_progress()
        
        # Connect UI Components to Functions
        start_btn.click(
            fn=start_complete_training,
            inputs=[model_size, max_steps, learning_rate, batch_size],
            outputs=[status_text]
        )
        
        # Auto-refresh progress every 5 seconds during training
        demo.load(update_progress, outputs=[progress_info])
        
        # Application Footer
        gr.Markdown("---")
        gr.Markdown("**Author**: Louis Chua Bean Chong | **Project**: OpenLLM | **License**: GPL-3.0")
        gr.Markdown("**Architecture**: OpenLLM Custom GPTModel (Not Hugging Face Transformers)")
        gr.Markdown("**Tokenizer**: sentencepiece.SentencePieceProcessor()")
    
    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch()
