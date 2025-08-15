#!/usr/bin/env python3
"""
OpenLLM Training Space Application - Gradio 4.44.1 Compatible

This is a complete Gradio application that provides actual model training functionality
for OpenLLM models. It loads the 7k model, trains it for additional steps, and pushes
the results to Hugging Face Hub. Updated for Gradio 4.44.1 compatibility.

Author: Louis Chua Bean Chong
License: GPL-3.0
Version: 2.0.2
Last Updated: 2024
"""

import gradio as gr
import torch
import os
import time
from typing import Dict, Any, Optional
import threading
from dataclasses import dataclass

# Import training dependencies
try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        TrainingArguments, 
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import load_dataset
    from huggingface_hub import HfApi
    TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"Training dependencies not available: {e}")
    TRAINING_AVAILABLE = False

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
    Complete training implementation for OpenLLM models.
    
    This class handles the entire training pipeline including:
    - Model and tokenizer loading
    - Dataset preparation
    - Training execution
    - Model saving and uploading
    """
    
    def __init__(self):
        """Initialize the trainer with default settings."""
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_thread = None
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
        Load the pre-trained OpenLLM model and tokenizer.
        
        Args:
            model_size: Size of the model to load ("small", "medium", "large")
            
        Returns:
            Status message indicating success or failure
        """
        try:
            # Map model size to actual model repository
            model_mapping = {
                "small": "lemms/openllm-small-extended-7k",
                "medium": "lemms/openllm-medium-extended-7k",  # Placeholder
                "large": "lemms/openllm-large-extended-7k"     # Placeholder
            }
            
            model_name = model_mapping.get(model_size, "lemms/openllm-small-extended-7k")
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision for memory efficiency
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            return f"✅ Successfully loaded {model_size} model from {model_name}"
            
        except Exception as e:
            return f"❌ Failed to load model: {str(e)}"
    
    def prepare_dataset(self) -> str:
        """
        Load and prepare the training dataset.
        
        Returns:
            Status message indicating success or failure
        """
        try:
            # Load the training dataset
            dataset = load_dataset("lemms/openllm-training-data")
            
            # Tokenize the dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )
            
            tokenized_dataset = dataset["train"].map(
                tokenize_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            
            self.dataset = tokenized_dataset
            
            return f"✅ Successfully prepared dataset with {len(tokenized_dataset)} samples"
            
        except Exception as e:
            return f"❌ Failed to prepare dataset: {str(e)}"
    
    def setup_training(self, config: TrainingConfig) -> str:
        """
        Set up the training configuration and trainer.
        
        Args:
            config: Training configuration object
            
        Returns:
            Status message indicating success or failure
        """
        try:
            # Create output directory
            os.makedirs(config.output_dir, exist_ok=True)
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=config.output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                max_steps=config.max_steps,
                save_steps=config.save_steps,
                logging_steps=config.logging_steps,
                warmup_steps=config.warmup_steps,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                evaluation_strategy="no",  # Disable evaluation for faster training
                save_strategy="steps",
                logging_dir=f"{config.output_dir}/logs",
                report_to=None,  # Disable wandb/tensorboard reporting
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
                dataloader_num_workers=0,  # Reduce memory usage
            )
            
            # Set up data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # We're doing causal language modeling, not masked
            )
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            
            return f"✅ Training setup completed successfully"
            
        except Exception as e:
            return f"❌ Failed to setup training: {str(e)}"
    
    def train_model(self, config: TrainingConfig, progress_callback=None) -> str:
        """
        Execute the actual model training.
        
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
            
            # Start training
            train_result = self.trainer.train()
            
            # Update final progress
            self.training_progress["status"] = "Completed"
            self.training_progress["current_step"] = config.max_steps
            self.training_progress["loss"] = train_result.training_loss
            
            return f"✅ Training completed successfully! Final loss: {train_result.training_loss:.4f}"
            
        except Exception as e:
            self.training_progress["status"] = "Failed"
            return f"❌ Training failed: {str(e)}"
        finally:
            self.is_training = False
    
    def save_and_upload_model(self, config: TrainingConfig) -> str:
        """
        Save the trained model and upload it to Hugging Face Hub.
        
        Args:
            config: Training configuration object
            
        Returns:
            Status message indicating success or failure
        """
        try:
            # Save the model locally
            self.trainer.save_model()
            self.tokenizer.save_pretrained(config.output_dir)
            
            # Generate model name for upload
            model_name = f"openllm-{config.model_size}-extended-8k"
            repo_id = f"lemms/{model_name}"
            
            # Upload to Hugging Face Hub
            if self.hf_api:
                # Upload model files
                self.hf_api.upload_folder(
                    folder_path=config.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add trained OpenLLM {config.model_size} model (8k steps)"
                )
                
                return f"✅ Model saved and uploaded to https://huggingface.co/{repo_id}"
            else:
                return f"✅ Model saved locally to {config.output_dir}"
                
        except Exception as e:
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
        title="OpenLLM Training Space - Gradio 4.44.1",
        theme=gr.themes.Soft()
    ) as demo:
        
        # Application Header
        gr.Markdown("# 🚀 OpenLLM Training Space - Complete Implementation")
        gr.Markdown("### *Real Model Training Interface - Gradio 4.44.1*")
        gr.Markdown("---")
        
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
                    value="Ready to start training" if TRAINING_AVAILABLE else "Training dependencies not available",
                    label="Current Status",
                    interactive=False,
                    lines=5
                )
                
                # Progress Information - Updated for Gradio 4.44.1 compatibility
                progress_info = gr.JSON(
                    value=trainer.get_training_progress(),
                    label="Training Progress",
                    interactive=False  # Now supported in Gradio 4.44.1
                )
                
                # Training Control Buttons
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Training", variant="primary", disabled=not TRAINING_AVAILABLE)
                    stop_btn = gr.Button("⏹️ Stop Training", variant="stop", disabled=not TRAINING_AVAILABLE)
        
        # Instructions Section
        gr.Markdown("## 📋 Complete Training Instructions")
        gr.Markdown("""
        This interface provides **real model training** functionality with Gradio 4.44.1:
        
        ### **Step 1: Configure Parameters**
        - **Model Size**: Select the base model to train from (7k models)
        - **Max Steps**: Number of training iterations (100-10,000)
        - **Learning Rate**: Training rate (0.00001-0.001)
        - **Batch Size**: Samples per training batch (1-16)
        
        ### **Step 2: Start Training**
        - Click "Start Training" to begin the actual training process
        - The system will:
          1. Load the 7k model from Hugging Face Hub
          2. Prepare the training dataset
          3. Execute training for the specified steps
          4. Save and upload the trained model
        
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
            Execute the complete training process with real model training.
            """
            if not TRAINING_AVAILABLE:
                return "❌ Training dependencies not available. Please check the installation."
            
            try:
                # Create training configuration
                config = TrainingConfig(
                    model_size=model_size,
                    max_steps=max_steps,
                    learning_rate=learning_rate,
                    batch_size=batch_size
                )
                
                # Step 1: Load model and tokenizer
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
        gr.Markdown(f"**Training Available**: {'✅ Yes' if TRAINING_AVAILABLE else '❌ No'}")
        gr.Markdown("**Gradio Version**: 4.44.1")
    
    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch()
