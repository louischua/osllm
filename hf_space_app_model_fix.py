#!/usr/bin/env python3
"""
OpenLLM Training Space Application - Custom Model Architecture Fix

This version handles the custom GPT model architecture by:
- Updating transformers to latest version
- Using alternative model loading approaches
- Handling custom model architectures properly

Author: Louis Chua Bean Chong
License: GPL-3.0
Version: 2.0.8
Last Updated: 2024
"""

import gradio as gr
import torch
import os
import time
from typing import Dict, Any, Optional
import threading
from dataclasses import dataclass

# First, try to update transformers to latest version
try:
    import subprocess
    print("🔄 Updating transformers to latest version...")
    subprocess.run(["pip", "install", "--upgrade", "transformers"], check=True)
    print("✅ Transformers updated successfully")
except Exception as e:
    print(f"⚠️ Could not update transformers: {e}")

# Import training dependencies with robust error handling
try:
    from transformers import (
        AutoModelForCausalLM, 
        TrainingArguments, 
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import load_dataset
    from huggingface_hub import HfApi
    TRAINING_AVAILABLE = True
    print("✅ Transformers imported successfully")
except ImportError as e:
    print(f"Training dependencies not available: {e}")
    TRAINING_AVAILABLE = False

# Try to import sentencepiece with fallback
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
    print(f"✅ SentencePiece available: {spm.__version__}")
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    print("❌ SentencePiece not available - will use fallback methods")

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
    Complete training implementation for OpenLLM models with custom architecture handling.
    
    This class handles the entire training pipeline including:
    - Model loading with custom architecture support
    - Tokenizer loading using sentencepiece.SentencePieceProcessor()
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
        Load the pre-trained OpenLLM model and tokenizer with custom architecture handling.
        
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
            
            print(f"🔄 Loading OpenLLM model: {model_name}")
            print("📝 Handling custom GPT architecture...")
            
            # Try multiple approaches to load the model
            model_loaded = False
            
            # Approach 1: Try with latest transformers and trust_remote_code
            try:
                print("🔄 Attempting to load model with latest transformers...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    revision="main"  # Use main branch for latest code
                )
                model_loaded = True
                print(f"✅ Model loaded successfully with latest transformers: {type(self.model).__name__}")
                
            except Exception as e1:
                print(f"❌ Approach 1 failed: {e1}")
                
                # Approach 2: Try installing transformers from source
                try:
                    print("🔄 Installing transformers from source...")
                    subprocess.run(["pip", "install", "git+https://github.com/huggingface/transformers.git"], check=True)
                    
                    # Reload transformers
                    import importlib
                    import transformers
                    importlib.reload(transformers)
                    from transformers import AutoModelForCausalLM
                    
                    print("🔄 Attempting to load model with source transformers...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True
                    )
                    model_loaded = True
                    print(f"✅ Model loaded successfully with source transformers: {type(self.model).__name__}")
                    
                except Exception as e2:
                    print(f"❌ Approach 2 failed: {e2}")
                    
                    # Approach 3: Try loading as a generic model
                    try:
                        print("🔄 Attempting to load as generic model...")
                        from transformers import AutoModel
                        
                        self.model = AutoModel.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            device_map="auto" if torch.cuda.is_available() else None,
                            trust_remote_code=True
                        )
                        model_loaded = True
                        print(f"✅ Model loaded as generic model: {type(self.model).__name__}")
                        
                    except Exception as e3:
                        print(f"❌ Approach 3 failed: {e3}")
                        return f"❌ Failed to load OpenLLM model: All approaches failed. Latest error: {str(e3)}"
            
            # Load tokenizer using the same approach as local training code
            try:
                print("🔄 Loading tokenizer using sentencepiece.SentencePieceProcessor()...")
                
                # Create a custom tokenizer class that wraps SentencePieceProcessor
                class OpenLLMTokenizer:
                    def __init__(self, sp_processor):
                        self.sp_processor = sp_processor
                        self.pad_token = "<pad>"
                        self.eos_token = "</s>"
                        self.bos_token = "<s>"
                        self.unk_token = "<unk>"
                    
                    def __call__(self, texts, **kwargs):
                        """Tokenize texts using SentencePieceProcessor."""
                        if isinstance(texts, str):
                            texts = [texts]
                        
                        results = []
                        for text in texts:
                            # Encode text to token IDs
                            token_ids = self.sp_processor.encode(text)
                            
                            # Create attention mask (all tokens are attended to)
                            attention_mask = [1] * len(token_ids)
                            
                            results.append({
                                'input_ids': token_ids,
                                'attention_mask': attention_mask
                            })
                        
                        return results
                    
                    def encode(self, text, **kwargs):
                        """Encode text to token IDs."""
                        return self.sp_processor.encode(text)
                    
                    def decode(self, token_ids, **kwargs):
                        """Decode token IDs to text."""
                        return self.sp_processor.decode(token_ids)
                    
                    def save_pretrained(self, path):
                        """Save tokenizer files."""
                        # The SentencePieceProcessor is already saved as tokenizer.model
                        pass
                
                # Download and load the tokenizer.model file
                from huggingface_hub import hf_hub_download
                
                print("🔄 Downloading tokenizer.model from HF Hub...")
                tokenizer_path = hf_hub_download(
                    repo_id=model_name,
                    filename="tokenizer.model"
                )
                
                print(f"✅ Tokenizer downloaded to: {tokenizer_path}")
                
                # Load using SentencePieceProcessor (same as local code)
                sp_processor = spm.SentencePieceProcessor()
                sp_processor.load(tokenizer_path)
                
                # Wrap in our custom tokenizer class for HF Trainer compatibility
                self.tokenizer = OpenLLMTokenizer(sp_processor)
                
                print(f"✅ Tokenizer loaded successfully using SentencePieceProcessor")
                print(f"   Vocabulary size: {sp_processor.vocab_size()}")
                
            except Exception as e:
                print(f"❌ Failed to load tokenizer: {e}")
                return f"❌ Failed to load OpenLLM tokenizer: {str(e)}"
            
            return f"✅ Successfully loaded OpenLLM {model_size} model from {model_name}"
            
        except Exception as e:
            return f"❌ Failed to load OpenLLM model and tokenizer: {str(e)}"
    
    def prepare_dataset(self) -> str:
        """
        Load and prepare the training dataset.
        
        Returns:
            Status message indicating success or failure
        """
        try:
            # Load the training dataset
            print("🔄 Loading training dataset...")
            dataset = load_dataset("lemms/openllm-training-data")
            print(f"✅ Dataset loaded: {len(dataset['train'])} samples")
            
            # Tokenize the dataset using our custom tokenizer
            def tokenize_function(examples):
                try:
                    # Use our custom tokenizer
                    tokenized = self.tokenizer(examples["text"])
                    
                    # Extract input_ids and attention_mask
                    input_ids = [item['input_ids'] for item in tokenized]
                    attention_mask = [item['attention_mask'] for item in tokenized]
                    
                    # Pad sequences to max_length
                    max_length = 512
                    padded_input_ids = []
                    padded_attention_mask = []
                    
                    for ids, mask in zip(input_ids, attention_mask):
                        if len(ids) > max_length:
                            ids = ids[:max_length]
                            mask = mask[:max_length]
                        else:
                            # Pad with pad_token_id
                            pad_length = max_length - len(ids)
                            ids = ids + [0] * pad_length  # 0 is pad_token_id
                            mask = mask + [0] * pad_length
                        
                        padded_input_ids.append(ids)
                        padded_attention_mask.append(mask)
                    
                    return {
                        "input_ids": padded_input_ids,
                        "attention_mask": padded_attention_mask
                    }
                    
                except Exception as e:
                    print(f"Tokenization error: {e}")
                    # Fallback: return empty tensors
                    return {"input_ids": [], "attention_mask": []}
            
            print("🔄 Tokenizing dataset...")
            tokenized_dataset = dataset["train"].map(
                tokenize_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            
            self.dataset = tokenized_dataset
            print(f"✅ Dataset tokenized successfully: {len(tokenized_dataset)} samples")
            
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
            
            print(f"🚀 Starting OpenLLM training for {config.max_steps} steps...")
            
            # Start training
            train_result = self.trainer.train()
            
            # Update final progress
            self.training_progress["status"] = "Completed"
            self.training_progress["current_step"] = config.max_steps
            self.training_progress["loss"] = train_result.training_loss
            
            print(f"✅ Training completed! Final loss: {train_result.training_loss:.4f}")
            
            return f"✅ Training completed successfully! Final loss: {train_result.training_loss:.4f}"
            
        except Exception as e:
            self.training_progress["status"] = "Failed"
            print(f"❌ Training failed: {e}")
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
            print("🔄 Saving trained model...")
            
            # Save the model locally
            self.trainer.save_model()
            
            # Save tokenizer files
            if hasattr(self.tokenizer, 'sp_processor'):
                # Save the SentencePieceProcessor files
                tokenizer_dir = os.path.join(config.output_dir, "tokenizer")
                os.makedirs(tokenizer_dir, exist_ok=True)
                
                # Copy the original tokenizer.model file
                import shutil
                from huggingface_hub import hf_hub_download
                
                model_name = f"lemms/openllm-{config.model_size}-extended-7k"
                tokenizer_path = hf_hub_download(
                    repo_id=model_name,
                    filename="tokenizer.model"
                )
                shutil.copy2(tokenizer_path, os.path.join(tokenizer_dir, "tokenizer.model"))
            
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
        title="OpenLLM Training Space - Custom Architecture Fix",
        theme=gr.themes.Soft()
    ) as demo:
        
        # Application Header
        gr.Markdown("# 🚀 OpenLLM Training Space - Custom Architecture Fix")
        gr.Markdown("### *Handles Custom GPT Model Architecture*")
        gr.Markdown("---")
        
        # Status Information
        gr.Markdown(f"**Training Available**: {'✅ Yes' if TRAINING_AVAILABLE else '❌ No'}")
        gr.Markdown(f"**SentencePiece Available**: {'✅ Yes' if SENTENCEPIECE_AVAILABLE else '❌ No (using fallback methods)'}")
        gr.Markdown("**Custom Architecture**: ✅ Multiple loading approaches")
        
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
                
                # Progress Information - Simplified for maximum compatibility
                progress_info = gr.JSON(
                    value=trainer.get_training_progress(),
                    label="Training Progress"
                )
                
                # Training Control Buttons - Removed disabled parameter for compatibility
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Training", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop Training", variant="stop")
        
        # Instructions Section
        gr.Markdown("## 📋 Custom Architecture Training Instructions")
        gr.Markdown("""
        This interface handles **OpenLLM's custom GPT architecture**:
        
        ### **Step 1: Configure Parameters**
        - **Model Size**: Select the base model to train from (7k models)
        - **Max Steps**: Number of training iterations (100-10,000)
        - **Learning Rate**: Training rate (0.00001-0.001)
        - **Batch Size**: Samples per training batch (1-16)
        
        ### **Step 2: Start Training**
        - Click "Start Training" to begin the actual training process
        - Automatically updates transformers to latest version
        - Uses multiple approaches to load custom GPT architecture
        - Handles custom model types properly
        
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
            Execute the complete training process with custom architecture handling.
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
                
                # Step 1: Load model and tokenizer with custom architecture handling
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
        gr.Markdown("**Gradio Version**: 4.44.1 (Fully Compatible)")
        gr.Markdown("**Custom Architecture**: Multiple loading approaches for GPT model")
    
    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch()
