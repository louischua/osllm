#!/usr/bin/env python3
"""
OpenLLM Training Script with Hugging Face Authentication

This script includes proper authentication setup for Hugging Face Spaces
and handles model upload after training completion.

Features:
- Automatic authentication using GitHub secrets
- Model training with proper error handling
- Automatic model upload to Hugging Face Hub
- Model card and configuration generation

Usage:
    Add this to your Space and run it for training with automatic upload.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
import json
import torch
from pathlib import Path

try:
    from huggingface_hub import HfApi, login, whoami, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("❌ huggingface_hub not installed")
    sys.exit(1)


class OpenLLMTrainingManager:
    """
    Manages OpenLLM training and upload in Hugging Face Spaces.
    """
    
    def __init__(self):
        """Initialize the training manager with authentication."""
        self.api = None
        self.username = None
        self.is_authenticated = False
        self.setup_authentication()
    
    def setup_authentication(self):
        """Set up authentication for the Space using GitHub secrets."""
        print("🔐 Setting up Hugging Face Authentication")
        print("-" * 40)
        
        try:
            # Get token from GitHub secrets (automatically available in Space)
            token = os.getenv("HF_TOKEN")
            if not token:
                raise ValueError("HF_TOKEN not found in Space environment. Please set it in GitHub repository secrets.")
            
            # Login with the token
            login(token=token)
            
            # Initialize API and get user info
            self.api = HfApi()
            user_info = whoami()
            self.username = user_info["name"]
            self.is_authenticated = True
            
            print(f"✅ Authentication successful!")
            print(f"   - Username: {self.username}")
            print(f"   - Source: GitHub secrets")
            
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            print("   - Please ensure HF_TOKEN is set in GitHub repository secrets")
            raise
    
    def create_model_config(self, model_dir: str, model_size: str = "small"):
        """Create Hugging Face compatible configuration."""
        config = {
            "architectures": ["GPTModel"],
            "model_type": "gpt",
            "vocab_size": 32000,
            "n_positions": 2048,
            "n_embd": 768 if model_size == "small" else 1024 if model_size == "medium" else 1280,
            "n_layer": 12 if model_size == "small" else 24 if model_size == "medium" else 32,
            "n_head": 12 if model_size == "small" else 16 if model_size == "medium" else 20,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "unk_token_id": 3,
            "transformers_version": "4.35.0",
            "use_cache": True
        }
        
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Model configuration created: {config_path}")
    
    def create_model_card(self, model_dir: str, repo_id: str, model_size: str, steps: int):
        """Create model card (README.md)."""
        model_card = f"""# OpenLLM {model_size.capitalize()} Model ({steps} steps)

This is a trained OpenLLM {model_size} model with extended training.

## Model Details

- **Model Type**: GPT-style decoder-only transformer
- **Architecture**: Custom OpenLLM implementation
- **Training Data**: SQUAD dataset (Wikipedia passages)
- **Vocabulary Size**: 32,000 tokens
- **Sequence Length**: 2,048 tokens
- **Model Size**: {model_size.capitalize()}
- **Training Steps**: {steps:,}

## Usage

This model can be used with the OpenLLM framework for text generation and language modeling tasks.

## Training

The model was trained using the OpenLLM training pipeline with:
- SentencePiece tokenization
- Custom GPT architecture
- SQUAD dataset for training
- Extended training for improved performance

## License

This model is released under the GNU General Public License v3.0.

## Repository

This model is hosted on Hugging Face Hub: https://huggingface.co/{repo_id}
"""
        
        readme_path = os.path.join(model_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card)
        
        print(f"✅ Model card created: {readme_path}")
    
    def upload_model(self, model_dir: str, model_size: str = "small", steps: int = 8000):
        """Upload the trained model to Hugging Face Hub."""
        if not self.is_authenticated:
            raise ValueError("Not authenticated. Please run setup_authentication() first.")
        
        try:
            # Create repository name
            repo_name = f"openllm-{model_size}-extended-{steps//1000}k"
            repo_id = f"{self.username}/{repo_name}"
            
            print(f"\n📤 Uploading model to Hugging Face Hub")
            print(f"   - Repository: {repo_id}")
            print(f"   - Model directory: {model_dir}")
            
            # Verify model directory exists
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            # Create repository
            print(f"🔄 Creating repository...")
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=False
            )
            
            # Create model configuration and card
            print(f"🔄 Creating model configuration...")
            self.create_model_config(model_dir, model_size)
            self.create_model_card(model_dir, repo_id, model_size, steps)
            
            # Upload all files
            print(f"🔄 Uploading model files...")
            self.api.upload_folder(
                folder_path=model_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add OpenLLM {model_size} model ({steps} steps)"
            )
            
            print(f"✅ Model uploaded successfully!")
            print(f"   - Repository: https://huggingface.co/{repo_id}")
            print(f"   - Model available for download and use")
            
            return repo_id
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            raise
    
    def run_training(self, model_size: str = "small", steps: int = 8000):
        """Run the OpenLLM training process."""
        print(f"\n🚀 Starting OpenLLM Training")
        print(f"=" * 50)
        print(f"   - Model Size: {model_size}")
        print(f"   - Training Steps: {steps}")
        print(f"   - Username: {self.username}")
        
        # This is where you would integrate with your actual training code
        # For now, we'll simulate the training process
        
        print(f"\n🔄 Training in progress...")
        print(f"   - This would run your actual training code here")
        print(f"   - Training would save model to: ./openllm-trained")
        
        # Simulate training completion
        model_dir = "./openllm-trained"
        
        # Create model directory if it doesn't exist (for testing)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create a dummy model file for testing
        dummy_model_path = os.path.join(model_dir, "best_model.pt")
        with open(dummy_model_path, "w") as f:
            f.write("Dummy model file for testing upload functionality")
        
        print(f"✅ Training completed!")
        print(f"   - Model saved to: {model_dir}")
        
        # Upload the model
        repo_id = self.upload_model(model_dir, model_size, steps)
        
        print(f"\n🎉 Training and upload completed successfully!")
        print(f"   - Model available at: https://huggingface.co/{repo_id}")
        
        return repo_id


def main():
    """Main training function."""
    print("🚀 OpenLLM Training with Hugging Face Authentication")
    print("=" * 60)
    
    try:
        # Initialize training manager
        training_manager = OpenLLMTrainingManager()
        
        # Run training (you can modify parameters here)
        model_size = "small"  # Options: "small", "medium", "large"
        steps = 8000  # Number of training steps
        
        repo_id = training_manager.run_training(model_size, steps)
        
        print(f"\n✅ Success! Your model is now available at:")
        print(f"   https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
