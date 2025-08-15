#!/usr/bin/env python3
"""
Fix Training Upload Script

This script fixes the Hugging Face authentication issues in the training pipeline
and provides a proper upload mechanism for trained models.

Features:
- Proper Hugging Face authentication setup
- Repository creation with correct permissions
- Model upload with proper error handling
- Integration with existing training pipeline

Usage:
    python fix_training_upload.py --model-dir ./openllm-trained --repo-name my-model

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from huggingface_hub import HfApi, login, whoami, create_repo
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("❌ huggingface_hub not installed. Installing...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub>=0.34.0"])
        from huggingface_hub import HfApi, login, whoami, create_repo
        from huggingface_hub.utils import HfHubHTTPError
        HF_AVAILABLE = True
        print("✅ huggingface_hub installed successfully")
    except Exception as e:
        print(f"❌ Failed to install huggingface_hub: {e}")
        sys.exit(1)


class TrainingUploadFixer:
    """
    Fixes Hugging Face authentication and upload issues in the training pipeline.
    
    This class provides methods to:
    1. Set up proper authentication
    2. Create repositories with correct permissions
    3. Upload trained models with proper error handling
    4. Generate Hugging Face compatible model files
    """
    
    def __init__(self):
        """Initialize the upload fixer."""
        self.api = None
        self.username = None
        self.is_authenticated = False
        
    def setup_authentication(self) -> bool:
        """
        Set up proper Hugging Face authentication.
        
        Returns:
            True if authentication successful, False otherwise
        """
        print("🔐 Setting up Hugging Face Authentication")
        print("-" * 40)
        
        try:
            # Try to get current user info first
            user_info = whoami()
            self.username = user_info["name"]
            self.api = HfApi()
            self.is_authenticated = True
            
            print(f"✅ Already authenticated as: {self.username}")
            return True
            
        except Exception as e:
            print(f"❌ Not authenticated: {e}")
            print("🔄 Attempting to authenticate...")
            
            # Try environment variable
            token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
            if token:
                try:
                    login(token=token)
                    user_info = whoami()
                    self.username = user_info["name"]
                    self.api = HfApi()
                    self.is_authenticated = True
                    
                    print(f"✅ Authenticated with token as: {self.username}")
                    return True
                    
                except Exception as token_error:
                    print(f"❌ Token authentication failed: {token_error}")
            
            # Interactive authentication
            print("\n📝 Please provide your Hugging Face token:")
            print("1. Go to https://huggingface.co/settings/tokens")
            print("2. Click 'New token'")
            print("3. Give it a name (e.g., 'OpenLLM Training')")
            print("4. Select 'Write' role")
            print("5. Copy the generated token")
            print()
            
            try:
                token = input("Enter your Hugging Face token: ").strip()
                if token:
                    login(token=token)
                    user_info = whoami()
                    self.username = user_info["name"]
                    self.api = HfApi()
                    self.is_authenticated = True
                    
                    print(f"✅ Authenticated as: {self.username}")
                    return True
                    
            except KeyboardInterrupt:
                print("\n❌ Authentication cancelled")
            except Exception as e:
                print(f"❌ Authentication failed: {e}")
            
            return False
    
    def create_huggingface_config(self, model_dir: str, model_size: str = "small") -> Dict[str, Any]:
        """
        Create Hugging Face compatible configuration files.
        
        Args:
            model_dir: Directory containing the trained model
            model_size: Size of the model (small, medium, large)
            
        Returns:
            Dictionary with configuration details
        """
        print(f"📝 Creating Hugging Face configuration...")
        
        # Model configuration
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
        
        # Save config.json
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to: {config_path}")
        return config
    
    def create_model_card(self, model_dir: str, repo_name: str, model_size: str = "small") -> str:
        """
        Create a model card (README.md) for the Hugging Face repository.
        
        Args:
            model_dir: Directory containing the trained model
            repo_name: Name of the repository
            model_size: Size of the model
            
        Returns:
            Path to the created model card
        """
        print(f"📄 Creating model card...")
        
        # Model card content
        model_card = f"""# OpenLLM {model_size.capitalize()} Model

This is a trained OpenLLM {model_size} model with extended training.

## Model Details

- **Model Type**: GPT-style decoder-only transformer
- **Architecture**: Custom OpenLLM implementation
- **Training Data**: SQUAD dataset (Wikipedia passages)
- **Vocabulary Size**: 32,000 tokens
- **Sequence Length**: 2,048 tokens
- **Model Size**: {model_size.capitalize()}

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

## Citation

If you use this model, please cite the OpenLLM project:

```bibtex
@software{{openllm,
  title={{OpenLLM: Open Source Language Model Training}},
  author={{Louis Chua Bean Chong}},
  year={{2024}},
  url={{https://github.com/lemms/openllm}}
}}
```

## Repository

This model is hosted on Hugging Face Hub: https://huggingface.co/{repo_name}
"""
        
        # Save README.md
        readme_path = os.path.join(model_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card)
        
        print(f"✅ Model card saved to: {readme_path}")
        return readme_path
    
    def upload_model(self, model_dir: str, repo_name: str, model_size: str = "small") -> bool:
        """
        Upload the trained model to Hugging Face Hub.
        
        Args:
            model_dir: Directory containing the trained model
            repo_name: Name of the repository (without username)
            model_size: Size of the model
            
        Returns:
            True if upload successful, False otherwise
        """
        if not self.is_authenticated:
            print("❌ Not authenticated. Please run setup_authentication() first.")
            return False
        
        print(f"📤 Uploading model to Hugging Face Hub...")
        print(f"   - Model directory: {model_dir}")
        print(f"   - Repository: {self.username}/{repo_name}")
        print(f"   - Model size: {model_size}")
        
        try:
            # Create repository
            repo_id = f"{self.username}/{repo_name}"
            print(f"🔄 Creating repository: {repo_id}")
            
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=False
            )
            
            print(f"✅ Repository created: {repo_id}")
            
            # Create Hugging Face compatible files
            self.create_huggingface_config(model_dir, model_size)
            self.create_model_card(model_dir, repo_id, model_size)
            
            # Upload all files
            print(f"🔄 Uploading model files...")
            self.api.upload_folder(
                folder_path=model_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add OpenLLM {model_size} model with extended training"
            )
            
            print(f"✅ Model uploaded successfully!")
            print(f"   - Repository: https://huggingface.co/{repo_id}")
            print(f"   - Model files: {len(list(Path(model_dir).rglob('*')))} files")
            
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
    def fix_existing_upload(self, model_dir: str, repo_name: str = None) -> bool:
        """
        Fix an existing failed upload by re-uploading the model.
        
        Args:
            model_dir: Directory containing the trained model
            repo_name: Optional repository name (will generate if not provided)
            
        Returns:
            True if fix successful, False otherwise
        """
        print(f"🔧 Fixing existing upload...")
        print(f"   - Model directory: {model_dir}")
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            print(f"❌ Model directory not found: {model_dir}")
            return False
        
        # Generate repository name if not provided
        if not repo_name:
            repo_name = f"openllm-small-extended-8k"
        
        # Set up authentication
        if not self.setup_authentication():
            print("❌ Authentication failed")
            return False
        
        # Upload model
        return self.upload_model(model_dir, repo_name, "small")


def main():
    """Main function to run the upload fixer."""
    parser = argparse.ArgumentParser(
        description="Fix Hugging Face upload issues for OpenLLM training"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./openllm-trained",
        help="Directory containing the trained model (default: ./openllm-trained)"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        help="Repository name (without username). If not provided, will use default naming."
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "medium", "large"],
        default="small",
        help="Size of the model (default: small)"
    )
    parser.add_argument(
        "--setup-auth-only",
        action="store_true",
        help="Only set up authentication, don't upload"
    )
    
    args = parser.parse_args()
    
    print("🚀 OpenLLM - Training Upload Fixer")
    print("=" * 50)
    
    # Initialize fixer
    fixer = TrainingUploadFixer()
    
    # Set up authentication
    if not fixer.setup_authentication():
        print("❌ Authentication setup failed")
        print("\n🔧 Troubleshooting:")
        print("1. Get a Hugging Face token from https://huggingface.co/settings/tokens")
        print("2. Set environment variable: export HUGGING_FACE_HUB_TOKEN=your_token")
        print("3. Or run: huggingface-cli login")
        sys.exit(1)
    
    if args.setup_auth_only:
        print("✅ Authentication setup completed successfully!")
        return True
    
    # Fix upload
    success = fixer.fix_existing_upload(args.model_dir, args.repo_name)
    
    if success:
        print("\n🎉 Upload fix completed successfully!")
        print(f"   - Your model is now available on Hugging Face Hub")
        print(f"   - You can use it in your applications")
    else:
        print("\n❌ Upload fix failed")
        print("   - Check the error messages above")
        print("   - Verify your model directory exists")
        print("   - Ensure you have proper permissions")
        sys.exit(1)
    
    return success


if __name__ == "__main__":
    main()
