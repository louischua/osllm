#!/usr/bin/env python3
"""
Integration Guide: Add Authentication to Existing Training Code

This script shows how to integrate Hugging Face authentication into your
existing OpenLLM training code. Copy the relevant parts into your training script.

Usage:
    Use this as a reference to update your existing training code.
"""

import os
import sys
import json

try:
    from huggingface_hub import HfApi, login, whoami, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("❌ huggingface_hub not installed")
    sys.exit(1)


def setup_hf_authentication():
    """
    Set up Hugging Face authentication using GitHub secrets.
    Add this function to your training script.
    """
    print("🔐 Setting up Hugging Face Authentication")
    print("-" * 40)
    
    try:
        # Get token from GitHub secrets
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not found. Please set it in GitHub repository secrets.")
        
        # Login
        login(token=token)
        
        # Get user info
        api = HfApi()
        user_info = whoami()
        username = user_info["name"]
        
        print(f"✅ Authentication successful!")
        print(f"   - Username: {username}")
        print(f"   - Source: GitHub secrets")
        
        return api, username
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        raise


def upload_model_after_training(api, username, model_dir, model_size="small", steps=8000):
    """
    Upload the trained model to Hugging Face Hub.
    Call this function after your training completes.
    """
    try:
        # Create repository name
        repo_name = f"openllm-{model_size}-extended-{steps//1000}k"
        repo_id = f"{username}/{repo_name}"
        
        print(f"\n📤 Uploading model to {repo_id}")
        
        # Create repository
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        
        # Create model configuration
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
        
        # Create model card
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

## License
This model is released under the GNU General Public License v3.0.

## Repository
This model is hosted on Hugging Face Hub: https://huggingface.co/{repo_id}
"""
        
        readme_path = os.path.join(model_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card)
        
        # Upload all files
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add OpenLLM {model_size} model ({steps} steps)"
        )
        
        print(f"✅ Model uploaded successfully!")
        print(f"   - Repository: https://huggingface.co/{repo_id}")
        
        return repo_id
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise


# ============================================================================
# INTEGRATION EXAMPLE: How to modify your existing training code
# ============================================================================

def example_integration():
    """
    Example of how to integrate authentication into your existing training code.
    """
    print("🚀 Example: Integrating Authentication into Training")
    print("=" * 55)
    
    # Step 1: Set up authentication at the start
    print("\n1️⃣ Setting up authentication...")
    api, username = setup_hf_authentication()
    
    # Step 2: Your existing training code goes here
    print("\n2️⃣ Running your existing training code...")
    print("   - This is where your actual training happens")
    print("   - Training saves model to: ./openllm-trained")
    
    # Simulate training completion
    model_dir = "./openllm-trained"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create dummy model file
    with open(os.path.join(model_dir, "best_model.pt"), "w") as f:
        f.write("Dummy model file")
    
    print("   ✅ Training completed!")
    
    # Step 3: Upload model after training
    print("\n3️⃣ Uploading model...")
    repo_id = upload_model_after_training(
        api=api,
        username=username,
        model_dir=model_dir,
        model_size="small",
        steps=8000
    )
    
    print(f"\n🎉 Success! Model available at: https://huggingface.co/{repo_id}")


# ============================================================================
# CODE SNIPPETS FOR YOUR EXISTING TRAINING SCRIPT
# ============================================================================

def get_code_snippets():
    """Show code snippets to add to your existing training script."""
    snippets = """
# ============================================================================
# ADD THESE IMPORTS TO YOUR TRAINING SCRIPT
# ============================================================================

import os
from huggingface_hub import HfApi, login, whoami, create_repo
import json

# ============================================================================
# ADD THIS FUNCTION TO YOUR TRAINING SCRIPT
# ============================================================================

def setup_hf_authentication():
    \"\"\"Set up Hugging Face authentication using GitHub secrets.\"\"\"
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found. Please set it in GitHub repository secrets.")
    
    login(token=token)
    api = HfApi()
    user_info = whoami()
    username = user_info["name"]
    
    print(f"✅ Authentication successful: {username}")
    return api, username

# ============================================================================
# ADD THIS FUNCTION TO YOUR TRAINING SCRIPT
# ============================================================================

def upload_model_after_training(api, username, model_dir, model_size="small", steps=8000):
    \"\"\"Upload the trained model to Hugging Face Hub.\"\"\"
    repo_name = f"openllm-{model_size}-extended-{steps//1000}k"
    repo_id = f"{username}/{repo_name}"
    
    # Create repository
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    # Upload all files
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add OpenLLM {model_size} model ({steps} steps)"
    )
    
    print(f"✅ Model uploaded: https://huggingface.co/{repo_id}")
    return repo_id

# ============================================================================
# MODIFY YOUR MAIN TRAINING FUNCTION
# ============================================================================

def main():
    # Step 1: Set up authentication
    api, username = setup_hf_authentication()
    
    # Step 2: Your existing training code
    # ... your training code here ...
    
    # Step 3: Upload after training
    model_dir = "./openllm-trained"  # Your model directory
    repo_id = upload_model_after_training(api, username, model_dir)
    
    print(f"🎉 Training and upload completed!")

if __name__ == "__main__":
    main()
"""
    return snippets


def main():
    """Main function to demonstrate integration."""
    print("🔧 Integration Guide: Add Authentication to Existing Training")
    print("=" * 65)
    
    # Show example integration
    example_integration()
    
    # Show code snippets
    print("\n" + "="*65)
    print("📝 CODE SNIPPETS FOR YOUR EXISTING TRAINING SCRIPT")
    print("="*65)
    print(get_code_snippets())


if __name__ == "__main__":
    main()
