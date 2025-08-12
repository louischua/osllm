#!/usr/bin/env python3
"""
Script to push the OpenLLM 6k model to Hugging Face Hub
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

def push_model_to_hf():
    """Push the 6k model to Hugging Face Hub."""
    
    # Model details
    model_name = "openllm-small-extended-6k"
    repo_id = f"lemms/{model_name}"
    model_path = "exports/huggingface-6k/huggingface"
    
    print(f"🚀 Pushing OpenLLM 6k model to Hugging Face Hub")
    print(f"Repository: {repo_id}")
    print(f"Model path: {model_path}")
    print("=" * 50)
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        return False
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Create repository (will fail if it already exists, which is fine)
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=False,
                exist_ok=True
            )
            print(f"✅ Repository created/verified: {repo_id}")
        except Exception as e:
            print(f"⚠️ Repository creation warning: {e}")
        
        # Upload the model files
        print(f"📤 Uploading model files...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add OpenLLM Small Extended 6k model",
            commit_description="""
            OpenLLM Small Extended model trained for 6,000 steps.
            
            - Model: GPT-style transformer (35.8M parameters)
            - Training: 6,000 steps on SQUAD Wikipedia passages
            - Tokenizer: SentencePiece BPE (32k vocabulary)
            - License: GPL-3.0 / Commercial available
            
            For more details, see: https://github.com/louischua/openllm
            """
        )
        
        print(f"✅ Model successfully uploaded to: https://huggingface.co/{repo_id}")
        print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to push model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = push_model_to_hf()
    if success:
        print("\n🎉 Model push completed successfully!")
    else:
        print("\n💥 Model push failed!")
        sys.exit(1)
