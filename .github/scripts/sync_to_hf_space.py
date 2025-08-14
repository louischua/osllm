#!/usr/bin/env python3
"""
Sync Script for OpenLLM Repository Structure

This script synchronizes the main GitHub repository with the Hugging Face Space
while maintaining separate model repositories.

Author: Louis Chua Bean Chong
License: GPL-3.0
"""

import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def setup_hf_space():
    """Setup the HF Space repository structure."""
    
    print("🔧 Setting up HF Space repository structure...")
    
    # Configuration
    space_repo = "lemms/openllm"
    api = HfApi()
    
    try:
        # Create or update the space
        create_repo(
            repo_id=space_repo,
            repo_type="space",
            exist_ok=True,
            private=False
        )
        print(f"✅ HF Space repository ready: {space_repo}")
        
    except Exception as e:
        print(f"⚠️ Space setup warning: {e}")

def create_space_structure():
    """Create the HF Space directory structure."""
    
    print("📁 Creating HF Space directory structure...")
    
    # Create temporary directory for HF Space
    with tempfile.TemporaryDirectory() as temp_dir:
        space_dir = Path(temp_dir) / "hf_space"
        space_dir.mkdir()
        
        # Create directory structure
        (space_dir / "training").mkdir()
        (space_dir / "models").mkdir()
        (space_dir / "scripts").mkdir()
        (space_dir / "configs").mkdir()
        
        # Copy core files needed for training
        core_files = [
            "core/src/model.py",
            "core/src/train_model.py", 
            "core/src/train_tokenizer.py",
            "core/src/data_loader.py",
            "core/src/evaluate_model.py",
            "configs/small_model.json",
            "configs/medium_model.json",
            "configs/large_model.json"
        ]
        
        for file_path in core_files:
            if os.path.exists(file_path):
                dest_path = space_dir / "training" / Path(file_path).name
                shutil.copy2(file_path, dest_path)
                print(f"📋 Copied: {file_path} -> {dest_path}")
        
        # Create requirements.txt for HF Space
        requirements_content = """# Core ML dependencies
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
tokenizers>=0.13.0
sentencepiece>=0.1.99
huggingface_hub>=0.34.0
accelerate>=0.20.0

# Gradio for UI
gradio>=4.0.0

# Utilities
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
psutil>=5.9.0
"""
        
        with open(space_dir / "requirements.txt", "w") as f:
            f.write(requirements_content)
        
        print("✅ HF Space structure created successfully")

def main():
    """Main sync function."""
    
    print("🔄 Starting OpenLLM repository sync...")
    
    # Setup HF Space
    setup_hf_space()
    
    # Create space structure
    create_space_structure()
    
    print("✅ Sync process completed!")

if __name__ == "__main__":
    main()
