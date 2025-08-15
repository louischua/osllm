#!/usr/bin/env python3
"""
Manual dependency installation script for Hugging Face Spaces.

This script installs missing dependencies that might not be properly
installed through requirements.txt. Run this in the Space terminal.

Author: Louis Chua Bean Chong
License: GPL-3.0
"""

import subprocess
import sys
import os

def run_command(command, description):
    """
    Run a shell command and handle errors.
    
    Args:
        command: Command to run
        description: Description of what the command does
    """
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
        else:
            print(f"❌ Failed: {description}")
            print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False
    
    return True

def main():
    """Install all required dependencies manually."""
    print("🚀 Manual Dependency Installation for OpenLLM Training")
    print("=" * 60)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Core dependencies that might be missing
    dependencies = [
        ("sentencepiece>=0.1.99", "SentencePiece tokenization library (CRITICAL for OpenLLM)"),
        ("transformers>=4.30.0", "Hugging Face Transformers library"),
        ("datasets>=2.12.0", "Hugging Face Datasets library"),
        ("tokenizers>=0.13.0", "Fast tokenization library"),
        ("huggingface_hub>=0.34.0", "Hugging Face Hub integration"),
        ("accelerate>=0.20.0", "Distributed training acceleration"),
        ("torch>=2.0.0", "PyTorch deep learning framework"),
        ("gradio==4.44.1", "Gradio UI framework (fixed version)"),
        ("numpy>=1.24.0", "Numerical computing library"),
        ("pandas>=2.0.0", "Data manipulation library"),
        ("tqdm>=4.65.0", "Progress bars"),
        ("requests>=2.31.0", "HTTP library"),
    ]
    
    print(f"\n📦 Installing {len(dependencies)} dependencies...")
    
    success_count = 0
    for package, description in dependencies:
        command = f"pip install {package}"
        if run_command(command, description):
            success_count += 1
    
    print(f"\n" + "=" * 60)
    print(f"🎯 Installation Summary:")
    print(f"✅ Successful: {success_count}/{len(dependencies)}")
    print(f"❌ Failed: {len(dependencies) - success_count}/{len(dependencies)}")
    
    if success_count == len(dependencies):
        print("\n🎉 All dependencies installed successfully!")
        print("💡 You can now try the training again.")
    else:
        print("\n⚠️ Some dependencies failed to install.")
        print("💡 Check the error messages above and try again.")
    
    # Test critical imports
    print(f"\n🧪 Testing critical imports...")
    test_imports = [
        ("sentencepiece", "SentencePiece (CRITICAL)"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("torch", "PyTorch"),
        ("gradio", "Gradio"),
    ]
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} - Import successful")
        except ImportError as e:
            print(f"❌ {name} - Import failed: {e}")
    
    print(f"\n🔧 Manual installation complete!")
    print("💡 If imports still fail, try restarting the Space.")

if __name__ == "__main__":
    main()
