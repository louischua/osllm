#!/usr/bin/env python3
"""
Deployment script for OpenLLM Training Space
Uploads the training space files to Hugging Face Space
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return None

def deploy_to_huggingface_space():
    """Deploy the training space to Hugging Face"""
    
    print("🚀 Deploying OpenLLM Training Space to Hugging Face...")
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found. Please run this script from the training_space directory.")
        return False
    
    # Check if huggingface-cli is installed
    if run_command("huggingface-cli --version", "Checking Hugging Face CLI") is None:
        print("❌ Hugging Face CLI not found. Please install it with: pip install huggingface_hub")
        return False
    
    # Login to Hugging Face (if not already logged in)
    print("🔐 Checking Hugging Face authentication...")
    login_check = run_command("huggingface-cli whoami", "Checking authentication")
    if login_check is None:
        print("⚠️ Not logged in to Hugging Face. Please run: huggingface-cli login")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        return False
    
    print(f"✅ Logged in as: {login_check.strip()}")
    
    # Upload files to the space
    space_name = "lemms/openllm"
    
    print(f"📤 Uploading files to {space_name}...")
    
    # Upload app.py
    if run_command(f"huggingface-cli upload {space_name} app.py app.py", "Uploading app.py") is None:
        return False
    
    # Upload requirements.txt
    if run_command(f"huggingface-cli upload {space_name} requirements.txt requirements.txt", "Uploading requirements.txt") is None:
        return False
    
    # Upload README.md
    if run_command(f"huggingface-cli upload {space_name} README.md README.md", "Uploading README.md") is None:
        return False
    
    print("🎉 Deployment completed successfully!")
    print(f"🌐 Your training space is available at: https://huggingface.co/spaces/{space_name}")
    print("⏳ The space will take a few minutes to build and become available.")
    
    return True

def main():
    """Main deployment function"""
    print("=" * 60)
    print("🚀 OpenLLM Training Space Deployment")
    print("=" * 60)
    
    # Check if we're in the training_space directory
    current_dir = Path.cwd()
    if current_dir.name != "training_space":
        print("❌ Please run this script from the training_space directory")
        print(f"Current directory: {current_dir}")
        print("Expected directory: .../training_space")
        return
    
    # Check required files
    required_files = ["app.py", "requirements.txt", "README.md"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return
    
    print("✅ All required files found")
    
    # Deploy to Hugging Face
    if deploy_to_huggingface_space():
        print("\n🎉 Deployment successful!")
        print("\n📋 Next steps:")
        print("1. Wait 5-10 minutes for the space to build")
        print("2. Visit https://huggingface.co/spaces/lemms/openllm")
        print("3. Test the training interface")
        print("4. Share with the community!")
    else:
        print("\n❌ Deployment failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
