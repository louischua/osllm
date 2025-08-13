#!/usr/bin/env python3
"""
Script to update the README.md file on Hugging Face Hub with proper YAML metadata

This script fixes the YAML metadata warning by uploading the corrected README.md
file to the Hugging Face Hub repository.

Author: Louis Chua Bean Chong
License: GPL-3.0
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_file

def update_hf_readme():
    """
    Update the README.md file on Hugging Face Hub with proper YAML metadata.
    
    This function uploads the corrected README.md file to fix the YAML metadata warning.
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Model details
    repo_id = "lemms/openllm-small-extended-7k"
    readme_path = "exports/huggingface-7k/huggingface/README.md"
    
    print(f"🔧 Updating README.md on Hugging Face Hub")
    print(f"Repository: {repo_id}")
    print(f"README path: {readme_path}")
    print("=" * 60)
    
    # Check if README file exists
    if not os.path.exists(readme_path):
        print(f"❌ README file not found: {readme_path}")
        return False
    
    try:
        # Initialize Hugging Face API
        print("🔐 Initializing Hugging Face API...")
        api = HfApi()
        
        # Upload the corrected README.md file
        print(f"📤 Uploading corrected README.md...")
        
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Fix YAML metadata warning in README.md",
            commit_description="""
            Add proper YAML frontmatter to README.md to resolve the metadata warning.
            
            Changes:
            - Added YAML metadata section with model information
            - Included language, license, tags, datasets, and metrics
            - Added model-index with performance results
            - Specified library_name and pipeline_tag
            
            This resolves the warning: "empty or missing yaml metadata in repo card"
            """
        )
        
        print(f"✅ README.md successfully updated on: https://huggingface.co/{repo_id}")
        print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
        print(f"📝 YAML metadata warning should now be resolved")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to update README: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to execute the README update process.
    
    This function provides user feedback and handles the update workflow
    with proper error handling and status reporting.
    """
    
    print("🔧 OpenLLM 7k Model README Update Script")
    print("=" * 50)
    
    # Check if huggingface_hub is available
    try:
        import huggingface_hub
        print(f"✅ Hugging Face Hub library found: {huggingface_hub.__version__}")
    except ImportError:
        print("❌ Hugging Face Hub library not found!")
        print("   Please install it with: pip install huggingface_hub")
        return False
    
    # Execute the update
    success = update_hf_readme()
    
    if success:
        print("\n🎉 README update completed successfully!")
        print("\n📋 What was fixed:")
        print("   1. Added YAML frontmatter with model metadata")
        print("   2. Included proper tags and license information")
        print("   3. Added model performance metrics")
        print("   4. Specified library and pipeline information")
        print("\n🔍 Check the model page to verify the warning is gone!")
    else:
        print("\n💥 README update failed!")
        print("\n🔧 Troubleshooting:")
        print("   1. Check your Hugging Face authentication")
        print("   2. Verify the README file exists")
        print("   3. Ensure you have write access to the repository")
        print("   4. Check internet connection")
        sys.exit(1)

if __name__ == "__main__":
    main()
