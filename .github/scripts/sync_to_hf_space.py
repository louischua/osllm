#!/usr/bin/env python3
"""
Synchronization Script for OpenLLM Repository Structure

This script provides automated synchronization between the main GitHub repository 
and the Hugging Face Space repository. It maintains a clean separation of concerns
while ensuring that core training functionality is properly distributed across
different platforms.

The script performs the following key operations:
1. Sets up the Hugging Face Space repository structure
2. Copies essential training files from the main repo to the HF Space
3. Creates appropriate directory structures for training infrastructure
4. Generates HF Space-specific configuration files
5. Maintains version consistency between repositories

This approach enables:
- Centralized development in the main GitHub repository
- Distributed training capabilities in Hugging Face Spaces
- Automated synchronization to reduce manual maintenance
- Clear separation between core code and training infrastructure

Author: Louis Chua Bean Chong
License: GPL-3.0
Version: 1.0.0
Last Updated: 2024
Last Sync Test: 2025-01-27
"""

import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def setup_hf_space():
    """
    Initialize and configure the Hugging Face Space repository.

    This function establishes the connection to Hugging Face Hub and ensures
    that the target Space repository exists and is properly configured for
    the OpenLLM training infrastructure.

    The function performs the following operations:
    - Initializes the Hugging Face API client
    - Creates or updates the target Space repository
    - Configures repository settings for public access
    - Handles any setup errors gracefully

    Returns:
        None

    Raises:
        Exception: If the Space repository cannot be created or configured
    """

    print("🔧 Setting up HF Space repository structure...")
    print("   This process ensures the target Space exists and is properly configured")

    # Configuration for the target Hugging Face Space
    # This should match the Space URL: https://huggingface.co/spaces/lemms/openllm
    space_repo = "lemms/openllm"

    # Initialize the Hugging Face API client for repository operations
    # This client handles authentication and API communication
    api = HfApi()

    try:
        # Create or update the Space repository
        # The exist_ok=True parameter ensures we don't fail if the repo already exists
        # private=False makes the Space publicly accessible for community use
        create_repo(
            repo_id=space_repo,
            repo_type="space",  # Specifies this is a Space, not a model or dataset
            exist_ok=True,  # Don't fail if repository already exists
            private=False,  # Make the Space publicly accessible
        )
        print(f"✅ HF Space repository ready: {space_repo}")
        print(f"   Repository URL: https://huggingface.co/spaces/{space_repo}")

    except Exception as e:
        # Handle any errors during Space setup
        # This could include authentication issues, network problems, or API limits
        print(f"⚠️ Space setup warning: {e}")
        print("   The sync process will continue, but manual Space setup may be required")


def create_space_structure():
    """
    Create the complete directory structure and files for the Hugging Face Space.

    This function builds the entire file structure needed for the OpenLLM training
    Space, including copying essential files from the main repository and creating
    Space-specific configuration files.

    The function performs the following operations:
    - Creates a temporary working directory for file operations
    - Establishes the required directory structure for training infrastructure
    - Copies core training files from the main repository
    - Generates HF Space-specific configuration files
    - Creates dependency and documentation files
    - Uploads all files to the HF Space repository

    Directory Structure Created:
    ├── training/          # Core training modules copied from main repo
    ├── models/           # Local storage for trained models
    ├── scripts/          # Training and utility scripts
    ├── configs/          # Model configuration files
    ├── requirements.txt  # Python dependencies for the Space
    └── README.md         # Space documentation

    Returns:
        None

    Note:
        This function uses a temporary directory to avoid conflicts with
        existing files and ensures clean file operations.
    """

    print("📁 Creating HF Space directory structure...")
    print("   Building complete file structure for training infrastructure")

    # Create a temporary directory for file operations
    # This ensures we don't interfere with existing files and provides a clean workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the main Space directory within the temporary workspace
        space_dir = Path(temp_dir) / "hf_space"
        space_dir.mkdir()

        print(f"   Working directory: {space_dir}")

        # Create the essential directory structure for the training Space
        # Each directory serves a specific purpose in the training workflow
        training_dir = space_dir / "training"
        models_dir = space_dir / "models"
        scripts_dir = space_dir / "scripts"
        configs_dir = space_dir / "configs"

        # Create all required directories
        training_dir.mkdir()
        models_dir.mkdir()
        scripts_dir.mkdir()
        configs_dir.mkdir()

        print("   ✅ Created directory structure:")
        print(f"      - {training_dir.name}/ (core training modules)")
        print(f"      - {models_dir.name}/ (trained model storage)")
        print(f"      - {scripts_dir.name}/ (utility scripts)")
        print(f"      - {configs_dir.name}/ (model configurations)")

        # Define the core files that need to be copied from the main repository
        # These files contain the essential training functionality
        core_files = [
            "core/src/model.py",  # Model architecture and implementation
            "core/src/train_model.py",  # Main training pipeline
            "core/src/train_tokenizer.py",  # Tokenizer training functionality
            "core/src/data_loader.py",  # Data loading and preprocessing
            "core/src/evaluate_model.py",  # Model evaluation and metrics
            "configs/small_model.json",  # Small model configuration
            "configs/medium_model.json",  # Medium model configuration
            "configs/large_model.json",  # Large model configuration
        ]

        # Copy each core file to the training directory
        # This ensures the HF Space has access to all necessary training components
        print("   📋 Copying core training files...")
        for file_path in core_files:
            if os.path.exists(file_path):
                # Copy the file to the training directory, preserving the filename
                dest_path = training_dir / Path(file_path).name
                shutil.copy2(file_path, dest_path)
                print(f"      ✅ Copied: {file_path} -> {dest_path}")
            else:
                print(f"      ⚠️ File not found: {file_path}")

        # Create the requirements.txt file for the HF Space
        # This file specifies all Python dependencies needed for training
        requirements_content = """# Core Machine Learning Dependencies
# PyTorch - Deep learning framework for model training and inference
torch>=2.0.0

# Hugging Face Ecosystem - Model loading, training, and tokenization
transformers>=4.30.0      # Pre-trained models and training utilities
datasets>=2.12.0          # Dataset loading and processing
tokenizers>=0.13.0        # Fast tokenization library
sentencepiece>=0.1.99     # SentencePiece tokenization
huggingface_hub>=0.34.0   # Hugging Face Hub integration
accelerate>=0.20.0        # Distributed training acceleration

# User Interface - Gradio for web-based training interface
gradio>=4.0.0             # Web UI framework for ML applications

# Data Processing and Utilities
numpy>=1.24.0             # Numerical computing library
pandas>=2.0.0             # Data manipulation and analysis
tqdm>=4.65.0              # Progress bars for long-running operations
psutil>=5.9.0             # System and process utilities

# Note: These versions are compatible with Hugging Face Spaces
# and provide stable training performance
"""

        # Copy HF Space-specific files from the main repository
        # These files are essential for the Space to function properly
        print("   📋 Copying HF Space-specific files...")

        # Copy app.py (Gradio interface) - Use the fixed version
        if os.path.exists("hf_space_app_fixed.py"):
            app_dest = space_dir / "app.py"
            shutil.copy2("hf_space_app_fixed.py", app_dest)
            print(f"      ✅ Copied: hf_space_app_fixed.py -> {app_dest}")
        elif os.path.exists("hf_space_app.py"):
            app_dest = space_dir / "app.py"
            shutil.copy2("hf_space_app.py", app_dest)
            print(f"      ✅ Copied: hf_space_app.py -> {app_dest}")
        else:
            print("      ⚠️ File not found: hf_space_app_fixed.py or hf_space_app.py")

        # Copy README.md (Space documentation)
        if os.path.exists("hf_space_README.md"):
            readme_dest = space_dir / "README.md"
            shutil.copy2("hf_space_README.md", readme_dest)
            print(f"      ✅ Copied: hf_space_README.md -> {readme_dest}")
        else:
            print("      ⚠️ File not found: hf_space_README.md")

        # Copy requirements.txt (Space dependencies)
        if os.path.exists("hf_space_requirements.txt"):
            req_dest = space_dir / "requirements.txt"
            shutil.copy2("hf_space_requirements.txt", req_dest)
            print(f"      ✅ Copied: hf_space_requirements.txt -> {req_dest}")
        else:
            # Fallback to generated requirements if HF Space file doesn't exist
            requirements_path = space_dir / "requirements.txt"
            with open(requirements_path, "w") as f:
                f.write(requirements_content)
            print(f"      ✅ Created: {requirements_path}")
            print("         Contains all necessary dependencies for training")

        print("✅ HF Space structure created successfully")
        print("   The Space is now ready for file upload and deployment")

        # Step 3: Upload files to HF Space
        print("\n📋 Step 3: Uploading files to Hugging Face Space...")
        upload_files_to_hf_space(space_dir)


def upload_files_to_hf_space(space_dir):
    """
    Upload all files from the local space directory to the Hugging Face Space.

    This function uploads all the prepared files to the target HF Space repository,
    ensuring that the Space has all the necessary training infrastructure and files.

    Args:
        space_dir (Path): Path to the local space directory containing files to upload

    Returns:
        None

    Note:
        This function uses the HF_TOKEN environment variable for authentication
        and uploads files to the lemms/openllm Space repository.
    """

    print("📤 Uploading files to Hugging Face Space...")
    print("   This process uploads all training infrastructure files to the HF Space")

    # Configuration for the target Hugging Face Space
    space_repo = "lemms/openllm"

    # Initialize the Hugging Face API client
    api = HfApi()

    try:
        # Upload all files from the space directory
        # This uploads the entire directory structure to the HF Space
        api.upload_folder(
            folder_path=str(space_dir),
            repo_id=space_repo,
            repo_type="space",
            commit_message="feat: Sync training infrastructure from main repository",
        )

        print(f"✅ Files uploaded successfully to {space_repo}")
        print(f"   Space URL: https://huggingface.co/spaces/{space_repo}")
        print("   All training infrastructure files are now available in the Space")

    except Exception as e:
        print(f"❌ Error uploading files to HF Space: {e}")
        print("   Please check the HF_TOKEN and repository permissions")
        raise


def main():
    """
    Main synchronization function that orchestrates the entire sync process.

    This function coordinates the complete synchronization workflow between
    the main GitHub repository and the Hugging Face Space. It ensures that
    all necessary components are properly set up and configured.

    The function performs the following operations in sequence:
    1. Sets up the Hugging Face Space repository
    2. Creates the complete file structure
    3. Copies essential training files
    4. Generates configuration files
    5. Reports completion status

    This function is designed to be called from GitHub Actions or manually
    to maintain synchronization between repositories.

    Returns:
        None

    Note:
        This function provides comprehensive logging to track the sync process
        and identify any issues that may arise during synchronization.
    """

    print("🔄 Starting OpenLLM repository sync process...")
    print("=" * 60)
    print("This process will synchronize core training functionality")
    print("from the main GitHub repository to the Hugging Face Space.")
    print("=" * 60)

    # Step 1: Initialize the Hugging Face Space repository
    # This ensures the target Space exists and is properly configured
    print("\n📋 Step 1: Setting up Hugging Face Space repository...")
    setup_hf_space()

    # Step 2: Create the complete file structure for the Space
    # This includes copying files, generating configuration, and uploading to HF
    print("\n📋 Step 2: Creating and uploading Space file structure...")
    create_space_structure()

    # Report successful completion
    print("\n" + "=" * 60)
    print("✅ OpenLLM repository sync process completed successfully!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("   1. Check the Hugging Face Space for uploaded files")
    print("   2. Test the training functionality in the Space")
    print("   3. Monitor the Space for any issues or improvements needed")
    print("   4. Future updates will be automatically synced")
    print("\n🔗 Space URL: https://huggingface.co/spaces/lemms/openllm")


if __name__ == "__main__":
    # Execute the main synchronization function when the script is run directly
    # This allows the script to be used both as a module and as a standalone tool
    main()
