#!/usr/bin/env python3
"""
Hugging Face Space Authentication Setup (GitHub Secrets)

This script helps set up proper authentication for Hugging Face Spaces
using GitHub secrets to ensure training and model uploads work correctly.

Features:
- Sets up authentication for Hugging Face Spaces using GitHub secrets
- Tests repository creation and upload capabilities
- Provides configuration for Space environments
- Validates authentication before training

Usage:
    python setup_hf_space_auth.py

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
import json
from pathlib import Path

try:
    from huggingface_hub import HfApi, login, whoami, create_repo
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("❌ huggingface_hub not installed")
    sys.exit(1)


class HuggingFaceSpaceAuthSetup:
    """
    Sets up authentication for Hugging Face Spaces training and upload using GitHub secrets.
    """
    
    def __init__(self):
        """Initialize the Space authentication setup."""
        self.api = None
        self.username = None
        self.is_authenticated = False
        self.space_config = {}
        
    def setup_space_authentication(self) -> bool:
        """
        Set up authentication specifically for Hugging Face Spaces using GitHub secrets.
        
        Returns:
            True if authentication successful, False otherwise
        """
        print("🚀 Hugging Face Space Authentication Setup (GitHub Secrets)")
        print("=" * 65)
        
        # Check if we're in a Hugging Face Space
        is_space = self._check_if_in_space()
        
        if is_space:
            print("✅ Running in Hugging Face Space environment")
            return self._setup_space_env_auth()
        else:
            print("ℹ️ Running in local environment")
            return self._setup_local_auth()
    
    def _check_if_in_space(self) -> bool:
        """Check if we're running in a Hugging Face Space."""
        space_env_vars = [
            "SPACE_ID",
            "SPACE_HOST",
            "SPACE_REPO_ID",
            "HF_HUB_ENABLE_HF_TRANSFER"
        ]
        
        for var in space_env_vars:
            if os.getenv(var):
                print(f"   - Found Space environment variable: {var}")
                return True
        
        return False
    
    def _setup_space_env_auth(self) -> bool:
        """Set up authentication in Hugging Face Space environment using GitHub secrets."""
        print("\n🔐 Setting up Space Environment Authentication (GitHub Secrets)")
        print("-" * 60)
        
        # In Spaces with GitHub integration, authentication is handled via HF_TOKEN from GitHub secrets
        token = os.getenv("HF_TOKEN")
        if not token:
            print("❌ HF_TOKEN not found in Space environment")
            print("   - This should be set in your GitHub repository secrets")
            print("   - Go to your GitHub repository → Settings → Secrets and variables → Actions")
            print("   - Add HF_TOKEN with your Hugging Face token")
            print("   - The Space will automatically have access to this secret")
            return False
        
        try:
            # Login with the token
            login(token=token)
            
            # Test authentication
            self.api = HfApi()
            user_info = whoami()
            
            self.username = user_info["name"]
            self.is_authenticated = True
            
            print(f"✅ Space authentication successful!")
            print(f"   - Username: {self.username}")
            print(f"   - Token: {token[:8]}...{token[-4:]}")
            print(f"   - Source: GitHub secrets")
            
            # Store Space configuration
            self.space_config = {
                "is_space": True,
                "username": self.username,
                "space_id": os.getenv("SPACE_ID"),
                "space_host": os.getenv("SPACE_HOST"),
                "space_repo_id": os.getenv("SPACE_REPO_ID"),
                "auth_source": "github_secrets"
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Space authentication failed: {e}")
            return False
    
    def _setup_local_auth(self) -> bool:
        """Set up authentication in local environment."""
        print("\n🔐 Setting up Local Environment Authentication")
        print("-" * 45)
        
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
            
            print("❌ Authentication failed")
            return False
    
    def test_repository_creation(self) -> bool:
        """
        Test repository creation capabilities.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_authenticated:
            print("❌ Not authenticated")
            return False
        
        print(f"\n🧪 Testing Repository Creation")
        print("-" * 35)
        
        try:
            # Create test repository
            repo_name = "test-openllm-space-auth"
            repo_id = f"{self.username}/{repo_name}"
            
            print(f"🔄 Creating test repository: {repo_id}")
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=True
            )
            
            print(f"✅ Test repository created successfully")
            
            # Clean up - delete the test repository
            from huggingface_hub import delete_repo
            print(f"🔄 Cleaning up test repository...")
            delete_repo(repo_id=repo_id, repo_type="model")
            print(f"✅ Test repository deleted")
            
            return True
            
        except Exception as e:
            print(f"❌ Repository creation test failed: {e}")
            return False
    
    def test_model_upload(self) -> bool:
        """
        Test model upload capabilities.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_authenticated:
            print("❌ Not authenticated")
            return False
        
        print(f"\n📤 Testing Model Upload")
        print("-" * 25)
        
        try:
            # Create test file
            test_file = "test_upload.txt"
            with open(test_file, "w") as f:
                f.write("This is a test file for Hugging Face Space upload verification.\n")
                f.write("Generated by setup_hf_space_auth.py (GitHub Secrets)\n")
            
            # Create test repository
            repo_name = "test-upload-openllm-space"
            repo_id = f"{self.username}/{repo_name}"
            
            from huggingface_hub import create_repo, delete_repo
            
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=True
            )
            
            # Upload test file
            print(f"🔄 Uploading test file to {repo_id}...")
            self.api.upload_file(
                path_or_fileobj=test_file,
                path_in_repo="test_file.txt",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Test upload from OpenLLM Space auth setup (GitHub Secrets)"
            )
            
            print(f"✅ Test upload successful!")
            
            # Clean up
            print(f"🔄 Cleaning up...")
            delete_repo(repo_id=repo_id, repo_type="model")
            os.remove(test_file)
            print(f"✅ Cleanup completed")
            
            return True
            
        except Exception as e:
            print(f"❌ Upload test failed: {e}")
            # Clean up test file if it exists
            if os.path.exists("test_upload.txt"):
                os.remove("test_upload.txt")
            return False
    
    def create_space_config(self) -> bool:
        """
        Create configuration file for Hugging Face Space.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_authenticated:
            print("❌ Not authenticated")
            return False
        
        print(f"\n📝 Creating Space Configuration")
        print("-" * 30)
        
        config = {
            "authentication": {
                "username": self.username,
                "authenticated": self.is_authenticated,
                "method": "github_secrets" if self.space_config.get("auth_source") == "github_secrets" else "local_token"
            },
            "space_config": self.space_config,
            "model_upload": {
                "default_repo_prefix": f"{self.username}/openllm",
                "supported_sizes": ["small", "medium", "large"],
                "upload_enabled": True
            },
            "training": {
                "checkpoint_upload": True,
                "final_model_upload": True,
                "create_model_card": True
            }
        }
        
        # Save configuration
        config_path = ".hf_space_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Space configuration saved to: {config_path}")
        return True
    
    def generate_space_instructions(self) -> str:
        """Generate instructions for setting up Hugging Face Space with GitHub secrets."""
        instructions = f"""
🎯 Hugging Face Space Setup Instructions (GitHub Secrets)

1. **Set up GitHub Repository Secrets:**
   - Go to your GitHub repository: https://github.com/your-username/your-repo
   - Click on "Settings" tab
   - Click on "Secrets and variables" → "Actions"
   - Click "New repository secret"
   - Add a new secret:
     - Name: HF_TOKEN
     - Value: Your Hugging Face token (get from https://huggingface.co/settings/tokens)

2. **Verify Authentication:**
   - Run this script in your Space to verify authentication works
   - The script will test repository creation and upload capabilities
   - GitHub secrets are automatically available in Hugging Face Spaces

3. **Training Configuration:**
   - Your training script should use the authentication setup
   - Model uploads will go to: https://huggingface.co/{self.username}/openllm-*
   - Checkpoints will be saved during training

4. **Expected Results:**
   - Training will complete successfully
   - Model will be uploaded to Hugging Face Hub
   - Repository will be created with proper model files
   - Model card and configuration will be generated

5. **Troubleshooting:**
   - If upload fails, check HF_TOKEN is set correctly in GitHub secrets
   - Verify token has "Write" permissions
   - Check Space logs for detailed error messages
   - Ensure your Space is connected to the GitHub repository
        """
        return instructions


def main():
    """Main function to run the Space authentication setup."""
    print("🚀 OpenLLM - Hugging Face Space Authentication Setup (GitHub Secrets)")
    print("=" * 70)
    
    # Initialize setup
    auth_setup = HuggingFaceSpaceAuthSetup()
    
    # Set up authentication
    if not auth_setup.setup_space_authentication():
        print("\n❌ Authentication setup failed")
        print("\n🔧 Troubleshooting:")
        print("1. Get a Hugging Face token from https://huggingface.co/settings/tokens")
        print("2. In GitHub: Set HF_TOKEN in Repository secrets (Settings → Secrets and variables → Actions)")
        print("3. Locally: Set HUGGING_FACE_HUB_TOKEN environment variable")
        sys.exit(1)
    
    print(f"\n✅ Authentication successful!")
    print(f"   - Username: {auth_setup.username}")
    
    # Test repository creation
    repo_success = auth_setup.test_repository_creation()
    
    # Test model upload
    upload_success = auth_setup.test_model_upload()
    
    # Create configuration
    config_success = auth_setup.create_space_config()
    
    # Summary
    print(f"\n📊 Setup Results")
    print("-" * 20)
    print(f"✅ Authentication: PASSED")
    print(f"{'✅' if repo_success else '❌'} Repository Creation: {'PASSED' if repo_success else 'FAILED'}")
    print(f"{'✅' if upload_success else '❌'} Model Upload: {'PASSED' if upload_success else 'FAILED'}")
    print(f"{'✅' if config_success else '❌'} Configuration: {'PASSED' if config_success else 'FAILED'}")
    
    if repo_success and upload_success:
        print(f"\n🎉 Space authentication setup completed successfully!")
        print(f"   - You can now run training in Hugging Face Spaces")
        print(f"   - Model uploads will work correctly")
        print(f"   - Your models will be uploaded to: https://huggingface.co/{auth_setup.username}")
        
        # Show Space instructions
        print(auth_setup.generate_space_instructions())
    else:
        print(f"\n⚠️ Some tests failed. Check the error messages above.")
        sys.exit(1)
    
    return True


if __name__ == "__main__":
    main()
