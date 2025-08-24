#!/usr/bin/env python3
"""
Hugging Face Authentication Setup Script

This script helps set up proper authentication for Hugging Face Hub operations
including model uploads, repository creation, and API access.

Features:
- Multiple authentication methods (token, CLI, environment variables)
- Authentication validation and testing
- Repository creation testing
- Clear error messages and troubleshooting

Usage:
    python setup_hf_auth.py [--token YOUR_TOKEN] [--test-upload]

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from huggingface_hub import HfApi, login, whoami
    from huggingface_hub.utils import HfHubHTTPError

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("❌ huggingface_hub not installed. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub>=0.34.0"])
        from huggingface_hub import HfApi, login, whoami
        from huggingface_hub.utils import HfHubHTTPError

        HF_AVAILABLE = True
        print("✅ huggingface_hub installed successfully")
    except Exception as e:
        print(f"❌ Failed to install huggingface_hub: {e}")
        sys.exit(1)


class HuggingFaceAuthSetup:
    """
    Comprehensive Hugging Face authentication setup and validation.

    This class provides methods to:
    1. Set up authentication using various methods
    2. Validate authentication status
    3. Test repository creation and upload capabilities
    4. Provide clear error messages and troubleshooting steps
    """

    def __init__(self):
        """Initialize the authentication setup."""
        self.api = None
        self.username = None
        self.is_authenticated = False

    def setup_authentication(self, token: Optional[str] = None) -> Dict[str, Any]:
        """
        Set up Hugging Face authentication using the best available method.

        Args:
            token: Optional Hugging Face token. If not provided, will try other methods.

        Returns:
            Dictionary with authentication status and details
        """
        print("🔐 Setting up Hugging Face Authentication")
        print("=" * 50)

        # Method 1: Use provided token
        if token:
            print(f"🔄 Method 1: Using provided token...")
            result = self._authenticate_with_token(token)
            if result["success"]:
                return result

        # Method 2: Check environment variable
        env_token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        if env_token:
            print(f"🔄 Method 2: Using environment variable token...")
            result = self._authenticate_with_token(env_token)
            if result["success"]:
                return result

        # Method 3: Try CLI login
        print(f"🔄 Method 3: Checking CLI login...")
        result = self._check_cli_login()
        if result["success"]:
            return result

        # Method 4: Interactive token input
        print(f"🔄 Method 4: Interactive token input...")
        result = self._interactive_token_setup()
        if result["success"]:
            return result

        # All methods failed
        return {
            "success": False,
            "error": "All authentication methods failed",
            "troubleshooting": self._get_troubleshooting_steps(),
        }

    def _authenticate_with_token(self, token: str) -> Dict[str, Any]:
        """
        Authenticate using a Hugging Face token.

        Args:
            token: Hugging Face API token

        Returns:
            Dictionary with authentication result
        """
        try:
            # Login with token
            login(token=token)

            # Test authentication
            self.api = HfApi()
            user_info = whoami()

            self.username = user_info["name"]
            self.is_authenticated = True

            print(f"✅ Authentication successful!")
            print(f"   - Username: {self.username}")
            print(f"   - Token: {token[:8]}...{token[-4:]}")

            return {
                "success": True,
                "username": self.username,
                "method": "token",
                "message": f"Authenticated as {self.username}",
            }

        except Exception as e:
            print(f"❌ Token authentication failed: {e}")
            return {"success": False, "error": str(e), "method": "token"}

    def _check_cli_login(self) -> Dict[str, Any]:
        """
        Check if user is already logged in via CLI.

        Returns:
            Dictionary with authentication result
        """
        try:
            # Try to get current user info
            user_info = whoami()

            self.api = HfApi()
            self.username = user_info["name"]
            self.is_authenticated = True

            print(f"✅ CLI authentication found!")
            print(f"   - Username: {self.username}")

            return {
                "success": True,
                "username": self.username,
                "method": "cli",
                "message": f"Already authenticated as {self.username}",
            }

        except Exception as e:
            print(f"❌ CLI authentication not found: {e}")
            return {"success": False, "error": str(e), "method": "cli"}

    def _interactive_token_setup(self) -> Dict[str, Any]:
        """
        Interactive token setup with user input.

        Returns:
            Dictionary with authentication result
        """
        print("\n📝 Interactive Token Setup")
        print("-" * 30)
        print("To get your Hugging Face token:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Click 'New token'")
        print("3. Give it a name (e.g., 'OpenLLM Training')")
        print("4. Select 'Write' role")
        print("5. Copy the generated token")
        print()

        try:
            token = input("Enter your Hugging Face token: ").strip()
            if not token:
                return {"success": False, "error": "No token provided", "method": "interactive"}

            return self._authenticate_with_token(token)

        except KeyboardInterrupt:
            print("\n❌ Token setup cancelled")
            return {"success": False, "error": "Setup cancelled by user", "method": "interactive"}

    def test_repository_creation(self, repo_name: str = "test-repo") -> Dict[str, Any]:
        """
        Test repository creation capabilities.

        Args:
            repo_name: Name of test repository to create

        Returns:
            Dictionary with test result
        """
        if not self.is_authenticated:
            return {"success": False, "error": "Not authenticated"}

        print(f"\n🧪 Testing Repository Creation")
        print("-" * 40)

        try:
            from huggingface_hub import create_repo, delete_repo

            # Create test repository
            repo_id = f"{self.username}/{repo_name}"
            print(f"🔄 Creating test repository: {repo_id}")

            create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=True,  # Make it private for testing
            )

            print(f"✅ Test repository created successfully: {repo_id}")

            # Clean up - delete the test repository
            print(f"🔄 Cleaning up test repository...")
            delete_repo(repo_id=repo_id, repo_type="model")
            print(f"✅ Test repository deleted")

            return {
                "success": True,
                "message": f"Repository creation test passed for {self.username}",
                "repo_id": repo_id,
            }

        except Exception as e:
            print(f"❌ Repository creation test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "troubleshooting": self._get_repo_troubleshooting_steps(),
            }

    def test_model_upload(self, test_file_path: str = None) -> Dict[str, Any]:
        """
        Test model upload capabilities.

        Args:
            test_file_path: Path to test file to upload. If None, creates a dummy file.

        Returns:
            Dictionary with test result
        """
        if not self.is_authenticated:
            return {"success": False, "error": "Not authenticated"}

        print(f"\n📤 Testing Model Upload")
        print("-" * 30)

        try:
            # Create test file if not provided
            if not test_file_path:
                test_file_path = "test_upload.txt"
                with open(test_file_path, "w") as f:
                    f.write("This is a test file for Hugging Face upload verification.\n")
                    f.write("Generated by OpenLLM authentication setup.\n")

            # Create test repository
            repo_id = f"{self.username}/test-upload-repo"

            from huggingface_hub import create_repo, delete_repo

            create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)

            # Upload test file
            print(f"🔄 Uploading test file to {repo_id}...")
            self.api.upload_file(
                path_or_fileobj=test_file_path,
                path_in_repo="test_file.txt",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Test upload from OpenLLM auth setup",
            )

            print(f"✅ Test upload successful!")
            print(f"   - Repository: {repo_id}")
            print(f"   - File: test_file.txt")

            # Clean up
            print(f"🔄 Cleaning up test repository...")
            delete_repo(repo_id=repo_id, repo_type="model")

            # Remove test file if we created it
            if test_file_path == "test_upload.txt":
                os.remove(test_file_path)

            print(f"✅ Test upload completed and cleaned up")

            return {
                "success": True,
                "message": f"Upload test passed for {self.username}",
                "repo_id": repo_id,
            }

        except Exception as e:
            print(f"❌ Upload test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "troubleshooting": self._get_upload_troubleshooting_steps(),
            }

    def _get_troubleshooting_steps(self) -> str:
        """Get general troubleshooting steps for authentication issues."""
        return """
🔧 Troubleshooting Steps:

1. **Get a Hugging Face Token:**
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Give it a name (e.g., "OpenLLM Training")
   - Select "Write" role for full access
   - Copy the generated token

2. **Set Environment Variable:**
   ```bash
   # Linux/macOS
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   
   # Windows PowerShell
   $env:HUGGING_FACE_HUB_TOKEN="your_token_here"
   
   # Windows Command Prompt
   set HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

3. **Use CLI Login:**
   ```bash
   huggingface-cli login
   # Then enter your token when prompted
   ```

4. **Verify Token Permissions:**
   - Make sure the token has "Write" role
   - Check that your account has permission to create repositories
   - Ensure you're not rate limited

5. **Check Network Connection:**
   - Verify internet connection
   - Check if huggingface.co is accessible
   - Try with a different network if needed
        """

    def _get_repo_troubleshooting_steps(self) -> str:
        """Get troubleshooting steps for repository creation issues."""
        return """
🔧 Repository Creation Troubleshooting:

1. **Check Account Permissions:**
   - Verify your account can create repositories
   - Check if you've reached repository limits
   - Ensure account is not suspended

2. **Repository Name Issues:**
   - Repository names must be lowercase
   - No special characters except hyphens and underscores
   - Must be unique within your namespace

3. **Network Issues:**
   - Check internet connection
   - Try again in a few minutes
   - Use a different network if needed

4. **Token Permissions:**
   - Ensure token has "Write" role
   - Generate a new token if needed
   - Check token hasn't expired
        """

    def _get_upload_troubleshooting_steps(self) -> str:
        """Get troubleshooting steps for upload issues."""
        return """
🔧 Upload Troubleshooting:

1. **File Size Limits:**
   - Check file size (max 5GB per file)
   - Split large files if needed
   - Use Git LFS for large files

2. **Repository Permissions:**
   - Ensure you have write access to the repository
   - Check if repository exists and is accessible
   - Verify repository type (model, dataset, space)

3. **Network Issues:**
   - Check internet connection stability
   - Try uploading smaller files first
   - Use a different network if needed

4. **Rate Limiting:**
   - Check if you've hit rate limits
   - Wait a few minutes and try again
   - Consider using a different account
        """

    def save_authentication_config(self, config_path: str = ".hf_auth_config") -> bool:
        """
        Save authentication configuration for future use.

        Args:
            config_path: Path to save configuration file

        Returns:
            True if saved successfully, False otherwise
        """
        if not self.is_authenticated:
            print("❌ Cannot save config: not authenticated")
            return False

        try:
            config = {
                "username": self.username,
                "authenticated": self.is_authenticated,
                "method": "token",  # We'll always use token method
                "timestamp": str(Path().absolute()),
            }

            with open(config_path, "w") as f:
                import json

                json.dump(config, f, indent=2)

            print(f"✅ Authentication config saved to: {config_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to save config: {e}")
            return False


def main():
    """Main function to run the authentication setup."""
    parser = argparse.ArgumentParser(
        description="Set up Hugging Face authentication for OpenLLM training"
    )
    parser.add_argument("--token", type=str, help="Hugging Face token to use for authentication")
    parser.add_argument(
        "--test-upload",
        action="store_true",
        help="Test repository creation and upload capabilities",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save authentication configuration for future use",
    )
    parser.add_argument(
        "--setup-auth-only",
        action="store_true",
        help="Only set up authentication, don't test upload",
    )

    args = parser.parse_args()

    print("🚀 OpenLLM - Hugging Face Authentication Setup")
    print("=" * 60)

    # Initialize authentication setup
    auth_setup = HuggingFaceAuthSetup()

    # Set up authentication
    result = auth_setup.setup_authentication(token=args.token)

    if not result["success"]:
        print(f"\n❌ Authentication failed: {result['error']}")
        if "troubleshooting" in result:
            print(result["troubleshooting"])
        sys.exit(1)

    print(f"\n✅ Authentication successful!")
    print(f"   - Username: {result['username']}")
    print(f"   - Method: {result['method']}")

    # If only setting up auth, exit here
    if args.setup_auth_only:
        print(f"\n🎉 Authentication setup completed successfully!")
        print(f"   - You can now upload models to Hugging Face Hub")
        print(f"   - Your models will be uploaded to: https://huggingface.co/{result['username']}")
        return True

    # Test repository creation if requested
    if args.test_upload:
        repo_result = auth_setup.test_repository_creation()
        if not repo_result["success"]:
            print(f"\n❌ Repository creation test failed: {repo_result['error']}")
            if "troubleshooting" in repo_result:
                print(repo_result["troubleshooting"])
            sys.exit(1)

        upload_result = auth_setup.test_model_upload()
        if not upload_result["success"]:
            print(f"\n❌ Upload test failed: {upload_result['error']}")
            if "troubleshooting" in upload_result:
                print(upload_result["troubleshooting"])
            sys.exit(1)

        print(f"\n🎉 All tests passed! You're ready to upload models.")

    # Save configuration if requested
    if args.save_config:
        auth_setup.save_authentication_config()

    print(f"\n🎉 Hugging Face authentication setup completed successfully!")
    print(f"   - You can now upload models to Hugging Face Hub")
    print(f"   - Your models will be uploaded to: https://huggingface.co/{result['username']}")

    return True


if __name__ == "__main__":
    main()
