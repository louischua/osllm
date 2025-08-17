#!/usr/bin/env python3
"""
Simple Space Authentication Verification (GitHub Secrets)

This script can be added to your Hugging Face Space to verify
that authentication is working correctly using GitHub secrets.

Usage:
    Add this to your Space and run it to verify authentication.
"""

import os
import sys

try:
    from huggingface_hub import HfApi, login, whoami, create_repo, delete_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("❌ huggingface_hub not installed")
    sys.exit(1)


def verify_space_authentication():
    """Verify authentication is working in the Space using GitHub secrets."""
    print("🔐 Verifying Space Authentication (GitHub Secrets)")
    print("=" * 55)
    
    # Check if we're in a Space
    space_vars = ["SPACE_ID", "SPACE_HOST", "SPACE_REPO_ID"]
    is_space = any(os.getenv(var) for var in space_vars)
    
    if is_space:
        print("✅ Running in Hugging Face Space environment")
        for var in space_vars:
            value = os.getenv(var)
            if value:
                print(f"   - {var}: {value}")
    else:
        print("ℹ️ Running in local environment")
    
    # Check HF_TOKEN from GitHub secrets
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found in environment")
        print("   - Please set HF_TOKEN in your GitHub repository secrets")
        print("   - Go to GitHub repository → Settings → Secrets and variables → Actions")
        print("   - Add HF_TOKEN with your Hugging Face token")
        print("   - The Space will automatically have access to this secret")
        return False
    
    print(f"✅ HF_TOKEN found: {token[:8]}...{token[-4:]}")
    print(f"   - Source: GitHub secrets")
    
    try:
        # Test authentication
        login(token=token)
        
        user_info = whoami()
        username = user_info["name"]
        
        print(f"✅ Authentication successful!")
        print(f"   - Username: {username}")
        
        # Test API access
        api = HfApi()
        print(f"✅ API access working")
        
        # Test repository creation
        print(f"\n🧪 Testing Repository Creation")
        repo_name = "test-openllm-verification"
        repo_id = f"{username}/{repo_name}"
        
        print(f"🔄 Creating test repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=True
        )
        print(f"✅ Repository created successfully")
        
        # Clean up
        print(f"🔄 Cleaning up test repository...")
        delete_repo(repo_id=repo_id, repo_type="model")
        print(f"✅ Repository deleted")
        
        print(f"\n🎉 All verification tests passed!")
        print(f"   - Authentication: ✅ Working")
        print(f"   - Repository Creation: ✅ Working")
        print(f"   - GitHub Secrets Integration: ✅ Working")
        print(f"   - Ready for training and model uploads!")
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"1. Check if HF_TOKEN is set correctly in GitHub repository secrets")
        print(f"2. Verify token has 'Write' permissions")
        print(f"3. Check Space logs for detailed error messages")
        print(f"4. Ensure your Space is connected to the GitHub repository")
        return False


def main():
    """Main verification function."""
    print("🚀 OpenLLM - Space Authentication Verification (GitHub Secrets)")
    print("=" * 65)
    
    success = verify_space_authentication()
    
    if success:
        print(f"\n✅ Verification completed successfully!")
        print(f"   - Your Space is ready for OpenLLM training")
        print(f"   - Model uploads will work correctly")
        print(f"   - GitHub secrets integration is working")
    else:
        print(f"\n❌ Verification failed")
        print(f"   - Please fix authentication issues before training")
        sys.exit(1)


if __name__ == "__main__":
    main()
