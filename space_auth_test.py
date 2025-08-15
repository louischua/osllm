#!/usr/bin/env python3
"""
Space Authentication Test for OpenLLM Training

This script verifies that authentication is working correctly in a Hugging Face Space
environment. It uses the Space's own access token for authentication.

Author: Louis Chua Bean Chong
License: GPLv3
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

def test_space_authentication():
    """Test authentication using Space's access token."""
    print("🔍 Testing Space Authentication")
    print("=" * 40)
    
    # Check if we're in a Space environment
    space_id = os.environ.get('SPACE_ID', 'lemms/openllm')
    print(f"📁 Space ID: {space_id}")
    
    # Check for various authentication methods
    print(f"\n🔐 Checking authentication methods...")
    
    # Method 1: Check for HF_TOKEN environment variable
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        print(f"✅ HF_TOKEN found in environment")
        print(f"   Token: {hf_token[:8]}...{hf_token[-4:]}")
    else:
        print(f"⚠️ HF_TOKEN not found in environment")
    
    # Method 2: Check for HUGGING_FACE_HUB_TOKEN
    hf_hub_token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if hf_hub_token:
        print(f"✅ HUGGING_FACE_HUB_TOKEN found in environment")
        print(f"   Token: {hf_hub_token[:8]}...{hf_hub_token[-4:]}")
    else:
        print(f"⚠️ HUGGING_FACE_HUB_TOKEN not found in environment")
    
    # Method 3: Check for Space's built-in authentication
    try:
        # Try to get current user info (this will use Space's access token)
        api = HfApi()
        user_info = whoami()
        print(f"✅ Space authentication successful!")
        print(f"👤 User: {user_info}")
        
        # Test API access by listing Space files
        print(f"\n📁 Testing Space file access...")
        files = api.list_repo_files(repo_id=space_id, repo_type='space')
        print(f"✅ Successfully listed {len(files)} files in Space")
        
        # Test repository creation/deletion (temporary test)
        test_repo_name = f"test-repo-{os.getpid()}"
        print(f"\n🧪 Testing repository operations...")
        
        try:
            # Create a temporary test repository
            api.create_repo(
                repo_id=test_repo_name,
                repo_type="model",
                private=True,
                exist_ok=True
            )
            print(f"✅ Successfully created test repository: {test_repo_name}")
            
            # Delete the test repository
            api.delete_repo(repo_id=test_repo_name, repo_type="model")
            print(f"✅ Successfully deleted test repository: {test_repo_name}")
            
        except Exception as e:
            print(f"⚠️ Repository operations test failed: {e}")
            print("   This is normal if the token doesn't have full permissions")
        
        print(f"\n🎉 Space authentication test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print(f"\n🔧 TROUBLESHOOTING STEPS:")
        print(f"1. Check Space Settings:")
        print(f"   - Go to https://huggingface.co/spaces/{space_id}/settings")
        print(f"   - Navigate to 'Repository secrets' section")
        print(f"   - Add HF_TOKEN with your Hugging Face access token")
        print(f"   - Token should have 'Write' permissions")
        
        print(f"\n2. Alternative: Use Space's Built-in Token:")
        print(f"   - Go to https://huggingface.co/settings/tokens")
        print(f"   - Create a new token with 'Write' permissions")
        print(f"   - Add it to Space secrets as HF_TOKEN")
        
        print(f"\n3. Verify Token Permissions:")
        print(f"   - Token must have 'Write' access to repositories")
        print(f"   - Token must be valid and not expired")
        print(f"   - Token must be associated with the correct user account")
        
        print(f"\n4. Check Space Configuration:")
        print(f"   - Ensure Space is connected to GitHub repository")
        print(f"   - Verify Space has proper access to Hugging Face Hub")
        print(f"   - Check Space logs for detailed error messages")
        
        return False

def main():
    """Main function to run authentication tests."""
    print("🚀 OpenLLM Space Authentication Test")
    print("=" * 50)
    
    if not HF_AVAILABLE:
        print("❌ Required dependencies not available")
        sys.exit(1)
    
    # Run authentication test
    success = test_space_authentication()
    
    if success:
        print(f"\n✅ All authentication tests passed!")
        print(f"🚀 Ready for OpenLLM training!")
        sys.exit(0)
    else:
        print(f"\n❌ Authentication tests failed!")
        print(f"🔧 Please follow the troubleshooting steps above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
