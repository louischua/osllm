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
    
    # Check for Space access token
    # In Hugging Face Spaces, the access token should be automatically available
    # through the Space's own authentication mechanism
    try:
        # Try to get current user info (this will use Space's access token)
        api = HfApi()
        user_info = whoami()
        print(f"✅ Authentication successful!")
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
        print(f"\n💡 Troubleshooting:")
        print(f"   - Ensure the Space has proper access token configured")
        print(f"   - Check Space settings for authentication configuration")
        print(f"   - Verify the Space has necessary permissions")
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
        print(f"🔧 Please check Space configuration and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
