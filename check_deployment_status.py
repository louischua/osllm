#!/usr/bin/env python3
"""
Check Deployment Status Script

This script helps verify the deployment status and provides guidance
on monitoring the GitHub Actions workflow.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
from pathlib import Path

def check_local_files():
    """Check if all required files are present locally."""
    print("🔍 Checking Local Files for Deployment")
    print("=" * 45)
    
    required_files = [
        "app.py",
        "requirements.txt", 
        "space_auth_test.py",
        "openllm_training_with_auth.py",
        "integrate_auth_into_training.py",
        "setup_hf_space_auth.py",
        "verify_space_auth.py",
        ".github/workflows/deploy-to-space.yml"
    ]
    
    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_present = False
    
    return all_present

def check_workflow_configuration():
    """Check if the workflow is properly configured."""
    print("\n🔧 Checking Workflow Configuration")
    print("=" * 40)
    
    workflow_file = ".github/workflows/deploy-to-space.yml"
    if os.path.exists(workflow_file):
        print(f"✅ Workflow file exists: {workflow_file}")
        
        # Check for key configuration
        with open(workflow_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        checks = [
            ("Trigger on push to main", "branches: [ main, master ]" in content),
            ("Python setup", "actions/setup-python" in content),
            ("Hugging Face Hub", "huggingface_hub" in content),
            ("HF_TOKEN secret", "HF_TOKEN" in content),
            ("SPACE_ID secret", "SPACE_ID" in content),
            ("File deployment", "upload_file" in content)
        ]
        
        for check_name, is_present in checks:
            status = "✅" if is_present else "❌"
            print(f"   {status} {check_name}")
            
        return True
    else:
        print(f"❌ Workflow file missing: {workflow_file}")
        return False

def provide_monitoring_guidance():
    """Provide guidance on monitoring the deployment."""
    print("\n📊 Monitoring GitHub Actions Deployment")
    print("=" * 45)
    
    print("""
🚀 To check if the deployment is working:

1. **GitHub Actions Status:**
   - Go to: https://github.com/louischua/osllm/actions
   - Look for "Deploy to Hugging Face Space" workflow
   - Check if it's running or completed successfully
   - Click on the workflow to see detailed logs

2. **Space Deployment Status:**
   - Visit: https://huggingface.co/spaces/lemms/openllm
   - Check if the Space interface has updated
   - Look for the new web interface with tabs

3. **Expected Workflow Steps:**
   ✅ Checkout repository
   ✅ Set up Python 3.10
   ✅ Install dependencies
   ✅ Deploy to Hugging Face Space
   ✅ Verify deployment

4. **Common Issues to Check:**
   - HF_TOKEN secret is set in GitHub repository secrets
   - SPACE_ID secret is set to "lemms/openllm"
   - Space has write permissions for the token
   - No network connectivity issues

5. **Success Indicators:**
   - Workflow completes without errors
   - Files appear in your Space
   - Space web interface loads with new tabs
   - Authentication test works
""")

def check_space_connectivity():
    """Check if we can connect to the Space."""
    print("\n🌐 Checking Space Connectivity")
    print("=" * 35)
    
    try:
        import requests
        space_url = "https://huggingface.co/spaces/lemms/openllm"
        response = requests.get(space_url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Space is accessible: {space_url}")
            print("   - Space is running and responding")
        else:
            print(f"⚠️ Space returned status: {response.status_code}")
            print(f"   - URL: {space_url}")
            
    except ImportError:
        print("ℹ️ requests module not available for connectivity check")
    except Exception as e:
        print(f"❌ Cannot connect to Space: {e}")
        print(f"   - URL: https://huggingface.co/spaces/lemms/openllm")

def main():
    """Main function to check deployment status."""
    print("🚀 GitHub to Space Deployment Status Check")
    print("=" * 50)
    
    # Check local files
    files_ok = check_local_files()
    
    # Check workflow configuration
    workflow_ok = check_workflow_configuration()
    
    # Check space connectivity
    check_space_connectivity()
    
    # Provide monitoring guidance
    provide_monitoring_guidance()
    
    # Summary
    print("\n📋 Deployment Status Summary")
    print("=" * 30)
    
    if files_ok and workflow_ok:
        print("✅ Local setup is ready for deployment")
        print("✅ Workflow configuration looks correct")
        print("\n🎯 Next Steps:")
        print("1. Check GitHub Actions: https://github.com/louischua/osllm/actions")
        print("2. Monitor workflow execution")
        print("3. Verify Space deployment: https://huggingface.co/spaces/lemms/openllm")
        print("4. Test authentication and training")
    else:
        print("❌ Some issues found - please check the details above")
        print("🔧 Fix the issues before deployment")
    
    print("\n📞 If you need help:")
    print("- Check GitHub Actions logs for specific errors")
    print("- Verify GitHub secrets are set correctly")
    print("- Ensure Space permissions are configured")

if __name__ == "__main__":
    main()
