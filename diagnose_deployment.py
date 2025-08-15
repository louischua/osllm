#!/usr/bin/env python3
"""
Deployment Diagnostic Script

This script helps identify potential issues that could cause GitHub Actions
deployment failures.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
import subprocess
from pathlib import Path

def check_required_files():
    """Check if all required files for deployment exist."""
    print("🔍 Checking Required Files for Deployment")
    print("=" * 45)
    
    required_files = [
        "app.py",
        "requirements.txt",
        "space_auth_test.py",
        "openllm_training_with_auth.py",
        "integrate_auth_into_training.py",
        "setup_hf_space_auth.py",
        "verify_space_auth.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ {len(missing_files)} required files missing!")
        return False
    else:
        print(f"\n✅ All {len(required_files)} required files present")
        return True

def check_python_syntax():
    """Check Python syntax for key files."""
    print("\n🔍 Checking Python Syntax")
    print("=" * 30)
    
    python_files = [
        "app.py",
        "space_auth_test.py",
        "openllm_training_with_auth.py"
    ]
    
    syntax_errors = []
    
    for file_path in python_files:
        if not os.path.exists(file_path):
            print(f"⚠️ {file_path} - File not found")
            continue
            
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'py_compile', file_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ {file_path} - Valid syntax")
            else:
                print(f"❌ {file_path} - Syntax error: {result.stderr.strip()}")
                syntax_errors.append(file_path)
                
        except Exception as e:
            print(f"❌ {file_path} - Error checking syntax: {e}")
            syntax_errors.append(file_path)
    
    if syntax_errors:
        print(f"\n❌ {len(syntax_errors)} files have syntax errors!")
        return False
    else:
        print(f"\n✅ All Python files have valid syntax")
        return True

def check_requirements_txt():
    """Check requirements.txt file."""
    print("\n🔍 Checking Requirements.txt")
    print("=" * 30)
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt - MISSING")
        return False
    
    try:
        with open("requirements.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            print("❌ requirements.txt - EMPTY")
            return False
        
        lines = content.strip().split('\n')
        print(f"✅ requirements.txt - {len(lines)} lines")
        
        # Check for basic structure
        has_dependencies = any(line.strip() and not line.strip().startswith('#') for line in lines)
        if has_dependencies:
            print("✅ Contains dependencies")
        else:
            print("❌ No dependencies found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def check_workflow_files():
    """Check GitHub Actions workflow files."""
    print("\n🔍 Checking Workflow Files")
    print("=" * 30)
    
    workflow_files = [
        ".github/workflows/ci.yml",
        ".github/workflows/deploy-to-space.yml",
        ".github/workflows/sync-hf-space.yml"
    ]
    
    missing_workflows = []
    
    for workflow_file in workflow_files:
        if os.path.exists(workflow_file):
            size = os.path.getsize(workflow_file)
            print(f"✅ {workflow_file} ({size} bytes)")
        else:
            print(f"❌ {workflow_file} - MISSING")
            missing_workflows.append(workflow_file)
    
    if missing_workflows:
        print(f"\n❌ {len(missing_workflows)} workflow files missing!")
        return False
    else:
        print(f"\n✅ All workflow files present")
        return True

def check_git_status():
    """Check git status and recent commits."""
    print("\n🔍 Checking Git Status")
    print("=" * 25)
    
    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            if result.stdout.strip():
                print("⚠️ Uncommitted changes detected:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"  {line}")
            else:
                print("✅ No uncommitted changes")
        
        # Check recent commits
        result = subprocess.run(
            ['git', 'log', '--oneline', '-5'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("\n📋 Recent commits:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"  {line}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking git status: {e}")
        return False

def check_space_connectivity():
    """Check if we can reach the Hugging Face Space."""
    print("\n🔍 Checking Space Connectivity")
    print("=" * 35)
    
    try:
        import requests
        
        space_url = "https://huggingface.co/spaces/lemms/openllm"
        response = requests.get(space_url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Space is accessible: {space_url}")
            print("   - Space is running and responding")
            return True
        else:
            print(f"⚠️ Space returned status: {response.status_code}")
            print(f"   - URL: {space_url}")
            return False
            
    except ImportError:
        print("ℹ️ requests module not available for connectivity check")
        return True
    except Exception as e:
        print(f"❌ Cannot connect to Space: {e}")
        print(f"   - URL: https://huggingface.co/spaces/lemms/openllm")
        return False

def main():
    """Main diagnostic function."""
    print("🚀 Deployment Diagnostic for OpenLLM")
    print("=" * 50)
    
    checks = [
        ("Required Files", check_required_files),
        ("Python Syntax", check_python_syntax),
        ("Requirements.txt", check_requirements_txt),
        ("Workflow Files", check_workflow_files),
        ("Git Status", check_git_status),
        ("Space Connectivity", check_space_connectivity)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n📊 Diagnostic Summary")
    print("=" * 25)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\n🎯 Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Deployment should work.")
        print("\n💡 If deployment still fails, check:")
        print("   - GitHub Actions logs for specific error messages")
        print("   - GitHub repository secrets (HF_TOKEN, SPACE_ID)")
        print("   - Space permissions and resource limits")
    else:
        print("❌ Some checks failed. Please fix issues before deployment.")
        print("\n🔧 Common fixes:")
        print("   - Add missing required files")
        print("   - Fix Python syntax errors")
        print("   - Update requirements.txt")
        print("   - Commit and push changes")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
