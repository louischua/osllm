#!/usr/bin/env python3
"""
Test Deployment Files

This script tests all the files required for deployment to ensure they exist
and have valid Python syntax.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
import py_compile
from pathlib import Path


def test_file_exists(file_path):
    """Test if a file exists."""
    if os.path.exists(file_path):
        print(f"✅ {file_path} exists")
        return True
    else:
        print(f"❌ {file_path} missing")
        return False


def test_file_content(file_path):
    """Test if a file has content."""
    if os.path.getsize(file_path) > 0:
        print(f"✅ {file_path} has content ({os.path.getsize(file_path)} bytes)")
        return True
    else:
        print(f"❌ {file_path} is empty")
        return False


def test_python_syntax(file_path):
    """Test if a Python file has valid syntax."""
    try:
        py_compile.compile(file_path, doraise=True)
        print(f"✅ {file_path} has valid Python syntax")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ {file_path} has syntax errors: {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path} error: {e}")
        return False


def main():
    """Main function to test all deployment files."""
    print("🔍 Testing Deployment Files")
    print("=" * 50)
    
    # Required files for deployment
    required_files = [
        "app.py",
        "requirements.txt", 
        "space_auth_test.py",
        "openllm_training_with_auth.py",
        "integrate_auth_into_training.py",
        "setup_hf_space_auth.py",
        "verify_space_auth.py"
    ]
    
    # Python files that need syntax validation
    python_files = [
        "app.py",
        "space_auth_test.py",
        "openllm_training_with_auth.py",
        "integrate_auth_into_training.py",
        "setup_hf_space_auth.py",
        "verify_space_auth.py"
    ]
    
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📋 Files in directory: {len(os.listdir('.'))}")
    print()
    
    # Test file existence
    print("🔍 Testing file existence...")
    missing_files = []
    for file_path in required_files:
        if not test_file_exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("\n✅ All required files exist")
    
    # Test file content
    print("\n🔍 Testing file content...")
    empty_files = []
    for file_path in required_files:
        if not test_file_content(file_path):
            empty_files.append(file_path)
    
    if empty_files:
        print(f"\n❌ Empty files: {empty_files}")
        return False
    
    print("\n✅ All files have content")
    
    # Test Python syntax
    print("\n🔍 Testing Python syntax...")
    syntax_errors = []
    for file_path in python_files:
        if not test_python_syntax(file_path):
            syntax_errors.append(file_path)
    
    if syntax_errors:
        print(f"\n❌ Syntax errors in: {syntax_errors}")
        return False
    
    print("\n✅ All Python files have valid syntax")
    
    # Test Python version
    print(f"\n🐍 Python version: {sys.version}")
    print(f"🐍 Python executable: {sys.executable}")
    
    print("\n🎉 All deployment file tests passed!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
