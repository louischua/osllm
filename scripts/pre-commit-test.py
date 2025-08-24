#!/usr/bin/env python3
"""
Pre-commit Testing Script for OpenLLM

This script validates files and dependencies before pushing to GitHub
to prevent workflow failures.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path


def check_python_syntax():
    """Check Python syntax for all .py files."""
    print("🔍 Checking Python syntax...")

    python_files = list(Path(".").rglob("*.py"))
    failed_files = []

    for py_file in python_files:
        try:
            # Skip virtual environment and git directories
            if "venv" in str(py_file) or ".git" in str(py_file):
                continue

            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(py_file)], capture_output=True, text=True
            )

            if result.returncode == 0:
                print(f"  ✅ {py_file}")
            else:
                print(f"  ❌ {py_file}: {result.stderr.strip()}")
                failed_files.append(py_file)

        except Exception as e:
            print(f"  ❌ {py_file}: {e}")
            failed_files.append(py_file)

    if failed_files:
        print(f"❌ {len(failed_files)} files have syntax errors")
        return False
    else:
        print(f"✅ All {len(python_files)} Python files have valid syntax")
        return True


def check_required_files():
    """Check if required files exist."""
    print("🔍 Checking required files...")

    required_files = [
        "README.md",
        "requirements.txt",
        ".github/workflows/ci.yml",
        ".github/workflows/deploy-to-space.yml",
        ".github/workflows/sync-hf-space.yml",
    ]

    optional_files = ["app.py", "space_auth_test.py", "openllm_training_with_auth.py"]

    missing_required = []
    missing_optional = []

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            missing_required.append(file_path)

    for file_path in optional_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️ {file_path} - NOT FOUND (optional)")
            missing_optional.append(file_path)

    if missing_required:
        print(f"❌ {len(missing_required)} required files missing")
        return False
    else:
        print(f"✅ All required files present")
        return True


def check_dependencies():
    """Check if dependencies can be installed."""
    print("🔍 Checking dependencies...")

    try:
        # Try to install requirements
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            print("✅ Dependencies can be installed")
            return True
        else:
            print(f"❌ Dependency installation failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Dependency check timed out")
        return False
    except Exception as e:
        print(f"❌ Dependency check failed: {e}")
        return False


def check_imports():
    """Check if key modules can be imported."""
    print("🔍 Checking imports...")

    key_modules = ["huggingface_hub", "gradio", "torch", "transformers", "numpy", "pandas"]

    failed_imports = []

    for module in key_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)

    if failed_imports:
        print(f"❌ {len(failed_imports)} modules failed to import")
        return False
    else:
        print(f"✅ All {len(key_modules)} modules can be imported")
        return True


def check_workflow_syntax():
    """Check GitHub Actions workflow syntax."""
    print("🔍 Checking workflow syntax...")

    workflow_files = [
        ".github/workflows/ci.yml",
        ".github/workflows/deploy-to-space.yml",
        ".github/workflows/sync-hf-space.yml",
    ]

    failed_workflows = []

    for workflow_file in workflow_files:
        if not os.path.exists(workflow_file):
            print(f"  ❌ {workflow_file} - MISSING")
            failed_workflows.append(workflow_file)
            continue

        try:
            # Basic YAML syntax check
            with open(workflow_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for basic YAML structure
            if "name:" in content and "on:" in content and "jobs:" in content:
                print(f"  ✅ {workflow_file}")
            else:
                print(f"  ❌ {workflow_file} - Invalid YAML structure")
                failed_workflows.append(workflow_file)

        except Exception as e:
            print(f"  ❌ {workflow_file}: {e}")
            failed_workflows.append(workflow_file)

    if failed_workflows:
        print(f"❌ {len(failed_workflows)} workflows have issues")
        return False
    else:
        print(f"✅ All {len(workflow_files)} workflows have valid syntax")
        return True


def main():
    """Main function to run all checks."""
    print("🚀 Pre-commit Testing for OpenLLM")
    print("=" * 50)

    checks = [
        ("Python Syntax", check_python_syntax),
        ("Required Files", check_required_files),
        ("Dependencies", check_dependencies),
        ("Imports", check_imports),
        ("Workflow Syntax", check_workflow_syntax),
    ]

    results = []

    for check_name, check_func in checks:
        print(f"\n📋 {check_name}")
        print("-" * 30)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results.append((check_name, False))

    # Summary
    print("\n📊 Pre-commit Test Summary")
    print("=" * 30)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")

    print(f"\n🎯 Results: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All checks passed! Ready to commit.")
        return 0
    else:
        print("❌ Some checks failed. Please fix issues before committing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
