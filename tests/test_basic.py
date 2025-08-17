#!/usr/bin/env python3
"""
Basic Tests for OpenLLM

Simple tests that don't require complex imports to ensure CI pipeline works.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
import unittest
from pathlib import Path


class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality tests."""

    def test_python_version(self):
        """Test that we're running Python 3.8+."""
        version = sys.version_info
        self.assertGreaterEqual(version.major, 3)
        self.assertGreaterEqual(version.minor, 8)
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")

    def test_required_files_exist(self):
        """Test that required files exist."""
        required_files = [
            "README.md",
            "requirements.txt",
            "deployment/huggingface/space_app_main.py",  # app.py was moved
            "deployment/huggingface/space_auth.py",      # space_auth_test.py was moved
            "scripts/training/training_with_auth.py",    # openllm_training_with_auth.py was moved
        ]

        for file_path in required_files:
            self.assertTrue(os.path.exists(file_path), f"Required file {file_path} not found")
            print(f"✅ {file_path} exists")

    def test_core_directory_structure(self):
        """Test that core directory structure exists."""
        core_path = Path("core/src")
        self.assertTrue(core_path.exists(), "core/src directory not found")

        # Check for key files
        key_files = ["model.py", "train_model.py", "inference_server.py", "data_loader.py"]

        for file_name in key_files:
            file_path = core_path / file_name
            self.assertTrue(file_path.exists(), f"Core file {file_name} not found")
            print(f"✅ core/src/{file_name} exists")

    def test_requirements_file_content(self):
        """Test that requirements.txt has content."""
        with open("requirements.txt", "r") as f:
            content = f.read().strip()

        self.assertGreater(len(content), 0, "requirements.txt is empty")
        self.assertIn("huggingface_hub", content, "huggingface_hub not in requirements.txt")
        self.assertIn("gradio", content, "gradio not in requirements.txt")
        print("✅ requirements.txt has valid content")

    def test_workflow_files_exist(self):
        """Test that GitHub workflow files exist."""
        workflow_files = [
            ".github/workflows/ci.yml",
            ".github/workflows/deploy-to-space.yml",
            ".github/workflows/sync-hf-space.yml",
        ]

        for file_path in workflow_files:
            self.assertTrue(os.path.exists(file_path), f"Workflow file {file_path} not found")
            print(f"✅ {file_path} exists")


if __name__ == "__main__":
    unittest.main()
