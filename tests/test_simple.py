#!/usr/bin/env python3
"""
Simple Tests for OpenLLM

Very basic tests that don't require any project imports.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import unittest
import os
import sys
import json
from pathlib import Path


class TestSimpleFunctionality(unittest.TestCase):
    """Simple functionality tests that don't require project imports."""
    
    def test_python_basics(self):
        """Test basic Python functionality."""
        # Test basic math
        self.assertEqual(2 + 2, 4)
        self.assertEqual(10 * 5, 50)
        
        # Test string operations
        text = "Hello World"
        self.assertEqual(text.upper(), "HELLO WORLD")
        self.assertEqual(len(text), 11)
        
        # Test list operations
        numbers = [1, 2, 3, 4, 5]
        self.assertEqual(sum(numbers), 15)
        self.assertEqual(len(numbers), 5)
        
        print("✅ Basic Python functionality works")
    
    def test_file_system_access(self):
        """Test basic file system operations."""
        # Test current directory
        current_dir = os.getcwd()
        self.assertIsInstance(current_dir, str)
        self.assertGreater(len(current_dir), 0)
        
        # Test that we can list files
        files = os.listdir(current_dir)
        self.assertIsInstance(files, list)
        self.assertGreater(len(files), 0)
        
        print(f"✅ File system access works in: {current_dir}")
    
    def test_path_operations(self):
        """Test pathlib operations."""
        # Test path creation
        test_path = Path("test_file.txt")
        self.assertIsInstance(test_path, Path)
        
        # Test path joining
        joined_path = Path("dir1") / "dir2" / "file.txt"
        self.assertEqual(str(joined_path), "dir1/dir2/file.txt")
        
        print("✅ Path operations work")
    
    def test_json_operations(self):
        """Test JSON operations."""
        # Test JSON encoding
        data = {"name": "test", "value": 42, "list": [1, 2, 3]}
        json_str = json.dumps(data)
        self.assertIsInstance(json_str, str)
        
        # Test JSON decoding
        decoded_data = json.loads(json_str)
        self.assertEqual(decoded_data["name"], "test")
        self.assertEqual(decoded_data["value"], 42)
        
        print("✅ JSON operations work")
    
    def test_system_info(self):
        """Test system information access."""
        # Test Python version
        version = sys.version_info
        self.assertGreaterEqual(version.major, 3)
        self.assertGreaterEqual(version.minor, 8)
        
        # Test platform info
        platform = sys.platform
        self.assertIsInstance(platform, str)
        self.assertGreater(len(platform), 0)
        
        print(f"✅ System info: Python {version.major}.{version.minor}.{version.micro} on {platform}")
    
    def test_environment_variables(self):
        """Test environment variable access."""
        # Test that we can access environment variables
        env_vars = os.environ
        self.assertIsInstance(env_vars, dict)
        
        # Test specific environment variables that should exist
        if 'PATH' in env_vars:
            self.assertIsInstance(env_vars['PATH'], str)
            print("✅ Environment variables accessible")
        else:
            print("⚠️ PATH environment variable not found")
    
    def test_import_capability(self):
        """Test that we can import standard library modules."""
        # Test importing common modules
        try:
            import datetime
            import random
            import math
            import re
            print("✅ Standard library imports work")
        except ImportError as e:
            self.fail(f"Failed to import standard library: {e}")


if __name__ == "__main__":
    unittest.main()
