#!/usr/bin/env python3
# Copyright (C) 2024 Louis Chua Bean Chong
#
# This file is part of OpenLLM.
#
# OpenLLM is dual-licensed:
# 1. For open source use: GNU General Public License v3.0
# 2. For commercial use: Commercial License (contact for details)
#
# See LICENSE and docs/LICENSES.md for full license information.

"""
Main test runner for OpenLLM.

This script runs all tests in the test suite and provides a comprehensive report.
It can be run individually or as part of a CI/CD pipeline.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --verbose          # Run with verbose output
    python tests/run_tests.py --coverage         # Run with coverage report
    python tests/run_tests.py test_model         # Run specific test module
"""

import unittest
import sys
import os
import argparse
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add the core/src directory to the path
core_src_path = project_root / "core" / "src"
sys.path.insert(0, str(core_src_path))


def run_tests(test_pattern=None, verbose=False, coverage=False):
    """
    Run the test suite.
    
    Args:
        test_pattern (str): Pattern to match test files/modules
        verbose (bool): Whether to run tests in verbose mode
        coverage (bool): Whether to generate coverage report
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Discover tests
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    
    if test_pattern:
        # Run specific test module
        suite = loader.loadTestsFromName(f"tests.{test_pattern}")
    else:
        # Run all tests
        suite = loader.discover(str(test_dir), pattern="test_*.py")
    
    # Configure test runner
    if verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    # Run tests with coverage if requested
    if coverage:
        try:
            import coverage
            cov = coverage.Coverage()
            cov.start()
        except ImportError:
            print("Warning: coverage module not found. Running tests without coverage.")
            coverage = False
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Stop coverage if it was started
    if coverage and 'cov' in locals():
        cov.stop()
        cov.save()
        
        # Generate coverage report
        print("\n" + "="*60)
        print("COVERAGE REPORT")
        print("="*60)
        cov.report()
        
        # Generate HTML report
        cov.html_report(directory='htmlcov')
        print(f"\nHTML coverage report generated in 'htmlcov' directory")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        
        # Print failure details
        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"  {test}: {traceback}")
        
        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"  {test}: {traceback}")
        
        return False


def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        'torch',
        'numpy',
        'fastapi',
        'pydantic',
        'sentencepiece'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are available")
    return True


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run OpenLLM test suite")
    parser.add_argument(
        'test_pattern',
        nargs='?',
        help='Pattern to match test files/modules (e.g., test_model)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Run tests in verbose mode'
    )
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Generate coverage report'
    )
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        return 0 if check_dependencies() else 1
    
    # Check dependencies before running tests
    if not check_dependencies():
        return 1
    
    # Run tests
    success = run_tests(
        test_pattern=args.test_pattern,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
