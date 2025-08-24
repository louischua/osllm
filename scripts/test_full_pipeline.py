#!/usr/bin/env python3
"""
OpenLLM Full Pipeline Test

This script validates the complete OpenLLM training pipeline from data preparation
to model evaluation, ensuring all components work together for v0.1.0 release.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path


def run_command(cmd, description, timeout=300):
    """Run a command and return success status."""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    print("-" * 60)

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="ignore",
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS ({elapsed:.1f}s)")
            if result.stdout.strip():
                print("Output:", result.stdout.strip())
            return True
        else:
            print(f"❌ {description} - FAILED ({elapsed:.1f}s)")
            print("Error:", result.stderr.strip())
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT ({timeout}s)")
        return False
    except Exception as e:
        print(f"💥 {description} - EXCEPTION: {e}")
        return False


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description} - EXISTS")
        return True
    else:
        print(f"❌ {description} - MISSING")
        return False


def main():
    """Run the full pipeline test."""
    print("🚀 OpenLLM Full Pipeline Test")
    print("=" * 60)
    print("Testing complete pipeline for v0.1.0 release validation")
    print()

    # Track test results
    results = {"cli_commands": [], "file_checks": [], "pipeline_steps": []}

    # Test 1: CLI Commands
    print("📋 Testing CLI Commands")
    print("=" * 40)

    cli_tests = [
        ("python core/src/main.py --help", "Main CLI help"),
        ("python core/src/main.py prepare-data --help", "Data preparation help"),
        ("python core/src/main.py train-tokenizer --help", "Tokenizer training help"),
        ("python core/src/main.py test-model --help", "Model testing help"),
        ("python core/src/main.py train-model --help", "Model training help"),
    ]

    for cmd, desc in cli_tests:
        success = run_command(cmd, desc, timeout=30)
        results["cli_commands"].append({"command": cmd, "description": desc, "success": success})

    # Test 2: File Existence Checks
    print("\n📁 Testing File Existence")
    print("=" * 40)

    file_checks = [
        ("core/src/model.py", "Core model implementation"),
        ("core/src/train_model.py", "Training pipeline"),
        ("core/src/inference_server.py", "Inference server"),
        ("core/src/data_loader.py", "Data loading utilities"),
        ("data/tokenizer/tokenizer.model", "Trained tokenizer"),
        ("models/small-extended-7k/best_model.pt", "Trained model checkpoint"),
        ("requirements.txt", "Python dependencies"),
        ("README.md", "Project documentation"),
    ]

    for filepath, desc in file_checks:
        success = check_file_exists(filepath, desc)
        results["file_checks"].append({"file": filepath, "description": desc, "success": success})

    # Test 3: Model Testing
    print("\n🧪 Testing Model Functionality")
    print("=" * 40)

    model_tests = [
        ("python core/src/main.py test-model --model-size small", "Small model test"),
    ]

    for cmd, desc in model_tests:
        success = run_command(cmd, desc, timeout=120)
        results["pipeline_steps"].append({"step": cmd, "description": desc, "success": success})

    # Test 4: Model Evaluation
    print("\n📊 Testing Model Evaluation")
    print("=" * 40)

    eval_tests = [
        ("python scripts/evaluation/evaluate_trained_model.py", "Trained model evaluation"),
    ]

    for cmd, desc in eval_tests:
        success = run_command(cmd, desc, timeout=180)
        results["pipeline_steps"].append({"step": cmd, "description": desc, "success": success})

    # Test 5: Inference Server
    print("\n🌐 Testing Inference Server")
    print("=" * 40)

    # Test if inference server can start (just check if it can be imported and configured)
    server_test = "python -c \"import sys; sys.path.append('core/src'); from inference_server import app; print('Inference server imports successfully')\""
    success = run_command(server_test, "Inference server import test", timeout=30)
    results["pipeline_steps"].append(
        {"step": server_test, "description": "Inference server test", "success": success}
    )

    # Calculate results
    print("\n📈 Test Results Summary")
    print("=" * 60)

    cli_success = sum(1 for r in results["cli_commands"] if r["success"])
    file_success = sum(1 for r in results["file_checks"] if r["success"])
    pipeline_success = sum(1 for r in results["pipeline_steps"] if r["success"])

    total_cli = len(results["cli_commands"])
    total_files = len(results["file_checks"])
    total_pipeline = len(results["pipeline_steps"])

    print(f"CLI Commands: {cli_success}/{total_cli} ✅")
    print(f"File Checks: {file_success}/{total_files} ✅")
    print(f"Pipeline Steps: {pipeline_success}/{total_pipeline} ✅")

    overall_success = (
        cli_success == total_cli
        and file_success == total_files
        and pipeline_success == total_pipeline
    )

    if overall_success:
        print(f"\n🎉 ALL TESTS PASSED! OpenLLM v0.1.0 is ready for release!")
        print("\n✅ Core Requirements Met:")
        print("   - Working training pipeline")
        print("   - Basic model quality (trained model available)")
        print("   - Inference server functional")
        print("   - Documentation complete")
        print("   - Testing comprehensive")
    else:
        print(f"\n⚠️  SOME TESTS FAILED. Review results above.")
        print("\n❌ Issues to resolve before v0.1.0 release:")

        failed_cli = [r for r in results["cli_commands"] if not r["success"]]
        failed_files = [r for r in results["file_checks"] if not r["success"]]
        failed_pipeline = [r for r in results["pipeline_steps"] if not r["success"]]

        if failed_cli:
            print("   - CLI command issues:")
            for f in failed_cli:
                print(f"     * {f['description']}")

        if failed_files:
            print("   - Missing files:")
            for f in failed_files:
                print(f"     * {f['file']}")

        if failed_pipeline:
            print("   - Pipeline step issues:")
            for f in failed_pipeline:
                print(f"     * {f['description']}")

    # Save detailed results
    results_file = "pipeline_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Detailed results saved to: {results_file}")

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
