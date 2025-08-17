#!/usr/bin/env python3
"""
Test script to verify all dependencies are properly installed for OpenLLM training.

This script checks if all required libraries are available and can be imported
without errors. It's useful for debugging dependency issues in Hugging Face Spaces.

Author: Louis Chua Bean Chong
License: GPL-3.0
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """
    Test if a module can be imported successfully.
    
    Args:
        module_name: Name of the module to import
        package_name: Optional package name for pip installation reference
        
    Returns:
        bool: True if import successful, False otherwise
    """
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - Import successful")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - Import failed: {e}")
        if package_name:
            print(f"   💡 Try installing with: pip install {package_name}")
        return False

def main():
    """Test all required dependencies for OpenLLM training."""
    print("🔍 Testing OpenLLM Training Dependencies")
    print("=" * 50)
    
    # Core ML Framework
    print("\n📊 Core Machine Learning Framework:")
    test_import("torch", "torch")
    test_import("torchvision", "torchvision")
    test_import("torchaudio", "torchaudio")
    
    # Hugging Face Ecosystem
    print("\n🤗 Hugging Face Ecosystem:")
    test_import("transformers", "transformers")
    test_import("datasets", "datasets")
    test_import("tokenizers", "tokenizers")
    test_import("sentencepiece", "sentencepiece")  # CRITICAL for OpenLLM
    test_import("huggingface_hub", "huggingface_hub")
    test_import("accelerate", "accelerate")
    
    # UI Framework
    print("\n🎨 User Interface Framework:")
    test_import("gradio", "gradio")
    
    # Data Processing
    print("\n📈 Data Processing and Scientific Computing:")
    test_import("numpy", "numpy")
    test_import("pandas", "pandas")
    test_import("scipy", "scipy")
    
    # Progress and Monitoring
    print("\n📊 Progress and Monitoring:")
    test_import("tqdm", "tqdm")
    test_import("psutil", "psutil")
    
    # Memory and Performance Optimization
    print("\n⚡ Memory and Performance Optimization:")
    test_import("bitsandbytes", "bitsandbytes")
    test_import("peft", "peft")
    
    # Logging and Debugging
    print("\n📝 Logging and Debugging:")
    test_import("wandb", "wandb")
    test_import("tensorboard", "tensorboard")
    
    # Additional Utilities
    print("\n🔧 Additional Utilities:")
    test_import("requests", "requests")
    test_import("PIL", "pillow")
    test_import("matplotlib", "matplotlib")
    test_import("seaborn", "seaborn")
    
    # Development and Testing
    print("\n🧪 Development and Testing:")
    test_import("pytest", "pytest")
    test_import("black", "black")
    test_import("flake8", "flake8")
    
    print("\n" + "=" * 50)
    print("🎯 Dependency Test Complete!")
    print("\n💡 If any dependencies failed to import:")
    print("   1. Check the error messages above")
    print("   2. Install missing packages with pip")
    print("   3. Restart the Hugging Face Space")
    print("   4. Run this test again to verify")

if __name__ == "__main__":
    main()
