#!/usr/bin/env python3
"""
Simple test script to check sentencepiece installation and import.

This script specifically tests the sentencepiece library which is critical
for OpenLLM model tokenization.

Author: Louis Chua Bean Chong
License: GPL-3.0
"""

import sys
import subprocess

def test_sentencepiece():
    """Test sentencepiece installation and import."""
    print("🔍 Testing SentencePiece Installation")
    print("=" * 40)
    
    # Test 1: Check if sentencepiece is installed via pip
    print("\n📦 Checking pip installation...")
    try:
        result = subprocess.run(
            ["pip", "show", "sentencepiece"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            print("✅ sentencepiece is installed via pip")
            print(f"Info:\n{result.stdout}")
        else:
            print("❌ sentencepiece is NOT installed via pip")
            print("Installing sentencepiece...")
            install_result = subprocess.run(
                ["pip", "install", "sentencepiece>=0.1.99"],
                capture_output=True,
                text=True
            )
            if install_result.returncode == 0:
                print("✅ sentencepiece installed successfully")
            else:
                print(f"❌ Failed to install sentencepiece: {install_result.stderr}")
    except Exception as e:
        print(f"❌ Error checking pip: {e}")
    
    # Test 2: Try to import sentencepiece
    print("\n🐍 Testing Python import...")
    try:
        import sentencepiece
        print("✅ sentencepiece import successful")
        print(f"Version: {sentencepiece.__version__}")
    except ImportError as e:
        print(f"❌ sentencepiece import failed: {e}")
        return False
    
    # Test 3: Test SentencePieceTokenizer specifically
    print("\n🔤 Testing SentencePieceTokenizer...")
    try:
        from transformers import AutoTokenizer
        print("✅ AutoTokenizer import successful")
        
        # Try to load a simple tokenizer to test
        print("Testing tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Simple test
        print("✅ Basic tokenizer loading successful")
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("🎯 SentencePiece Test Complete!")
    return True

def test_openllm_model():
    """Test loading the OpenLLM model specifically."""
    print("\n🚀 Testing OpenLLM Model Loading")
    print("=" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Loading OpenLLM small model...")
        model_name = "lemms/openllm-small-extended-7k"
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer loaded successfully")
        
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✅ Model loaded successfully")
        
        print(f"\n🎉 OpenLLM model test successful!")
        print(f"Model: {model_name}")
        print(f"Tokenizer type: {type(tokenizer).__name__}")
        print(f"Model type: {type(model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenLLM model test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 SentencePiece and OpenLLM Model Test")
    print("=" * 50)
    
    # Test sentencepiece
    sp_success = test_sentencepiece()
    
    # Test OpenLLM model if sentencepiece works
    if sp_success:
        model_success = test_openllm_model()
        if model_success:
            print("\n🎉 All tests passed! Training should work now.")
        else:
            print("\n⚠️ SentencePiece works but model loading failed.")
    else:
        print("\n❌ SentencePiece test failed. Need to fix dependencies first.")
    
    print("\n💡 Next steps:")
    print("1. If tests failed, run: python install_dependencies.py")
    print("2. If tests passed, try the training again")
    print("3. If still having issues, restart the Space")
