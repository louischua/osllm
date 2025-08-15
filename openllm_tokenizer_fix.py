#!/usr/bin/env python3
"""
OpenLLM Custom Tokenizer Fix Script

This script demonstrates the correct way to load OpenLLM models with their
custom tokenizer classes using trust_remote_code=True.

Author: Louis Chua Bean Chong
License: GPL-3.0
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_openllm_loading():
    """Test loading OpenLLM model with custom tokenizer."""
    
    model_name = "lemms/openllm-small-extended-7k"
    
    print("🔍 Testing OpenLLM Custom Tokenizer Loading")
    print("=" * 50)
    print(f"Model: {model_name}")
    print("Note: OpenLLM uses custom tokenizer classes")
    print()
    
    try:
        # Load tokenizer with trust_remote_code for custom classes
        print("🔄 Loading custom tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,  # CRITICAL for custom tokenizer classes
            use_fast=False          # Use slow tokenizer for compatibility
        )
        print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")
        
        # Load model with trust_remote_code
        print("🔄 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True   # CRITICAL for custom model classes
        )
        print(f"✅ Model loaded: {type(model).__name__}")
        
        print("\n🎉 OpenLLM loading successful!")
        print("The key is using trust_remote_code=True for custom classes")
        
        return True
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        return False

if __name__ == "__main__":
    test_openllm_loading()
