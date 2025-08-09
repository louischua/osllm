#!/usr/bin/env python3
"""
PyTorch Model Loader for OpenLLM

Usage:
    from load_model import load_model, generate_text
    
    model, tokenizer, config = load_model(".")
    text = generate_text(model, tokenizer, "Hello world", max_length=50)
    print(text)
"""

import torch
import json
import sentencepiece as spm
from pathlib import Path

def load_model(model_dir="."):
    """Load OpenLLM model from PyTorch export."""
    model_dir = Path(model_dir)
    
    # Load config
    with open(model_dir / "config.json", 'r') as f:
        config_data = json.load(f)
    
    model_config = config_data['model_config']
    
    # Recreate model architecture (you'll need to have the model.py file)
    # This is a simplified loader - in practice you'd import your GPTModel class
    print(f"Model config: {model_config}")
    print("Note: You need to import and create the actual model class")
    
    # Load model state
    checkpoint = torch.load(model_dir / "model.pt", map_location='cpu')
    
    # Load tokenizer
    tokenizer = smp.SentencePieceProcessor()
    tokenizer.load(str(model_dir / "tokenizer.model"))
    
    return None, tokenizer, model_config  # Placeholder

def generate_text(model, tokenizer, prompt, max_length=100):
    """Generate text using the loaded model."""
    # Implement text generation
    return f"Generated text for: {prompt}"

if __name__ == "__main__":
    model, tokenizer, config = load_model()
    print(f"Model loaded with {config.get('vocab_size', 'unknown')} vocabulary size")
