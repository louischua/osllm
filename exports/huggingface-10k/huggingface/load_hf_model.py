#!/usr/bin/env python3
"""
Hugging Face Compatible Loader for OpenLLM

Usage:
    # Using transformers library (if you implement custom model class)
    # from transformers import AutoModel, AutoTokenizer
    # model = AutoModel.from_pretrained(".")
    # tokenizer = AutoTokenizer.from_pretrained(".")

    # Manual loading
    from load_hf_model import load_model_manual
    model, tokenizer = load_model_manual(".")
"""

import torch
import json
import sentencepiece as smp
from pathlib import Path

def load_model_manual(model_dir="."):
    """Manually load model in HF format."""
    model_dir = Path(model_dir)

    # Load config
    with open(model_dir / "config.json", 'r') as f:
        config = json.load(f)

    # Load model weights
    state_dict = torch.load(model_dir / "pytorch_model.bin", map_location='cpu')

    # Load tokenizer
    tokenizer = smp.SentencePieceProcessor()
    tokenizer.load(str(model_dir / "tokenizer.model"))

    print(f"Loaded model: {config['model_type']} with {config['n_layer']} layers")
    print(f"Vocabulary size: {config['vocab_size']}")

    return state_dict, tokenizer

if __name__ == "__main__":
    state_dict, tokenizer = load_model_manual()
    print(f"Model weights loaded: {len(state_dict)} parameters")
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size()}")
