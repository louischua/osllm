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
    """
    Manually load model in HF format.
    
    This function loads OpenLLM models exported in Hugging Face compatible format.
    It handles both the model weights and tokenizer, providing a complete loading solution.
    
    Args:
        model_dir (str): Path to the HF-format model directory
        
    Returns:
        tuple: (model, tokenizer) where:
            - model: Loaded PyTorch model ready for inference
            - tokenizer: SentencePiece tokenizer for text processing
            
    Implementation Details:
        - Loads HF-compatible configuration and weights
        - Recreates model architecture based on config
        - Sets up tokenizer with proper vocabulary
        - Handles graceful fallback if model classes unavailable
    """
    model_dir = Path(model_dir)
    
    try:
        # Load HF-compatible configuration
        # This contains model architecture and metadata
        with open(model_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        print(f"📂 Loading HF model from: {model_dir}")
        print(f"🏗️  Model type: {config['model_type']} with {config['n_layer']} layers")
        print(f"📊 Vocabulary size: {config['vocab_size']}")
        print(f"🎯 Training steps: {config.get('training_steps', 'Unknown')}")
        
        # Load model weights
        # HF format stores weights in pytorch_model.bin
        weights_path = model_dir / "pytorch_model.bin"
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Try to import and create model architecture
        try:
            # Add core/src to path to import model classes
            import sys
            core_src_path = Path(__file__).parent.parent.parent / "core" / "src"
            if core_src_path.exists():
                sys.path.insert(0, str(core_src_path))
            
            from model import create_model
            
            # Determine model size from configuration
            # Map HF config to our model sizes
            n_layer = config.get('n_layer', 12)
            
            if n_layer <= 6:
                model_size = "small"
            elif n_layer <= 12:
                model_size = "medium"
            else:
                model_size = "large"
            
            print(f"🎯 Detected model size: {model_size}")
            
            # Create model architecture
            # This recreates the same architecture used during training
            model = create_model(model_size)
            
            # Load the trained weights
            # Restore model to its trained state
            model.load_state_dict(state_dict)
            model.eval()  # Set to evaluation mode
            
            print(f"✅ Model architecture created and weights loaded")
            print(f"📈 Parameters: {model.get_num_params():,}")
            
        except ImportError as e:
            print(f"⚠️  Warning: Could not import model classes: {e}")
            print("    Returning state_dict only - model architecture not available")
            
            # Return state_dict as fallback
            class StatedictPlaceholder:
                def __init__(self, state_dict, config):
                    self.state_dict = state_dict
                    self.config = config
                    
                def __len__(self):
                    return len(self.state_dict)
                    
                def get_num_params(self):
                    return sum(p.numel() for p in self.state_dict.values())
            
            model = StatedictPlaceholder(state_dict, config)
        
        # Load tokenizer
        # This handles text-to-token and token-to-text conversion
        tokenizer_path = model_dir / "tokenizer.model"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        tokenizer = smp.SentencePieceProcessor()
        tokenizer.load(str(tokenizer_path))
        
        print(f"🔤 Tokenizer loaded: {tokenizer.vocab_size()} vocabulary")
        print(f"🎉 HF model loading completed successfully!")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ HF model loading failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to load HF model: {e}")

if __name__ == "__main__":
    model, tokenizer = load_model_manual()
    
    # Display model information
    if hasattr(model, 'get_num_params'):
        print(f"Model parameters: {model.get_num_params():,}")
    elif hasattr(model, '__len__'):
        print(f"Model weights loaded: {len(model)} parameters")
    else:
        print("Model loaded (parameter count unavailable)")
    
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size()}")
    
    # Test tokenization
    test_text = "Hello world"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"Tokenization test: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")
