#!/usr/bin/env python3
"""
Test script for lightweight model implementation.
"""

import sys
import os

# Add core/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'src'))

def test_lightweight_model():
    """Test the lightweight model creation and basic functionality."""
    try:
        print("🧪 Testing lightweight model implementation...")
        
        # Test imports
        print("📦 Testing imports...")
        from model import GPTConfig, GPTModel
        import torch
        print("✅ Imports successful")
        
        # Create minimal config
        print("⚙️ Creating minimal config...")
        config = GPTConfig.small()
        config.n_embd = 128  # Very small for testing
        config.n_layer = 2   # Very small for testing
        config.vocab_size = 1000  # Small vocabulary
        config.block_size = 64    # Small context
        print("✅ Config created")
        
        # Create real model
        print("🏗️ Creating real model...")
        model = GPTModel(config)
        model.eval()
        print("✅ Model created")
        
        # Create minimal tokenizer
        print("🔤 Creating minimal tokenizer...")
        class MinimalTokenizer:
            def __init__(self):
                self.vocab_size = 1000
                
            def encode(self, text):
                # Simple character-based encoding for testing
                return [ord(c) % 1000 for c in text[:50]]  # Limit to 50 chars
                
            def decode(self, tokens):
                # Simple character-based decoding for testing
                return ''.join([chr(t % 256) for t in tokens if t < 256])
                
            def vocab_size(self):
                return 1000
        
        tokenizer = MinimalTokenizer()
        print("✅ Tokenizer created")
        
        # Test basic functionality
        print("🧪 Testing basic functionality...")
        
        # Test tokenization
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"  Tokenization: '{test_text}' -> {tokens} -> '{decoded}'")
        
        # Test model forward pass
        input_ids = tokenizer.encode("Test")
        if len(input_ids) == 0:
            input_ids = [1]  # Default token
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(input_tensor)
        
        print(f"  Model forward pass: input shape {input_tensor.shape} -> output shape {logits.shape}")
        
        # Test generation
        print("🎯 Testing generation...")
        generated = input_ids.copy()
        for i in range(5):  # Generate 5 tokens
            if len(generated) >= config.block_size:
                break
            
            # Create input tensor
            input_tensor = torch.tensor([generated], dtype=torch.long)
            
            # Forward pass
            with torch.no_grad():
                logits, _ = model(input_tensor)
            
            # Get next token
            next_token_logits = logits[0, -1, :] / 0.7  # temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
        
        # Decode generated text
        generated_text = tokenizer.decode(generated[len(input_ids):])
        print(f"  Generated text: '{generated_text}'")
        
        print("✅ All tests passed! Lightweight model works correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lightweight_model()
    sys.exit(0 if success else 1)
