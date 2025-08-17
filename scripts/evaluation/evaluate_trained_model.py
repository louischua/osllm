#!/usr/bin/env python3
"""
Test script for trained OpenLLM model
"""

import os
import sys
import torch
import sentencepiece as spm

# Set console encoding for Windows compatibility
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add the core/src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core', 'src'))

from model import create_model, GPTModel
from train_model import ModelTrainer


def load_trained_model(checkpoint_path, model_size="small", device="cpu"):
    """Load a trained model from checkpoint."""
    print(f"🔧 Loading trained model from {checkpoint_path}")
    
    # Create model architecture
    model = create_model(model_size)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Final training loss: {checkpoint.get('best_loss', 'N/A'):.4f}")
    print(f"   Training steps: {checkpoint.get('step', 'N/A')}")
    print(f"   Model parameters: {model.get_num_params():,}")
    
    return model, checkpoint


def test_text_generation(model, tokenizer, prompt, max_tokens=100, temperature=0.7, top_k=50):
    """Generate text using the trained model."""
    print(f"\n🎯 Testing text generation...")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}, Temperature: {temperature}, Top-k: {top_k}")
    print("-" * 50)
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    print(f"Input tokens: {input_ids}")
    print(f"Input length: {len(input_ids)} tokens")
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_tensor, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode output
    generated_ids = output[0].tolist()
    full_text = tokenizer.decode(generated_ids)
    generated_text = tokenizer.decode(generated_ids[len(input_ids):])
    
    print(f"Generated text: '{generated_text}'")
    print(f"Full text: '{full_text}'")
    print(f"Total tokens generated: {len(generated_ids) - len(input_ids)}")
    
    return full_text, generated_text


def evaluate_perplexity(model, tokenizer, test_text, max_length=512):
    """Calculate perplexity on test text."""
    print(f"\n📊 Evaluating perplexity...")
    
    # Tokenize test text
    tokens = tokenizer.encode(test_text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    # Create input and target tensors
    input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
    target_ids = torch.tensor([tokens[1:]], dtype=torch.long)
    
    print(f"Test sequence length: {len(tokens)} tokens")
    
    # Calculate loss
    with torch.no_grad():
        logits, loss = model(input_ids, target_ids)
    
    perplexity = torch.exp(loss).item()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    
    return loss.item(), perplexity


def main():
    print("🧪 Testing Trained OpenLLM Model")
    print("=" * 50)
    
    # Configuration
    device = "cpu"
    model_size = "small"
    checkpoint_path = "models/small-extended-7k/best_model.pt"
    tokenizer_path = "data/tokenizer/tokenizer.model"
    
    # Check if files exist
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        return
    
    # Load tokenizer
    print(f"🔤 Loading tokenizer from {tokenizer_path}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size():,})")
    
    # Load trained model
    model, checkpoint = load_trained_model(checkpoint_path, model_size, device)
    
    # Test text generation
    test_prompts = [
        "The history of artificial intelligence",
        "Machine learning algorithms",
        "The future of technology",
        "In a world where",
        "Scientists have discovered"
    ]
    
    for prompt in test_prompts:
        full_text, generated = test_text_generation(
            model, tokenizer, prompt, 
            max_tokens=50, temperature=0.7, top_k=40
        )
        print()
    
    # Test perplexity
    test_texts = [
        "Artificial intelligence is a rapidly growing field that encompasses machine learning, deep learning, and neural networks.",
        "The development of large language models has revolutionized natural language processing and generation.",
        "Computer science involves the study of algorithms, data structures, and computational systems."
    ]
    
    total_loss = 0
    total_perplexity = 0
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: '{text[:50]}...'")
        loss, ppl = evaluate_perplexity(model, tokenizer, text)
        total_loss += loss
        total_perplexity += ppl
    
    avg_loss = total_loss / len(test_texts)
    avg_perplexity = total_perplexity / len(test_texts)
    
    print(f"\n📈 Overall Results:")
    print(f"   Average loss: {avg_loss:.4f}")
    print(f"   Average perplexity: {avg_perplexity:.2f}")
    print(f"   Model quality: {'Good' if avg_perplexity < 50 else 'Needs improvement' if avg_perplexity < 100 else 'Poor'}")


if __name__ == "__main__":
    main()