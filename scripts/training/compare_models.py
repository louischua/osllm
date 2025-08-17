#!/usr/bin/env python3
"""
Compare different training checkpoints to evaluate progress
"""

import os
import sys
import torch
import sentencepiece as spm
import json

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'src'))

from model import create_model


def load_model_and_stats(checkpoint_path, model_size="small", device="cpu"):
    """Load model and extract training statistics."""
    if not os.path.exists(checkpoint_path):
        return None, None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create and load model
    model = create_model(model_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    
    # Extract stats
    stats = {
        'steps': checkpoint.get('step', 0),
        'loss': checkpoint.get('best_loss', float('inf')),
        'parameters': model.get_num_params()
    }
    
    return model, stats


def evaluate_model_quality(model, tokenizer, test_texts, device="cpu"):
    """Evaluate model on test texts."""
    total_loss = 0
    total_tokens = 0
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
            
        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits, loss = model(input_ids, target_ids)
        
        total_loss += loss.item() * len(tokens)
        total_tokens += len(tokens)
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def generate_sample(model, tokenizer, prompt, max_tokens=50, device="cpu"):
    """Generate sample text from model."""
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=max_tokens, temperature=0.7, top_k=40)
    
    generated_ids = output[0].tolist()
    generated_text = tokenizer.decode(generated_ids[len(input_ids):])
    
    return generated_text


def main():
    print("🔬 Model Comparison & Training Progress Analysis")
    print("=" * 60)
    
    device = "cpu"
    tokenizer_path = "data/tokenizer/tokenizer.model"
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    
    # Model checkpoints to compare
    models_to_compare = [
        ("2K Steps", "models/small-model-fixed/best_model.pt"),
        ("4K Steps", "models/small-extended-4k/best_model.pt"),
    ]
    
    # Test data for evaluation
    test_texts = [
        "Artificial intelligence is a rapidly growing field of computer science.",
        "Machine learning algorithms can learn patterns from data automatically.",
        "Natural language processing helps computers understand human language.",
        "Deep learning uses neural networks with multiple layers for complex tasks.",
        "The development of large language models has transformed AI applications."
    ]
    
    test_prompt = "The future of artificial intelligence"
    
    results = []
    
    print(f"📊 Evaluating models on {len(test_texts)} test passages...")
    print(f"🎯 Testing generation with prompt: '{test_prompt}'")
    print("-" * 60)
    
    for name, checkpoint_path in models_to_compare:
        print(f"\n🧠 {name}: {checkpoint_path}")
        
        model, stats = load_model_and_stats(checkpoint_path, device=device)
        if model is None:
            print(f"❌ Failed to load {checkpoint_path}")
            continue
        
        # Evaluate on test texts
        avg_loss, perplexity = evaluate_model_quality(model, tokenizer, test_texts, device)
        
        # Generate sample text
        generated = generate_sample(model, tokenizer, test_prompt, device=device)
        
        result = {
            'name': name,
            'steps': stats['steps'],
            'training_loss': stats['loss'],
            'eval_loss': avg_loss,
            'perplexity': perplexity,
            'generated_text': generated[:100] + "..." if len(generated) > 100 else generated
        }
        results.append(result)
        
        print(f"  📈 Training Steps: {stats['steps']:,}")
        print(f"  📉 Training Loss: {stats['loss']:.4f}")
        print(f"  📊 Eval Loss: {avg_loss:.4f}")
        print(f"  📋 Perplexity: {perplexity:.2f}")
        print(f"  📝 Sample: '{result['generated_text']}'")
    
    # Analysis and recommendations
    print(f"\n" + "=" * 60)
    print("📈 PROGRESS ANALYSIS")
    print("=" * 60)
    
    if len(results) >= 2:
        prev_result = results[0]
        curr_result = results[1]
        
        loss_improvement = prev_result['training_loss'] - curr_result['training_loss']
        ppl_improvement = prev_result['perplexity'] - curr_result['perplexity']
        
        print(f"🎯 Training Progress:")
        print(f"  Loss improved by: {loss_improvement:.4f} ({loss_improvement/prev_result['training_loss']*100:.1f}%)")
        print(f"  Perplexity changed by: {ppl_improvement:.2f}")
        
        if loss_improvement > 0:
            print(f"  ✅ Model is improving with more training")
        else:
            print(f"  ⚠️  Training loss increased - may need different approach")
        
        print(f"\n📊 Quality Assessment:")
        if curr_result['perplexity'] < 100:
            quality = "Good"
        elif curr_result['perplexity'] < 500:
            quality = "Fair"
        else:
            quality = "Needs improvement"
        
        print(f"  Current model quality: {quality}")
        print(f"  Target perplexity: <50 for good quality")
        
        print(f"\n🚀 Recommendations:")
        if curr_result['perplexity'] > 500:
            print(f"  • Continue training to 8K-10K steps")
            print(f"  • Model still learning basic patterns")
        elif curr_result['perplexity'] > 100:
            print(f"  • Consider training to 8K steps")
            print(f"  • Evaluate data quality and diversity")
        else:
            print(f"  • Model quality is improving well")
            print(f"  • Consider fine-tuning or specialized training")
    
    # Save comparison results
    with open('model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: model_comparison.json")


if __name__ == "__main__":
    main()