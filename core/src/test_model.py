#!/usr/bin/env python3
"""
Model Architecture Testing and Validation Script

This script provides comprehensive testing and validation for the GPT model architecture.
It helps verify that the model is correctly implemented and can run on your hardware.

FEATURES:
- Model initialization testing
- Forward pass validation
- Memory usage analysis
- Tokenizer integration testing
- Performance benchmarking
- Hardware compatibility checks

Usage:
    python core/src/test_model.py --model_size medium
    python core/src/test_model.py --model_size small --test_generation
    python core/src/test_model.py --all_sizes --benchmark

Requirements:
    - torch
    - sentencepiece (for tokenizer integration)
    - Our trained tokenizer in data/tokenizer/

Author: OpenLLM Project
License: GPLv3
"""

import argparse
import json
import os
import time
import torch
import traceback
from pathlib import Path
from typing import Dict, List, Optional

# Import our model architecture
try:
    from model import GPTModel, GPTConfig, create_model
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from model import GPTModel, GPTConfig, create_model

# Import tokenizer if available
try:
    import sentencepiece as spm
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: SentencePiece not available. Tokenizer tests will be skipped.")


class ModelTester:
    """
    Comprehensive model testing class.
    
    Provides methods to test model initialization, forward passes, memory usage,
    and integration with the tokenizer.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the model tester.
        
        Args:
            device: Device to use ("cpu", "cuda", or "auto")
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🔧 Model Tester initialized")
        print(f"Device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        
        # Try to load tokenizer
        self.tokenizer = None
        self.load_tokenizer()
    
    def load_tokenizer(self) -> None:
        """Load the trained SentencePiece tokenizer if available."""
        if not TOKENIZER_AVAILABLE:
            return
            
        tokenizer_path = "data/tokenizer/tokenizer.model"
        if os.path.exists(tokenizer_path):
            try:
                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.load(tokenizer_path)
                print(f"✓ Tokenizer loaded: {tokenizer_path}")
                print(f"  Vocabulary size: {self.tokenizer.vocab_size():,}")
            except Exception as e:
                print(f"⚠️  Failed to load tokenizer: {e}")
        else:
            print(f"⚠️  Tokenizer not found at {tokenizer_path}")
    
    def test_model_initialization(self, model_size: str = "medium") -> Dict:
        """
        Test model initialization and basic properties.
        
        Args:
            model_size: Size of model to test
            
        Returns:
            dict: Test results
        """
        print(f"\n🧠 Testing {model_size.upper()} model initialization...")
        
        try:
            # Create model
            start_time = time.time()
            model = create_model(model_size)
            init_time = time.time() - start_time
            
            # Move to device
            model = model.to(self.device)
            
            # Basic checks
            param_count = model.get_num_params()
            config = model.config
            
            print(f"✓ Model created successfully")
            print(f"  Parameters: {param_count:,}")
            print(f"  Layers: {config.n_layer}")
            print(f"  Heads: {config.n_head}")
            print(f"  Embedding dim: {config.n_embd}")
            print(f"  Block size: {config.block_size}")
            print(f"  Initialization time: {init_time:.2f}s")
            
            return {
                "success": True,
                "model_size": model_size,
                "parameters": param_count,
                "config": config.__dict__,
                "init_time": init_time,
                "device": str(next(model.parameters()).device)
            }
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def test_forward_pass(self, model: GPTModel, batch_size: int = 2, seq_len: int = 64) -> Dict:
        """
        Test model forward pass with synthetic data.
        
        Args:
            model: Model to test
            batch_size: Batch size for test
            seq_len: Sequence length for test
            
        Returns:
            dict: Test results
        """
        print(f"\n🔄 Testing forward pass (batch={batch_size}, seq_len={seq_len})...")
        
        try:
            model.eval()
            
            # Create synthetic input
            x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
            x = x.to(self.device)
            
            # Test inference mode
            start_time = time.time()
            with torch.no_grad():
                logits, _ = model(x)
            inference_time = time.time() - start_time
            
            # Test training mode with targets
            model.train()
            targets = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
            targets = targets.to(self.device)
            
            start_time = time.time()
            logits_train, loss = model(x, targets)
            train_time = time.time() - start_time
            
            print(f"✓ Forward pass successful")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {logits.shape}")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Inference time: {inference_time:.4f}s")
            print(f"  Training time: {train_time:.4f}s")
            
            return {
                "success": True,
                "input_shape": list(x.shape),
                "output_shape": list(logits.shape),
                "loss": loss.item(),
                "inference_time": inference_time,
                "training_time": train_time
            }
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def test_memory_usage(self, model: GPTModel, batch_sizes: List[int] = [1, 2, 4]) -> Dict:
        """
        Test memory usage for different batch sizes.
        
        Args:
            model: Model to test
            batch_sizes: List of batch sizes to test
            
        Returns:
            dict: Memory usage results
        """
        print(f"\n💾 Testing memory usage...")
        
        results = {}
        
        for batch_size in batch_sizes:
            try:
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Get initial memory
                if torch.cuda.is_available():
                    initial_memory = torch.cuda.memory_allocated() / (1024**2)
                else:
                    initial_memory = 0
                
                # Forward pass
                seq_len = min(512, model.config.block_size)
                x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
                x = x.to(self.device)
                
                with torch.no_grad():
                    logits, _ = model(x)
                
                # Get peak memory
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
                    memory_used = peak_memory - initial_memory
                else:
                    memory_used = model.estimate_memory_usage(batch_size, seq_len)["total_inference_mb"]
                
                results[f"batch_{batch_size}"] = {
                    "memory_mb": memory_used,
                    "memory_per_sample": memory_used / batch_size
                }
                
                print(f"  Batch size {batch_size}: {memory_used:.1f}MB ({memory_used/batch_size:.1f}MB per sample)")
                
            except Exception as e:
                print(f"  Batch size {batch_size}: Failed - {e}")
                results[f"batch_{batch_size}"] = {"error": str(e)}
        
        return results
    
    def test_tokenizer_integration(self, model: GPTModel) -> Dict:
        """
        Test integration with the trained tokenizer.
        
        Args:
            model: Model to test
            
        Returns:
            dict: Integration test results
        """
        print(f"\n🔤 Testing tokenizer integration...")
        
        if self.tokenizer is None:
            print("⚠️  No tokenizer available, skipping integration test")
            return {"success": False, "reason": "No tokenizer available"}
        
        try:
            # Test sentences
            test_sentences = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is transforming technology.",
                "GPT models use transformer architecture for language modeling."
            ]
            
            results = []
            
            for sentence in test_sentences:
                # Tokenize
                tokens = self.tokenizer.encode(sentence)
                token_tensor = torch.tensor([tokens]).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    logits, _ = model(token_tensor)
                
                # Get predictions for next token
                next_token_logits = logits[0, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=0)
                top5_tokens = torch.topk(next_token_probs, 5)
                
                # Decode top predictions
                top5_decoded = []
                for token_id in top5_tokens.indices:
                    try:
                        decoded = self.tokenizer.decode([token_id.item()])
                        prob = top5_tokens.values[len(top5_decoded)].item()
                        top5_decoded.append((decoded, prob))
                    except:
                        top5_decoded.append(("<??>", 0.0))
                
                results.append({
                    "input": sentence,
                    "tokens": len(tokens),
                    "top_predictions": top5_decoded
                })
                
                print(f"✓ '{sentence[:30]}...' -> {len(tokens)} tokens")
                print(f"  Top prediction: '{top5_decoded[0][0]}' ({top5_decoded[0][1]:.3f})")
            
            return {
                "success": True,
                "vocab_size_match": self.tokenizer.vocab_size() == model.config.vocab_size,
                "test_results": results
            }
            
        except Exception as e:
            print(f"❌ Tokenizer integration failed: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def test_generation(self, model: GPTModel, prompt: str = "The future of AI") -> Dict:
        """
        Test text generation capabilities.
        
        Args:
            model: Model to test
            prompt: Starting prompt for generation
            
        Returns:
            dict: Generation test results
        """
        print(f"\n✍️  Testing text generation...")
        
        if self.tokenizer is None:
            print("⚠️  No tokenizer available, skipping generation test")
            return {"success": False, "reason": "No tokenizer available"}
        
        try:
            # Tokenize prompt
            tokens = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([tokens]).to(self.device)
            
            print(f"Prompt: '{prompt}'")
            print(f"Generating...")
            
            # Generate
            start_time = time.time()
            output = model.generate(
                input_tensor,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50
            )
            generation_time = time.time() - start_time
            
            # Decode output
            generated_tokens = output[0].tolist()
            generated_text = self.tokenizer.decode(generated_tokens)
            
            print(f"✓ Generated text: '{generated_text}'")
            print(f"  Generation time: {generation_time:.2f}s")
            print(f"  Tokens per second: {50/generation_time:.1f}")
            
            return {
                "success": True,
                "prompt": prompt,
                "generated_text": generated_text,
                "generation_time": generation_time,
                "tokens_per_second": 50 / generation_time
            }
            
        except Exception as e:
            print(f"❌ Text generation failed: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def run_comprehensive_test(self, model_size: str = "medium") -> Dict:
        """
        Run all tests for a given model size.
        
        Args:
            model_size: Size of model to test
            
        Returns:
            dict: Complete test results
        """
        print(f"\n🔍 Running comprehensive test for {model_size.upper()} model")
        print("=" * 60)
        
        results = {"model_size": model_size, "device": self.device}
        
        # Test 1: Model initialization
        init_result = self.test_model_initialization(model_size)
        results["initialization"] = init_result
        
        if not init_result["success"]:
            return results
        
        # Create model for remaining tests
        model = create_model(model_size).to(self.device)
        
        # Test 2: Forward pass
        results["forward_pass"] = self.test_forward_pass(model)
        
        # Test 3: Memory usage
        results["memory_usage"] = self.test_memory_usage(model)
        
        # Test 4: Tokenizer integration
        results["tokenizer_integration"] = self.test_tokenizer_integration(model)
        
        # Test 5: Text generation
        results["generation"] = self.test_generation(model)
        
        return results


def load_model_config(model_size: str) -> Dict:
    """Load model configuration from JSON file."""
    config_path = f"configs/{model_size}_model.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def print_hardware_recommendations(model_size: str) -> None:
    """Print hardware recommendations for the given model size."""
    config = load_model_config(model_size)
    
    if config:
        print(f"\n💻 Hardware Recommendations for {model_size.upper()} model:")
        print(f"  Parameters: {config.get('parameters', 'Unknown')}")
        print(f"  Recommended: {config.get('recommended_hardware', 'Unknown')}")
        
        if "memory_estimates" in config:
            mem = config["memory_estimates"]
            print(f"  Memory usage: ~{mem.get('parameters_mb', '?')}MB parameters")
            print(f"  Training: ~{mem.get('training_mb_per_sample', '?')}MB per sample")
            print(f"  Inference: ~{mem.get('inference_mb_per_sample', '?')}MB per sample")
        
        if "cpu_training_notes" in config:
            cpu_notes = config["cpu_training_notes"]
            if cpu_notes.get("feasible"):
                print(f"  CPU Training: Feasible but slow ({cpu_notes.get('expected_training_time', '?')})")
            else:
                print(f"  CPU Training: Not recommended - {cpu_notes.get('reason', 'Too large')}")


def main():
    """Main function to handle command line testing."""
    parser = argparse.ArgumentParser(
        description="Test and validate GPT model architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test medium model
  python core/src/test_model.py --model_size medium
  
  # Test all model sizes
  python core/src/test_model.py --all_sizes
  
  # Test with text generation
  python core/src/test_model.py --model_size small --test_generation
  
  # Show hardware recommendations
  python core/src/test_model.py --recommendations
        """
    )
    
    parser.add_argument(
        "--model_size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size to test (default: medium)"
    )
    
    parser.add_argument(
        "--all_sizes", 
        action="store_true",
        help="Test all model sizes"
    )
    
    parser.add_argument(
        "--test_generation",
        action="store_true", 
        help="Include text generation test"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for testing (default: auto)"
    )
    
    parser.add_argument(
        "--recommendations",
        action="store_true",
        help="Show hardware recommendations for all model sizes"
    )
    
    parser.add_argument(
        "--save_results",
        help="Save test results to JSON file"
    )
    
    args = parser.parse_args()
    
    print("🧪 GPT Model Architecture Tester")
    print("=" * 50)
    
    # Show hardware recommendations
    if args.recommendations:
        for size in ["small", "medium", "large"]:
            print_hardware_recommendations(size)
        return
    
    # Initialize tester
    tester = ModelTester(device=args.device)
    
    # Run tests
    all_results = {}
    
    if args.all_sizes:
        test_sizes = ["small", "medium", "large"]
    else:
        test_sizes = [args.model_size]
    
    for size in test_sizes:
        results = tester.run_comprehensive_test(size)
        all_results[size] = results
        
        # Print summary
        print(f"\n📊 {size.upper()} Model Test Summary:")
        print(f"  Initialization: {'✓' if results['initialization']['success'] else '❌'}")
        print(f"  Forward Pass: {'✓' if results.get('forward_pass', {}).get('success') else '❌'}")
        print(f"  Memory Test: {'✓' if 'memory_usage' in results else '❌'}")
        print(f"  Tokenizer: {'✓' if results.get('tokenizer_integration', {}).get('success') else '❌'}")
        print(f"  Generation: {'✓' if results.get('generation', {}).get('success') else '❌'}")
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Results saved to {args.save_results}")
    
    print(f"\n🎉 Testing completed!")


if __name__ == "__main__":
    main()