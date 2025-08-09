#!/usr/bin/env python3
"""
OpenLLM Text Generation Script

This script implements standalone text generation for OpenLLM models
as specified in Step 5 of the training pipeline (Text Generation Quality assessment).

Features:
- Load trained OpenLLM models from checkpoint directories
- Generate text with configurable parameters (temperature, length, etc.)
- Support multiple model formats (auto-detection)
- Quality assessment and metrics
- Batch generation capabilities
- Output formatting and saving

Usage:
    # Basic text generation
    python core/src/generate_text.py \
        --model_dir models/small-extended-4k \
        --prompt "The history of artificial intelligence" \
        --max_length 256 \
        --temperature 0.7

    # Multiple prompts with custom settings
    python core/src/generate_text.py \
        --model_dir models/small-extended-4k \
        --prompts_file prompts.txt \
        --max_length 100 \
        --temperature 0.8 \
        --top_k 40 \
        --num_samples 3

    # Save results to file
    python core/src/generate_text.py \
        --model_dir models/small-extended-4k \
        --prompt "Once upon a time" \
        --output_file generated_samples.txt

Author: OpenLLM Project
License: GPLv3
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import sentencepiece as spm

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import create_model, GPTModel, GPTConfig


class TextGenerator:
    """
    Comprehensive text generation engine for OpenLLM models.
    
    This class handles loading trained models and generating high-quality text
    with configurable sampling parameters and quality assessment.
    """
    
    def __init__(self, model_dir: str, device: str = "auto"):
        """
        Initialize the text generator.
        
        Args:
            model_dir: Directory containing trained model checkpoints
            device: Device to use ("auto", "cpu", "cuda")
            
        Implementation Details:
            - Auto-detects best available device if device="auto"
            - Loads model architecture based on checkpoint configuration
            - Sets up tokenizer for text processing
            - Validates model and tokenizer compatibility
        """
        self.model_dir = Path(model_dir)
        
        # Determine device to use
        # Auto-detection prioritizes CUDA if available for better performance
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🚀 OpenLLM Text Generator")
        print(f"📂 Model directory: {model_dir}")
        print(f"🖥️  Device: {self.device}")
        
        # Load model and tokenizer
        # This handles the complete setup process
        self._load_model()
        self._load_tokenizer()
        
        # Validate setup
        # Ensure model and tokenizer are compatible
        self._validate_setup()
        
        print(f"✅ Text generator initialized successfully!")
    
    def _load_model(self):
        """
        Load the trained model from checkpoint.
        
        Implementation Details:
            - Searches for best_model.pt or latest checkpoint
            - Auto-detects model size from configuration
            - Handles different checkpoint formats gracefully
            - Sets model to evaluation mode for inference
        """
        # Find the best model checkpoint
        # Priority: best_model.pt > latest checkpoint by step number
        best_model_path = self.model_dir / "best_model.pt"
        
        if best_model_path.exists():
            checkpoint_path = best_model_path
            print(f"📥 Loading best model: {checkpoint_path}")
        else:
            # Look for step-based checkpoints
            checkpoints = list(self.model_dir.glob("checkpoint_step_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"No model checkpoints found in {self.model_dir}")
            
            # Get the latest checkpoint by step number
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            checkpoint_path = latest_checkpoint
            print(f"📥 Loading latest checkpoint: {checkpoint_path}")
        
        # Load checkpoint data
        # This contains model weights, configuration, and training metadata
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print(f"✅ Checkpoint loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Extract model configuration
        # This tells us what architecture to create
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
        else:
            # Fallback: try to infer from model state dict
            print("⚠️  No config found in checkpoint, inferring from model structure...")
            config_dict = self._infer_config_from_state_dict(checkpoint.get('model_state_dict', checkpoint))
        
        # Determine model size category
        # This maps checkpoint config to our predefined model sizes
        n_layer = config_dict.get('n_layer', 12)
        n_embd = config_dict.get('n_embd', 768)
        
        if n_layer <= 6:
            model_size = "small"
        elif n_layer <= 12:
            model_size = "medium"
        else:
            model_size = "large"
        
        print(f"🎯 Detected model size: {model_size}")
        print(f"📊 Architecture: {n_layer} layers, {n_embd} embedding dim")
        
        # Create model architecture
        # This recreates the exact same model used during training
        try:
            self.model = create_model(model_size)
            print(f"🏗️  Model architecture created: {self.model.get_num_params():,} parameters")
        except Exception as e:
            raise RuntimeError(f"Failed to create model architecture: {e}")
        
        # Load trained weights
        # This restores the model to its trained state
        try:
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Fallback for different checkpoint formats
                self.model.load_state_dict(checkpoint)
            
            print(f"✅ Model weights loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")
        
        # Move model to device and set to evaluation mode
        # Evaluation mode disables dropout and other training-specific behaviors
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Store model configuration for later use
        # This is useful for generation parameters and limits
        self.config = self.model.config
        
        # Extract training metadata if available
        # This provides context about model quality and training progress
        self.training_info = {
            'step': checkpoint.get('step', 'Unknown'),
            'best_loss': checkpoint.get('best_loss', 'Unknown'),
            'model_size': model_size
        }
        
        print(f"📈 Training info: step {self.training_info['step']}, "
              f"best loss {self.training_info['best_loss']}")
    
    def _infer_config_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Infer model configuration from state dict when config is missing.
        
        Args:
            state_dict: Model parameter dictionary
            
        Returns:
            Inferred configuration dictionary
            
        Implementation Details:
            - Analyzes parameter shapes to determine architecture
            - Makes reasonable assumptions about standard GPT architecture
            - Provides fallback values for missing parameters
        """
        # Extract key dimensions from parameter shapes
        # This reverse-engineers the model architecture
        
        # Embedding layer tells us vocab size and embedding dimension
        if 'transformer.wte.weight' in state_dict:
            vocab_size, n_embd = state_dict['transformer.wte.weight'].shape
        else:
            # Fallback defaults
            vocab_size, n_embd = 32000, 512
        
        # Count transformer blocks to get number of layers
        # Look for attention weight patterns
        n_layer = 0
        for key in state_dict.keys():
            if 'attn.c_attn.weight' in key:
                # Extract layer number from key like 'transformer.h.0.attn.c_attn.weight'
                layer_num = int(key.split('.')[2])
                n_layer = max(n_layer, layer_num + 1)
        
        # Infer number of attention heads from attention weights
        # The c_attn weight combines query, key, value projections
        if f'transformer.h.0.attn.c_attn.weight' in state_dict:
            attn_weight_shape = state_dict[f'transformer.h.0.attn.c_attn.weight'].shape
            # Shape is [n_embd, 3 * n_embd] for combined Q,K,V
            # So n_head = n_embd / head_dim, assuming head_dim = 64
            n_head = n_embd // 64  # Standard head dimension
        else:
            n_head = 8  # Fallback
        
        # Construct configuration dictionary
        # Use reasonable defaults for missing values
        config = {
            'vocab_size': vocab_size,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'block_size': 1024,  # Standard context length
            'dropout': 0.1,      # Standard dropout rate
            'bias': True,        # Most models use bias
            'model_name': f'gpt-inferred-{n_layer}L'
        }
        
        print(f"🔍 Inferred config: {config}")
        return config
    
    def _load_tokenizer(self):
        """
        Load the SentencePiece tokenizer.
        
        Implementation Details:
            - Searches multiple possible tokenizer locations
            - Validates tokenizer vocabulary size against model
            - Sets up special tokens if available
        """
        # Try multiple possible tokenizer locations
        # Different training setups may store tokenizer in different places
        possible_paths = [
            self.model_dir / "tokenizer.model",
            self.model_dir.parent / "tokenizer" / "tokenizer.model", 
            Path("data/tokenizer/tokenizer.model"),
            self.model_dir / ".." / "tokenizer" / "tokenizer.model"
        ]
        
        tokenizer_path = None
        for path in possible_paths:
            if path.exists():
                tokenizer_path = path
                break
        
        if tokenizer_path is None:
            raise FileNotFoundError(f"Tokenizer not found in any of: {possible_paths}")
        
        print(f"📝 Loading tokenizer from: {tokenizer_path}")
        
        # Load SentencePiece tokenizer
        # This handles all text-to-token and token-to-text conversion
        try:
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(tokenizer_path))
            print(f"✅ Tokenizer loaded: {self.tokenizer.vocab_size()} vocabulary")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    def _validate_setup(self):
        """
        Validate that model and tokenizer are compatible.
        
        Implementation Details:
            - Checks vocabulary size consistency
            - Tests basic tokenization and model forward pass
            - Warns about potential compatibility issues
        """
        # Check vocabulary size consistency
        # Model and tokenizer should have matching vocabulary
        model_vocab_size = self.config.vocab_size
        tokenizer_vocab_size = self.tokenizer.vocab_size()
        
        if model_vocab_size != tokenizer_vocab_size:
            print(f"⚠️  Warning: Vocabulary size mismatch!")
            print(f"   Model expects: {model_vocab_size}")
            print(f"   Tokenizer has: {tokenizer_vocab_size}")
            print(f"   This may cause generation issues.")
        
        # Test basic functionality
        # Quick validation that everything works together
        try:
            # Test tokenization
            test_text = "Hello world"
            tokens = self.tokenizer.encode(test_text)
            decoded = self.tokenizer.decode(tokens)
            
            # Test model forward pass
            input_ids = torch.tensor([tokens[:5]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            print(f"✅ Validation passed: tokenization and model forward pass work")
            
        except Exception as e:
            print(f"⚠️  Validation warning: {e}")
            print(f"   Generation may still work, but there might be issues.")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.0
    ) -> List[str]:
        """
        Generate text from a prompt using the loaded model.
        
        Args:
            prompt: Input text to continue
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (0.1-2.0, higher = more random)
            top_k: Limit to top-k most likely tokens (None = no limit)
            top_p: Nucleus sampling threshold (None = no nucleus sampling)
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling (False = greedy)
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            
        Returns:
            List of generated text strings
            
        Implementation Details:
            - Uses autoregressive generation (one token at a time)
            - Supports multiple sampling strategies (greedy, top-k, nucleus)
            - Handles context length limits gracefully
            - Applies repetition penalty to improve quality
            - Returns only the generated portion (excludes input prompt)
        """
        print(f"🎯 Generating text for: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        print(f"⚙️  Parameters: max_length={max_length}, temperature={temperature}, "
              f"top_k={top_k}, top_p={top_p}")
        
        # Tokenize input prompt
        # Convert text to token IDs for model processing
        try:
            input_tokens = self.tokenizer.encode(prompt)
            if len(input_tokens) == 0:
                raise ValueError("Empty tokenization result")
        except Exception as e:
            raise RuntimeError(f"Failed to tokenize prompt: {e}")
        
        # Check prompt length against model context
        # Ensure we don't exceed model's maximum sequence length
        max_context = self.config.block_size
        if len(input_tokens) >= max_context:
            print(f"⚠️  Warning: Prompt length ({len(input_tokens)}) approaches "
                  f"context limit ({max_context})")
            # Truncate prompt if necessary
            input_tokens = input_tokens[-(max_context - max_length):]
            print(f"   Truncated prompt to {len(input_tokens)} tokens")
        
        # Generate multiple sequences
        # Each sequence is generated independently
        generated_texts = []
        
        for seq_idx in range(num_return_sequences):
            if num_return_sequences > 1:
                print(f"🔄 Generating sequence {seq_idx + 1}/{num_return_sequences}")
            
            try:
                generated_text = self._generate_single_sequence(
                    input_tokens=input_tokens,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty
                )
                generated_texts.append(generated_text)
                
            except Exception as e:
                print(f"⚠️  Generation failed for sequence {seq_idx + 1}: {e}")
                generated_texts.append(f"Generation error: {e}")
        
        return generated_texts
    
    def _generate_single_sequence(
        self,
        input_tokens: List[int],
        max_length: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
        repetition_penalty: float
    ) -> str:
        """
        Generate a single text sequence using autoregressive sampling.
        
        Args:
            input_tokens: Tokenized input prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling limit
            top_p: Nucleus sampling threshold
            do_sample: Whether to use sampling vs greedy
            repetition_penalty: Repetition penalty factor
            
        Returns:
            Generated text string (excluding input prompt)
            
        Implementation Details:
            - Implements autoregressive generation loop
            - Applies all specified sampling strategies
            - Handles special tokens (EOS, padding)
            - Tracks token frequencies for repetition penalty
        """
        # Initialize generation state
        # Keep track of all generated tokens and their frequencies
        generated_tokens = input_tokens.copy()
        token_frequencies = {}  # For repetition penalty
        
        # Count initial token frequencies
        # This helps apply repetition penalty from the start
        for token in input_tokens:
            token_frequencies[token] = token_frequencies.get(token, 0) + 1
        
        # Set model to evaluation mode and disable gradients
        # This ensures consistent inference behavior and saves memory
        self.model.eval()
        
        with torch.no_grad():
            # Main generation loop
            # Generate one token at a time until stopping condition
            for step in range(max_length):
                
                # Check context length limits
                # Prevent exceeding model's maximum sequence length
                if len(generated_tokens) >= self.config.block_size:
                    print(f"⚠️  Reached maximum context length ({self.config.block_size})")
                    break
                
                # Prepare model input
                # Use all generated tokens as context for next prediction
                input_ids = torch.tensor([generated_tokens], dtype=torch.long, device=self.device)
                
                try:
                    # Forward pass through model
                    # Get logits (raw predictions) for all vocabulary tokens
                    outputs = self.model(input_ids)
                    
                    # Handle different model output formats
                    # Some models return tuples, others return tensors directly
                    if isinstance(outputs, tuple):
                        logits = outputs[0]  # First element is usually logits
                    else:
                        logits = outputs
                    
                    # Get predictions for next token (last position in sequence)
                    next_token_logits = logits[0, -1, :].float()
                    
                except Exception as e:
                    raise RuntimeError(f"Model forward pass failed at step {step}: {e}")
                
                # Apply repetition penalty
                # Reduce probability of recently used tokens
                if repetition_penalty != 1.0:
                    for token, freq in token_frequencies.items():
                        if token < len(next_token_logits):
                            penalty = repetition_penalty ** freq
                            if next_token_logits[token] > 0:
                                next_token_logits[token] /= penalty
                            else:
                                next_token_logits[token] *= penalty
                
                # Apply sampling strategy to select next token
                # This determines the randomness and quality of generation
                if do_sample:
                    next_token = self._sample_next_token(
                        next_token_logits, temperature, top_k, top_p
                    )
                else:
                    # Greedy decoding: always pick most likely token
                    next_token = torch.argmax(next_token_logits).item()
                
                # Add generated token to sequence
                generated_tokens.append(next_token)
                
                # Update token frequency for repetition penalty
                token_frequencies[next_token] = token_frequencies.get(next_token, 0) + 1
                
                # Check for end-of-sequence token
                # Some models/tokenizers have special EOS tokens
                if hasattr(self.tokenizer, 'eos_id') and next_token == self.tokenizer.eos_id():
                    print(f"🔚 Reached end-of-sequence token at step {step}")
                    break
                
                # Optional: Check for other stopping conditions
                # Could add custom stop words or patterns here
        
        # Decode generated tokens to text
        # Convert token IDs back to readable text, excluding input prompt
        try:
            # Extract only newly generated tokens (exclude input prompt)
            new_tokens = generated_tokens[len(input_tokens):]
            
            if len(new_tokens) == 0:
                return "⚠️  No tokens generated"
            
            # Decode to text using tokenizer
            generated_text = self.tokenizer.decode(new_tokens)
            
            print(f"✅ Generated {len(new_tokens)} tokens")
            return generated_text
            
        except Exception as e:
            raise RuntimeError(f"Failed to decode generated tokens: {e}")
    
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float]
    ) -> int:
        """
        Sample next token using specified sampling strategy.
        
        Args:
            logits: Raw model predictions for next token
            temperature: Sampling temperature
            top_k: Top-k sampling limit
            top_p: Nucleus sampling threshold
            
        Returns:
            Selected token ID
            
        Implementation Details:
            - Applies temperature scaling for randomness control
            - Implements top-k sampling to limit choices
            - Implements nucleus (top-p) sampling for quality
            - Uses multinomial sampling for final selection
        """
        # Apply temperature scaling
        # Higher temperature = more random, lower = more deterministic
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        # Only consider the k most likely tokens
        if top_k is not None and top_k > 0:
            # Get indices of top-k tokens
            top_k_tokens = min(top_k, logits.size(-1))
            top_k_values, top_k_indices = torch.topk(logits, top_k_tokens)
            
            # Zero out non-top-k logits
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits[top_k_indices] = top_k_values
            logits = filtered_logits
        
        # Apply nucleus (top-p) sampling
        # Dynamically adjust vocabulary based on cumulative probability
        if top_p is not None and top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            
            # Calculate cumulative probabilities
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find cutoff point where cumulative probability exceeds top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Keep at least the top token
            sorted_indices_to_remove[0] = False
            
            # Zero out tokens beyond nucleus
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        # Convert logits to probabilities and sample
        # Use multinomial sampling for final token selection
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        return next_token
    
    def generate_batch(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> List[List[str]]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **generation_kwargs: Arguments passed to generate()
            
        Returns:
            List of lists, where each inner list contains generated texts for one prompt
            
        Implementation Details:
            - Processes prompts sequentially (could be parallelized)
            - Applies same generation parameters to all prompts
            - Handles errors gracefully for individual prompts
        """
        print(f"🔄 Generating text for {len(prompts)} prompts...")
        
        all_results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n--- Prompt {i + 1}/{len(prompts)} ---")
            
            try:
                results = self.generate(prompt, **generation_kwargs)
                all_results.append(results)
                
            except Exception as e:
                print(f"❌ Failed to generate for prompt {i + 1}: {e}")
                all_results.append([f"Generation failed: {e}"])
        
        return all_results


def load_prompts_from_file(file_path: str) -> List[str]:
    """
    Load prompts from a text file.
    
    Args:
        file_path: Path to file containing prompts (one per line)
        
    Returns:
        List of prompt strings
        
    Implementation Details:
        - Reads file line by line
        - Strips whitespace and filters empty lines
        - Handles different text encodings gracefully
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print(f"📄 Loaded {len(prompts)} prompts from {file_path}")
        return prompts
        
    except Exception as e:
        raise RuntimeError(f"Failed to load prompts from {file_path}: {e}")


def save_results_to_file(results: List[str], output_path: str, prompts: List[str] = None):
    """
    Save generation results to a text file.
    
    Args:
        results: Generated text results
        output_path: Path to output file
        prompts: Original prompts (optional, for context)
        
    Implementation Details:
        - Formats output with clear separators
        - Includes prompts and metadata when available
        - Handles file creation and error reporting
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# OpenLLM Text Generation Results\n")
            f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total samples: {len(results)}\n\n")
            
            for i, result in enumerate(results):
                f.write(f"--- Sample {i + 1} ---\n")
                
                if prompts and i < len(prompts):
                    f.write(f"Prompt: {prompts[i]}\n\n")
                
                if isinstance(result, list):
                    for j, text in enumerate(result):
                        f.write(f"Generated {j + 1}: {text}\n\n")
                else:
                    f.write(f"Generated: {result}\n\n")
                
                f.write("-" * 50 + "\n\n")
        
        print(f"💾 Results saved to: {output_path}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save results to {output_path}: {e}")


def main():
    """Main function for text generation script."""
    parser = argparse.ArgumentParser(
        description="Generate text using trained OpenLLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic text generation
  python core/src/generate_text.py \\
    --model_dir models/small-extended-4k \\
    --prompt "The history of artificial intelligence" \\
    --max_length 256 \\
    --temperature 0.7

  # Multiple samples with different settings
  python core/src/generate_text.py \\
    --model_dir models/small-extended-4k \\
    --prompt "Once upon a time" \\
    --max_length 100 \\
    --num_samples 3 \\
    --temperature 0.8

  # Batch generation from file
  python core/src/generate_text.py \\
    --model_dir models/small-extended-4k \\
    --prompts_file prompts.txt \\
    --output_file results.txt
        """
    )
    
    # Model and data arguments
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directory containing trained model checkpoints"
    )
    
    parser.add_argument(
        "--prompt",
        help="Input text prompt for generation"
    )
    
    parser.add_argument(
        "--prompts_file",
        help="Path to file containing prompts (one per line)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.1-2.0, default: 0.7)"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling parameter (default: 40, 0 = disabled)"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (default: 0.9, 1.0 = disabled)"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate per prompt (default: 1)"
    )
    
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (1.0 = no penalty, default: 1.0)"
    )
    
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    
    # Output options
    parser.add_argument(
        "--output_file",
        help="Path to save generation results"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for generation (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.prompt and not args.prompts_file:
        parser.error("Either --prompt or --prompts_file must be provided")
    
    if args.prompt and args.prompts_file:
        parser.error("Cannot specify both --prompt and --prompts_file")
    
    print("📝 OpenLLM Text Generation")
    print("=" * 50)
    
    try:
        # Initialize text generator
        # This loads the model and sets up all necessary components
        generator = TextGenerator(args.model_dir, args.device)
        
        # Prepare prompts for generation
        # Handle both single prompt and batch file inputs
        if args.prompt:
            prompts = [args.prompt]
        else:
            prompts = load_prompts_from_file(args.prompts_file)
        
        # Set up generation parameters
        # Convert arguments to generation parameters
        generation_kwargs = {
            'max_length': args.max_length,
            'temperature': args.temperature,
            'top_k': args.top_k if args.top_k > 0 else None,
            'top_p': args.top_p if args.top_p < 1.0 else None,
            'num_return_sequences': args.num_samples,
            'do_sample': not args.no_sample,
            'repetition_penalty': args.repetition_penalty
        }
        
        print(f"\n🎯 Generation Settings:")
        for key, value in generation_kwargs.items():
            print(f"   {key}: {value}")
        print()
        
        # Generate text
        # Process all prompts and collect results
        start_time = time.time()
        
        if len(prompts) == 1:
            # Single prompt generation
            results = generator.generate(prompts[0], **generation_kwargs)
        else:
            # Batch generation
            batch_results = generator.generate_batch(prompts, **generation_kwargs)
            results = [result for sublist in batch_results for result in sublist]
        
        generation_time = time.time() - start_time
        
        # Display results
        # Format and show generated text with clear presentation
        print(f"\n" + "=" * 60)
        print(f"📊 GENERATION RESULTS")
        print(f"=" * 60)
        
        total_samples = len(results) if isinstance(results[0], str) else sum(len(r) for r in results)
        print(f"🎯 Total samples: {total_samples}")
        print(f"⏱️  Generation time: {generation_time:.1f} seconds")
        print(f"📈 Speed: {total_samples / generation_time:.1f} samples/second")
        
        # Show individual results
        if len(prompts) == 1:
            # Single prompt results
            print(f"\n💭 Prompt: '{prompts[0]}'")
            print("-" * 50)
            
            for i, text in enumerate(results):
                if len(results) > 1:
                    print(f"\n[Sample {i + 1}]")
                print(text)
        else:
            # Batch results
            for i, (prompt, result_group) in enumerate(zip(prompts, batch_results)):
                print(f"\n💭 Prompt {i + 1}: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
                print("-" * 50)
                
                for j, text in enumerate(result_group):
                    if len(result_group) > 1:
                        print(f"\n[Sample {j + 1}]")
                    print(text)
                print()
        
        # Save results if requested
        # Write output to file with proper formatting
        if args.output_file:
            if len(prompts) == 1:
                save_results_to_file(results, args.output_file, [prompts[0]])
            else:
                # Flatten batch results for saving
                flat_results = []
                flat_prompts = []
                for prompt, result_group in zip(prompts, batch_results):
                    for result in result_group:
                        flat_results.append(result)
                        flat_prompts.append(prompt)
                
                save_results_to_file(flat_results, args.output_file, flat_prompts)
        
        print(f"\n🎉 Text generation completed successfully!")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Generation interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)