#!/usr/bin/env python3
# Copyright (C) 2024 Louis Chua Bean Chong
#
# This file is part of OpenLLM.
#
# OpenLLM is dual-licensed:
# 1. For open source use: GNU General Public License v3.0
# 2. For commercial use: Commercial License (contact for details)
#
# See LICENSE and docs/LICENSES.md for full license information.

"""
Train a SentencePiece tokenizer from scratch using the prepared training data.

OVERVIEW:
This script trains a SentencePiece tokenizer on the cleaned text data from the SQUAD dataset
or any other text corpus. SentencePiece is a subword tokenizer that works well for language
models and supports multiple languages without requiring pre-tokenization.

FEATURES:
- Supports BPE (Byte Pair Encoding) and Unigram tokenization algorithms
- Configurable vocabulary size (recommended: 8k-64k for LLMs)
- Handles special tokens (BOS, EOS, UNK, PAD)
- Outputs tokenizer model files compatible with Hugging Face
- Comprehensive statistics and vocabulary analysis

TOKENIZER OUTPUT:
- tokenizer.model: SentencePiece model file
- tokenizer.vocab: Human-readable vocabulary file
- tokenizer_config.json: Configuration for Hugging Face integration

Usage:
    python core/src/train_tokenizer.py --input data/clean/training_data.txt --vocab_size 32000

Advanced usage:
    python core/src/train_tokenizer.py \\
        --input data/clean/training_data.txt \\
        --vocab_size 32000 \\
        --model_type bpe \\
        --output_dir data/tokenizer/ \\
        --character_coverage 0.9995

Requirements:
    pip install sentencepiece

Example setup:
```bash
# If not already in virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\Activate.ps1  # Windows PowerShell

# Install SentencePiece
pip install sentencepiece

# Train tokenizer
python core/src/train_tokenizer.py --input data/clean/training_data.txt --vocab_size 32000
```

"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

try:
    import sentencepiece as spm
except ImportError:
    print("ERROR: SentencePiece not installed. Run: pip install sentencepiece")
    exit(1)


def validate_input_file(input_path: str) -> None:
    """
    Validate that the input training file exists and is readable.

    Args:
        input_path (str): Path to the training text file

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input file is empty or unreadable
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Training data file not found: {input_path}")

    # Check file size and readability
    file_size = os.path.getsize(input_path)
    if file_size == 0:
        raise ValueError(f"Training data file is empty: {input_path}")

    # Test that we can read the file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line.strip():
                raise ValueError(
                    "Training data file appears to be empty or contains only whitespace"
                )
    except UnicodeDecodeError as e:
        raise ValueError(f"Cannot read training data file as UTF-8: {e}")

    print(f"✓ Input file validated: {input_path} ({file_size:,} bytes)")


def count_training_sentences(input_path: str) -> int:
    """
    Count the number of training sentences/lines in the input file.

    Args:
        input_path (str): Path to the training text file

    Returns:
        int: Number of lines in the file
    """
    print("Counting training sentences...")
    with open(input_path, "r", encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())
    print(f"✓ Found {count:,} training sentences")
    return count


def train_sentencepiece_tokenizer(
    input_path: str,
    output_dir: str,
    vocab_size: int = 32000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    max_sentence_length: int = 4192,
    input_sentence_size: int = 10000000,
    shuffle_input_sentence: bool = True,
) -> Dict[str, Any]:
    """
    Train a SentencePiece tokenizer with the specified parameters.

    Args:
        input_path (str): Path to training text file
        output_dir (str): Directory to save tokenizer files
        vocab_size (int): Target vocabulary size (recommended: 8k-64k)
        model_type (str): Algorithm type ('bpe' or 'unigram')
        character_coverage (float): Character coverage (0.9995 for English, 1.0 for Japanese)
        max_sentence_length (int): Maximum sentence length in characters
        input_sentence_size (int): Maximum number of sentences to use for training
        shuffle_input_sentence (bool): Whether to shuffle input sentences

    Returns:
        Dict[str, Any]: Training statistics and configuration
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define output paths
    model_prefix = os.path.join(output_dir, "tokenizer")

    # SentencePiece training parameters
    train_params = [
        f"--input={input_path}",
        f"--model_prefix={model_prefix}",
        f"--vocab_size={vocab_size}",
        f"--model_type={model_type}",
        f"--character_coverage={character_coverage}",
        f"--max_sentence_length={max_sentence_length}",
        f"--input_sentence_size={input_sentence_size}",
        f"--shuffle_input_sentence={shuffle_input_sentence}",
        # Special tokens for language modeling
        "--pad_id=0",  # Padding token
        "--unk_id=1",  # Unknown token
        "--bos_id=2",  # Beginning of sequence
        "--eos_id=3",  # End of sequence
        # Additional useful parameters
        "--split_by_unicode_script=true",  # Better handling of mixed scripts
        "--split_by_whitespace=true",  # Split on whitespace
        "--remove_extra_whitespaces=true",  # Clean up whitespace
        "--normalization_rule_name=identity",  # Keep original text as-is
    ]

    print("\nTraining SentencePiece tokenizer...")
    print(f"  Algorithm: {model_type.upper()}")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Character coverage: {character_coverage}")
    print(f"  Output directory: {output_dir}")
    print(f"  Model files: {model_prefix}.model, {model_prefix}.vocab")

    # Record training start time
    start_time = time.time()

    # Train the tokenizer
    try:
        spm.SentencePieceTrainer.train(" ".join(train_params))
        training_time = time.time() - start_time
        print(f"✓ Tokenizer training completed in {training_time:.1f} seconds")
    except Exception as e:
        raise RuntimeError(f"SentencePiece training failed: {e}")

    # Verify output files were created
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"

    if not os.path.exists(model_file):
        raise RuntimeError(f"Expected model file not created: {model_file}")
    if not os.path.exists(vocab_file):
        raise RuntimeError(f"Expected vocab file not created: {vocab_file}")

    print(f"✓ Model file created: {model_file} ({os.path.getsize(model_file):,} bytes)")
    print(f"✓ Vocab file created: {vocab_file} ({os.path.getsize(vocab_file):,} bytes)")

    # Return training configuration and statistics
    config = {
        "model_type": model_type,
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "max_sentence_length": max_sentence_length,
        "training_time_seconds": training_time,
        "input_file": input_path,
        "output_directory": output_dir,
        "model_file": model_file,
        "vocab_file": vocab_file,
    }

    return config


def test_tokenizer(model_path: str, test_sentences: list = None) -> None:
    """
    Test the trained tokenizer on sample sentences to verify it works correctly.

    Args:
        model_path (str): Path to the trained .model file
        test_sentences (list): Optional list of test sentences
    """
    print("\nTesting trained tokenizer...")

    # Load the trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    # Default test sentences if none provided
    if test_sentences is None:
        test_sentences = [
            "Hello, world! This is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning and artificial intelligence are transforming technology.",
            "SentencePiece tokenization works well for language models.",
        ]

    print(f"Vocabulary size: {sp.vocab_size():,}")
    print(
        f"Special tokens: PAD={sp.pad_id()}, UNK={sp.unk_id()}, BOS={sp.bos_id()}, EOS={sp.eos_id()}"
    )

    print("\nTokenization examples:")
    for i, sentence in enumerate(test_sentences, 1):
        # Encode to token IDs and pieces
        token_ids = sp.encode(sentence)
        token_pieces = sp.encode(sentence, out_type=str)

        print(f"\n{i}. Input: {sentence}")
        print(f"   Tokens ({len(token_pieces)}): {token_pieces}")
        print(f"   IDs: {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")

        # Test decoding
        decoded = sp.decode(token_ids)
        print(f"   Decoded: {decoded}")

        # Verify round-trip encoding/decoding
        if decoded.strip() != sentence.strip():
            print("   ⚠️  Warning: Decode mismatch!")

    print("✓ Tokenizer testing completed")


def save_huggingface_config(output_dir: str, config: Dict[str, Any]) -> None:
    """
    Save a Hugging Face compatible tokenizer configuration file.

    Args:
        output_dir (str): Directory containing the tokenizer files
        config (Dict[str, Any]): Tokenizer configuration
    """
    # Create Hugging Face tokenizer config
    hf_config = {
        "tokenizer_class": "SentencePieceTokenizer",
        "model_type": config["model_type"],
        "vocab_size": config["vocab_size"],
        "model_file": "tokenizer.model",
        "special_tokens": {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
        },
        "special_token_ids": {
            "pad_token_id": 0,
            "unk_token_id": 1,
            "bos_token_id": 2,
            "eos_token_id": 3,
        },
    }

    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2, ensure_ascii=False)

    print(f"✓ Hugging Face config saved: {config_path}")


def main():
    """Main function to handle command line arguments and orchestrate tokenizer training."""
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer for language model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with SQUAD data
  python core/src/train_tokenizer.py --input data/clean/training_data.txt --vocab_size 32000

  # Advanced configuration
  python core/src/train_tokenizer.py \\
    --input data/clean/training_data.txt \\
    --vocab_size 32000 \\
    --model_type bpe \\
    --output_dir data/tokenizer/ \\
    --character_coverage 0.9995
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        required=True,
        help="Path to training text file (e.g., data/clean/training_data.txt)",
    )

    # Optional arguments with sensible defaults
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000, recommended: 8k-64k)",
    )

    parser.add_argument(
        "--model_type",
        choices=["bpe", "unigram"],
        default="bpe",
        help="Tokenization algorithm (default: bpe)",
    )

    parser.add_argument(
        "--output_dir",
        default="data/tokenizer/",
        help="Output directory for tokenizer files (default: data/tokenizer/)",
    )

    parser.add_argument(
        "--character_coverage",
        type=float,
        default=0.9995,
        help="Character coverage (default: 0.9995 for English)",
    )

    parser.add_argument(
        "--max_sentence_length",
        type=int,
        default=4192,
        help="Maximum sentence length in characters (default: 4192)",
    )

    parser.add_argument(
        "--no_test", action="store_true", help="Skip tokenizer testing after training"
    )

    args = parser.parse_args()

    print("🔤 SentencePiece Tokenizer Training")
    print("=" * 50)

    try:
        # Step 1: Validate input file
        validate_input_file(args.input)

        # Step 2: Count training data
        sentence_count = count_training_sentences(args.input)

        # Step 3: Train tokenizer
        config = train_sentencepiece_tokenizer(
            input_path=args.input,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            max_sentence_length=args.max_sentence_length,
        )

        # Step 4: Save Hugging Face compatible config
        save_huggingface_config(args.output_dir, config)

        # Step 5: Test tokenizer (unless skipped)
        if not args.no_test:
            model_path = os.path.join(args.output_dir, "tokenizer.model")
            test_tokenizer(model_path)

        # Step 6: Print summary
        print("\n🎉 Tokenizer training completed successfully!")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📊 Vocabulary size: {config['vocab_size']:,}")
        print(f"⏱️  Training time: {config['training_time_seconds']:.1f}s")
        print(f"📄 Training sentences: {sentence_count:,}")

        print("\nFiles created:")
        print(f"  • {config['model_file']} - SentencePiece model")
        print(f"  • {config['vocab_file']} - Vocabulary file")
        print(f"  • {os.path.join(args.output_dir, 'tokenizer_config.json')} - Hugging Face config")

        print("\nTo use this tokenizer in your language model:")
        print("  import sentencepiece as spm")
        print("  sp = spm.SentencePieceProcessor()")
        print(f"  sp.load('{config['model_file']}')")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
