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
Training Data Loader for Language Model Training

This module provides efficient data loading and batching for training GPT-style
language models. It handles text preprocessing, tokenization, and creates
batches suitable for autoregressive language modeling.

FEATURES:
- Memory-efficient text loading with sliding window
- Automatic tokenization using trained SentencePiece model
- Configurable sequence length and batch size
- CPU-optimized data loading for limited hardware
- Support for training data validation and statistics

MEMORY OPTIMIZATION:
- Streaming data loading (doesn't load entire dataset to memory)
- Configurable chunk sizes for large files
- Efficient tensor creation and batching
- Garbage collection hints for memory management

Usage:
    from data_loader import TextDataLoader

    loader = TextDataLoader(
        data_file="data/clean/training_data.txt",
        tokenizer_path="data/tokenizer/tokenizer.model",
        seq_len=512,
        batch_size=4
    )

    for batch in loader:
        input_ids, targets = batch
        # input_ids: (batch_size, seq_len)
        # targets: (batch_size, seq_len) - shifted by 1 for next token prediction

Author: Louis Chua Bean Chong
License: GPLv3
"""

import gc
import os
import random
import time
from typing import Iterator, List, Tuple

import torch

try:
    import sentencepiece as spm
except ImportError:
    print("ERROR: SentencePiece not installed. Run: pip install sentencepiece")
    exit(1)


class TextDataLoader:
    """
    Efficient data loader for autoregressive language model training.

    This class handles loading text data, tokenizing it using SentencePiece,
    and creating batches suitable for next-token prediction training.
    """

    def __init__(
        self,
        data_file: str,
        tokenizer_path: str,
        seq_len: int = 512,
        batch_size: int = 4,
        chunk_size: int = 1000000,  # Lines to read at once
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the data loader.

        Args:
            data_file: Path to training text file (one passage per line)
            tokenizer_path: Path to trained SentencePiece model
            seq_len: Maximum sequence length for training
            batch_size: Batch size for training
            chunk_size: Number of lines to read in memory at once
            shuffle: Whether to shuffle training examples
            seed: Random seed for reproducibility
        """
        self.data_file = data_file
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.seed = seed

        # Validate inputs
        self._validate_inputs()

        # Load tokenizer
        self.tokenizer = self._load_tokenizer()

        # Get data statistics
        self.total_lines = self._count_lines()
        self.current_line = 0

        # Initialize data attribute for testing compatibility
        # Load a small sample of data for testing purposes
        self.data = self._read_chunk(0, min(self.chunk_size, 100))  # Load up to 100 passages for testing

        # Set random seed for reproducibility
        random.seed(seed)

        print("📊 TextDataLoader initialized")
        print(f"  Data file: {data_file}")
        print(f"  Total passages: {self.total_lines:,}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Batch size: {batch_size}")
        print(f"  Vocabulary size: {self.tokenizer.vocab_size():,}")

    def _validate_inputs(self) -> None:
        """Validate input parameters and file paths."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Training data file not found: {self.data_file}")

        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer model not found: {self.tokenizer_path}")

        if self.seq_len <= 0:
            raise ValueError(f"Sequence length must be positive, got {self.seq_len}")

        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")

        if self.chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got {self.chunk_size}")

    def _load_tokenizer(self) -> spm.SentencePieceProcessor:
        """Load the trained SentencePiece tokenizer."""
        try:
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.load(self.tokenizer_path)
            return tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")

    def _count_lines(self) -> int:
        """Count total number of lines in the data file."""
        print("📏 Counting training passages...")
        start_time = time.time()

        line_count = 0
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Only count non-empty lines
                    line_count += 1

        count_time = time.time() - start_time
        print(f"✓ Found {line_count:,} passages in {count_time:.1f}s")

        return line_count

    def _read_chunk(self, start_line: int = 0, limit: int = None) -> List[str]:
        """
        Read a chunk of lines from the data file.

        Args:
            start_line: Line number to start reading from
            limit: Maximum number of lines to read (None for default chunk_size)

        Returns:
            List of text passages
        """
        chunk = []
        current_line = 0
        lines_read = 0
        max_lines = limit if limit is not None else self.chunk_size

        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                if current_line < start_line:
                    current_line += 1
                    continue

                text = line.strip()
                if text:  # Only include non-empty lines
                    chunk.append(text)
                    lines_read += 1

                    if lines_read >= max_lines:
                        break

                current_line += 1

        return chunk

    def _tokenize_texts(self, texts: List[str]) -> List[List[int]]:
        """
        Tokenize a list of text passages using SentencePiece tokenizer.

        This method converts raw text into token ID sequences suitable for language model training.
        It handles special tokens (BOS/EOS) and length constraints for efficient training.

        Text processing pipeline:
        1. Add BOS (Beginning of Sequence) token to mark sequence start
        2. Tokenize text using trained SentencePiece model (subword tokenization)
        3. Truncate sequences that exceed maximum length
        4. Add EOS (End of Sequence) token to mark sequence end

        Special token handling:
        - BOS token helps model learn to generate text from scratch
        - EOS token signals natural sequence endings
        - These tokens are crucial for proper autoregressive generation

        Args:
            texts: List of text passages (typically Wikipedia passages from SQUAD)
                  Each passage should be a complete, coherent text segment

        Returns:
            List of token ID sequences, where each sequence is a list of integers
            representing subword tokens from the SentencePiece vocabulary
        """
        tokenized = []

        for text in texts:
            try:
                # Add BOS (Beginning of Sequence) token at the start
                # BOS token ID=2 by default in SentencePiece, signals sequence start
                # This helps the model learn proper sequence initialization during generation
                tokens = [self.tokenizer.bos_id()] + self.tokenizer.encode(text)

                # Truncate sequences that exceed maximum context length
                # Reserve one position for EOS token by using (seq_len - 1)
                # This ensures we never exceed the model's context window during training
                if len(tokens) > self.seq_len - 1:
                    tokens = tokens[: self.seq_len - 1]
                    # NOTE: Truncation may cut off text mid-sentence, but this is acceptable
                    # for language modeling where the model learns from partial contexts

                # Add EOS (End of Sequence) token at the end
                # EOS token ID=1 by default in SentencePiece, signals sequence completion
                # This teaches the model when to stop generating text naturally
                tokens.append(self.tokenizer.eos_id())

                # Validate tokenization result
                if len(tokens) <= 2:  # Only BOS + EOS tokens, no actual content
                    print(f"⚠️  Skipping very short text: {text[:50]}...")
                    continue

                tokenized.append(tokens)

            except Exception as e:
                # Handle tokenization errors gracefully to avoid stopping training
                # Common causes: encoding issues, very long texts, special characters
                print(f"⚠️  Failed to tokenize passage: {text[:50]}... Error: {e}")
                continue

        # Log tokenization statistics for monitoring
        if tokenized:
            avg_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
            print(f"📊 Tokenized {len(tokenized)} passages, avg length: {avg_length:.1f} tokens")

        return tokenized

    def _create_training_examples(
        self, token_sequences: List[List[int]]
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Create training examples with input and target sequences.

        For autoregressive training, targets are inputs shifted by one position.

        Args:
            token_sequences: List of tokenized sequences

        Returns:
            List of (input_ids, target_ids) tuples
        """
        examples = []

        for tokens in token_sequences:
            if len(tokens) < 2:  # Need at least 2 tokens for input/target pair
                continue

            # For sequences longer than seq_len, create multiple examples with sliding window
            if len(tokens) > self.seq_len:
                # Create overlapping windows (50% overlap for better learning)
                stride = self.seq_len // 2
                for i in range(0, len(tokens) - self.seq_len, stride):
                    input_ids = tokens[i : i + self.seq_len]
                    target_ids = tokens[i + 1 : i + self.seq_len + 1]
                    examples.append((input_ids, target_ids))
            else:
                # Pad shorter sequences
                input_ids = tokens[:-1]  # All but last token
                target_ids = tokens[1:]  # All but first token

                # Pad to seq_len if necessary
                while len(input_ids) < self.seq_len:
                    input_ids.append(self.tokenizer.pad_id())
                    target_ids.append(-1)  # Use -1 for padding in targets (ignored in loss)

                # Truncate if still too long
                input_ids = input_ids[: self.seq_len]
                target_ids = target_ids[: self.seq_len]

                examples.append((input_ids, target_ids))

        return examples

    def _create_batch(
        self, examples: List[Tuple[List[int], List[int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a batch tensor from training examples.

        Args:
            examples: List of (input_ids, target_ids) tuples

        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        if not examples:
            raise ValueError("Cannot create batch from empty examples")

        batch_size = len(examples)

        # Initialize tensors
        input_ids = torch.zeros((batch_size, self.seq_len), dtype=torch.long)
        target_ids = torch.full((batch_size, self.seq_len), -1, dtype=torch.long)

        # Fill tensors
        for i, (inp, tgt) in enumerate(examples):
            input_ids[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
            target_ids[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)

        return input_ids, target_ids

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over training batches.

        Yields:
            Tuple of (input_ids, target_ids) tensors
        """
        self.current_line = 0

        while self.current_line < self.total_lines:
            # Read chunk of text
            texts = self._read_chunk(self.current_line)
            if not texts:
                break

            # Tokenize texts
            token_sequences = self._tokenize_texts(texts)

            # Create training examples
            examples = self._create_training_examples(token_sequences)

            # Shuffle examples if requested
            if self.shuffle:
                random.shuffle(examples)

            # Create batches
            for i in range(0, len(examples), self.batch_size):
                batch_examples = examples[i : i + self.batch_size]

                if len(batch_examples) == self.batch_size:  # Only yield full batches
                    try:
                        input_ids, target_ids = self._create_batch(batch_examples)
                        yield input_ids, target_ids
                    except Exception as e:
                        print(f"⚠️  Failed to create batch: {e}")
                        continue

            # Update progress
            self.current_line += len(texts)

            # Clean up memory
            del texts, token_sequences, examples
            gc.collect()

    def get_data_stats(self) -> dict:
        """
        Get statistics about the training data.

        Returns:
            Dictionary with data statistics
        """
        print("📊 Analyzing training data...")

        # Sample some data to get statistics
        sample_texts = self._read_chunk(0)[:100]  # Sample first 100 passages
        token_sequences = self._tokenize_texts(sample_texts)

        if token_sequences:
            sequence_lengths = [len(seq) for seq in token_sequences]
            avg_length = sum(sequence_lengths) / len(sequence_lengths)
            max_length = max(sequence_lengths)
            min_length = min(sequence_lengths)
        else:
            avg_length = max_length = min_length = 0

        # Estimate total tokens
        estimated_total_tokens = int(avg_length * self.total_lines)

        # Estimate number of batches per epoch
        examples_per_passage = max(1, avg_length // self.seq_len)
        total_examples = int(self.total_lines * examples_per_passage)
        batches_per_epoch = total_examples // self.batch_size

        stats = {
            "total_passages": self.total_lines,
            "avg_tokens_per_passage": avg_length,
            "min_tokens_per_passage": min_length,
            "max_tokens_per_passage": max_length,
            "estimated_total_tokens": estimated_total_tokens,
            "estimated_examples_per_epoch": total_examples,
            "estimated_batches_per_epoch": batches_per_epoch,
            "sequence_length": self.seq_len,
            "batch_size": self.batch_size,
            "vocabulary_size": self.tokenizer.vocab_size(),
        }

        print("✓ Data analysis complete:")
        print(f"  Total passages: {stats['total_passages']:,}")
        print(f"  Avg tokens per passage: {stats['avg_tokens_per_passage']:.1f}")
        print(f"  Estimated total tokens: {stats['estimated_total_tokens']:,}")
        print(f"  Estimated batches per epoch: {stats['estimated_batches_per_epoch']:,}")

        return stats


def test_data_loader():
    """Test function for the data loader."""
    print("🧪 Testing TextDataLoader...")

    # Test with small parameters
    try:
        loader = TextDataLoader(
            data_file="data/clean/training_data.txt",
            tokenizer_path="data/tokenizer/tokenizer.model",
            seq_len=128,
            batch_size=2,
            chunk_size=10,  # Small for testing
        )

        # Get data statistics
        _ = loader.get_data_stats()

        # Test iteration
        print("\n🔄 Testing batch iteration...")
        start_time = time.time()
        batch_count = 0

        for batch_idx, (input_ids, target_ids) in enumerate(loader):
            batch_count += 1

            print(f"Batch {batch_idx + 1}:")
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Target shape: {target_ids.shape}")
            print(f"  Sample input tokens: {input_ids[0][:10].tolist()}")
            print(f"  Sample target tokens: {target_ids[0][:10].tolist()}")

            if batch_idx >= 2:  # Only test first few batches
                break

        test_time = time.time() - start_time
        print("\n✓ Data loader test completed successfully!")
        print(f"  Processed {batch_count} batches in {test_time:.2f}s")
        print(f"  Average time per batch: {test_time/max(1, batch_count):.2f}s")

        return True

    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_data_loader()
