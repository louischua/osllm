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
GPT-style Language Model Architecture

This module implements a standard GPT (Generative Pre-trained Transformer) architecture
using pure PyTorch. The model is a decoder-only transformer designed for autoregressive
language modeling (next-token prediction).

ARCHITECTURE OVERVIEW:
- Token Embedding: Maps token IDs to dense vectors
- Positional Embedding: Adds position information to token embeddings
- Transformer Blocks: Stack of multi-head attention + feed-forward layers
- Layer Normalization: Pre-norm placement for training stability
- Output Head: Linear projection to vocabulary for next-token prediction

FEATURES:
- Configurable model size (small/medium/large)
- Dropout for regularization
- Causal (autoregressive) attention masking
- Compatible with our SentencePiece tokenizer
- Memory-efficient implementation for training on limited hardware

Usage:
    from model import GPTConfig, GPTModel

    config = GPTConfig(vocab_size=32000, n_layer=12, n_head=12, n_embd=768)
    model = GPTModel(config)

    # Forward pass
    logits = model(input_ids)  # Shape: (batch_size, seq_len, vocab_size)

Hardware Requirements:
- Small Model (25M params): 4-8GB RAM, CPU/integrated GPU
- Medium Model (117M params): 8-16GB RAM, dedicated GPU recommended
- Large Model (350M params): 16GB+ RAM, high-end GPU required

Author: Louis Chua Bean Chong
License: GPLv3
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    """
    Configuration class for GPT model hyperparameters.

    This class defines all the architectural parameters needed to instantiate
    a GPT model. Use the provided class methods to get pre-configured setups
    for different model sizes.
    """

    # Model architecture
    vocab_size: int = 32000  # Vocabulary size (from tokenizer)
    n_layer: int = 12  # Number of transformer layers
    n_head: int = 12  # Number of attention heads
    n_embd: int = 768  # Embedding dimension

    # Sequence and context
    block_size: int = 1024  # Maximum sequence length

    # Training hyperparameters
    dropout: float = 0.1  # Dropout probability
    bias: bool = True  # Use bias in linear layers

    # Model size identifier
    model_name: str = "gpt-medium"  # Human-readable model identifier

    @classmethod
    def small(cls) -> "GPTConfig":
        """Small model configuration (~25M parameters) - Good for CPU training"""
        return cls(
            vocab_size=32000,
            n_layer=6,
            n_head=8,
            n_embd=512,
            block_size=1024,
            dropout=0.1,
            model_name="gpt-small",
        )

    @classmethod
    def medium(cls) -> "GPTConfig":
        """Medium model configuration (~117M parameters) - Balanced performance"""
        return cls(
            vocab_size=32000,
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=2048,
            dropout=0.1,
            model_name="gpt-medium",
        )

    @classmethod
    def large(cls) -> "GPTConfig":
        """Large model configuration (~350M parameters) - High performance"""
        return cls(
            vocab_size=32000,
            n_layer=24,
            n_head=16,
            n_embd=1024,
            block_size=2048,
            dropout=0.1,
            model_name="gpt-large",
        )

    def estimate_parameters(self) -> int:
        """
        Estimate the total number of trainable parameters.

        Returns:
            int: Estimated parameter count
        """
        # Token embeddings
        token_emb = self.vocab_size * self.n_embd

        # Position embeddings
        pos_emb = self.block_size * self.n_embd

        # Transformer layers
        # Each layer: attention (4 * n_embd^2) + mlp (8 * n_embd^2) + layer_norms
        layer_params = self.n_layer * (12 * self.n_embd**2 + 4 * self.n_embd)

        # Output head
        output_head = self.vocab_size * self.n_embd

        total = token_emb + pos_emb + layer_params + output_head
        return total


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.

    This implements the core attention mechanism of the transformer, with causal
    masking to ensure autoregressive behavior (tokens can only attend to previous
    tokens, not future ones).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), "Embedding dim must be divisible by number of heads"

        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        # Key, query, value projections for all heads (batched)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask - lower triangular matrix
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of causal self-attention.

        This method implements the scaled dot-product attention mechanism with causal masking.
        The attention mechanism allows each token to attend to all previous tokens in the sequence,
        but not to future tokens, maintaining the autoregressive property essential for language modeling.

        Mathematical formulation:
            Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
            where Q, K, V are query, key, value matrices derived from input x

        Implementation details:
            - Uses batch matrix multiplication for efficiency
            - Applies causal mask to prevent future token attention
            - Implements multi-head attention by reshaping and parallel processing
            - Applies dropout for regularization during training

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
               Contains embedded token representations from previous layer

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Extract tensor dimensions for clear variable naming and validation
        # B = batch size (number of sequences processed in parallel)
        # T = sequence length (number of tokens in each sequence)
        # C = embedding dimensionality (n_embd from config)
        B, T, C = x.size()

        # Generate query, key, and value projections for all attention heads
        # The c_attn linear layer outputs 3 * n_embd features, which we split into Q, K, V
        # This batched approach is more efficient than separate linear layers
        # Input shape: (B, T, C) -> Output shape: (B, T, 3*C) -> Split to 3x (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape tensors for multi-head attention computation
        # Transform from (B, T, C) to (B, nh, T, hs) where:
        # - nh = number of heads (self.n_head)
        # - hs = head size (self.head_dim = C // nh)
        # The transpose(1, 2) moves the head dimension before sequence dimension for efficient computation
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Compute scaled dot-product attention scores
        # Matrix multiplication: Q @ K^T gives attention affinities between all token pairs
        # Scaling by 1/sqrt(head_dim) prevents softmax saturation for large embedding dimensions
        # Shape: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # The resulting (T, T) matrix represents attention weights from each token to every other token
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal masking to enforce autoregressive property
        # The causal mask ensures that token i can only attend to tokens j where j <= i
        # This prevents the model from "cheating" by looking at future tokens during training
        # We use -inf for masked positions so they become 0 after softmax
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # Convert attention scores to probabilities using softmax
        # Each row of the attention matrix now sums to 1, representing a probability distribution
        # over which tokens to attend to for each query position
        att = F.softmax(att, dim=-1)

        # Apply dropout to attention weights for regularization
        # This randomly zeros some attention connections during training to prevent overfitting
        att = self.attn_dropout(att)

        # Apply attention weights to value vectors
        # This weighted combination produces the actual output of the attention mechanism
        # Shape: (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        # Each output position is a weighted sum of all value vectors, with weights from attention
        y = att @ v

        # Concatenate multi-head outputs back to original embedding dimension
        # Transform from (B, nh, T, hs) back to (B, T, C) where C = nh * hs
        # The transpose moves head dimension back, and contiguous() ensures memory layout efficiency
        # This combines information from all attention heads into a single representation
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Apply final output projection and residual dropout
        # The output projection allows the model to learn how to best combine multi-head information
        # Residual dropout provides additional regularization before the residual connection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) for Transformer.

    This implements the position-wise feed-forward network that appears in each transformer layer.
    The MLP provides additional non-linear transformation capacity beyond what attention provides.

    Architecture:
        Input -> Linear(n_embd -> 4*n_embd) -> GELU -> Linear(4*n_embd -> n_embd) -> Dropout -> Output

    Design rationale:
        - 4x expansion is standard in transformers (from "Attention Is All You Need")
        - GELU activation provides smoother gradients than ReLU for language modeling
        - Dropout prevents overfitting in the feed-forward layers
        - Two linear layers allow complex non-linear transformations of attention outputs

    Parameters:
        - First linear layer: n_embd * 4*n_embd parameters (expansion)
        - Second linear layer: 4*n_embd * n_embd parameters (projection back)
        - Total: 8 * n_embd^2 parameters (significant portion of model size)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # First linear layer: expand embedding dimension by 4x
        # This expansion gives the network more representational capacity
        # The 4x factor is a standard choice that balances capacity vs efficiency
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)

        # GELU (Gaussian Error Linear Unit) activation function
        # GELU provides smoother gradients compared to ReLU and works better for language modeling
        # It's approximately: GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
        self.gelu = nn.GELU()

        # Second linear layer: project back to original embedding dimension
        # This projection allows the network to combine information from the expanded representation
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        # Dropout for regularization in the feed-forward network
        # Applied after the final projection to prevent overfitting
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.

        This method applies a two-layer MLP with GELU activation to transform
        the attention outputs. The MLP operates independently on each position
        in the sequence, providing position-wise non-linear transformations.

        Mathematical operation:
            MLP(x) = Dropout(Linear₂(GELU(Linear₁(x))))
            where Linear₁: R^n_embd -> R^4*n_embd and Linear₂: R^4*n_embd -> R^n_embd

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
               Contains attended representations from the attention layer

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
                         Contains transformed representations ready for residual connection
        """
        # First linear transformation: expand from n_embd to 4*n_embd dimensions
        # This expansion provides the network with a higher-dimensional space for computation
        # Shape: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, 4*n_embd)
        x = self.c_fc(x)

        # Apply GELU activation function for non-linearity
        # GELU is smoother than ReLU and provides better gradients for language modeling
        # It introduces non-linearity while maintaining differentiability everywhere
        x = self.gelu(x)

        # Second linear transformation: project back to original n_embd dimensions
        # This projection combines information from the expanded representation
        # Shape: (batch_size, seq_len, 4*n_embd) -> (batch_size, seq_len, n_embd)
        x = self.c_proj(x)

        # Apply dropout for regularization before residual connection
        # Dropout randomly zeros some neurons during training to prevent overfitting
        # This is particularly important in the feed-forward layers which have many parameters
        x = self.dropout(x)

        return x


class Block(nn.Module):
    """
    Single Transformer block.

    Consists of:
    1. Layer normalization
    2. Multi-head causal self-attention
    3. Residual connection
    4. Layer normalization
    5. MLP (feed-forward network)
    6. Residual connection

    Uses pre-norm architecture for better training stability.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Pre-norm attention with residual connection
        x = x + self.attn(self.ln_1(x))

        # Pre-norm MLP with residual connection
        x = x + self.mlp(self.ln_2(x))

        return x


class GPTModel(nn.Module):
    """
    Complete GPT Language Model.

    This is the main model class that combines all components:
    - Token and positional embeddings
    - Stack of transformer blocks
    - Final layer normalization
    - Language modeling head

    The model can be used for:
    - Training from scratch on text data
    - Fine-tuning on downstream tasks
    - Text generation (inference)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be specified"
        assert config.block_size is not None, "block_size must be specified"

        self.config = config

        # Embeddings
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),  # Position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),  # Transformer blocks
                ln_f=nn.LayerNorm(config.n_embd),  # Final layer norm
            )
        )

        # Language modeling head (maps hidden states to vocabulary)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights between token embeddings and output head (common practice)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        print(f"Model initialized: {self.config.model_name}")
        print(f"Parameters: {self.get_num_params():,}")
        print(f"Estimated: {self.config.estimate_parameters():,}")

    def _init_weights(self, module):
        """Initialize model weights using standard practices."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Count the number of parameters in the model.

        Args:
            non_embedding: If True, subtract embedding parameters

        Returns:
            int: Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the GPT model.

        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            targets: Optional target tokens for loss calculation (batch_size, seq_len)

        Returns:
            Tuple containing:
            - logits: Output logits of shape (batch_size, seq_len, vocab_size)
            - loss: Cross-entropy loss if targets provided, None otherwise
        """
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Sequence length {t} exceeds block size {self.config.block_size}"

        # Token embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)

        # Position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (t,)
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)

        # Combine embeddings and apply dropout
        x = self.transformer.drop(tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer normalization
        x = self.transformer.ln_f(x)

        # Language modeling head
        if targets is not None:
            # If we have targets, compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # If no targets, only compute logits for the last token (more efficient for generation)
            logits = self.lm_head(x[:, [-1], :])  # Note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            idx: Starting token indices (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k most likely tokens

        Returns:
            torch.Tensor: Generated sequence (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop sequence if it exceeds block size
                idx_cond = (
                    idx
                    if idx.size(1) <= self.config.block_size
                    else idx[:, -self.config.block_size :]
                )

                # Forward pass
                logits, _ = self(idx_cond)

                # Get logits for the last token and apply temperature
                logits = logits[:, -1, :] / temperature

                # Optionally crop to top-k most likely tokens
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Apply softmax and sample
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                idx = torch.cat((idx, idx_next), dim=1)

        self.train()  # Return to training mode
        return idx

    def estimate_memory_usage(self, batch_size: int = 1, seq_len: int = None) -> dict:
        """
        Estimate memory usage for training and inference.

        Args:
            batch_size: Batch size for estimation
            seq_len: Sequence length (defaults to block_size)

        Returns:
            dict: Memory usage estimates in MB
        """
        if seq_len is None:
            seq_len = self.config.block_size

        # Model parameters (weights)
        param_memory = self.get_num_params() * 4 / (1024**2)  # 4 bytes per float32

        # Activations (rough estimate)
        activation_memory = (
            batch_size * seq_len * self.config.n_embd * self.config.n_layer * 8  # Rough estimate
        ) / (1024**2)

        # Gradients (same size as parameters during training)
        gradient_memory = param_memory

        return {
            "parameters_mb": param_memory,
            "activations_mb": activation_memory,
            "gradients_mb": gradient_memory,
            "total_training_mb": param_memory + activation_memory + gradient_memory,
            "total_inference_mb": param_memory + activation_memory * 0.5,  # No gradients needed
        }


def create_model(model_size: str = "medium") -> GPTModel:
    """
    Factory function to create a GPT model with predefined configurations.

    Args:
        model_size: Size of model to create ("small", "medium", "large")

    Returns:
        GPTModel: Initialized model
    """
    configs = {
        "small": GPTConfig.small(),
        "medium": GPTConfig.medium(),
        "large": GPTConfig.large(),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    config = configs[model_size]
    model = GPTModel(config)

    return model


if __name__ == "__main__":
    # Example usage
    print("🧠 GPT Model Architecture")
    print("=" * 50)

    # Create models of different sizes
    for size in ["small", "medium", "large"]:
        print(f"\n{size.upper()} MODEL:")
        model = create_model(size)

        # Show memory estimates
        memory = model.estimate_memory_usage(batch_size=4, seq_len=512)
        print(
            f"Memory (4 batch, 512 seq): {memory['total_training_mb']:.1f}MB training, {memory['total_inference_mb']:.1f}MB inference"
        )

        # Test forward pass
        x = torch.randint(0, 32000, (2, 64))  # Batch size 2, sequence length 64
        with torch.no_grad():
            logits, _ = model(x)
        print(f"Test forward pass: {x.shape} -> {logits.shape} ✓")
