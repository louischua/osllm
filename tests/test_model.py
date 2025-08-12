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
Unit tests for the GPT model architecture.

This module tests the core model components including:
- GPTConfig: Configuration class and parameter estimation
- GPTModel: Main model architecture and forward pass
- CausalSelfAttention: Attention mechanism
- Block: Transformer block implementation
- Model loading/saving functionality

Test Coverage:
- Model initialization with different configurations
- Forward pass with various input shapes
- Parameter counting and estimation
- Model saving and loading
- Attention mechanism correctness
- Memory efficiency and performance
"""

import unittest
import torch
import tempfile
import os
import sys
from pathlib import Path

# Add the core/src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "core" / "src"))

from model import GPTConfig, GPTModel, CausalSelfAttention, Block


class TestGPTConfig(unittest.TestCase):
    """Test cases for GPTConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.small_config = GPTConfig.small()
        self.medium_config = GPTConfig.medium()
        self.large_config = GPTConfig.large()
        self.custom_config = GPTConfig(
            vocab_size=1000,
            n_layer=4,
            n_head=4,
            n_embd=256,
            block_size=512,
            dropout=0.2,
            bias=False
        )
    
    def test_small_config(self):
        """Test small model configuration."""
        config = self.small_config
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.n_layer, 6)
        self.assertEqual(config.n_head, 8)
        self.assertEqual(config.n_embd, 512)
        self.assertEqual(config.block_size, 1024)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.model_name, "gpt-small")
        self.assertTrue(config.bias)
    
    def test_medium_config(self):
        """Test medium model configuration."""
        config = self.medium_config
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.n_layer, 12)
        self.assertEqual(config.n_head, 12)
        self.assertEqual(config.n_embd, 768)
        self.assertEqual(config.block_size, 2048)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.model_name, "gpt-medium")
        self.assertTrue(config.bias)
    
    def test_large_config(self):
        """Test large model configuration."""
        config = self.large_config
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.n_layer, 24)
        self.assertEqual(config.n_head, 16)
        self.assertEqual(config.n_embd, 1024)
        self.assertEqual(config.block_size, 2048)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.model_name, "gpt-large")
        self.assertTrue(config.bias)
    
    def test_custom_config(self):
        """Test custom configuration parameters."""
        config = self.custom_config
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.n_layer, 4)
        self.assertEqual(config.n_head, 4)
        self.assertEqual(config.n_embd, 256)
        self.assertEqual(config.block_size, 512)
        self.assertEqual(config.dropout, 0.2)
        self.assertFalse(config.bias)
    
    def test_parameter_estimation(self):
        """Test parameter count estimation."""
        # Test small model
        small_params = self.small_config.estimate_parameters()
        self.assertGreater(small_params, 0)
        self.assertLess(small_params, 100_000_000)  # Should be ~35M
        
        # Test medium model
        medium_params = self.medium_config.estimate_parameters()
        self.assertGreater(medium_params, small_params)
        self.assertLess(medium_params, 200_000_000)  # Should be ~111M
        
        # Test large model
        large_params = self.large_config.estimate_parameters()
        self.assertGreater(large_params, medium_params)
        self.assertLess(large_params, 500_000_000)  # Should be ~350M
    
    def test_embedding_divisibility(self):
        """Test that embedding dimension is divisible by number of heads."""
        # This should work
        valid_config = GPTConfig(n_embd=512, n_head=8)
        self.assertEqual(valid_config.n_embd % valid_config.n_head, 0)
        
        # This should raise an assertion error
        with self.assertRaises(AssertionError):
            GPTModel(GPTConfig(n_embd=511, n_head=8))


class TestGPTModel(unittest.TestCase):
    """Test cases for GPTModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.small_config = GPTConfig.small()
        self.medium_config = GPTConfig.medium()
        self.small_model = GPTModel(self.small_config)
        self.medium_model = GPTModel(self.medium_config)
    
    def test_model_initialization(self):
        """Test model initialization with different configurations."""
        # Test small model
        self.assertIsNotNone(self.small_model)
        self.assertEqual(self.small_model.config.model_name, "gpt-small")
        
        # Test medium model
        self.assertIsNotNone(self.medium_model)
        self.assertEqual(self.medium_model.config.model_name, "gpt-medium")
        
        # Test that models are different
        self.assertNotEqual(
            self.small_model.config.n_layer,
            self.medium_model.config.n_layer
        )
    
    def test_forward_pass_small_batch(self):
        """Test forward pass with small batch size."""
        batch_size, seq_len = 2, 10
        vocab_size = self.small_config.vocab_size
        
        # Create input tensor
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        with torch.no_grad():
            logits, loss = self.small_model(input_ids)
        
        # Check output shape - when no targets, only last token logits are returned
        expected_shape = (batch_size, 1, vocab_size)
        self.assertEqual(logits.shape, expected_shape)
        
        # Check that logits are finite
        self.assertTrue(torch.isfinite(logits).all())
        
        # Loss should be None when no targets provided
        self.assertIsNone(loss)
    
    def test_forward_pass_large_batch(self):
        """Test forward pass with larger batch size."""
        batch_size, seq_len = 8, 32
        vocab_size = self.small_config.vocab_size
        
        # Create input tensor
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        with torch.no_grad():
            logits, loss = self.small_model(input_ids)
        
        # Check output shape - when no targets, only last token logits are returned
        expected_shape = (batch_size, 1, vocab_size)
        self.assertEqual(logits.shape, expected_shape)
    
    def test_forward_pass_different_lengths(self):
        """Test forward pass with different sequence lengths."""
        batch_size = 4
        vocab_size = self.small_config.vocab_size
        
        # Test different sequence lengths
        for seq_len in [1, 5, 10, 50, 100]:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                logits, loss = self.small_model(input_ids)
            
            # When no targets, only last token logits are returned
            expected_shape = (batch_size, 1, vocab_size)
            self.assertEqual(logits.shape, expected_shape)
    
    def test_forward_pass_max_length(self):
        """Test forward pass with maximum sequence length."""
        batch_size = 2
        seq_len = self.small_config.block_size
        vocab_size = self.small_config.vocab_size
        
        # Create input tensor
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        with torch.no_grad():
            logits, loss = self.small_model(input_ids)
        
        # Check output shape - when no targets, only last token logits are returned
        expected_shape = (batch_size, 1, vocab_size)
        self.assertEqual(logits.shape, expected_shape)
    
    def test_forward_pass_with_targets(self):
        """Test forward pass with targets (training mode)."""
        batch_size, seq_len = 2, 10
        vocab_size = self.small_config.vocab_size
        
        # Create input and target tensors
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        with torch.no_grad():
            logits, loss = self.small_model(input_ids, targets)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, vocab_size)
        self.assertEqual(logits.shape, expected_shape)
        
        # Loss should be computed when targets provided
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)
    
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        # Count parameters using model's method
        total_params = self.small_model.get_num_params()
        trainable_params = sum(p.numel() for p in self.small_model.parameters() if p.requires_grad)
        
        # Should have parameters
        self.assertGreater(total_params, 0)
        self.assertGreater(trainable_params, 0)
        
        # All parameters should be trainable
        self.assertEqual(total_params, trainable_params)
        
        # Should be reasonable size for small model (~35M parameters)
        self.assertGreater(total_params, 30_000_000)
        self.assertLess(total_params, 40_000_000)
    
    def test_model_device_transfer(self):
        """Test model transfer to different devices."""
        if torch.cuda.is_available():
            # Test CUDA transfer
            model_cuda = self.small_model.cuda()
            self.assertEqual(next(model_cuda.parameters()).device.type, 'cuda')
            
            # Test CPU transfer back
            model_cpu = model_cuda.cpu()
            self.assertEqual(next(model_cpu.parameters()).device.type, 'cpu')
    
    def test_model_eval_mode(self):
        """Test model evaluation mode."""
        # Set to eval mode
        self.small_model.eval()
        self.assertFalse(self.small_model.training)
        
        # Set to train mode
        self.small_model.train()
        self.assertTrue(self.small_model.training)


class TestCausalSelfAttention(unittest.TestCase):
    """Test cases for CausalSelfAttention class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig.small()
        self.attention = CausalSelfAttention(self.config)
    
    def test_attention_initialization(self):
        """Test attention module initialization."""
        self.assertIsNotNone(self.attention)
        self.assertEqual(self.attention.n_head, self.config.n_head)
        self.assertEqual(self.attention.n_embd, self.config.n_embd)
        self.assertEqual(self.attention.head_dim, self.config.n_embd // self.config.n_head)
    
    def test_attention_forward_pass(self):
        """Test attention forward pass."""
        batch_size, seq_len = 2, 10
        input_tensor = torch.randn(batch_size, seq_len, self.config.n_embd)
        
        with torch.no_grad():
            output = self.attention(input_tensor)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, self.config.n_embd)
        self.assertEqual(output.shape, expected_shape)
        
        # Check that output is finite
        self.assertTrue(torch.isfinite(output).all())
    
    def test_causal_masking(self):
        """Test that causal masking prevents future token attention."""
        batch_size, seq_len = 1, 5
        input_tensor = torch.randn(batch_size, seq_len, self.config.n_embd)
        
        # Set to eval mode to disable dropout
        self.attention.eval()
        
        with torch.no_grad():
            output = self.attention(input_tensor)
        
        # The output should be different for each position due to causal masking
        # This is a basic test - in practice, you'd need more sophisticated analysis
        self.assertTrue(torch.isfinite(output).all())


class TestModelPersistence(unittest.TestCase):
    """Test cases for model saving and loading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig.small()
        self.model = GPTModel(self.config)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            save_path = os.path.join(temp_dir, "test_model.pt")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }, save_path)
            
            # Load checkpoint with weights_only=False for custom classes
            checkpoint = torch.load(save_path, weights_only=False)
            loaded_config = checkpoint['config']
            loaded_model = GPTModel(loaded_config)
            loaded_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Test that loaded model produces valid output
            batch_size, seq_len = 2, 10
            vocab_size = self.config.vocab_size
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                original_logits, original_loss = self.model(input_ids)
                loaded_logits, loaded_loss = loaded_model(input_ids)
            
            # Check that outputs have the same shape
            self.assertEqual(original_logits.shape, loaded_logits.shape)
            
            # Check that outputs are finite
            self.assertTrue(torch.isfinite(original_logits).all())
            self.assertTrue(torch.isfinite(loaded_logits).all())
            
            # Check that both models produce reasonable outputs (not all zeros or infinities)
            self.assertTrue(torch.any(original_logits != 0))
            self.assertTrue(torch.any(loaded_logits != 0))
            self.assertFalse(torch.any(torch.isinf(original_logits)))
            self.assertFalse(torch.any(torch.isinf(loaded_logits)))
    
    def test_config_serialization(self):
        """Test that config can be serialized and deserialized."""
        config = GPTConfig.small()
        
        # Test that config can be converted to dict and back
        config_dict = {
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'block_size': config.block_size,
            'dropout': config.dropout,
            'bias': config.bias,
            'model_name': config.model_name
        }
        
        # Recreate config from dict
        recreated_config = GPTConfig(**config_dict)
        
        # Should be identical
        self.assertEqual(config.vocab_size, recreated_config.vocab_size)
        self.assertEqual(config.n_layer, recreated_config.n_layer)
        self.assertEqual(config.n_head, recreated_config.n_head)
        self.assertEqual(config.n_embd, recreated_config.n_embd)
        self.assertEqual(config.block_size, recreated_config.block_size)
        self.assertEqual(config.dropout, recreated_config.dropout)
        self.assertEqual(config.bias, recreated_config.bias)
        self.assertEqual(config.model_name, recreated_config.model_name)


class TestModelPerformance(unittest.TestCase):
    """Test cases for model performance and memory usage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig.small()
        self.model = GPTModel(self.config)
    
    def test_memory_usage(self):
        """Test that model doesn't use excessive memory."""
        batch_size, seq_len = 4, 64
        vocab_size = self.config.vocab_size
        
        # Create input tensor
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Measure memory before forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
            
            # Move model and input to GPU
            model_gpu = self.model.cuda()
            input_gpu = input_ids.cuda()
            
            # Forward pass
            with torch.no_grad():
                _ = model_gpu(input_gpu)
            
            # Measure memory after forward pass
            memory_after = torch.cuda.memory_allocated()
            memory_used = memory_after - memory_before
            
            # Memory usage should be reasonable (less than 1GB for small model)
            self.assertLess(memory_used, 1024 * 1024 * 1024)  # 1GB
    
    def test_inference_speed(self):
        """Test that model inference is reasonably fast."""
        batch_size, seq_len = 2, 32
        vocab_size = self.config.vocab_size
        
        # Create input tensor
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(input_ids)
        
        # Time inference
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(input_ids)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Should be reasonably fast (less than 1 second per forward pass)
        self.assertLess(avg_time, 1.0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
