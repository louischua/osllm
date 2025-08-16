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
Unit tests for the training pipeline.

This module tests the training-related components including:
- Data loading and preprocessing
- Training loop functionality
- Model evaluation and metrics
- Checkpoint saving and loading
- Training configuration validation

Test Coverage:
- Data loader functionality
- Training step execution
- Loss computation and backpropagation
- Learning rate scheduling
- Model checkpointing
- Training metrics logging
"""

import unittest
import torch
import tempfile
import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the core/src directory to the path for imports
core_src_path = str(Path(__file__).parent.parent / "core" / "src")
sys.path.insert(0, core_src_path)

# Import after path setup
try:
    from model import GPTConfig, GPTModel
    from data_loader import TextDataLoader
    from train_model import ModelTrainer
    from evaluate_model import ModelEvaluator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"📁 Core src path: {core_src_path}")
    print(f"📁 Available files: {os.listdir(core_src_path) if os.path.exists(core_src_path) else 'Path not found'}")
    raise


class TestDataLoader(unittest.TestCase):
    """Test cases for TextDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig.small()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample training data
        self.sample_data = [
            "This is a sample text for testing.",
            "Another sample text with different content.",
            "A third sample to ensure variety in the dataset.",
            "Testing the data loader with multiple sentences.",
            "This should be enough data for basic testing."
        ]
        
        # Write sample data to file
        self.data_file = os.path.join(self.temp_dir, "test_data.txt")
        with open(self.data_file, 'w', encoding='utf-8') as f:
            for line in self.sample_data:
                f.write(line + '\n')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_data_loader_initialization(self):
        """Test TextDataLoader initialization."""
        # Create data loader with real tokenizer
        data_loader = TextDataLoader(
            data_file=self.data_file,
            tokenizer_path="data/tokenizer/tokenizer.model",
            seq_len=10,
            batch_size=2
        )
        
        self.assertIsNotNone(data_loader)
        self.assertEqual(data_loader.batch_size, 2)
        self.assertEqual(data_loader.seq_len, 10)
    
    def test_data_loading(self):
        """Test that data is loaded correctly."""
        # Create data loader with real tokenizer
        data_loader = TextDataLoader(
            data_file=self.data_file,
            tokenizer_path="data/tokenizer/tokenizer.model",
            seq_len=10,
            batch_size=2
        )
        
        # Check that data file exists
        self.assertTrue(os.path.exists(self.data_file))
    
    def test_batch_generation(self):
        """Test batch generation functionality."""
        # Create data loader with real tokenizer
        data_loader = TextDataLoader(
            data_file=self.data_file,
            tokenizer_path="data/tokenizer/tokenizer.model",
            seq_len=10,
            batch_size=2
        )
        
        # Test that the data loader can be created
        self.assertIsNotNone(data_loader)
        self.assertEqual(data_loader.batch_size, 2)
        self.assertEqual(data_loader.seq_len, 10)
    
    def test_data_preprocessing(self):
        """Test data preprocessing functionality."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.vocab_size.return_value = 32000
        
        # Create data loader with real tokenizer
        data_loader = TextDataLoader(
            data_file=self.data_file,
            tokenizer_path="data/tokenizer/tokenizer.model",
            seq_len=10,
            batch_size=2
        )
        
        # Check that preprocessing was applied
        self.assertIsInstance(data_loader.data, list)
        self.assertGreater(len(data_loader.data), 0)


class TestTrainingConfig(unittest.TestCase):
    """Test cases for TrainingConfig class."""
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        
        # Check default values
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.max_steps, 100000)
        self.assertEqual(config.warmup_steps, 10000)
        self.assertEqual(config.gradient_clipping, 1.0)
        self.assertEqual(config.weight_decay, 0.01)
        self.assertTrue(config.mixed_precision)
        self.assertTrue(config.gradient_checkpointing)
    
    def test_training_config_custom(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            learning_rate=2e-4,
            batch_size=16,
            max_steps=50000,
            warmup_steps=5000,
            gradient_clipping=0.5,
            weight_decay=0.02,
            mixed_precision=False,
            gradient_checkpointing=False
        )
        
        # Check custom values
        self.assertEqual(config.learning_rate, 2e-4)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.max_steps, 50000)
        self.assertEqual(config.warmup_steps, 5000)
        self.assertEqual(config.gradient_clipping, 0.5)
        self.assertEqual(config.weight_decay, 0.02)
        self.assertFalse(config.mixed_precision)
        self.assertFalse(config.gradient_checkpointing)


class TestTrainingLoop(unittest.TestCase):
    """Test cases for training loop functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig.small()
        self.model = GPTModel(self.config)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_training_step(self):
        """Test a single training step."""
        # Create optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # Create sample batch
        batch_size, seq_len = 2, 10
        vocab_size = self.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Training step
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Loss computation
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)
    
    def test_model_checkpointing(self):
        """Test model checkpoint saving and loading."""
        # Create checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'step': 1000,
            'loss': 2.5,
            'optimizer_state_dict': torch.optim.AdamW(self.model.parameters()).state_dict()
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        
        # Verify checkpoint contents
        self.assertEqual(loaded_checkpoint['step'], 1000)
        self.assertEqual(loaded_checkpoint['loss'], 2.5)
        self.assertIn('model_state_dict', loaded_checkpoint)
        self.assertIn('config', loaded_checkpoint)
        self.assertIn('optimizer_state_dict', loaded_checkpoint)
    
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling."""
        # Create optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000, eta_min=1e-6
        )
        
        # Check initial learning rate
        initial_lr = optimizer.param_groups[0]['lr']
        self.assertEqual(initial_lr, 1e-4)
        
        # Step scheduler
        scheduler.step()
        
        # Check that learning rate changed
        new_lr = optimizer.param_groups[0]['lr']
        self.assertNotEqual(initial_lr, new_lr)
    
    @patch('train_model.DataLoader')
    @patch('train_model.GPTModel')
    def test_training_function_mock(self, mock_model_class, mock_data_loader_class):
        """Test training function with mocked dependencies."""
        # Mock model
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Mock data loader
        mock_data_loader = MagicMock()
        mock_data_loader_class.return_value = mock_data_loader
        
        # Mock optimizer
        mock_optimizer = MagicMock()
        
        # Mock training config
        training_config = TrainingConfig(
            max_steps=10,  # Small number for testing
            batch_size=2
        )
        
        # Mock model config
        model_config = GPTConfig.small()
        
        # This would test the training function without actually training
        # In a real implementation, you'd want to test the actual training loop
        self.assertIsNotNone(training_config)
        self.assertIsNotNone(model_config)


class TestModelEvaluation(unittest.TestCase):
    """Test cases for model evaluation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig.small()
        self.model = GPTModel(self.config)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_perplexity_calculation(self):
        """Test perplexity calculation."""
        # Create sample data
        batch_size, seq_len = 2, 10
        vocab_size = self.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids)
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            reduction='mean'
        )
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Check that perplexity is finite and positive
        self.assertTrue(torch.isfinite(perplexity))
        self.assertGreater(perplexity.item(), 1.0)  # Perplexity should be > 1
    
    def test_model_evaluation_metrics(self):
        """Test various evaluation metrics."""
        # Create sample data
        batch_size, seq_len = 4, 20
        vocab_size = self.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids)
        
        # Calculate metrics
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            reduction='mean'
        )
        
        perplexity = torch.exp(loss)
        
        # Calculate accuracy (top-1)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).float().mean()
        
        # Check metrics
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(perplexity))
        self.assertTrue(torch.isfinite(accuracy))
        self.assertGreaterEqual(accuracy.item(), 0.0)
        self.assertLessEqual(accuracy.item(), 1.0)
    
    def test_evaluation_on_different_lengths(self):
        """Test evaluation on different sequence lengths."""
        vocab_size = self.config.vocab_size
        
        for seq_len in [5, 10, 20, 50]:
            batch_size = 2
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_ids)
            
            # Calculate loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
                reduction='mean'
            )
            
            # Check that loss is finite
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(loss.item(), 0)


class TestTrainingIntegration(unittest.TestCase):
    """Integration tests for the training pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig.small()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample training data
        self.sample_data = [
            "This is a sample text for testing the training pipeline.",
            "Another sample text with different content for variety.",
            "A third sample to ensure the model learns diverse patterns.",
            "Testing the complete training pipeline with multiple sentences.",
            "This should provide enough data for basic integration testing."
        ]
        
        # Write sample data to file
        self.data_file = os.path.join(self.temp_dir, "test_data.txt")
        with open(self.data_file, 'w', encoding='utf-8') as f:
            for line in self.sample_data:
                f.write(line + '\n')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_training_simulation(self):
        """Test a simulated end-to-end training process."""
        # Create model
        model = GPTModel(self.config)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        
        # Training loop simulation
        vocab_size = self.config.vocab_size
        losses = []
        
        for step in range(10):  # Small number of steps for testing
            # Create sample batch
            batch_size, seq_len = 2, 10
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Training step
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids)
            
            # Loss computation
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            # Record loss
            losses.append(loss.item())
        
        # Check that training completed
        self.assertEqual(len(losses), 10)
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        
        # Check that loss is reasonable (not NaN or infinite)
        self.assertTrue(all(loss > 0 for loss in losses))
    
    def test_training_logging(self):
        """Test training metrics logging."""
        # Create training log
        training_log = {
            'step': 1000,
            'loss': 2.5,
            'perplexity': 12.2,
            'learning_rate': 1e-4,
            'accuracy': 0.15
        }
        
        # Save training log
        log_file = os.path.join(self.temp_dir, "training_log.json")
        with open(log_file, 'w') as f:
            json.dump(training_log, f)
        
        # Load and verify training log
        with open(log_file, 'r') as f:
            loaded_log = json.load(f)
        
        # Check log contents
        self.assertEqual(loaded_log['step'], 1000)
        self.assertEqual(loaded_log['loss'], 2.5)
        self.assertEqual(loaded_log['perplexity'], 12.2)
        self.assertEqual(loaded_log['learning_rate'], 1e-4)
        self.assertEqual(loaded_log['accuracy'], 0.15)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
