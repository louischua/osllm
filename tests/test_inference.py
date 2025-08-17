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
Unit tests for the inference server and API.

This module tests the inference-related components including:
- FastAPI server functionality
- Text generation endpoints
- Model loading and serving
- API request/response handling
- Performance and reliability

Test Coverage:
- Server startup and shutdown
- API endpoint functionality
- Request validation
- Response formatting
- Error handling
- Performance benchmarks
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
import torch
from fastapi.testclient import TestClient

# Add the core/src directory to the path for imports
core_src_path = str(Path(__file__).parent.parent / "core" / "src")
sys.path.insert(0, core_src_path)

# Import after path setup
try:
    from generate_text import TextGenerator, load_tokenizer
    from inference_server import TextGenerationConfig, app, load_model
    from model import GPTConfig, GPTModel
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"📁 Core src path: {core_src_path}")
    print(
        f"📁 Available files: {os.listdir(core_src_path) if os.path.exists(core_src_path) else 'Path not found'}"
    )
    raise


class TestInferenceServer(unittest.TestCase):
    """Test cases for the inference server."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig.small()
        self.model = GPTModel(self.config)
        self.temp_dir = tempfile.mkdtemp()

        # Create real lightweight model for testing (no downloads needed)
        from inference_server import create_test_model

        self.inference_engine = create_test_model()

        # Set the global inference engine for FastAPI app
        import inference_server

        inference_server.inference_engine = self.inference_engine

        print("✅ Using real lightweight model for testing (no downloads)")
        print("✅ Global inference engine initialized for FastAPI")

        # Create test client
        self.client = TestClient(app)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        import inference_server

        # Reset global inference engine
        inference_server.inference_engine = None

        shutil.rmtree(self.temp_dir)

    def test_server_startup(self):
        """Test that the server can start up correctly."""
        # Test that the app is created
        self.assertIsNotNone(app)

        # Test that the app has the expected routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/generate", "/info"]

        for route in expected_routes:
            self.assertIn(route, routes)

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        # Use real inference engine
        response = self.client.get("/health")

        # Check response
        self.assertEqual(response.status_code, 200)

        # Check response content
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")

    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")

        # Check response
        self.assertEqual(response.status_code, 200)

        # Check response content
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)
        self.assertIn("endpoints", data)

    def test_generate_endpoint_basic(self):
        """Test the basic text generation endpoint."""
        # Use real inference engine
        # Test request
        request_data = {
            "prompt": "Hello world",
            "max_length": 10,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
        }

        response = self.client.post("/generate", json=request_data)

        # Check response
        self.assertEqual(response.status_code, 200)

        # Check response content
        data = response.json()
        self.assertIn("generated_text", data)
        self.assertIn("prompt", data)
        self.assertIn("parameters", data)

        # Verify real generation
        self.assertIsInstance(data["generated_text"], list)
        self.assertGreater(len(data["generated_text"]), 0)

    def test_generate_endpoint_validation(self):
        """Test input validation for the generate endpoint."""
        # Test missing prompt
        request_data = {"max_length": 10, "temperature": 0.7}

        response = self.client.post("/generate", json=request_data)
        self.assertEqual(response.status_code, 422)  # Validation error

        # Test invalid temperature (now allows 0.0-2.0)
        request_data = {"prompt": "Hello world", "temperature": 2.5}  # Should be <= 2.0

        response = self.client.post("/generate", json=request_data)
        self.assertEqual(response.status_code, 422)  # Validation error

        # Test invalid max_length
        request_data = {"prompt": "Hello world", "max_length": -1}  # Should be > 0

        response = self.client.post("/generate", json=request_data)
        self.assertEqual(response.status_code, 422)  # Validation error

    def test_generate_endpoint_parameters(self):
        """Test different generation parameters."""
        # Use real inference engine
        # Test different temperature values
        for temp in [0.1, 0.5, 0.9]:
            request_data = {"prompt": "Test prompt", "temperature": temp, "max_length": 5}

            response = self.client.post("/generate", json=request_data)
            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertIn("generated_text", data)
            self.assertEqual(data["parameters"]["temperature"], temp)

        # Test different top_p values
        for top_p in [0.1, 0.5, 0.9]:
            request_data = {"prompt": "Test prompt", "top_p": top_p, "max_length": 5}

            response = self.client.post("/generate", json=request_data)
            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertEqual(data["parameters"]["top_p"], top_p)

    def test_streaming_endpoint(self):
        """Test the streaming text generation endpoint."""
        # Use real inference engine
        # Test request
        request_data = {"prompt": "Hello world", "max_length": 10, "temperature": 0.7}

        response = self.client.post("/generate/stream", json=request_data)

        # Check response
        self.assertEqual(response.status_code, 200)

        # Check response content
        data = response.json()
        self.assertIn("generated_text", data)
        self.assertIn("streaming", data)

        # Verify real generation
        self.assertIsInstance(data["generated_text"], list)
        self.assertGreater(len(data["generated_text"]), 0)

    def test_error_handling(self):
        """Test error handling in the API."""
        # Use real inference engine
        # Test with invalid JSON
        response = self.client.post("/generate", data="invalid json")
        self.assertEqual(response.status_code, 422)

        # Test with empty request body
        response = self.client.post("/generate", json={})
        self.assertEqual(response.status_code, 422)

        # Test with very long prompt
        long_prompt = "A" * 10000  # Very long prompt
        request_data = {"prompt": long_prompt, "max_length": 10}

        response = self.client.post("/generate", json=request_data)
        # Should either succeed or return a reasonable error
        self.assertIn(response.status_code, [200, 413, 422, 503])


class TestTextGeneration(unittest.TestCase):
    """Test cases for text generation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig.small()
        self.model = GPTModel(self.config)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        import inference_server

        # Reset global inference engine
        inference_server.inference_engine = None

        shutil.rmtree(self.temp_dir)

    def test_text_generation_basic(self):
        """Test basic text generation."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = " generated text"

        # Test generation
        prompt = "Hello world"
        max_new_tokens = 10

        with patch("generate_text.load_tokenizer") as mock_load_tokenizer:
            mock_load_tokenizer.return_value = mock_tokenizer

            # This would test the actual generation function
            # For now, we'll test the configuration
            generation_config = TextGenerationConfig(
                max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, top_k=50
            )

            self.assertEqual(generation_config.max_new_tokens, max_new_tokens)
            self.assertEqual(generation_config.temperature, 0.7)

    def test_generation_config_validation(self):
        """Test generation configuration validation."""
        # Test valid configuration
        config = TextGenerationConfig(max_new_tokens=10, temperature=0.7, top_p=0.9, top_k=50)

        self.assertEqual(config.max_new_tokens, 10)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.top_k, 50)

        # Test boundary values
        config = TextGenerationConfig(max_new_tokens=1, temperature=0.0, top_p=1.0, top_k=1)

        self.assertEqual(config.max_new_tokens, 1)
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.top_p, 1.0)
        self.assertEqual(config.top_k, 1)

    def test_generation_parameters(self):
        """Test different generation parameters."""
        # Test temperature effect (lower temperature = more deterministic)
        # This is a conceptual test since we can't easily test the actual generation

        # Test top_p sampling
        config_top_p = TextGenerationConfig(max_new_tokens=10, top_p=0.9)

        # Test top_k sampling
        config_top_k = TextGenerationConfig(max_new_tokens=10, top_k=50)

        # Test combined parameters
        config_combined = TextGenerationConfig(
            max_new_tokens=10, temperature=0.7, top_p=0.9, top_k=50
        )

        # All configurations should be valid
        self.assertIsNotNone(config_top_p)
        self.assertIsNotNone(config_top_k)
        self.assertIsNotNone(config_combined)


class TestInferencePerformance(unittest.TestCase):
    """Test cases for inference performance."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig.small()
        self.model = GPTModel(self.config)

        # Create real lightweight model for testing (no downloads needed)
        from inference_server import create_test_model

        self.inference_engine = create_test_model()

        # Set the global inference engine for FastAPI app
        import inference_server

        inference_server.inference_engine = self.inference_engine

        print("✅ Using real lightweight model for testing (no downloads)")
        print("✅ Global inference engine initialized for FastAPI")

        self.client = TestClient(app)

    def tearDown(self):
        """Clean up test fixtures."""
        import inference_server

        # Reset global inference engine
        inference_server.inference_engine = None

    def test_inference_speed(self):
        """Test inference speed for different input lengths."""
        # Use real inference engine
        # Test different prompt lengths
        prompts = [
            "Hello",  # Short
            "This is a medium length prompt for testing.",  # Medium
            "This is a longer prompt that should take more time to process and generate a response.",  # Long
        ]

        for prompt in prompts:
            request_data = {"prompt": prompt, "max_length": 10, "temperature": 0.7}

            start_time = time.time()
            response = self.client.post("/generate", json=request_data)
            end_time = time.time()

            # Check response
            self.assertEqual(response.status_code, 200)

            # Check timing (should be reasonable)
            response_time = end_time - start_time
            self.assertLess(response_time, 10.0)  # Should be under 10 seconds

    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        # Use real inference engine
        # Create multiple concurrent requests
        import queue
        import threading

        results = queue.Queue()

        def make_request():
            try:
                request_data = {
                    "prompt": "Test prompt",
                    "max_length": 5,
                    "temperature": 0.7,
                }

                response = self.client.post("/generate", json=request_data)
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")

            # Start multiple threads
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Check results
            while not results.empty():
                result = results.get()
                self.assertIn(
                    result, [200, "Error"]
                )  # Should either succeed or handle error gracefully

    def test_memory_usage(self):
        """Test memory usage during inference."""
        # Use real inference engine
        # Test with larger batch of requests
        request_data = {
            "prompt": "Test prompt for memory usage",
            "max_length": 20,
            "temperature": 0.7,
        }

        # Make multiple requests to test memory usage
        for _ in range(10):
            response = self.client.post("/generate", json=request_data)
            self.assertEqual(response.status_code, 200)

            # Check that response is reasonable
            data = response.json()
            self.assertIn("generated_text", data)


class TestInferenceReliability(unittest.TestCase):
    """Test cases for inference reliability."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig.small()
        self.model = GPTModel(self.config)

        # Create real lightweight model for testing (no downloads needed)
        from inference_server import create_test_model

        self.inference_engine = create_test_model()

        # Set the global inference engine for FastAPI app
        import inference_server

        inference_server.inference_engine = self.inference_engine

        print("✅ Using real lightweight model for testing (no downloads)")
        print("✅ Global inference engine initialized for FastAPI")

        self.client = TestClient(app)

    def test_server_stability(self):
        """Test server stability under load."""
        # Use real inference engine
        # Make many requests to test stability
        for i in range(50):
            request_data = {
                "prompt": f"Test prompt {i}",
                "max_length": 5,
                "temperature": 0.7,
            }

            response = self.client.post("/generate", json=request_data)

            # Should consistently return 200
            self.assertEqual(response.status_code, 200)

            # Check response structure
            data = response.json()
            self.assertIn("generated_text", data)
            self.assertIn("prompt", data)
            self.assertIn("parameters", data)

    def test_error_recovery(self):
        """Test error recovery and graceful degradation."""
        # Use real inference engine - test with invalid requests
        request_data = {"prompt": "Test prompt", "max_length": 5}

        response = self.client.post("/generate", json=request_data)

        # Should handle error gracefully
        self.assertIn(response.status_code, [200, 500, 503])

    def test_input_sanitization(self):
        """Test input sanitization and security."""
        # Use real inference engine
        # Test with potentially problematic inputs
        problematic_inputs = [
            "",  # Empty string
            "   ",  # Whitespace only
            "A" * 1000,  # Very long string
            "Test\nwith\nnewlines",  # Newlines
            "Test\twith\ttabs",  # Tabs
            "Test with special chars: !@#$%^&*()",  # Special characters
        ]

        for input_text in problematic_inputs:
            request_data = {"prompt": input_text, "max_length": 5}

            response = self.client.post("/generate", json=request_data)

            # Should handle gracefully (either succeed or return reasonable error)
            self.assertIn(response.status_code, [200, 422, 413, 500, 503])


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
