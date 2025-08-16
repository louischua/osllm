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
OpenLLM Inference Server

This script implements the REST API server for OpenLLM model inference
as specified in Step 6 of the training pipeline.

Features:
- FastAPI-based REST API
- Support for multiple model formats (PyTorch, Hugging Face, ONNX)
- Text generation with configurable parameters
- Health checks and metrics
- Production-ready deployment

Usage:
    python core/src/inference_server.py \
        --model_path exports/huggingface/ \
        --host 0.0.0.0 \
        --port 8000 \
        --max_length 512

API Endpoints:
    POST /generate - Generate text from prompt
    GET /health - Health check
    GET /info - Model information

Author: Louis Chua Bean Chong
License: GPLv3
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn

# FastAPI imports (open source)
try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("Install FastAPI: pip install fastapi uvicorn[standard]")

import os

# Import our modules
import sys

import numpy as np
import sentencepiece as smp
import torch

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import create_model


class TextGenerationConfig(BaseModel):
    """Configuration for text generation parameters."""

    max_new_tokens: int = Field(
        256, description="Maximum number of tokens to generate", ge=1, le=2048
    )
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.1, le=2.0)
    top_k: Optional[int] = Field(40, description="Top-k sampling parameter", ge=1, le=1000)
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling parameter", ge=0.1, le=1.0)
    num_return_sequences: int = Field(1, description="Number of sequences to generate", ge=1, le=5)
    stop_sequences: Optional[List[str]] = Field(
        None, description="Stop generation at these sequences"
    )


class GenerationRequest(BaseModel):
    """Request model for text generation."""

    prompt: str = Field(..., description="Input text prompt")
    max_length: int = Field(256, description="Maximum generation length", ge=1, le=2048)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.1, le=2.0)
    top_k: Optional[int] = Field(40, description="Top-k sampling parameter", ge=1, le=1000)
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling parameter", ge=0.1, le=1.0)
    num_return_sequences: int = Field(1, description="Number of sequences to generate", ge=1, le=5)
    stop_sequences: Optional[List[str]] = Field(
        None, description="Stop generation at these sequences"
    )


class GenerationResponse(BaseModel):
    """Response model for text generation."""

    generated_text: List[str] = Field(..., description="Generated text sequences")
    prompt: str = Field(..., description="Original prompt")
    generation_time: float = Field(..., description="Generation time in seconds")
    parameters: Dict[str, Any] = Field(..., description="Generation parameters used")


class ModelInfo(BaseModel):
    """Model information response."""

    model_name: str
    model_size: str
    parameters: int
    vocab_size: int
    max_length: int
    format: str
    loaded_at: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    uptime_seconds: float
    total_requests: int


class OpenLLMInference:
    """
    OpenLLM model inference engine.

    Supports multiple model formats and provides text generation capabilities.
    """

    def __init__(self, model_path: str, model_format: str = "auto"):
        """
        Initialize inference engine.

        Args:
            model_path: Path to exported model directory
            model_format: Model format (pytorch, huggingface, onnx, auto)
        """
        self.model_path = Path(model_path)
        self.model_format = model_format
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        self._load_model()

        # Statistics
        self.loaded_at = time.time()
        self.total_requests = 0

        print("🚀 OpenLLM Inference Engine initialized")
        print(f"  Model: {self.config.get('model_name', 'Unknown')}")
        print(f"  Format: {self.detected_format}")
        print(f"  Device: {self.device}")

    def _detect_format(self) -> str:
        """Auto-detect model format from directory contents."""
        if (self.model_path / "model.pt").exists():
            return "pytorch"
        elif (self.model_path / "pytorch_model.bin").exists():
            return "huggingface"
        elif (self.model_path / "model.onnx").exists():
            return "onnx"
        else:
            raise ValueError(f"Could not detect model format in {self.model_path}")

    def _load_model(self):
        """Load model based on detected format."""
        if self.model_format == "auto":
            self.detected_format = self._detect_format()
        else:
            self.detected_format = self.model_format

        print(f"📂 Loading {self.detected_format} model from {self.model_path}")

        if self.detected_format == "pytorch":
            self._load_pytorch_model()
        elif self.detected_format == "huggingface":
            self._load_huggingface_model()
        elif self.detected_format == "onnx":
            self._load_onnx_model()
        else:
            raise ValueError(f"Unsupported format: {self.detected_format}")

        # Load tokenizer
        self._load_tokenizer()

        print("✅ Model loaded successfully")

    def _load_pytorch_model(self):
        """Load PyTorch format model."""
        # Load config
        with open(self.model_path / "config.json", "r") as f:
            config_data = json.load(f)

        self.config = config_data["model_config"]

        # Load model
        checkpoint = torch.load(self.model_path / "model.pt", map_location=self.device)

        # Determine model size
        n_layer = self.config.get("n_layer", 12)
        if n_layer <= 6:
            model_size = "small"
        elif n_layer <= 12:
            model_size = "medium"
        else:
            model_size = "large"

        # Create model
        self.model = create_model(model_size)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _load_huggingface_model(self):
        """Load Hugging Face format model."""
        # Load config
        with open(self.model_path / "config.json", "r") as f:
            self.config = json.load(f)

        # Load model weights
        state_dict = torch.load(self.model_path / "pytorch_model.bin", map_location=self.device)

        # Determine model size
        n_layer = self.config.get("n_layer", 12)
        if n_layer <= 6:
            model_size = "small"
        elif n_layer <= 12:
            model_size = "medium"
        else:
            model_size = "large"

        # Create model
        self.model = create_model(model_size)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _load_onnx_model(self):
        """Load ONNX format model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX inference requires: pip install onnxruntime")

        # Load metadata
        with open(self.model_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.config = metadata["model_config"]

        # Create ONNX session
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )
        self.onnx_session = ort.InferenceSession(
            str(self.model_path / "model.onnx"), providers=providers
        )

        # ONNX models don't need device management
        self.device = "onnx"

    def _load_tokenizer(self):
        """Load tokenizer."""
        tokenizer_path = self.model_path / "tokenizer.model"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        self.tokenizer = smp.SentencePieceProcessor()
        self.tokenizer.load(str(tokenizer_path))

    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        num_return_sequences: int = 1,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
            stop_sequences: Stop generation at these sequences

        Returns:
            List of generated text sequences
        """
        self.total_requests += 1

        if self.detected_format == "onnx":
            return self._generate_onnx(
                prompt, max_length, temperature, top_k, num_return_sequences, stop_sequences
            )
        else:
            return self._generate_pytorch(
                prompt, max_length, temperature, top_k, top_p, num_return_sequences, stop_sequences
            )

    def _generate_pytorch(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        num_return_sequences: int,
        stop_sequences: Optional[List[str]],
    ) -> List[str]:
        """Generate using PyTorch model."""
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(
            [input_ids] * num_return_sequences, dtype=torch.long, device=self.device
        )

        # Generate
        with torch.no_grad():
            outputs = []
            for _ in range(num_return_sequences):
                # Use model's generate method if available
                if hasattr(self.model, "generate"):
                    output = self.model.generate(
                        input_tensor[:1],  # Single sequence
                        max_new_tokens=max_length,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    generated_ids = output[0].tolist()
                    generated_text = self.tokenizer.decode(generated_ids[len(input_ids) :])
                else:
                    # Fallback simple generation
                    generated_text = self._simple_generate(
                        input_tensor[:1], max_length, temperature
                    )

                # Apply stop sequences
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in generated_text:
                            generated_text = generated_text.split(stop_seq)[0]
                            break

                outputs.append(generated_text)

        return outputs

    def _generate_onnx(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        top_k: Optional[int],
        num_return_sequences: int,
        stop_sequences: Optional[List[str]],
    ) -> List[str]:
        """Generate using ONNX model."""
        outputs = []

        for _ in range(num_return_sequences):
            # Tokenize prompt
            tokens = self.tokenizer.encode(prompt)
            generated = tokens.copy()

            # Simple autoregressive generation
            for _ in range(max_length):
                if len(generated) >= 512:  # Max sequence length for ONNX
                    break

                # Prepare input (last 64 tokens to fit ONNX model)
                current_input = np.array([generated[-64:]], dtype=np.int64)

                # Run inference
                logits = self.onnx_session.run(None, {"input_ids": current_input})[0]
                next_token_logits = logits[0, -1, :]

                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))

                    # Apply top-k if specified
                    if top_k:
                        top_indices = np.argpartition(probs, -top_k)[-top_k:]
                        probs_filtered = np.zeros_like(probs)
                        probs_filtered[top_indices] = probs[top_indices]
                        probs = probs_filtered / np.sum(probs_filtered)

                    next_token = np.random.choice(len(probs), p=probs)
                else:
                    next_token = np.argmax(next_token_logits)

                generated.append(int(next_token))

            # Decode generated text
            generated_text = self.tokenizer.decode(generated[len(tokens) :])

            # Apply stop sequences
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                        break

            outputs.append(generated_text)

        return outputs

    def _simple_generate(
        self, input_tensor: torch.Tensor, max_length: int, temperature: float
    ) -> str:
        """Simple fallback generation method."""
        generated = input_tensor[0].tolist()

        for _ in range(max_length):
            if len(generated) >= self.config.get("block_size", 1024):
                break

            # Forward pass
            current_input = torch.tensor([generated], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits, _ = self.model(current_input)

            # Get next token logits and apply temperature
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

        # Decode only the generated part
        original_length = input_tensor.size(1)
        generated_tokens = generated[original_length:]
        return self.tokenizer.decode(generated_tokens)

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.config.get("model_name", "OpenLLM"),
            "model_size": self.config.get("model_size", "unknown"),
            "parameters": self.config.get("n_embd", 0)
            * self.config.get("n_layer", 0),  # Approximate
            "vocab_size": self.config.get("vocab_size", self.tokenizer.vocab_size()),
            "max_length": self.config.get("block_size", 1024),
            "format": self.detected_format,
            "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.loaded_at)),
        }

    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "uptime_seconds": time.time() - self.loaded_at,
            "total_requests": self.total_requests,
        }


# Global inference engine
inference_engine: Optional[OpenLLMInference] = None

# FastAPI app
app = FastAPI(
    title="OpenLLM Inference API",
    description="REST API for OpenLLM text generation",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup."""
    print("🚀 Starting OpenLLM Inference Server...")


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate text from prompt."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Generate text
        generated_texts = inference_engine.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            num_return_sequences=request.num_return_sequences,
            stop_sequences=request.stop_sequences,
        )

        generation_time = time.time() - start_time

        return GenerationResponse(
            generated_text=generated_texts,
            prompt=request.prompt,
            generation_time=generation_time,
            parameters={
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p,
                "num_return_sequences": request.num_return_sequences,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = inference_engine.get_info()
    return ModelInfo(**info)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if inference_engine is None:
        return HealthResponse(
            status="unhealthy", model_loaded=False, uptime_seconds=0.0, total_requests=0
        )

    health = inference_engine.get_health()
    return HealthResponse(**health)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "OpenLLM Inference API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "info": "/info",
    }


def main():
    """Main server function."""
    parser = argparse.ArgumentParser(
        description="OpenLLM Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with Hugging Face model
  python core/src/inference_server.py \\
    --model_path exports/huggingface/ \\
    --host 0.0.0.0 \\
    --port 8000

  # Start server with ONNX model
  python core/src/inference_server.py \\
    --model_path exports/onnx/ \\
    --format onnx \\
    --port 8001
        """,
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to exported model directory",
    )

    parser.add_argument(
        "--format",
        choices=["pytorch", "huggingface", "onnx", "auto"],
        default="auto",
        help="Model format (default: auto-detect)",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum generation length (default: 512)",
    )

    args = parser.parse_args()

    # Initialize inference engine
    global inference_engine
    inference_engine = OpenLLMInference(args.model_path, args.format)

    # Start server
    print(f"🚀 Starting server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


def load_model(model_path: str, model_format: str = "auto"):
    """
    Load model for testing purposes.

    This function is used by tests to load models without starting the full server.

    Args:
        model_path: Path to exported model directory
        model_format: Model format (pytorch, huggingface, onnx, auto)

    Returns:
        OpenLLMInference: Initialized inference engine
    """
    return OpenLLMInference(model_path, model_format)


if __name__ == "__main__":
    main()
