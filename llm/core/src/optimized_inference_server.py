#!/usr/bin/env python3
"""
Optimized OpenLLM Inference Server

This module provides an optimized inference server with:
- Model caching and memory management
- Request batching for improved throughput
- Response streaming for real-time generation
- Performance monitoring and metrics
- Load balancing and concurrent processing

Author: Louis Chua Bean Chong
License: GPLv3
"""

import asyncio
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any, AsyncGenerator
from collections import deque
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging
import psutil
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPTConfig, GPTModel
from quantization import QuantizedModel, quantize_model_dynamic


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedInferenceEngine:
    """
    Optimized inference engine with caching and batching.

    This engine provides high-performance inference with:
    - Model caching and memory management
    - Request batching for improved throughput
    - Quantization support for reduced memory usage
    - Performance monitoring and metrics
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        use_quantization: bool = True,
        cache_size: int = 1000,
        max_batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        Initialize optimized inference engine.

        Args:
            model_path: Path to the model
            device: Device to use ("auto", "cpu", "cuda")
            use_quantization: Whether to use quantization
            cache_size: Size of response cache
            max_batch_size: Maximum batch size for processing
            num_workers: Number of worker threads
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.use_quantization = use_quantization
        self.cache_size = cache_size
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.quantized_model = None
        self.response_cache = {}
        self.request_queue = deque()
        self.processing_lock = threading.Lock()

        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_generation_time": 0.0,
            "total_generation_time": 0.0,
            "requests_per_second": 0.0,
        }

        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # Load model
        self._load_model()

        logger.info(f"OptimizedInferenceEngine initialized on {self.device}")

    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)

    def _load_model(self):
        """Load and optimize the model."""
        try:
            logger.info(f"Loading model from {self.model_path}")

            # Load model configuration
            config_path = Path(self.model_path) / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                config = GPTConfig(**config_data)
            else:
                # Use default config
                config = GPTConfig.small()

            # Create model
            self.model = GPTModel(config, use_checkpoint=False)  # No checkpointing for inference

            # Load model weights
            model_path = Path(self.model_path) / "pytorch_model.bin"
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Model weights loaded successfully")
            else:
                logger.warning("No model weights found, using initialized weights")

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            # Apply quantization if requested
            if self.use_quantization and self.device.type == "cpu":
                logger.info("Applying dynamic quantization")
                self.quantized_model = QuantizedModel(self.model)
                self.quantized_model.quantize_dynamic()
                logger.info("Quantization completed")

            # Load tokenizer
            tokenizer_path = Path(self.model_path) / "tokenizer.model"
            if tokenizer_path.exists():
                import sentencepiece as spm

                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.load(str(tokenizer_path))
                logger.info("Tokenizer loaded successfully")
            else:
                logger.warning("No tokenizer found")

            logger.info("Model loading completed")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for request."""
        # Create a hash of the prompt and parameters
        import hashlib

        key_data = f"{prompt}_{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[List[str]]:
        """Check if response is cached."""
        if cache_key in self.response_cache:
            self.metrics["cache_hits"] += 1
            return self.response_cache[cache_key]
        else:
            self.metrics["cache_misses"] += 1
            return None

    def _update_cache(self, cache_key: str, response: List[str]):
        """Update response cache."""
        if len(self.response_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]

        self.response_cache[cache_key] = response

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text using the loaded tokenizer."""
        if self.tokenizer is None:
            # Fallback to simple tokenization
            return torch.tensor([ord(c) % 1000 for c in text], dtype=torch.long)

        tokens = self.tokenizer.encode_as_ids(text)
        return torch.tensor(tokens, dtype=torch.long)

    def _detokenize(self, tokens: torch.Tensor) -> str:
        """Detokenize tokens to text."""
        if self.tokenizer is None:
            # Fallback to simple detokenization
            return "".join([chr(t % 1000) for t in tokens.tolist()])

        return self.tokenizer.decode(tokens.tolist())

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
        Generate text with optimizations.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
            stop_sequences: Stop generation at these sequences

        Returns:
            List of generated texts
        """
        start_time = time.time()

        # Check cache first
        cache_key = self._get_cache_key(
            prompt, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p
        )
        cached_response = self._check_cache(cache_key)
        if cached_response:
            return cached_response

        # Tokenize input
        input_tokens = self._tokenize(prompt)
        input_tokens = input_tokens.unsqueeze(0).to(self.device)  # Add batch dimension

        # Generate text
        with torch.no_grad():
            if self.quantized_model and self.quantized_model.is_quantized:
                # Use quantized model
                generated_tokens = self.quantized_model.quantized_model.generate(
                    input_tokens,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=True,
                )
            else:
                # Use regular model
                generated_tokens = self.model.generate(
                    input_tokens,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=True,
                )

        # Detokenize
        generated_texts = []
        for i in range(num_return_sequences):
            # Extract generated part (remove input)
            generated_part = generated_tokens[0, len(input_tokens[0]) :]
            text = self._detokenize(generated_part)

            # Apply stop sequences
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in text:
                        text = text[: text.find(stop_seq)]
                        break

            generated_texts.append(text)

        # Update cache
        self._update_cache(cache_key, generated_texts)

        # Update metrics
        generation_time = time.time() - start_time
        self.metrics["total_requests"] += 1
        self.metrics["total_generation_time"] += generation_time
        self.metrics["avg_generation_time"] = (
            self.metrics["total_generation_time"] / self.metrics["total_requests"]
        )

        return generated_texts

    async def generate_async(
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
        Asynchronous text generation.

        Args:
            Same as generate()

        Returns:
            List of generated texts
        """
        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.generate,
            prompt,
            max_length,
            temperature,
            top_k,
            top_p,
            num_return_sequences,
            stop_sequences,
        )

    async def generate_stream(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream generated text token by token.

        Args:
            Same as generate()

        Yields:
            Generated text tokens
        """
        # Tokenize input
        input_tokens = self._tokenize(prompt)
        input_tokens = input_tokens.unsqueeze(0).to(self.device)

        # Generate tokens one by one
        current_tokens = input_tokens.clone()

        with torch.no_grad():
            for _ in range(max_length):
                # Get next token
                if self.quantized_model and self.quantized_model.is_quantized:
                    logits = self.quantized_model.quantized_model(current_tokens)
                else:
                    logits = self.model(current_tokens)

                # Sample next token
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Add to sequence
                current_tokens = torch.cat([current_tokens, next_token], dim=1)

                # Convert token to text
                token_text = self._detokenize(next_token[0])
                yield token_text

                # Check for stop sequences
                if stop_sequences:
                    full_text = self._detokenize(current_tokens[0, len(input_tokens[0]) :])
                    for stop_seq in stop_sequences:
                        if stop_seq in full_text:
                            return

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        memory_usage = psutil.virtual_memory().percent

        return {
            **self.metrics,
            "memory_usage_percent": memory_usage,
            "cache_size": len(self.response_cache),
            "max_cache_size": self.cache_size,
            "cache_hit_rate": (
                self.metrics["cache_hits"]
                / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
                if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
                else 0
            ),
            "device": str(self.device),
            "quantization_enabled": self.quantized_model is not None,
        }

    def cleanup(self):
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)

        # Clear cache
        self.response_cache.clear()

        logger.info("Inference engine cleaned up")


# Request/Response models
class GenerationRequest(BaseModel):
    """Request model for text generation."""

    prompt: str = Field(..., description="Input text prompt")
    max_length: int = Field(256, description="Maximum generation length", ge=1, le=2048)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_k: Optional[int] = Field(40, description="Top-k sampling parameter", ge=1, le=1000)
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling parameter", ge=0.1, le=1.0)
    num_return_sequences: int = Field(1, description="Number of sequences to generate", ge=1, le=5)
    stop_sequences: Optional[List[str]] = Field(
        None, description="Stop generation at these sequences"
    )


class GenerationResponse(BaseModel):
    """Response model for text generation."""

    generated_text: List[str]
    prompt: str
    generation_time: float
    parameters: Dict[str, Any]


class BatchGenerationRequest(BaseModel):
    """Request model for batch text generation."""

    prompts: List[str] = Field(..., description="List of input prompts")
    max_length: int = Field(256, description="Maximum generation length", ge=1, le=2048)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_k: Optional[int] = Field(40, description="Top-k sampling parameter", ge=1, le=1000)
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling parameter", ge=0.1, le=1.0)
    stop_sequences: Optional[List[str]] = Field(
        None, description="Stop generation at these sequences"
    )


class BatchGenerationResponse(BaseModel):
    """Response model for batch text generation."""

    generated_texts: List[List[str]]
    prompts: List[str]
    generation_time: float
    parameters: Dict[str, Any]


# Global inference engine
inference_engine: Optional[OptimizedInferenceEngine] = None

# FastAPI app
app = FastAPI(
    title="Optimized OpenLLM Inference API",
    description="High-performance REST API for OpenLLM text generation",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup."""
    logger.info("🚀 Starting Optimized OpenLLM Inference Server...")
    global inference_engine
    if inference_engine is None:
        logger.warning("No model loaded - server will return 503 for generation requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global inference_engine
    if inference_engine:
        inference_engine.cleanup()
    logger.info("Server shutdown complete")


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text from prompt with optimizations."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Generate text asynchronously
        generated_texts = await inference_engine.generate_async(
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
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/stream")
async def generate_text_stream(request: GenerationRequest):
    """Generate text with streaming response."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def generate_stream():
        try:
            async for token in inference_engine.generate_stream(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stop_sequences=request.stop_sequences,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/generate/batch", response_model=BatchGenerationResponse)
async def generate_text_batch(request: BatchGenerationRequest):
    """Generate text for multiple prompts in batch."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Process prompts in parallel
        tasks = []
        for prompt in request.prompts:
            task = inference_engine.generate_async(
                prompt=prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                num_return_sequences=1,
                stop_sequences=request.stop_sequences,
            )
            tasks.append(task)

        # Wait for all tasks to complete
        generated_texts = await asyncio.gather(*tasks)

        generation_time = time.time() - start_time

        return BatchGenerationResponse(
            generated_texts=generated_texts,
            prompts=request.prompts,
            generation_time=generation_time,
            parameters={
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p,
                "num_prompts": len(request.prompts),
            },
        )

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global inference_engine

    if inference_engine is None:
        return {"status": "unhealthy", "message": "Model not loaded"}

    try:
        # Quick generation test
        test_result = await inference_engine.generate_async(
            prompt="Hello", max_length=5, temperature=0.7
        )

        return {"status": "healthy", "model_loaded": True, "test_generation": len(test_result) > 0}

    except Exception as e:
        return {"status": "unhealthy", "message": f"Generation test failed: {str(e)}"}


@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    global inference_engine

    if inference_engine is None:
        return {"error": "Model not loaded"}

    return inference_engine.get_metrics()


@app.get("/info")
async def get_model_info():
    """Get model information."""
    global inference_engine

    if inference_engine is None:
        return {"error": "Model not loaded"}

    model = inference_engine.model
    if model is None:
        return {"error": "Model not available"}

    return {
        "model_name": model.config.model_name,
        "vocab_size": model.config.vocab_size,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
        "block_size": model.config.block_size,
        "parameters": model.get_num_params(),
        "device": str(inference_engine.device),
        "quantization_enabled": inference_engine.quantized_model is not None,
        "cache_size": len(inference_engine.response_cache),
        "max_cache_size": inference_engine.cache_size,
    }


def create_optimized_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "auto",
    use_quantization: bool = True,
    cache_size: int = 1000,
    max_batch_size: int = 32,
    num_workers: int = 4,
) -> FastAPI:
    """
    Create an optimized inference server.

    Args:
        model_path: Path to the model
        host: Server host
        port: Server port
        device: Device to use
        use_quantization: Whether to use quantization
        cache_size: Size of response cache
        max_batch_size: Maximum batch size
        num_workers: Number of worker threads

    Returns:
        FastAPI app instance
    """
    global inference_engine

    # Initialize inference engine
    inference_engine = OptimizedInferenceEngine(
        model_path=model_path,
        device=device,
        use_quantization=use_quantization,
        cache_size=cache_size,
        max_batch_size=max_batch_size,
        num_workers=num_workers,
    )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimized OpenLLM Inference Server")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--use_quantization", action="store_true", help="Use quantization")
    parser.add_argument("--cache_size", type=int, default=1000, help="Cache size")
    parser.add_argument("--max_batch_size", type=int, default=32, help="Max batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    args = parser.parse_args()

    # Create server
    app = create_optimized_server(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        device=args.device,
        use_quantization=args.use_quantization,
        cache_size=args.cache_size,
        max_batch_size=args.max_batch_size,
        num_workers=args.num_workers,
    )

    # Run server
    uvicorn.run(app, host=args.host, port=args.port)
