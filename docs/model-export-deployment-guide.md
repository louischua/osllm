# 🚀 OpenLLM Deployment Guide

This guide covers Step 6 of the training pipeline: Model Export & Deployment.

## Overview

After training your OpenLLM model, you can export it to various formats for different deployment scenarios:

- **PyTorch**: Native Python inference
- **Hugging Face**: Ecosystem compatibility
- **ONNX**: Cross-platform optimized inference
- **REST API**: Web service deployment

## Prerequisites

```bash
# Install deployment dependencies
pip install fastapi uvicorn[standard] onnx onnxruntime
```

## Export Formats

### 1. PyTorch Native Export

Best for Python-based applications and development.

```bash
python core/src/export_model.py \
  --model_dir models/small-extended-4k \
  --format pytorch \
  --output_dir exports/
```

**Output Structure:**
```
exports/pytorch/
├── model.pt              # Model weights and config
├── config.json           # Model configuration
├── tokenizer.model       # SentencePiece tokenizer
└── load_model.py         # Loading script
```

**Usage:**
```python
import torch
import sentencepiece as smp
import json

# Load model manually
checkpoint = torch.load("exports/pytorch/model.pt")
config = json.load(open("exports/pytorch/config.json"))

# Load tokenizer
tokenizer = smp.SentencePieceProcessor()
tokenizer.load("exports/pytorch/tokenizer.model")
```

### 2. Hugging Face Compatible Export

Best for integration with Hugging Face ecosystem and tools.

```bash
python core/src/export_model.py \
  --model_dir models/small-extended-4k \
  --format huggingface \
  --output_dir exports/
```

**Output Structure:**
```
exports/huggingface/
├── pytorch_model.bin      # Model weights (HF format)
├── config.json           # HF-compatible config
├── tokenizer.model       # SentencePiece tokenizer
├── tokenizer_config.json # Tokenizer configuration
├── generation_config.json # Generation parameters
└── load_hf_model.py      # Loading script
```

**Usage:**
```python
# Manual loading
import torch
import sentencepiece as smp

state_dict = torch.load("exports/huggingface/pytorch_model.bin")
tokenizer = smp.SentencePieceProcessor()
tokenizer.load("exports/huggingface/tokenizer.model")

# Future: Transformers integration
# from transformers import AutoModel, AutoTokenizer
# model = AutoModel.from_pretrained("exports/huggingface/")
# tokenizer = AutoTokenizer.from_pretrained("exports/huggingface/")
```

### 3. ONNX Export (Production Optimized)

Best for production inference across different platforms and languages.

```bash
python core/src/export_model.py \
  --model_dir models/small-extended-4k \
  --format onnx \
  --output_dir exports/ \
  --optimize_for_inference
```

**Output Structure:**
```
exports/onnx/
├── model.onnx           # ONNX model
├── metadata.json        # Model metadata
├── tokenizer.model      # SentencePiece tokenizer
└── onnx_inference.py    # Inference script
```

**Usage:**
```python
import onnxruntime as ort
import sentencepiece as smp
import numpy as np

# Load ONNX model
session = ort.InferenceSession("exports/onnx/model.onnx")

# Load tokenizer
tokenizer = smp.SentencePieceProcessor()
tokenizer.load("exports/onnx/tokenizer.model")

# Run inference
input_ids = np.array([[1, 2, 3]], dtype=np.int64)
outputs = session.run(None, {"input_ids": input_ids})
```

### 4. Export All Formats

```bash
python core/src/export_model.py \
  --model_dir models/small-extended-4k \
  --format all \
  --output_dir exports/
```

## REST API Deployment

### Local Development Server

```bash
python core/src/inference_server.py \
  --model_path exports/huggingface/ \
  --host 127.0.0.1 \
  --port 8000
```

### Production Deployment

```bash
# With multiple workers
python core/src/inference_server.py \
  --model_path exports/huggingface/ \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### API Usage

The server provides the following endpoints:

#### Generate Text
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "The future of AI is",
       "max_length": 100,
       "temperature": 0.7,
       "top_k": 40
     }'
```

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Model Info
```bash
curl http://localhost:8000/info
```

#### Interactive Documentation
Visit `http://localhost:8000/docs` for Swagger UI.

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model exports
COPY exports/ ./exports/
COPY core/src/ ./core/src/

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "core/src/inference_server.py", \
     "--model_path", "exports/huggingface/", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

### Build and Run
```bash
# Build image
docker build -t openllm-inference .

# Run container
docker run -p 8000:8000 openllm-inference
```

## Performance Optimization

### ONNX Optimizations
- **Graph Optimization**: Applies during export with `--optimize_for_inference`
- **Quantization**: Reduces model size and improves speed
- **Provider Selection**: CUDA vs CPU execution

### API Optimizations
- **Multiple Workers**: Use `--workers N` for parallel processing
- **Batch Processing**: Group multiple requests
- **Caching**: Cache tokenization results
- **Load Balancing**: Use nginx or similar for multiple instances

### Example nginx Configuration
```nginx
upstream openllm_backend {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://openllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring and Logging

### Health Monitoring
```bash
# Check server health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600.5,
  "total_requests": 150
}
```

### Custom Logging
```python
import logging

# Configure logging in inference_server.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openllm.log'),
        logging.StreamHandler()
    ]
)
```

## Security Considerations

### Production Checklist
- [ ] **Authentication**: Add API key authentication
- [ ] **Rate Limiting**: Prevent abuse
- [ ] **CORS**: Configure appropriate origins
- [ ] **HTTPS**: Use SSL/TLS in production
- [ ] **Input Validation**: Sanitize all inputs
- [ ] **Resource Limits**: Set memory and CPU limits

### Example Authentication
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    if token.credentials != "your-secret-api-key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return token

# Protect endpoints
@app.post("/generate")
async def generate_text(request: GenerationRequest, 
                       token: str = Depends(verify_token)):
    # ... generation logic
```

## Deployment Scenarios

### 1. Development/Research
- **Format**: PyTorch
- **Server**: Local FastAPI server
- **Use Case**: Testing, experimentation

### 2. Production Web Service
- **Format**: Hugging Face
- **Server**: FastAPI with multiple workers
- **Infrastructure**: Docker + nginx
- **Use Case**: Web applications, APIs

### 3. Edge/Mobile Deployment
- **Format**: ONNX (quantized)
- **Runtime**: ONNX Runtime
- **Use Case**: Mobile apps, edge devices

### 4. Batch Processing
- **Format**: PyTorch or ONNX
- **Infrastructure**: Kubernetes jobs
- **Use Case**: Large-scale text processing

## Troubleshooting

### Common Issues

**Export Fails:**
```bash
# Check model directory
ls -la models/small-extended-4k/

# Verify checkpoint exists
python -c "import torch; print(torch.load('models/small-extended-4k/best_model.pt').keys())"
```

**Server Won't Start:**
```bash
# Check dependencies
pip install fastapi uvicorn[standard]

# Test model loading
python -c "
import sys; sys.path.append('core/src')
from export_model import ModelExporter
exporter = ModelExporter('models/small-extended-4k', 'test')
print('Model loads successfully')
"
```

**Poor Performance:**
- Check GPU availability: `torch.cuda.is_available()`
- Monitor memory usage: `nvidia-smi` or Task Manager
- Reduce batch size or max_length
- Use ONNX format for faster inference

**API Errors:**
- Check logs: `tail -f openllm.log`
- Verify model format compatibility
- Test with simple requests first
- Check network connectivity

## Next Steps

After successful deployment:

1. **Monitor Performance**: Track latency, throughput, errors
2. **A/B Testing**: Compare different model versions
3. **Scaling**: Add more instances as needed
4. **Fine-tuning**: Adapt model for specific use cases
5. **Integration**: Connect to your applications

## Support

For issues and questions:
- Check the [troubleshooting section](#troubleshooting)
- Review API documentation at `/docs`
- Submit issues on GitHub
- Follow open source best practices

Remember: This is an open source project following the "Open Source First" philosophy outlined in `.cursorrules`.