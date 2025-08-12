# OpenLLM Deployment Guide

<!-- Copyright (C) 2024 Louis Chua Bean Chong -->
<!-- This file is part of OpenLLM - dual-licensed under GPLv3 and Commercial License -->

## 🚀 Production Deployment

### **🖥️ Using the Inference Server**

For production deployment and API access, use our FastAPI inference server:

#### **1. Start the Inference Server**

```bash
# Start the server with the pre-trained model
python core/src/main.py inference \
    --model-path exports/huggingface-6k/huggingface \
    --host 0.0.0.0 \
    --port 8000
```

#### **2. API Endpoints**

Once the server is running, you can access these endpoints:

**Text Generation:**
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "The history of artificial intelligence",
       "max_tokens": 50,
       "temperature": 0.7,
       "top_k": 40
     }'
```

**Model Information:**
```bash
curl "http://localhost:8000/model-info"
```

**Health Check:**
```bash
curl "http://localhost:8000/health"
```

#### **3. Python Client Example**

```python
import requests
import json

# Server configuration
SERVER_URL = "http://localhost:8000"

def generate_text(prompt, max_tokens=50, temperature=0.7, top_k=40):
    """Generate text using the OpenLLM inference server."""
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k
    }
    
    response = requests.post(f"{SERVER_URL}/generate", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result["generated_text"]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# Example usage
try:
    generated = generate_text("Machine learning algorithms")
    print(f"Generated: {generated}")
except Exception as e:
    print(f"Error: {e}")
```

### **🐳 Docker Deployment**

#### **Dockerfile**

```dockerfile
# Dockerfile for OpenLLM inference server
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "core/src/main.py", "inference", \
     "--model-path", "exports/huggingface-6k/huggingface", \
     "--host", "0.0.0.0", "--port", "8000"]
```

#### **Build and Run**

```bash
# Build the Docker image
docker build -t openllm-inference .

# Run the container
docker run -p 8000:8000 openllm-inference

# Run with custom configuration
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/exports/huggingface-6k/huggingface \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  openllm-inference
```

#### **Docker Compose**

```yaml
# docker-compose.yml
version: '3.8'

services:
  openllm-inference:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/exports/huggingface-6k/huggingface
      - HOST=0.0.0.0
      - PORT=8000
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    resources:
      limits:
        memory: 4G
        cpus: '2'
      requests:
        memory: 2G
        cpus: '1'
```

```bash
# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### **☸️ Kubernetes Deployment**

#### **Deployment Configuration**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openllm-inference
  labels:
    app: openllm-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: openllm-inference
  template:
    metadata:
      labels:
        app: openllm-inference
    spec:
      containers:
      - name: openllm
        image: openllm-inference:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/exports/huggingface-6k/huggingface"
        - name: HOST
          value: "0.0.0.0"
        - name: PORT
          value: "8000"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### **Service Configuration**

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: openllm-service
  labels:
    app: openllm-inference
spec:
  selector:
    app: openllm-inference
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: LoadBalancer
```

#### **Ingress Configuration (Optional)**

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: openllm-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: openllm.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: openllm-service
            port:
              number: 80
```

#### **Deploy to Kubernetes**

```bash
# Apply configurations
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods
kubectl get services
kubectl get ingress

# View logs
kubectl logs -f deployment/openllm-inference

# Scale deployment
kubectl scale deployment openllm-inference --replicas=3
```

### **📊 Monitoring and Logging**

#### **Health Checks**

```python
import requests
import time

def monitor_server(server_url="http://localhost:8000"):
    """Monitor server health and performance."""
    
    try:
        # Health check
        health = requests.get(f"{server_url}/health")
        print(f"Health: {health.status_code}")
        
        # Model info
        info = requests.get(f"{server_url}/model-info")
        print(f"Model Info: {info.json()}")
        
        # Performance test
        start_time = time.time()
        response = requests.post(f"{server_url}/generate", json={
            "prompt": "Test",
            "max_tokens": 10
        })
        end_time = time.time()
        
        print(f"Response time: {end_time - start_time:.2f}s")
        print(f"Status: {response.status_code}")
        
    except Exception as e:
        print(f"Error: {e}")

# Run monitoring
monitor_server()
```

#### **Prometheus Metrics (Optional)**

```python
# Add Prometheus metrics to your FastAPI app
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import FastAPI

# Define metrics
REQUEST_COUNT = Counter('openllm_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('openllm_request_duration_seconds', 'Request latency')

app = FastAPI()

@app.middleware("http")
async def add_metrics(request, call_next):
    REQUEST_COUNT.inc()
    start_time = time.time()
    response = await call_next(request)
    REQUEST_LATENCY.observe(time.time() - start_time)
    return response

@app.get("/metrics")
async def metrics():
    return generate_latest()
```

#### **Logging Configuration**

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('openllm.log', maxBytes=10000000, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### **🔧 Environment Configuration**

#### **Environment Variables**

```bash
# .env file
MODEL_PATH=exports/huggingface-6k/huggingface
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
MAX_WORKERS=4
BATCH_SIZE=1
```

#### **Configuration File**

```yaml
# config.yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout: 30

model:
  path: exports/huggingface-6k/huggingface
  max_length: 512
  device: auto

generation:
  max_tokens: 50
  temperature: 0.7
  top_k: 40
  top_p: 0.9

logging:
  level: INFO
  file: openllm.log
  max_size: 10MB
  backup_count: 5
```

### **🔒 Security Considerations**

#### **Authentication**

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # Implement your token verification logic here
    if not is_valid_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return token

@app.post("/generate")
async def generate_text(request: GenerateRequest, token: str = Depends(verify_token)):
    # Your generation logic here
    pass
```

#### **Rate Limiting**

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/generate")
@limiter.limit("10/minute")
async def generate_text(request: GenerateRequest):
    # Your generation logic here
    pass
```

## 📚 Additional Resources

- **[User Guide](user-guide.md)** - Complete usage instructions
- **[Training Guide](training-guide.md)** - How to train your own models
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
