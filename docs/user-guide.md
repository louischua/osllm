# OpenLLM User Guide

<!-- Copyright (C) 2024 Louis Chua Bean Chong -->
<!-- This file is part of OpenLLM - dual-licensed under GPLv3 and Commercial License -->

## 🎯 Getting Started with OpenLLM

### **Option 1: Use Pre-trained Model (Recommended for Beginners)**

1. **Install Dependencies:**
```bash
pip install transformers torch sentencepiece requests
```

2. **Load and Use the Model:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model
model_name = "lemms/openllm-small-extended-6k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
def generate_text(prompt, max_length=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=40,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
result = generate_text("The future of artificial intelligence")
print(result)
```

### **Option 2: Use Inference Server (Recommended for Production)**

1. **Clone and Setup:**
```bash
git clone https://github.com/louischua/openllm.git
cd openllm
pip install -r requirements.txt
```

2. **Start the Server:**
```bash
python core/src/main.py inference \
    --model-path exports/huggingface-6k/huggingface \
    --host 0.0.0.0 \
    --port 8000
```

3. **Use the API:**
```python
import requests

def query_model(prompt, max_tokens=50):
    response = requests.post("http://localhost:8000/generate", json={
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_k": 40
    })
    return response.json()["generated_text"]

# Test the model
result = query_model("Explain machine learning")
print(result)
```

## 🔧 Advanced Usage

### **Custom Generation Parameters**

```python
# Advanced generation with custom parameters
def advanced_generate(prompt, **kwargs):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    generation_config = {
        "max_new_tokens": kwargs.get("max_tokens", 50),
        "temperature": kwargs.get("temperature", 0.7),
        "top_k": kwargs.get("top_k", 40),
        "top_p": kwargs.get("top_p", 0.9),
        "do_sample": kwargs.get("do_sample", True),
        "num_beams": kwargs.get("num_beams", 1),
        "pad_token_id": tokenizer.eos_token_id
    }
    
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, **generation_config)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Examples
creative_text = advanced_generate("Write a story about", temperature=0.9, max_tokens=100)
focused_text = advanced_generate("Explain quantum physics", temperature=0.3, max_tokens=80)
```

### **Batch Processing**

```python
def batch_generate(prompts, max_tokens=50):
    """Generate text for multiple prompts efficiently."""
    results = []
    
    for prompt in prompts:
        result = generate_text(prompt, max_tokens)
        results.append({"prompt": prompt, "generated": result})
    
    return results

# Example
prompts = [
    "The history of computers",
    "Machine learning applications",
    "Future of technology"
]

batch_results = batch_generate(prompts)
for result in batch_results:
    print(f"Prompt: {result['prompt']}")
    print(f"Generated: {result['generated']}\n")
```

## 📊 Model Performance & Testing

Our pre-trained model has been thoroughly tested and evaluated:

### **🧪 Test Results (6k Model)**
- **Model Size:** 35.8M parameters
- **Training Steps:** 6,000
- **Final Training Loss:** 5.4302
- **Average Perplexity:** 816.04
- **Context Length:** 512 tokens
- **Tokenizer:** SentencePiece BPE (32k vocabulary)

### **🎯 Sample Generation Results**
```
Prompt: "The history of artificial intelligence"
Generated: "is the only 'core' of the two main classes. The two-speaking areas is the largest largest city in the world..."

Prompt: "Machine learning algorithms"
Generated: ", and is the most popular-term development of the world's economic and cultural rights. By 2014, the United Kingdom..."

Prompt: "The future of technology"
Generated: "has been the first one of the most popular culture. These institutions were established in the 1980s..."
```

### **📈 Performance Characteristics**
- **Strengths:** Coherent text structure, proper tokenization, stable generation
- **Areas for Improvement:** High perplexity, repetitive output, context drift
- **Best Use Cases:** Basic text generation, educational purposes, research experiments

## 🔍 Troubleshooting

### **Common Issues and Solutions**

1. **Out of Memory Errors:**
```python
# Reduce model precision
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # Use half precision
    device_map="auto"  # Auto device mapping
)
```

2. **Slow Generation:**
```python
# Optimize generation parameters
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_k=40,
    use_cache=True,  # Enable caching
    pad_token_id=tokenizer.eos_token_id
)
```

3. **Tokenization Issues:**
```python
# Handle long sequences
def safe_tokenize(text, max_length=512):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return torch.tensor([tokens])
```

## 📚 Additional Resources

- **[Deployment Guide](deployment-guide.md)** - Production deployment instructions
- **[Training Guide](training-guide.md)** - How to train your own models
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## 🤝 Getting Help

- 📝 **Issues:** [GitHub Issues](https://github.com/louischua/openllm/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/louischua/openllm/discussions)
- 📧 **Email:** louischua@gmail.com
