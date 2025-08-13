# OpenLLM: Open Source Large Language Model

<!-- Copyright (C) 2024 Louis Chua Bean Chong -->
<!-- This file is part of OpenLLM - dual-licensed under GPLv3 and Commercial License -->

## 🌟 Overview

OpenLLM is an open source project to develop a powerful, flexible, and modular large language model (LLM) that is openly licensed under GPLv3 for research and community use, with a commercial license available for enterprise applications.

### **🎯 Current Status**

✅ **Pre-trained Models Available:** 
   - [lemms/openllm-small-extended-6k](https://huggingface.co/lemms/openllm-small-extended-6k) (6,000 steps)
   - [lemms/openllm-small-extended-7k](https://huggingface.co/lemms/openllm-small-extended-7k) (7,000 steps)  
✅ **Inference Server:** FastAPI-based production-ready server  
✅ **Training Pipeline:** Complete end-to-end training workflow  
✅ **Documentation:** Comprehensive guides and examples  
✅ **Test Suite:** Comprehensive unit and integration tests

## 🚀 Key Features

- ✔️ Pretraining and fine-tuning pipeline
- ✔️ Tokenizer training with SentencePiece or BPE
- ✔️ Support for multilingual datasets
- ✔️ Transformer-based architecture (GPT-like)
- ✔️ Model quantization and export for inference
- ✔️ Integration with Hugging Face, PyTorch, and ONNX
- ✔️ CLI and RESTful API for inference
- 🔒 Enterprise: RLHF trainer, fine-tuning UI, inference server orchestration (Kubernetes)

## 🧠 Design Goals

- Fully transparent and reproducible LLM stack
- Plug-and-play components (tokenizer, model, trainer)
- Scalable to billions of parameters
- Simple to extend with downstream tasks

## 📂 Folder Structure

```
osllm-1/
├── compare_models.py           # Model comparison and benchmarking utility
├── configs/                    # Model configuration files
│   ├── large_model.json       # Large model hyperparameters
│   ├── medium_model.json      # Medium model hyperparameters
│   └── small_model.json       # Small model hyperparameters
├── core/                       # Open source components (training, tokenization, inference)
│   ├── LICENSE                 # Core module license
│   ├── README.md              # Core module documentation
│   └── src/                   # Python source files
│       ├── data_loader.py              # Dataset loading and preprocessing
│       ├── download_and_prepare.py     # SQUAD dataset downloader & processor
│       ├── enterprise_integration.py  # Enterprise feature integration
│       ├── evaluate_model.py          # Model evaluation and metrics
│       ├── export_model.py            # Model export to various formats
│       ├── generate_text.py           # Text generation utilities
│       ├── inference_server.py        # FastAPI inference server
│       ├── main.py                    # Main CLI interface
│       ├── model.py                   # Transformer model architecture
│       ├── test_model.py              # Model testing utilities
│       ├── train_model.py             # Model training pipeline
│       └── train_tokenizer.py         # SentencePiece tokenizer trainer
├── data/                       # Training data and model artifacts
│   ├── raw/                    # Downloaded raw data (temporary)
│   ├── clean/                  # Processed training text
│   │   └── training_data.txt   # ~41k Wikipedia passages from SQUAD
│   └── tokenizer/              # Trained tokenizer files
│       ├── tokenizer_config.json # Tokenizer configuration
│       ├── tokenizer.model     # Trained SentencePiece model
│       └── tokenizer.vocab     # Vocabulary file
├── docs/                       # Documentation and community guidelines
│   ├── CODE_OF_CONDUCT.md      # Community guidelines
│   ├── CONTRIBUTING.md         # Contribution guidelines
│   ├── COPYRIGHT_HEADER.txt    # Standard copyright header
│   ├── deployment_guide.md     # Deployment instructions
│   ├── LICENSES.md            # Licensing information
│   └── training_pipeline.md   # Complete training guide
├── tests/                      # Comprehensive test suite
│   ├── __init__.py            # Test package initialization
│   ├── README.md              # Test suite documentation and guidelines
│   ├── requirements-test.txt  # Test dependencies
│   ├── run_tests.py           # Main test runner with coverage reporting
│   ├── test_model.py          # Model architecture and configuration tests
│   ├── test_training.py       # Training pipeline and data loader tests
│   └── test_inference.py      # Inference server and API tests
├── enterprise/                 # Enterprise-only modules
│   └── README.md              # Enterprise features documentation
├── exports/                    # Exported model formats
│   ├── huggingface/           # Hugging Face compatible exports
│   │   ├── config.json        # Model configuration
│   │   ├── generation_config.json # Generation parameters
│   │   ├── load_hf_model.py   # Hugging Face loader script
│   │   ├── pytorch_model.bin  # Model weights
│   │   ├── tokenizer_config.json # Tokenizer config
│   │   └── tokenizer.model    # Tokenizer model
│   └── pytorch/               # PyTorch native exports
│       ├── config.json        # Model configuration
│       ├── load_model.py      # PyTorch loader script
│       ├── model.pt          # Model state dict
│       └── tokenizer.model   # Tokenizer model
├── LICENSES/                   # License files
│   ├── LICENSE-COMMERCIAL     # Commercial license terms
│   ├── LICENSE-DUAL-INFO      # Dual licensing information
│   ├── LICENSE-GPL-3.0        # GPL-3.0 license text
│   └── README.md             # License documentation
├── models/                     # Trained models and checkpoints
├── pyproject.toml             # Python project configuration
├── requirements.txt           # Python dependencies
├── SECURITY.md               # Security policy and reporting
└── test_trained_model.py     # Model testing script
```

## 🛠️ Tech Stack

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- SentencePiece
- FastAPI for inference API

## 🚀 Getting Started

### **🎯 Quick Start: Use Our Pre-trained Models**

We have multiple pre-trained OpenLLM models available on Hugging Face that you can use immediately:

**🔗 Available Models:**
- **[lemms/openllm-small-extended-6k](https://huggingface.co/lemms/openllm-small-extended-6k)** (6,000 training steps)
- **[lemms/openllm-small-extended-7k](https://huggingface.co/lemms/openllm-small-extended-7k)** (7,000 training steps) - **Latest & Recommended**

**💡 Note:** The quick start guide downloads models directly from Hugging Face, so it works for all users!

#### **🚀 Option 1: Using the Latest 7k Model (Recommended)**

```python
# Install dependencies
pip install torch sentencepiece huggingface_hub transformers

# Load the latest 7k model using Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer from Hugging Face
model_name = "lemms/openllm-small-extended-7k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
prompt = "The future of artificial intelligence"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

#### **🔧 Option 2: Using Custom Loader (Advanced)**

```python
# Install dependencies
pip install torch sentencepiece huggingface_hub

# Load the model using our custom loader
import torch
import sentencepiece as spm
from huggingface_hub import hf_hub_download
import sys
import os

# Add the core/src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'src'))
from model import create_model

# Download and load tokenizer from Hugging Face
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(hf_hub_download("lemms/openllm-small-extended-7k", "tokenizer.model"))

# Download and load model from Hugging Face
model = create_model("small")
checkpoint = torch.load(hf_hub_download("lemms/openllm-small-extended-7k", "pytorch_model.bin"), map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

# Generate text
text = "The future of AI"
tokens = tokenizer.encode(text)
inputs = torch.tensor([tokens])

with torch.no_grad():
    outputs = model.generate(inputs, max_new_tokens=50, temperature=0.7)
    
generated_text = tokenizer.decode(outputs[0].tolist())
print(generated_text)
```

#### **🌐 Option 3: Using the Inference Server**

```bash
# Start the FastAPI inference server
python core/src/inference_server.py \
    --model_path exports/huggingface-7k/huggingface \
    --port 8000

# Make API calls
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "The future of renewable energy",
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

### **📚 Documentation**

- **[📖 User Guide](docs/user-guide.md)** - Complete usage instructions and examples
- **[🚀 Deployment Guide](docs/deployment-guide.md)** - Production deployment with Docker & Kubernetes
- **[🏗️ Training Guide](docs/training_pipeline.md)** - Train your own models from scratch
- **[🗺️ Roadmap](docs/roadmap.md)** - Development roadmap and future plans

### **🧪 Testing**

Our comprehensive test suite ensures code quality and reliability:

- **[🧪 Test Suite](tests/)** - Complete test coverage for all components
- **[📊 Test Coverage](tests/README.md)** - Detailed testing documentation and guidelines
- **[⚡ Quick Test Run](tests/run_tests.py)** - Easy test execution with coverage reporting

**Run the tests:**
```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run all tests
python tests/run_tests.py

# Run specific test modules
python -m pytest tests/test_model.py -v
python -m pytest tests/test_training.py -v
python -m pytest tests/test_inference.py -v
```

**Test Coverage:**
- ✅ **Model Architecture** - GPT model, attention, and configuration tests
- ✅ **Training Pipeline** - Data loading, training loop, and evaluation
- ✅ **Inference Server** - API endpoints, text generation, and performance
- ✅ **Integration Tests** - End-to-end workflow validation

### **📊 Model Performance & Comparison**

| Model | Parameters | Training Steps | Context Length | Use Case | Performance |
|-------|------------|----------------|----------------|----------|-------------|
| **Small 6K** | 35.8M | 6,000 | 1,024 | Basic text generation | Good coherence |
| **Small 7K** | 35.8M | 7,000 | 1,024 | **Extended training** | **Improved quality** |

**Model Specifications:**
- **Architecture:** GPT-style Transformer
- **Layers:** 6 transformer layers
- **Heads:** 8 attention heads
- **Embedding Dimension:** 512
- **Vocabulary Size:** 32,000 tokens
- **Tokenizer:** SentencePiece BPE (32k vocabulary)

**Performance Metrics:**
- **Inference Speed:** ~50 tokens/second on CPU, ~200 tokens/second on GPU
- **Memory Usage:** ~2GB VRAM during training, ~1GB for inference
- **Model Size:** 161MB (pytorch_model.bin)

**💡 Pro Tip:** For production use, check out our [deployment guide](docs/deployment-guide.md) for Docker and Kubernetes setup!

### **🎯 Model Capabilities & Use Cases**

**Text Generation Tasks:**
- ✅ **Paragraph Generation** - Coherent, context-aware text generation
- ✅ **Question Answering** - Basic factual responses from training data
- ✅ **Text Summarization** - Short text summarization capabilities
- ✅ **Language Understanding** - Context-aware responses and reasoning

**Recommended Applications:**
- **Research & Education** - Learning about language models and AI
- **Prototyping** - Quick development of text generation features
- **Content Creation** - Basic text generation for creative writing
- **Chatbots** - Simple conversational AI applications

**Model Limitations:**
- **Context Length:** Limited to 1,024 tokens
- **Training Data:** Wikipedia passages only (limited domain)
- **Model Size:** Small model with basic reasoning capabilities
- **Factual Accuracy:** Not guaranteed for current events

**💡 For Advanced Use Cases:** Consider training larger models or fine-tuning for specific domains.

### **🏗️ Model Development & Training**

**Training Process:**
- **Dataset:** Wikipedia passages from SQuAD dataset (~41k passages)
- **Tokenization:** SentencePiece with 32k vocabulary
- **Training Objective:** Next token prediction (causal language modeling)
- **Optimizer:** AdamW with learning rate scheduling
- **Hardware:** Consumer GPU with gradient accumulation

**Model Evolution:**
- **4K Model:** Basic training foundation
- **6K Model:** Improved coherence and quality
- **7K Model:** Extended training for better performance

**Training Metrics:**
- **Final Loss:** ~2.1 (cross-entropy)
- **Training Time:** ~7 hours on consumer GPU
- **Memory Usage:** ~2GB VRAM during training

**🔬 Research & Development:** All training code, data processing, and model architecture are open source and fully reproducible.

### **🤗 Hugging Face Integration**

**Model Distribution:**
- **Repository:** [lemms/openllm-small-extended-7k](https://huggingface.co/lemms/openllm-small-extended-7k)
- **Format:** Fully compatible with Hugging Face Transformers
- **License:** GPL-3.0 / Commercial available
- **Documentation:** Comprehensive README with usage examples

**Easy Integration:**
```python
# One-line model loading
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("lemms/openllm-small-extended-7k")
model = AutoModelForCausalLM.from_pretrained("lemms/openllm-small-extended-7k")
```

**Community Benefits:**
- ✅ **Easy Access** - Download models directly from Hugging Face Hub
- ✅ **Standard Format** - Compatible with the entire Hugging Face ecosystem
- ✅ **Version Control** - Track model versions and improvements
- ✅ **Community Sharing** - Share and discover models easily

## 💼 Licensing

OpenLLM is **dual-licensed** to provide maximum flexibility:

### 🆓 GPLv3 (Free for Open Source)
- ✅ **Perfect for:** Research, education, open source projects
- ✅ **Free to use** and modify
- ⚠️ **Requirement:** Share modifications under GPL

### 💼 Commercial License
- ✅ **Perfect for:** Proprietary software, SaaS, enterprise
- ✅ **No copyleft** restrictions
- ✅ **Keep modifications private**
- ✅ **Enterprise support included**

**Quick Guide:**
- **Open source project?** → Use GPLv3 (free)
- **Commercial product?** → Get commercial license
- **Not sure?** → Start with GPLv3, upgrade later

**License Files:**
- [`LICENSE`](LICENSE) - GPL-3.0 license text (GitHub recognized)
- [`LICENSES/LICENSE-COMMERCIAL`](LICENSES/LICENSE-COMMERCIAL) - Commercial license terms
- [`docs/LICENSES.md`](docs/LICENSES.md) - Complete dual licensing guide

💬 **Commercial licensing:** Contact us at [louischua@gmail.com]

## 🗺️ **Development Roadmap**

For detailed information about our development plans, milestones, and future features, see our comprehensive roadmap:

**📋 [Complete Development Roadmap](docs/roadmap.md)**

This includes:
- ✅ **Completed Features** - What we've built so far
- 🚧 **In Progress** - Current development work
- 🔮 **Planned Features** - Future capabilities
- 🎯 **Priority Milestones** - Version releases and timelines
- 🏆 **Competitive Analysis** - Market positioning
- ⚠️ **Risk Assessment** - Challenges and mitigation strategies

## 🤝 Contributing

We welcome contributions from the community! Please read our:
- [Contributing Guide](docs/CONTRIBUTING.md) - How to contribute to OpenLLM
- [Code of Conduct](docs/CODE_OF_CONDUCT.md) - Community guidelines and standards

For questions or support, feel free to:
- 📝 Open an [issue](https://github.com/louischua/openllm/issues)
- 💬 Start a [discussion](https://github.com/louischua/openllm/discussions)
- 📧 Email us at [louischua@gmail.com]
