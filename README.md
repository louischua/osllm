# OpenLLM: Open Source Large Language Model

<!-- Copyright (C) 2024 Louis Chua Bean Chong -->
<!-- This file is part of OpenLLM - dual-licensed under GPLv3 and Commercial License -->

## 🌟 Overview

OpenLLM is an open source project to develop a powerful, flexible, and modular large language model (LLM) that is openly licensed under GPLv3 for research and community use, with a commercial license available for enterprise applications.

### **🎯 Current Status**

✅ **Pre-trained Model Available:** [lemms/openllm-small-extended-6k](https://huggingface.co/lemms/openllm-small-extended-6k)  
✅ **Inference Server:** FastAPI-based production-ready server  
✅ **Training Pipeline:** Complete end-to-end training workflow  
✅ **Documentation:** Comprehensive guides and examples  
✅ **Testing:** Model evaluation and benchmarking tools

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

### **🎯 Quick Start: Use Our Pre-trained Model**

We have a pre-trained OpenLLM model available on Hugging Face that you can use immediately:

**🔗 Model:** [lemms/openllm-small-extended-6k](https://huggingface.co/lemms/openllm-small-extended-6k)

**💡 Note:** The quick start guide now downloads the model and tokenizer directly from Hugging Face, so it works for all users!

```python
# Install and use the pre-trained model
pip install torch sentencepiece huggingface_hub

# Load the model and tokenizer from Hugging Face
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
tokenizer.load(hf_hub_download("lemms/openllm-small-extended-6k", "tokenizer.model"))

# Download and load model from Hugging Face
model = create_model("small")
checkpoint = torch.load(hf_hub_download("lemms/openllm-small-extended-6k", "pytorch_model.bin"), map_location="cpu")
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

### **📚 Documentation**

- **[📖 User Guide](docs/user-guide.md)** - Complete usage instructions and examples
- **[🚀 Deployment Guide](docs/deployment-guide.md)** - Production deployment with Docker & Kubernetes
- **[🏗️ Training Guide](docs/training_pipeline.md)** - Train your own models from scratch
- **[🗺️ Roadmap](docs/roadmap.md)** - Development roadmap and future plans

### **📊 Model Performance**

- **Model Size:** 35.8M parameters
- **Training Steps:** 6,000
- **Context Length:** 512 tokens
- **Tokenizer:** SentencePiece BPE (32k vocabulary)

**💡 Pro Tip:** For production use, check out our [deployment guide](docs/deployment-guide.md) for Docker and Kubernetes setup!

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
