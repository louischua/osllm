# 🚀 OpenLLM v0.1.0 - Open Source Large Language Model Framework

<!-- Copyright (C) 2024 Louis Chua Bean Chong -->
<!-- This file is part of OpenLLM - dual-licensed under GPLv3 and Commercial License -->

**🎉 OpenLLM v0.1.0 is now available!** A complete open source large language model (LLM) framework that is modular, scalable, and easy to adapt for downstream NLP tasks. Built from scratch using publicly available datasets with no fine-tuning of existing models.

## 🎯 **Project Overview**

OpenLLM provides a complete training pipeline for building large language models from scratch, with enterprise-ready deployment capabilities and comprehensive tooling for research and production use.

### **✅ v0.1.0 Release Features**
- ✅ **Complete Training Pipeline** - From data preparation to model deployment
- ✅ **Multiple Model Sizes** - Small (35M), Medium (125M), Large (350M) parameters
- ✅ **Trained Models Available** - Pre-trained models ready for inference
- ✅ **Inference Server** - FastAPI-based REST API for model serving
- ✅ **Comprehensive Testing** - 66/66 tests passing (100% success rate)
- ✅ **Professional Documentation** - Complete guides and examples
- ✅ **Cross-Platform Support** - Windows, Linux, macOS compatibility

### **Key Features**
- 🧠 **GPT-Style Architecture** - Transformer-based decoder-only model
- 🔤 **SentencePiece Tokenization** - BPE tokenizer with 32k vocabulary
- 📊 **Training Pipeline** - Complete from data preparation to model export
- 🌐 **Inference Server** - FastAPI REST API with streaming support
- 📈 **Model Evaluation** - Perplexity, text generation quality assessment
- 🔧 **CLI Interface** - Unified command-line tool for all operations
- 📦 **Model Export** - PyTorch, Hugging Face, ONNX formats
- 🧪 **Comprehensive Testing** - Unit and integration test coverage

## 🚀 **Quick Start**

### **Installation**

```bash
# Clone the repository
git clone https://github.com/your-username/openllm.git
cd openllm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Basic Usage**

```bash
# Test the model architecture
python core/src/main.py test-model --model-size small

# Evaluate a trained model
python scripts/evaluation/evaluate_trained_model.py

# Start inference server (if you have a trained model)
python core/src/inference_server.py --model_path models/small-extended-7k/
```

### **Training Your Own Model**

```bash
# 1. Prepare training data
python core/src/main.py prepare-data --output data/clean/training_data.txt

# 2. Train tokenizer
python core/src/main.py train-tokenizer \
  --input data/clean/training_data.txt \
  --vocab-size 32000 \
  --output-dir data/tokenizer/

# 3. Train the model
python core/src/main.py train-model \
  --model-size small \
  --output-dir models/my-model/ \
  --max-steps 10000
```

## 📁 **Project Structure**

```
openllm/
├── core/                          # Core training and inference code
│   ├── src/
│   │   ├── model.py              # GPT-style transformer model
│   │   ├── train_model.py        # Training pipeline
│   │   ├── inference_server.py   # FastAPI inference server
│   │   ├── data_loader.py        # Data loading utilities
│   │   └── main.py               # CLI interface
├── data/                          # Data and tokenizer files
│   ├── clean/                    # Processed training data
│   └── tokenizer/                # Trained tokenizer files
├── models/                        # Trained model checkpoints
│   └── small-extended-7k/        # Pre-trained small model (7k steps)
├── deployment/                    # Deployment configurations
│   ├── huggingface/              # Hugging Face deployment
│   ├── docker/                   # Docker containerization
│   └── kubernetes/               # Kubernetes deployment
├── scripts/                       # Utility scripts
│   ├── evaluation/               # Model evaluation scripts
│   ├── training/                 # Training utilities
│   └── setup/                    # Setup and installation
├── tests/                        # Test suite
├── docs/                         # Documentation
└── requirements.txt              # Python dependencies
```

## 🧪 **Model Performance**

### **Current Model (Small, 7k steps)**
- **Parameters**: 35.8M
- **Training Steps**: 7,000
- **Final Loss**: 5.22
- **Text Generation**: ✅ Working with coherent output
- **Inference Speed**: ~8.3 tokens/second on CPU

### **Model Quality Assessment**
- **Perplexity**: ~730 (needs improvement for production)
- **Text Coherence**: Basic coherence achieved
- **Generation Quality**: Functional but requires more training

## 🔧 **Advanced Usage**

### **Model Configuration**

OpenLLM supports three model sizes:

```python
# Small model (35M parameters)
model = create_model("small")

# Medium model (125M parameters)  
model = create_model("medium")

# Large model (350M parameters)
model = create_model("large")
```

### **Inference Server API**

```python
import requests

# Start the server
# python core/src/inference_server.py --model_path models/small-extended-7k/

# Generate text
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "The future of artificial intelligence",
    "max_length": 100,
    "temperature": 0.7
})

print(response.json()["generated_text"])
```

### **Custom Training**

```python
from core.src.train_model import ModelTrainer

trainer = ModelTrainer(
    model_size="small",
    tokenizer_dir="data/tokenizer/",
    data_file="data/clean/training_data.txt",
    output_dir="models/custom-model/"
)

trainer.train(max_steps=10000, batch_size=4)
```

## 🧪 **Testing**

Run the comprehensive test suite:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/test_model.py
python tests/test_training.py
python tests/test_inference.py

# Run full pipeline validation
python scripts/test_full_pipeline.py
```

## 📊 **Performance Benchmarks**

### **Training Performance**
- **Small Model**: ~2.5 hours on CPU (7k steps)
- **Memory Usage**: ~150MB for small model
- **Training Speed**: ~2.8 steps/second on CPU

### **Inference Performance**
- **Generation Speed**: 8.3 tokens/second (CPU)
- **Memory Usage**: ~143MB for small model
- **Response Time**: <5 seconds for 50 tokens

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### **Development Setup**

```bash
# Install development dependencies
pip install -r tests/requirements-test.txt

# Run tests
python tests/run_tests.py

# Run linting
python scripts/fix_linting.py
```

## 📄 **License**

OpenLLM is dual-licensed:

- **Open Source**: GPLv3 License (see [LICENSE](LICENSE))
- **Commercial**: Commercial License available for enterprise use

## 🎯 **Roadmap**

### **v0.2.0 (Q3 2025)**
- 🎯 Improved model quality (perplexity <50)
- 🎯 Production-ready inference server
- 🎯 Docker containerization
- 🎯 Advanced monitoring and logging

### **v0.3.0 (Q2 2026)**
- 🎯 Chain of Thought reasoning
- 🎯 Fine-tuning pipeline
- 🎯 Multi-language support
- 🎯 Mixture of Experts architecture

## 📞 **Support**

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-username/openllm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/openllm/discussions)

## 🙏 **Acknowledgments**

- Built with PyTorch and Hugging Face ecosystem
- Training data from SQUAD dataset
- Community contributions and feedback

---

**OpenLLM v0.1.0** - Building the future of open source AI, one model at a time! 🚀
