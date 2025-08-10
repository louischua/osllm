# OpenLLM: Open Source Large Language Model

<!-- Copyright (C) 2024 Louis Chua Bean Chong -->
<!-- This file is part of OpenLLM - dual-licensed under GPLv3 and Commercial License -->

## 🌟 Overview

OpenLLM is an open source project to develop a powerful, flexible, and modular large language model (LLM) that is openly licensed under GPLv3 for research and community use, with a commercial license available for enterprise applications.

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
├── core/             # Open source components (training, tokenization, inference)
│   └── src/          # Python source files
│       ├── download_and_prepare.py    # SQUAD dataset downloader & processor
│       └── train_tokenizer.py         # SentencePiece tokenizer trainer
├── data/             # Training data and model artifacts
│   ├── raw/          # Downloaded raw data (temporary)
│   ├── clean/        # Processed training text
│   │   └── training_data.txt          # ~41k Wikipedia passages from SQUAD
│   └── tokenizer/    # Trained tokenizer files
├── enterprise/       # Enterprise-only modules (e.g., dashboard, RLHF UI)
├── docs/             # Documentation and community guidelines
│   └── training_pipeline.md           # Complete training guide
└── .github/          # GitHub config (PR template, funding, etc.)
```

## 🛠️ Tech Stack

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- SentencePiece
- FastAPI for inference API

## 🚀 Getting Started: Training Your Own Foundation Model

**📚 Follow the Complete Training Pipeline**

To understand how to use OpenLLM scripts to generate foundational models from scratch, please follow our comprehensive training guide:

**👉 [Training Pipeline Documentation](docs/training_pipeline.md)**

This step-by-step guide covers the complete process:

### 📋 **Pipeline Overview:**
1. **📊 Data Preparation** - Download and process SQUAD dataset (~41k Wikipedia passages)
2. **🔤 Tokenizer Training** - Train SentencePiece BPE tokenizer on your text corpus
3. **🏗️ Model Architecture** - Set up GPT-style transformer (Small/Medium/Large configs)
4. **🎯 Model Training** - Pre-train your language model with modern optimization
5. **📊 Evaluation** - Assess model quality with perplexity and downstream tasks
6. **📦 Export & Deploy** - Save models for inference (PyTorch/HuggingFace/ONNX formats)

### ⚡ **Quick Start Commands:**
```bash
# 1. Prepare training data
python core/src/download_and_prepare.py

# 2. Train tokenizer  
python core/src/train_tokenizer.py --input data/clean/training_data.txt --vocab_size 32000

# 3. Train small model (recommended for beginners)
python core/src/main.py train-model --model-size small --data-file data/clean/training_data.txt --tokenizer-dir data/tokenizer --output-dir models/my-model

# 4. Evaluate trained model
python core/src/main.py evaluate --model_path models/my-model --metrics perplexity,generation

# 5. Export for inference
python core/src/main.py export --model_dir models/my-model --format huggingface --output_dir exports/
```

### 🎯 **Model Sizes Available:**
- **Small (~25M params)** - Great for learning and CPU training
- **Medium (~117M params)** - Balanced performance, GPU recommended  
- **Large (~350M params)** - High quality, requires powerful GPU

### 📖 **Essential Documentation:**
- **[Complete Training Guide](docs/training_pipeline.md)** - Detailed step-by-step instructions
- **[Model Architecture](core/src/model.py)** - GPT implementation details
- **[CLI Usage](core/src/main.py)** - All available commands and options

**💡 Pro Tip:** Start with the small model configuration to familiarize yourself with the training process, then scale up to larger models as needed!

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

## 🤝 Contributing

We welcome contributions from the community! Please read our:
- [Contributing Guide](docs/CONTRIBUTING.md) - How to contribute to OpenLLM
- [Code of Conduct](docs/CODE_OF_CONDUCT.md) - Community guidelines and standards

For questions or support, feel free to:
- 📝 Open an [issue](https://github.com/louischua/openllm/issues)
- 💬 Start a [discussion](https://github.com/louischua/openllm/discussions)
- 📧 Email us at [louischua@gmail.com]
