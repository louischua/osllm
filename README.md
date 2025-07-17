# OpenLLM: Open Source Large Language Model

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
core/                 # Open source components (training, tokenization, inference)
core/src/             # Python source files
enterprise/           # Enterprise-only modules (e.g., dashboard, RLHF UI)
docs/                 # Documentation and community guidelines
.github/              # GitHub config (PR template, funding, etc.)
```

## 🛠️ Tech Stack

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- SentencePiece
- FastAPI for inference API

## 💼 Licensing

OpenLLM is dual-licensed:

- GPLv3 for community and academic use
- Commercial license for closed-source or enterprise deployment

Contact us at [sales@yourcompany.com](mailto:sales@yourcompany.com) for licensing details.
