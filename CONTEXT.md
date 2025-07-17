# Cursor AI Context — OpenLLM Project Description

## Objective

Build an open source large language model (LLM) framework that is modular, scalable, and easy to adapt for downstream NLP tasks. The system should be:

- Open source under GPLv3
- Commercially licensable for enterprise users
- Capable of serving both researchers and production teams

## Core Components

- Tokenizer: Trainable using SentencePiece or custom BPE
- Model: Transformer-based LLM, GPT-style decoder
- Training Pipeline: Distributed training, mixed precision, gradient checkpointing
- Inference: FastAPI-based REST endpoint, ONNX export
- Enterprise Extensions (not in core repo):
  - RLHF trainer and reward model
  - Admin UI for managing fine-tuning runs
  - Scalable inference server with Kubernetes support

## Key Design Choices

- Use Hugging Face ecosystem to reduce boilerplate
- Maintain modularity between pretraining and fine-tuning
- Build CLI tools for training and inference
- Allow offline usage and checkpoint sharing

## Folder Overview

- `core/`: Open source logic and training components
- `enterprise/`: Placeholder for commercial-only features (private repo)
- `docs/`: CONTRIBUTING, licensing, setup guides
- `.github/`: PR templates and funding links
