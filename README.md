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

## 📋 Project Roadmap & To-Do List

### ✅ **Completed Features**

#### Core Training Pipeline
- ✅ **Data Processing** - SQUAD dataset download and cleaning (~41k passages)
- ✅ **Tokenizer Training** - SentencePiece BPE tokenizer with 32k vocabulary
- ✅ **Model Architecture** - GPT-style transformer (Small/Medium/Large configs)
- ✅ **Training Loop** - Complete training with optimization, checkpointing, logging
- ✅ **Model Evaluation** - Perplexity, text generation quality, downstream tasks
- ✅ **Model Export** - PyTorch native, Hugging Face compatible, ONNX formats
- ✅ **CLI Interface** - Unified command-line tool for all operations

#### Advanced Features
- ✅ **Inference Server** - FastAPI REST API for model serving
- ✅ **Text Generation** - Advanced sampling with temperature, top-k, top-p
- ✅ **Enterprise Integration** - Plugin system for commercial-only features
- ✅ **Comprehensive Documentation** - Training pipeline, API docs, examples

#### Project Infrastructure
- ✅ **Dual Licensing** - GPL-3.0 + Commercial license structure
- ✅ **Professional Documentation** - Code of Conduct, Contributing guidelines
- ✅ **GitHub Templates** - Issue templates, PR templates
- ✅ **Copyright Attribution** - Proper licensing headers in all source files

### 🚧 **In Progress**

#### Model Improvements
- 🔄 **Extended Training** - Scaling models to higher quality (6k+ steps)
- 🔄 **Performance Optimization** - Memory efficiency and training speed
- 🔄 **Hardware Support** - GPU optimization and multi-GPU training

#### Testing & Quality
- 🔄 **Test Suite** - Comprehensive unit and integration tests
- 🔄 **CI/CD Pipeline** - Automated testing and deployment
- 🔄 **Model Benchmarking** - Standardized evaluation protocols

### 🔮 **Planned Features**

#### Core Enhancements
- 📝 **Multi-Language Support** - Training on multilingual datasets
- 📝 **Custom Datasets** - Support for user-provided training data
- 📝 **Advanced Architectures** - Support for newer transformer variants
- 📝 **Distributed Training** - Multi-node training for large models
- 📝 **Mixed Precision** - FP16/BF16 training for efficiency

#### Advanced Training
- 📝 **Fine-tuning Pipeline** - Task-specific model adaptation
- 📝 **RLHF (Reinforcement Learning from Human Feedback)** - Alignment training
- 📝 **Instruction Tuning** - Chat/instruction-following capabilities
- 📝 **Parameter-Efficient Fine-tuning** - LoRA, AdaLoRA, QLoRA support
- 📝 **Chain of Thought Reasoning** - Advanced reasoning capabilities (see detailed roadmap below)
- 📝 **Multi-Modal Foundation Models** - Vision-Language models (see detailed roadmap below)

#### Multi-Modal Capabilities
- 📝 **Image Understanding** - Process and understand visual content
- 📝 **Vision-Language Integration** - Combined image and text processing
- 📝 **Document AI** - OCR, layout understanding, document analysis
- 📝 **Video Processing** - Temporal visual understanding and generation
- 📝 **Audio Integration** - Speech recognition and audio-text alignment
- 📝 **Multi-Modal Generation** - Text-to-image, image-to-text capabilities

##### 🎯 **Multi-Modal Development Roadmap**

**Phase 1: Foundation (Q1 2025)**
- 📝 **Vision Encoder Integration** - Add CLIP-style vision encoders
- 📝 **Image Preprocessing Pipeline** - Standardized image processing and augmentation
- 📝 **Vision-Text Tokenization** - Unified tokenization for text and image patches
- 📝 **Cross-Modal Attention** - Attention mechanisms between vision and text
- 📝 **Multi-Modal Data Loader** - Efficient loading of image-text pairs

**Phase 2: Core Models (Q2 2025)**
- 📝 **Vision-Language Pre-training** - Large-scale image-text pre-training
- 📝 **Multi-Modal Architecture** - Unified transformer for vision and language
- 📝 **Image Captioning** - Generate descriptions from images
- 📝 **Visual Question Answering** - Answer questions about images
- 📝 **Multi-Modal Embeddings** - Shared representation space for images and text

**Phase 3: Advanced Capabilities (Q3 2025)**
- 📝 **Document Understanding** - Layout analysis, table extraction, form processing
- 📝 **OCR Integration** - Text extraction from images and documents
- 📝 **Chart and Graph Analysis** - Understanding data visualizations
- 📝 **Multi-Modal Reasoning** - Complex reasoning across modalities
- 📝 **Fine-Grained Visual Understanding** - Object detection, segmentation integration

**Phase 4: Generation & Production (Q4 2025)**
- 📝 **Text-to-Image Generation** - Generate images from text descriptions
- 📝 **Image Editing** - Modify images based on text instructions
- 📝 **Multi-Modal Chat** - Conversational AI with image understanding
- 📝 **Production Inference** - Optimized multi-modal model serving
- 📝 **API Integration** - REST APIs for multi-modal capabilities

**Phase 5: Advanced Modalities (2026)**
- 📝 **Video Understanding** - Temporal modeling and video analysis
- 📝 **Audio Integration** - Speech recognition and audio-visual alignment
- 📝 **3D Understanding** - Point clouds, 3D scene understanding
- 📝 **Multi-Modal Memory** - Long-term memory across modalities
- 📝 **Real-Time Processing** - Live video/audio stream processing

##### 🛠️ **Technical Prerequisites for Multi-Modal**

**Infrastructure Requirements:**
- 📝 **GPU Memory Optimization** - Efficient handling of large image data
- 📝 **Distributed Training** - Multi-node training for large multi-modal models
- 📝 **Mixed Precision** - FP16/BF16 for memory efficiency
- 📝 **Model Parallelism** - Split large models across multiple GPUs
- 📝 **Data Pipeline Optimization** - Fast loading of image-text datasets

**Architecture Components:**
- 📝 **Vision Transformers (ViT)** - Image patch embedding and processing
- 📝 **Cross-Attention Layers** - Information flow between modalities
- 📝 **Positional Encodings** - 2D positional encoding for images
- 📝 **Multi-Modal Fusion** - Effective combination of different modalities
- 📝 **Adaptive Tokenization** - Dynamic sequence lengths for different modalities

**Dataset Integration:**
- 📝 **COCO Dataset Support** - Image captioning and object detection
- 📝 **Visual Genome** - Dense visual understanding annotations
- 📝 **Conceptual Captions** - Large-scale image-text pairs
- 📝 **TextVQA** - Visual question answering datasets
- 📝 **DocVQA** - Document understanding datasets
- 📝 **Custom Dataset Pipeline** - User-provided multi-modal data

**Evaluation Framework:**
- 📝 **Multi-Modal Benchmarks** - CLIP score, FID, BLEU for captioning
- 📝 **Visual Understanding Metrics** - VQA accuracy, object detection mAP
- 📝 **Cross-Modal Retrieval** - Image-text retrieval evaluation
- 📝 **Human Evaluation** - Quality assessment for generated content
- 📝 **Bias Detection** - Fairness evaluation across modalities

#### Production Features
- 📝 **Model Quantization** - INT8/INT4 quantization for deployment
- 📝 **Batch Inference** - Optimized batch processing
- 📝 **Streaming Generation** - Real-time text streaming
- 📝 **Model Caching** - Intelligent model loading and caching

#### Enterprise Features (Commercial License)
- 📝 **Web Dashboard** - Training monitoring and management UI
- 📝 **Kubernetes Deployment** - Scalable cloud deployment
- 📝 **Advanced Analytics** - Training metrics and performance monitoring
- 📝 **Enterprise Support** - Priority support and consulting
- 📝 **Custom Training Services** - Professional model training assistance

#### Developer Experience
- 📝 **Jupyter Notebooks** - Interactive tutorials and examples
- 📝 **Docker Containers** - Pre-configured development environments
- 📝 **Model Hub Integration** - Easy sharing and discovery of trained models
- 📝 **Auto-Documentation** - Automated API documentation generation

#### Research & Experimentation
- 📝 **Experiment Tracking** - Integration with MLflow/Weights & Biases
- 📝 **Hyperparameter Optimization** - Automated hyperparameter tuning
- 📝 **Architecture Search** - Neural architecture search capabilities
- 📝 **Research Baselines** - Standard benchmarks and comparisons

#### Chain of Thought Reasoning
- 📝 **Step-by-Step Reasoning** - Explicit reasoning process generation
- 📝 **Mathematical Problem Solving** - Advanced arithmetic and algebra
- 📝 **Logical Reasoning** - Deductive and inductive reasoning capabilities
- 📝 **Complex Problem Decomposition** - Breaking down multi-step problems
- 📝 **Self-Correction** - Error detection and reasoning refinement

##### 🧠 **Chain of Thought Development Roadmap**

**Phase 1: Foundation CoT (Q2 2025)**
- 📝 **Basic CoT Training Data** - Curate step-by-step reasoning datasets
- 📝 **CoT Prompt Engineering** - Design effective reasoning prompts
- 📝 **Simple Math CoT** - Basic arithmetic with explicit steps
- 📝 **CoT Evaluation Framework** - Metrics for reasoning quality assessment
- 📝 **Reasoning Template System** - Standardized reasoning patterns

**Phase 2: Advanced Reasoning (Q3 2025)**
- 📝 **Multi-Step Problem Solving** - Complex mathematical reasoning
- 📝 **Logical Inference** - Deductive and inductive reasoning training
- 📝 **Causal Reasoning** - Understanding cause-and-effect relationships
- 📝 **Analogical Reasoning** - Pattern recognition and analogy application
- 📝 **Self-Consistency Training** - Multiple reasoning path consistency

**Phase 3: Specialized Reasoning (Q4 2025)**
- 📝 **Scientific Reasoning** - Physics, chemistry, biology problem solving
- 📝 **Programming Logic** - Code generation with reasoning steps
- 📝 **Legal Reasoning** - Case analysis and legal argumentation
- 📝 **Common Sense Reasoning** - Everyday knowledge application
- 📝 **Abstract Reasoning** - Pattern completion and logical puzzles

**Phase 4: Self-Improving CoT (Q1 2026)**
- 📝 **Self-Correction Mechanisms** - Detecting and fixing reasoning errors
- 📝 **Confidence Estimation** - Assessing reasoning quality and certainty
- 📝 **Dynamic CoT Generation** - Adaptive reasoning depth based on complexity
- 📝 **Meta-Reasoning** - Reasoning about reasoning processes
- 📝 **Reasoning Path Optimization** - Finding most efficient solution paths

**Phase 5: Advanced CoT Applications (Q2 2026)**
- 📝 **Multi-Modal CoT** - Reasoning with images, diagrams, and text
- 📝 **Collaborative Reasoning** - Multi-agent reasoning systems
- 📝 **Real-Time CoT** - Interactive step-by-step problem solving
- 📝 **Domain-Specific CoT** - Specialized reasoning for specific fields
- 📝 **CoT Explainability** - Human-interpretable reasoning explanations

##### 🛠️ **Technical Requirements for CoT**

**Training Infrastructure:**
- 📝 **CoT Dataset Creation** - Large-scale step-by-step reasoning data
- 📝 **Reasoning Annotation Tools** - Human annotation for reasoning quality
- 📝 **Multi-Turn Training** - Extended sequence modeling for reasoning chains
- 📝 **Curriculum Learning** - Progressive difficulty in reasoning tasks
- 📝 **Reinforcement Learning** - Reward models for reasoning quality

**Architecture Enhancements:**
- 📝 **Extended Context Windows** - Support for long reasoning sequences
- 📝 **Reasoning Memory** - Maintain reasoning state across steps
- 📝 **Attention Mechanisms** - Focus on relevant reasoning components
- 📝 **Hierarchical Planning** - High-level to low-level reasoning decomposition
- 📝 **Reasoning State Tracking** - Monitor progress through problem-solving

**Data Sources & Benchmarks:**
- 📝 **GSM8K** - Grade school math word problems
- 📝 **MATH Dataset** - Competition-level mathematics
- 📝 **StrategyQA** - Multi-step reasoning questions
- 📝 **LogiQA** - Logical reasoning benchmarks
- 📝 **BigBench CoT** - Diverse reasoning task evaluation
- 📝 **Custom CoT Datasets** - Domain-specific reasoning problems

**Evaluation Metrics:**
- 📝 **Reasoning Accuracy** - Correctness of final answers
- 📝 **Step Quality** - Validity of intermediate reasoning steps
- 📝 **Coherence Metrics** - Logical flow of reasoning chains
- 📝 **Efficiency Measures** - Reasoning path length and optimality
- 📝 **Human Evaluation** - Expert assessment of reasoning quality

**CoT Training Techniques:**
- 📝 **Few-Shot CoT** - In-context learning with reasoning examples
- 📝 **Zero-Shot CoT** - "Let's think step by step" prompting
- 📝 **Self-Consistency** - Multiple reasoning paths for robustness
- 📝 **Tree of Thoughts** - Exploring multiple reasoning branches
- 📝 **Program-Aided Language Models** - Code execution for precise computation

**Integration Capabilities:**
- 📝 **CoT APIs** - RESTful endpoints for reasoning services
- 📝 **Interactive CoT** - Step-by-step user interaction
- 📝 **CoT Visualization** - Graphical reasoning flow display
- 📝 **Reasoning Export** - Save and share reasoning processes
- 📝 **CoT Fine-Tuning** - Domain-specific reasoning adaptation

### 🎯 **Priority Milestones**

#### **v0.2.0 - Production Ready**
- Enhanced model quality and stability
- Comprehensive testing and CI/CD
- Docker containerization
- Performance optimizations

#### **v0.3.0 - Advanced Training**
- Fine-tuning capabilities
- Multi-language support
- Distributed training
- Advanced evaluation metrics
- Basic Chain of Thought reasoning

#### **v0.4.0 - Multi-Modal Foundation**
- Vision encoder integration (CLIP-style)
- Image-text preprocessing pipeline
- Basic vision-language models
- Image captioning and VQA
- Mathematical CoT reasoning

#### **v0.5.0 - Multi-Modal & Advanced CoT**
- Document understanding and OCR
- Advanced multi-modal reasoning
- Cross-modal attention mechanisms
- Logical and causal reasoning
- Self-consistency training

#### **v0.6.0 - Specialized Reasoning**
- Scientific and programming logic
- Legal and common sense reasoning
- Abstract reasoning capabilities
- Production multi-modal inference
- CoT evaluation frameworks

#### **v1.0.0 - Enterprise Ready**
- RLHF and instruction tuning
- Multi-modal chat and generation
- Self-correcting CoT systems
- Production-grade inference
- Enterprise dashboard
- Professional support services

#### **v1.5.0 - Advanced Multi-Modal & CoT**
- Text-to-image generation
- Video and audio processing
- Multi-modal Chain of Thought
- Real-time reasoning applications
- 3D understanding capabilities

#### **v2.0.0 - AI Reasoning Platform**
- Collaborative reasoning systems
- Meta-reasoning capabilities
- Domain-specific CoT adaptation
- Advanced explainability
- Real-time interactive reasoning

### 🤝 **How to Contribute**

We welcome contributions to any of these areas! Here's how you can help:

- **🐛 Bug Fixes** - Report and fix issues in existing features
- **📝 Documentation** - Improve guides, tutorials, and API docs
- **🔬 Research** - Experiment with new architectures and training methods
- **🚀 Features** - Implement items from our planned features list
- **🧪 Testing** - Add tests and improve code quality
- **💼 Enterprise** - Contribute to commercial-licensed features

See our [Contributing Guide](docs/CONTRIBUTING.md) for detailed instructions!

## 🤝 Contributing

We welcome contributions from the community! Please read our:
- [Contributing Guide](docs/CONTRIBUTING.md) - How to contribute to OpenLLM
- [Code of Conduct](docs/CODE_OF_CONDUCT.md) - Community guidelines and standards

For questions or support, feel free to:
- 📝 Open an [issue](https://github.com/louischua/openllm/issues)
- 💬 Start a [discussion](https://github.com/louischua/openllm/discussions)
- 📧 Email us at [louischua@gmail.com]
