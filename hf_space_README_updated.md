---
title: OpenLLM Training Space
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: gpl-3.0
---

# OpenLLM Training Space

This space provides complete training infrastructure for OpenLLM models with real model training functionality.

## Features

- 🎯 **Real Model Training**: Actual PyTorch training with Transformers
- 📊 **Training Monitoring**: Live progress tracking and loss monitoring
- 🔄 **Model Versioning**: Automatic model saving and uploading to HF Hub
- 📈 **Performance Tracking**: Training metrics and completion status
- 🚀 **Gradio 4.44.1**: Latest UI framework with enhanced compatibility

## Complete Training Pipeline

### What Happens When You Click "Start Training":

1. **📥 Model Loading**: Loads the 7k model from `lemms/openllm-small-extended-7k`
2. **📊 Dataset Preparation**: Loads and tokenizes training data from `lemms/openllm-training-data`
3. **⚙️ Training Setup**: Configures PyTorch Trainer with your parameters
4. **🚀 Real Training**: Executes actual model training for specified steps
5. **💾 Save & Upload**: Saves trained model and uploads to HF Hub as `lemms/openllm-{size}-extended-8k`

### Training Configuration Options:

- **Model Size**: small, medium, large (currently supports small)
- **Max Steps**: 100-10,000 training iterations
- **Learning Rate**: 0.00001-0.001 (configurable)
- **Batch Size**: 1-16 samples per batch

### Expected Results:

- **Training Time**: 10-30 minutes for 1000 steps (depending on HF Space resources)
- **Output Model**: `lemms/openllm-small-extended-8k` (or other sizes)
- **Model Files**: Complete PyTorch model with tokenizer and configuration

## Model Repositories

- [📚 7k Small Model](https://huggingface.co/lemms/openllm-small-extended-7k)
- [🎯 8k Small Model](https://huggingface.co/lemms/openllm-small-extended-8k)
- [📊 Training Dataset](https://huggingface.co/datasets/lemms/openllm-training-data)

## Technical Details

- **Framework**: PyTorch with Transformers
- **UI**: Gradio 4.44.1 (latest stable version)
- **Training**: Mixed precision (FP16) for efficiency
- **Memory**: Optimized for HF Spaces with gradient accumulation
- **Dependencies**: Complete ML stack with all training utilities

## Usage

1. **Configure Parameters**: Set model size, steps, learning rate, and batch size
2. **Start Training**: Click "Start Training" to begin the complete pipeline
3. **Monitor Progress**: Watch real-time status updates and training progress
4. **Access Results**: Find your trained model in the HF Hub repository

## License

GPL-3.0 - See [LICENSE](LICENSE) for details.

## Author

**Louis Chua Bean Chong** - OpenLLM Project Maintainer
