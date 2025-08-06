# OpenLLM Core Module

The core module contains the essential open-source components for training and using the OpenLLM language model. This is the heart of the project, providing all the tools needed to build a language model from scratch.

## 🌟 Overview

The core module implements a complete pipeline for:
- **Data Collection**: Download and process high-quality training data from SQUAD dataset
- **Tokenization**: Train SentencePiece tokenizers with BPE or Unigram algorithms  
- **Model Training**: Train GPT-style transformer language models (coming soon)
- **Inference**: Generate text with trained models (coming soon)
- **Evaluation**: Assess model performance on benchmarks (coming soon)

## 📁 Module Structure

```
core/
├── src/                              # Source code
│   ├── main.py                       # Unified CLI entry point
│   ├── download_and_prepare.py       # SQUAD dataset processor
│   └── train_tokenizer.py            # SentencePiece tokenizer trainer
├── LICENSE                           # Module-specific license
└── README.md                         # This documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Prepare Training Data

```bash
# Download and process SQUAD dataset (~5-10 minutes)
python core/src/main.py prepare-data

# This creates: data/clean/training_data.txt (~41k Wikipedia passages)
```

### 3. Train Tokenizer

```bash
# Train a 32k vocabulary BPE tokenizer (~5-15 minutes)
python core/src/main.py train-tokenizer --input data/clean/training_data.txt --vocab-size 32000

# This creates: data/tokenizer/tokenizer.model and supporting files
```

### 4. Unified CLI Interface

```bash
# See all available commands
python core/src/main.py --help

# Get help for specific commands
python core/src/main.py train-tokenizer --help
```

## 📚 Detailed Component Documentation

### Data Preparation (`download_and_prepare.py`)

Downloads and processes the Stanford Question Answering Dataset (SQUAD) to create high-quality training text.

**Features:**
- Downloads SQUAD v1.1 and v2.0 (train + dev splits)
- Extracts Wikipedia context passages from JSON structure
- Cleans and normalizes text while preserving sentence structure
- Filters passages by minimum word count
- Outputs clean text ready for tokenizer training

**Usage:**
```bash
# Direct usage
python core/src/download_and_prepare.py

# Via CLI
python core/src/main.py prepare-data --output data/clean/training_data.txt --min-words 10
```

**Output:**
- `data/clean/training_data.txt` - ~41,202 Wikipedia passages (~100-150MB)

### Tokenizer Training (`train_tokenizer.py`)

Trains SentencePiece tokenizers using BPE (Byte Pair Encoding) or Unigram algorithms.

**Features:**
- Supports BPE and Unigram tokenization algorithms
- Configurable vocabulary sizes (recommended: 8k-64k)
- Proper special token handling (PAD, UNK, BOS, EOS)
- Hugging Face compatible output format
- Built-in testing and validation
- Comprehensive statistics and quality reports

**Usage:**
```bash
# Basic training
python core/src/train_tokenizer.py --input data/clean/training_data.txt --vocab-size 32000

# Advanced configuration
python core/src/train_tokenizer.py \
  --input data/clean/training_data.txt \
  --vocab-size 32000 \
  --model-type bpe \
  --output-dir data/tokenizer/ \
  --character-coverage 0.9995

# Via CLI
python core/src/main.py train-tokenizer --input data/clean/training_data.txt --vocab-size 32000
```

**Output:**
- `data/tokenizer/tokenizer.model` - SentencePiece model file
- `data/tokenizer/tokenizer.vocab` - Human-readable vocabulary
- `data/tokenizer/tokenizer_config.json` - Hugging Face configuration

### Main CLI (`main.py`)

Unified command-line interface that provides access to all OpenLLM functionality through a single entry point.

**Available Commands:**
- `prepare-data` - Download and prepare SQUAD training data
- `train-tokenizer` - Train SentencePiece tokenizers
- `train-model` - Train language models (coming soon)
- `inference` - Generate text with trained models (coming soon)
- `evaluate` - Evaluate model performance (coming soon)

**Features:**
- Consistent argument parsing across all components
- Comprehensive help and documentation
- Error handling and status reporting
- Progress tracking and statistics
- Modular design for easy extension

**Usage:**
```bash
# See all commands
python core/src/main.py --help

# Full pipeline example
python core/src/main.py prepare-data
python core/src/main.py train-tokenizer --input data/clean/training_data.txt --vocab-size 32000

# Get command-specific help
python core/src/main.py train-tokenizer --help
```

## 🛠️ Development

### Code Organization

The core module follows these principles:
- **Modularity**: Each component can be used independently
- **CLI-First**: All functionality accessible via command line
- **Documentation**: Comprehensive docstrings and examples
- **Error Handling**: Graceful failure with helpful messages
- **Testing**: Built-in validation and quality checks

### Adding New Components

To add new functionality to the core module:

1. **Create the module**: Add new `.py` file in `core/src/`
2. **Implement CLI**: Add command parser in `main.py`
3. **Add documentation**: Update this README with usage examples
4. **Test thoroughly**: Include validation and error handling

### Dependencies

Core module dependencies are minimal and focused:
- `requests` - HTTP downloads
- `tqdm` - Progress bars
- `sentencepiece` - Tokenizer training
- Standard library modules only

## 📊 Performance & Scalability

### Data Processing
- **SQUAD Download**: ~200MB, 5-10 minutes
- **Text Processing**: ~41k passages in 2-3 minutes
- **Memory Usage**: Low, streaming processing

### Tokenizer Training
- **Small Vocab (8k)**: ~30 seconds on CPU
- **Medium Vocab (32k)**: ~1-2 minutes on CPU  
- **Large Vocab (64k)**: ~3-5 minutes on CPU
- **Memory Usage**: <1GB RAM for most configurations

### Future Model Training
- **Small Model (25M params)**: 2-4 hours on single GPU
- **Medium Model (117M params)**: 6-12 hours on single GPU
- **Large Model (350M params)**: 12-24 hours on multi-GPU
- **Memory Requirements**: 4-16GB GPU RAM depending on model size

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom data directory
export OPENLLM_DATA_DIR=/path/to/data

# Optional: Set custom cache directory  
export OPENLLM_CACHE_DIR=/path/to/cache
```

### Configuration Files
- Model configs will be stored in `configs/` directory (coming soon)
- Tokenizer configs auto-generated in `data/tokenizer/`
- Training configs will support YAML/JSON formats (coming soon)

## 🧪 Testing

```bash
# Test data preparation
python core/src/main.py prepare-data --output test_data.txt --min-words 5

# Test tokenizer training
python core/src/main.py train-tokenizer --input test_data.txt --vocab-size 1000 --output-dir test_tokenizer/

# Clean up test files
rm test_data.txt
rm -rf test_tokenizer/
```

## 📝 License

The core module is licensed under GPLv3 for open source use. See `LICENSE` file for details.

## 🤝 Contributing

1. Follow the existing code style and documentation patterns
2. Add comprehensive tests for new functionality
3. Update this README with new features
4. Ensure CLI help text is clear and comprehensive
5. Test on multiple platforms (Linux, macOS, Windows)

## 🆘 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure you're in the project root directory
cd /path/to/osllm-1
python core/src/main.py --help
```

**Download Failures:**
```bash
# Check internet connection and retry
python core/src/main.py prepare-data --min-words 5
```

**Out of Memory:**
```bash
# Reduce vocabulary size for tokenizer training
python core/src/main.py train-tokenizer --vocab-size 16000
```

**Permission Errors:**
```bash
# Ensure write access to data directories
mkdir -p data/{raw,clean,tokenizer}
```

### Getting Help

- Check command-specific help: `python core/src/main.py <command> --help`
- Review the training pipeline documentation: `docs/training_pipeline.md`
- Check the main project README: `README.md`
