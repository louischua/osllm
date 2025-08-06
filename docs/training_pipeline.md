# 🚀 OpenLLM Training Pipeline: Complete Guide

This document provides a comprehensive, step-by-step guide for training the OpenLLM model from scratch. This pipeline builds a production-ready language model using only open source data and tools, without relying on any pre-trained models.

## 📋 Pipeline Overview

Our training pipeline consists of these key stages:
1. **Data Collection & Preparation** - Download and clean training text from SQUAD dataset
2. **Tokenizer Training** - Train a SentencePiece BPE tokenizer on the text corpus
3. **Model Architecture** - Define the transformer architecture (GPT-style decoder)
4. **Model Training** - Pre-train the language model on the tokenized text
5. **Evaluation & Validation** - Test model performance and quality
6. **Model Export** - Save final model for inference

**Expected Timeline:** ~2-4 hours for full pipeline (depending on hardware)
**Hardware Requirements:** GPU recommended for model training, CPU sufficient for data prep and tokenization

---

## 🗂️ Step 1: Data Collection & Preparation

**Objective:** Download high-quality training text from the Stanford Question Answering Dataset (SQUAD)

**Why SQUAD?** 
- Contains ~41k high-quality Wikipedia passages across diverse topics
- Well-structured, clean text ideal for language model training
- Publicly available and free to use
- Better quality than many web-scraped corpora

### 1.1 Download and Process SQUAD Dataset

```bash
# Activate your virtual environment first
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\Activate.ps1  # Windows PowerShell

# Install required dependencies for data processing
pip install requests tqdm

# Download and process SQUAD dataset (~200MB download, ~5-10 minutes)
python core/src/download_and_prepare.py
```

**What this does:**
- Downloads 4 JSON files: SQUAD v1.1 & v2.0 (train + dev splits)
- Extracts Wikipedia context passages from the JSON structure
- Cleans text: normalizes whitespace, removes formatting artifacts
- Filters out passages shorter than 10 words
- Outputs a single text file with one passage per line

**Expected Output:**
```
data/clean/training_data.txt    # ~41,202 Wikipedia passages (~100-150MB)
```

### 1.2 Verify Data Quality

```bash
# Check the processed data
wc -l data/clean/training_data.txt      # Should show ~41,202 lines
head -n 5 data/clean/training_data.txt  # Preview first 5 passages
```

**Data Quality Metrics:**
- **Volume:** ~41k passages, ~100-150MB of clean text
- **Diversity:** Wikipedia articles across history, science, culture, etc.
- **Quality:** Hand-curated, encyclopedia-quality writing
- **Language:** Primarily English, formal writing style

---

## 🔤 Step 2: Tokenizer Training

**Objective:** Train a SentencePiece BPE tokenizer that converts text into tokens for model training

**Why SentencePiece BPE?**
- Subword tokenization handles out-of-vocabulary words gracefully
- Works well across multiple languages without pre-tokenization
- Industry standard for modern language models (GPT, BERT, etc.)
- Efficient inference and training performance

### 2.1 Install SentencePiece

```bash
# Install SentencePiece library
pip install sentencepiece
```

### 2.2 Train the Tokenizer

```bash
# Basic tokenizer training (recommended for most use cases)
python core/src/train_tokenizer.py \
  --input data/clean/training_data.txt \
  --vocab_size 32000 \
  --model_type bpe \
  --output_dir data/tokenizer/

# Advanced configuration with custom parameters
python core/src/train_tokenizer.py \
  --input data/clean/training_data.txt \
  --vocab_size 32000 \
  --model_type bpe \
  --output_dir data/tokenizer/ \
  --character_coverage 0.9995 \
  --max_sentence_length 4192
```

**Tokenizer Parameters Explained:**
- `--vocab_size 32000`: Size of vocabulary (32k is good balance of efficiency vs. quality)
- `--model_type bpe`: Byte Pair Encoding algorithm (vs. unigram)
- `--character_coverage 0.9995`: Cover 99.95% of characters (good for English)
- `--max_sentence_length 4192`: Maximum input length in characters

**Expected Output:**
```
data/tokenizer/tokenizer.model         # SentencePiece model file (~1-2MB)
data/tokenizer/tokenizer.vocab         # Human-readable vocabulary (~1MB)  
data/tokenizer/tokenizer_config.json   # Hugging Face compatible config
```

### 2.3 Test Your Tokenizer

The training script automatically tests the tokenizer, but you can test it manually:

```python
import sentencepiece as spm

# Load the trained tokenizer
sp = spm.SentencePieceProcessor()
sp.load('data/tokenizer/tokenizer.model')

# Test tokenization
text = "Hello world! This is a test."
tokens = sp.encode(text, out_type=str)
token_ids = sp.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")
print(f"Decoded: {sp.decode(token_ids)}")
```

**Quality Checks:**
- Vocabulary should be exactly 32,000 tokens
- Common words should be single tokens ("the", "and", "is")
- Rare words should be split into subwords ("unusual" → ["un", "usual"])
- Round-trip encoding/decoding should preserve original text

---

## 🏗️ Step 3: Model Architecture Design

**Objective:** Define the transformer architecture for our language model

**Architecture Choice:** GPT-style decoder-only transformer
- **Why?** Proven architecture for autoregressive language modeling
- **Benefits:** Simple, scalable, well-understood, good performance

### 3.1 Model Specifications

```python
# Recommended model configurations for different scales

# Small Model (Laptop/CPU Training)
{
    "vocab_size": 32000,
    "d_model": 512,           # Hidden dimension
    "n_layers": 6,            # Number of transformer layers  
    "n_heads": 8,             # Number of attention heads
    "d_ff": 2048,             # Feed-forward dimension
    "max_seq_len": 1024,      # Maximum sequence length
    "dropout": 0.1,           # Dropout rate
    "parameters": "~25M"      # Approximate parameter count
}

# Medium Model (GPU Training)
{
    "vocab_size": 32000,
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "d_ff": 3072,
    "max_seq_len": 2048,
    "dropout": 0.1,
    "parameters": "~117M"
}

# Large Model (Multi-GPU/High-end)
{
    "vocab_size": 32000,
    "d_model": 1024,
    "n_layers": 24,
    "n_heads": 16,
    "d_ff": 4096,
    "max_seq_len": 2048,
    "dropout": 0.1,
    "parameters": "~350M"
}
```

### 3.2 Implementation Notes

**Model Components:**
- **Token Embedding:** Maps token IDs to dense vectors
- **Positional Encoding:** Adds position information to embeddings
- **Transformer Layers:** Multi-head attention + feed-forward networks
- **Layer Normalization:** Stabilizes training
- **Output Head:** Projects to vocabulary for next-token prediction

**Implementation Framework:** PyTorch + Hugging Face Transformers
- Leverages optimized attention implementations
- Compatible with distributed training
- Easy integration with existing tools

---

## 🎯 Step 4: Model Training

**Objective:** Train the language model using distributed training and modern optimization techniques

**Training Strategy:** Autoregressive language modeling (predict next token)

### 4.1 Training Configuration

```bash
# Recommended training parameters
{
    "batch_size": 32,              # Adjust based on GPU memory
    "learning_rate": 1e-4,         # Conservative starting point
    "warmup_steps": 10000,         # Learning rate warmup
    "max_steps": 100000,           # Total training steps
    "gradient_clipping": 1.0,      # Prevent gradient explosion
    "weight_decay": 0.01,          # L2 regularization
    "optimizer": "AdamW",          # Modern optimizer choice
    "lr_scheduler": "cosine",      # Learning rate decay
    "mixed_precision": True,       # FP16 for efficiency
    "gradient_checkpointing": True # Save memory
}
```

### 4.2 Training Command (To Be Implemented)

```bash
# This is the planned interface for model training
python core/src/train_model.py \
  --tokenizer_dir data/tokenizer/ \
  --data_file data/clean/training_data.txt \
  --model_config configs/medium_model.json \
  --output_dir models/openllm-medium \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --max_steps 100000 \
  --save_every 10000 \
  --eval_every 5000
```

### 4.3 Training Monitoring

**Key Metrics to Track:**
- **Loss:** Should decrease steadily (target: <2.5 for good models)
- **Perplexity:** Exp(loss), should decrease (target: <12 for good models)
- **Throughput:** Tokens/second processed
- **Memory Usage:** GPU utilization and VRAM consumption

**Training Progress:**
- **Phase 1 (0-20k steps):** Initial learning, loss drops rapidly
- **Phase 2 (20k-50k steps):** Steady improvement, loss plateaus
- **Phase 3 (50k+ steps):** Fine-tuning, diminishing returns

---

## 📊 Step 5: Evaluation & Validation

**Objective:** Assess model quality and performance across different tasks

### 5.1 Intrinsic Evaluation

**Perplexity on Held-out Data:**
```bash
# Evaluate on validation set
python core/src/evaluate_model.py \
  --model_dir models/openllm-medium \
  --eval_data data/clean/validation_data.txt \
  --metrics perplexity,loss
```

**Text Generation Quality:**
```bash
# Generate sample text to assess quality
python core/src/generate_text.py \
  --model_dir models/openllm-medium \
  --prompt "The history of artificial intelligence" \
  --max_length 256 \
  --temperature 0.7
```

### 5.2 Downstream Task Evaluation

**Reading Comprehension (SQUAD):**
- Fine-tune on SQUAD question-answering
- Measure F1 score and exact match accuracy

**Text Classification:**
- Test on sentiment analysis benchmarks
- Evaluate few-shot learning capabilities

**Common Sense Reasoning:**
- Evaluate on multiple choice questions
- Test logical reasoning abilities

### 5.3 Quality Benchmarks

**Good Model Indicators:**
- Perplexity < 12 on held-out Wikipedia text
- Coherent text generation for 100+ tokens
- Basic grammatical correctness
- Factual consistency within generated text

---

## 📦 Step 6: Model Export & Deployment

**Objective:** Prepare the trained model for production inference

### 6.1 Model Formats

**PyTorch Native:**
```bash
# Save PyTorch checkpoint
python core/src/export_model.py \
  --model_dir models/openllm-medium \
  --format pytorch \
  --output_dir exports/pytorch/
```

**Hugging Face Compatible:**
```bash
# Export to Hugging Face format
python core/src/export_model.py \
  --model_dir models/openllm-medium \
  --format huggingface \
  --output_dir exports/huggingface/
```

**ONNX (Production Inference):**
```bash
# Export to ONNX for optimized inference
python core/src/export_model.py \
  --model_dir models/openllm-medium \
  --format onnx \
  --output_dir exports/onnx/ \
  --optimize_for_inference
```

### 6.2 Inference API Setup

```bash
# Start REST API server
python core/src/inference_server.py \
  --model_path exports/huggingface/ \
  --host 0.0.0.0 \
  --port 8000 \
  --max_length 512
```

---

## 🛠️ Troubleshooting & Best Practices

### Common Issues

**Out of Memory Errors:**
- Reduce batch size: `--batch_size 16` → `--batch_size 8`
- Enable gradient checkpointing: `--gradient_checkpointing`
- Use gradient accumulation: `--accumulate_grad_batches 4`

**Slow Training:**
- Enable mixed precision: `--mixed_precision`
- Increase batch size if GPU memory allows
- Use multiple GPUs with distributed training

**Poor Model Quality:**
- Increase model size (more layers/dimensions)
- Train for more steps
- Verify data quality and preprocessing
- Adjust learning rate (try 5e-5 or 2e-4)

### Performance Optimization

**Hardware Recommendations:**
- **Minimum:** 8GB GPU (RTX 3070, GTX 1080 Ti)
- **Recommended:** 16GB+ GPU (RTX 4080, A4000)
- **Optimal:** Multi-GPU setup (A100, H100)

**Training Speed Tips:**
- Use larger batch sizes when possible
- Enable compilation: `torch.compile()` in PyTorch 2.0+
- Use efficient data loading with multiple workers
- Consider gradient accumulation for large effective batch sizes

### Data Quality Best Practices

**Training Data Guidelines:**
- Aim for 10M+ tokens for small models, 100M+ for larger models
- Ensure diverse, high-quality text sources
- Remove duplicates and low-quality content
- Balance different domains and writing styles

**Tokenizer Best Practices:**
- Use 32k vocabulary for most use cases
- Test on representative text samples
- Verify special token handling (BOS, EOS, PAD, UNK)
- Consider domain-specific vocabulary for specialized models

---

## 📈 Expected Results & Timeline

### Training Timeline
- **Data Preparation:** 10-30 minutes
- **Tokenizer Training:** 5-15 minutes  
- **Model Training:** 2-24 hours (depending on model size and hardware)
- **Evaluation:** 30-60 minutes
- **Export:** 5-15 minutes

### Performance Targets

**Small Model (25M parameters):**
- Perplexity: 15-25
- Training time: 2-4 hours on single GPU
- Memory: 4-8GB GPU RAM

**Medium Model (117M parameters):**
- Perplexity: 10-15
- Training time: 6-12 hours on single GPU
- Memory: 8-16GB GPU RAM

**Large Model (350M parameters):**
- Perplexity: 8-12
- Training time: 12-24 hours on multi-GPU
- Memory: 16GB+ GPU RAM

---

## 🔄 Next Steps

After completing the basic training pipeline:

1. **Fine-tuning:** Adapt the model for specific tasks (Q&A, summarization, etc.)
2. **RLHF:** Implement reinforcement learning from human feedback
3. **Scaling:** Train larger models on more data
4. **Optimization:** Quantization, pruning, distillation
5. **Deployment:** Production serving with load balancing and monitoring

This pipeline provides a solid foundation for building and training your own language model from scratch using only open source tools and data.