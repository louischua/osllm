# Hugging Face Hub Training Guide

<!-- Copyright (C) 2024 Louis Chua Bean Chong -->
<!-- This file is part of OpenLLM - dual-licensed under GPLv3 and Commercial License -->

## 🌟 Overview

This guide explains how to train the OpenLLM model on Hugging Face Hub's training infrastructure. This allows you to leverage HF Hub's compute resources and training features without needing local GPU hardware.

## 🎯 Training Objective

Continue training the OpenLLM Small Extended 7K model for an additional 1000 steps to create an 8K model:

- **Starting Point**: 7,000 training steps (existing model)
- **Target**: 8,000 training steps (new model)
- **Additional Steps**: 1,000 steps
- **Expected Duration**: 1-2 hours on HF Hub

## 🚀 Setup Instructions

### **Step 1: Prepare Your Repository**

1. **Fork or Clone** the OpenLLM repository to your Hugging Face account
2. **Upload Training Data** to HF Hub as a dataset
3. **Ensure Model Access** to the 7k model: `lemms/openllm-small-extended-7k`

### **Step 2: Configure Training Job**

#### **Training Configuration**

```python
# Training parameters
model_size = "small"
data_file = "/workspace/data/clean/training_data.txt"
tokenizer_dir = "/workspace/data/tokenizer/"
input_model_dir = "/workspace/models/small-extended-7k"
output_dir = "/workspace/models/small-extended-8k"
max_steps = 1000  # Additional steps
```

#### **HF Hub Training Arguments**

```python
TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=0,  # No warmup for continued training
    logging_steps=50,
    save_steps=200,
    eval_steps=100,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id="lemms/openllm-small-extended-8k",
    hub_strategy="end",
    report_to=["tensorboard"],
    seed=42,
    max_grad_norm=1.0,
)
```

### **Step 3: Launch Training Job**

#### **Option A: Using HF Hub Web Interface**

1. **Go to** [Hugging Face Training Hub](https://huggingface.co/training)
2. **Select** "New Training Job" or "Start Training"
3. **Configure**:
   - **Repository**: Your forked OpenLLM repo
   - **Script**: `hf_training_script.py`
   - **Model**: `lemms/openllm-small-extended-7k`
   - **Dataset**: Your training data
   - **Compute**: CPU (recommended) or GPU
   - **Duration**: 2 hours

**Alternative**: Go to your repository on HF Hub and look for the "Train" button.

#### **Option B: Using HF CLI**

```bash
# Install HF CLI
pip install huggingface_hub

# Login to HF Hub
huggingface-cli login

# Launch training job
huggingface-cli launch-training \
    --repo-id your-username/openllm \
    --script hf_training_script.py \
    --compute cpu \
    --duration 2h
```

#### **Option C: Using HF Hub API**

```python
from huggingface_hub import HfApi

api = HfApi()

# Create training job
api.create_training_job(
    repo_id="your-username/openllm",
    script="hf_training_script.py",
    compute="cpu",
    duration="2h",
    model="lemms/openllm-small-extended-7k"
)
```

## 📊 Training Process

### **What Happens During Training**

1. **Model Loading**: Loads the 7k model checkpoint
2. **Data Preparation**: Processes training data
3. **Continued Training**: Trains for 1000 additional steps
4. **Checkpointing**: Saves progress every 200 steps
5. **Evaluation**: Evaluates every 100 steps
6. **Model Upload**: Pushes final model to HF Hub

### **Monitoring Training**

- **Tensorboard**: Real-time training metrics
- **HF Hub Dashboard**: Training progress and logs
- **Email Notifications**: Job completion alerts

### **Expected Output**

```
🚀 Starting OpenLLM training on Hugging Face Hub
📥 Loading existing model from /workspace/models/small-extended-7k
✅ Loaded 7k model checkpoint
📂 Loading training data...
🔤 Tokenizing dataset...
🔥 Starting training...
Step 100/1000: loss=2.05, lr=3.00e-04
Step 200/1000: loss=2.03, lr=3.00e-04
...
Step 1000/1000: loss=1.98, lr=3.00e-04
✅ Training completed!
📤 Pushing model to lemms/openllm-small-extended-8k
```

## 🔧 Configuration Files

### **Required Files**

1. **`hf_training_script.py`** - Main training script
2. **`requirements-hf-training.txt`** - Dependencies
3. **`core/src/`** - Model architecture and data loader
4. **Training data** - Uploaded to HF Hub as dataset

### **Optional Files**

1. **`hf_training_config.py`** - Configuration utilities
2. **`docs/hf_training_guide.md`** - This guide

## 📈 Expected Results

### **Model Improvements**

- **Loss Reduction**: From ~2.1 to ~1.98
- **Better Coherence**: Improved text generation quality
- **Enhanced Understanding**: Better context awareness

### **Model Specifications**

- **Architecture**: GPT-style Transformer
- **Parameters**: 35,823,616 (35.8M)
- **Layers**: 6 transformer layers
- **Heads**: 8 attention heads
- **Training Steps**: 8,000 (7k + 1k additional)
- **Context Length**: 1,024 tokens

## 🛠️ Troubleshooting

### **Common Issues**

1. **Out of Memory**
   - Reduce batch size: `per_device_train_batch_size=2`
   - Increase gradient accumulation: `gradient_accumulation_steps=8`

2. **Training Too Slow**
   - Use GPU compute instead of CPU
   - Reduce evaluation frequency: `eval_steps=200`

3. **Model Not Loading**
   - Check model path: `/workspace/models/small-extended-7k`
   - Verify checkpoint file: `best_model.pt`

4. **Data Loading Issues**
   - Check data file path
   - Verify tokenizer directory
   - Ensure dataset format compatibility

### **Debug Commands**

```bash
# Check available files
ls -la /workspace/

# Verify model checkpoint
python -c "import torch; ckpt=torch.load('/workspace/models/small-extended-7k/best_model.pt'); print('Checkpoint loaded successfully')"

# Test data loading
python -c "from data_loader import TextDataLoader; dl=TextDataLoader('data/clean/training_data.txt', 'data/tokenizer/'); print('Data loader initialized')"
```

## 📋 Post-Training Steps

### **Model Verification**

1. **Download** the trained model from HF Hub
2. **Test** generation capabilities
3. **Compare** with 7k model performance
4. **Update** documentation

### **Model Distribution**

1. **Export** to Hugging Face format
2. **Push** to model repository
3. **Update** README with 8k information
4. **Share** with community

### **Documentation Updates**

1. **Update** main README.md
2. **Add** 8k model to comparison table
3. **Create** usage examples
4. **Update** performance metrics

## 💡 Best Practices

### **Training Optimization**

- **Monitor** loss curves for overfitting
- **Adjust** learning rate if needed
- **Use** early stopping for efficiency
- **Save** checkpoints regularly

### **Resource Management**

- **Choose** appropriate compute resources
- **Monitor** memory usage
- **Optimize** batch sizes
- **Use** gradient accumulation

### **Quality Assurance**

- **Validate** model outputs
- **Compare** with previous versions
- **Test** on diverse prompts
- **Document** any issues

## 🔗 Useful Links

- **[Hugging Face Training Hub](https://huggingface.co/training)** - Training platform
- **[HF Hub Documentation](https://huggingface.co/docs/hub/training)** - Official docs
- **[HF Hub Training Guide](https://huggingface.co/docs/hub/training)** - Training tutorials
- **[OpenLLM Repository](https://github.com/louischua/openllm)** - Source code
- **[7k Model](https://huggingface.co/lemms/openllm-small-extended-7k)** - Starting model

## 📞 Support

For issues with HF Hub training:

- **HF Hub Issues**: Report on HF Hub platform
- **GitHub Issues**: Open issue in OpenLLM repo
- **Email**: louischua@gmail.com

---

**Author**: Louis Chua Bean Chong  
**Project**: OpenLLM - Open Source Large Language Model  
**Version**: 0.1.0  
**Last Updated**: 2024
