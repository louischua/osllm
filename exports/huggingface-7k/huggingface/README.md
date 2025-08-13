---
language:
- en
license:
- gpl-3.0
- other
tags:
- text-generation
- language-model
- open-source
- gpt
- transformer
- causal-lm
datasets:
- squad
metrics:
- perplexity
- loss
library_name: transformers
pipeline_tag: text-generation
model-index:
- name: OpenLLM Small Extended 7K
  results:
  - task:
      type: text-generation
    dataset:
      type: squad
      name: Wikipedia passages from SQuAD
    metrics:
      - type: loss
        value: 2.1
      - type: perplexity
        value: 8.2
---

# OpenLLM Small Extended 7K Model

<!-- Copyright (C) 2024 Louis Chua Bean Chong -->
<!-- This file is part of OpenLLM - dual-licensed under GPLv3 and Commercial License -->

## 🌟 Model Overview

This is the **OpenLLM Small Extended 7K** model, a 35.8M parameter GPT-style language model trained for 7,000 steps on Wikipedia passages from the SQuAD dataset. This model represents the latest iteration of our small model architecture with extended training.

### **📊 Model Specifications**

- **Architecture**: GPT-style Transformer
- **Parameters**: 35,823,616 (35.8M)
- **Layers**: 6 transformer layers
- **Heads**: 8 attention heads
- **Embedding Dimension**: 512
- **Vocabulary Size**: 32,000 tokens
- **Context Length**: 1,024 tokens
- **Training Steps**: 7,000
- **Model Size**: Small

### **🎯 Training Details**

- **Dataset**: Wikipedia passages from SQuAD dataset (~41k passages)
- **Tokenization**: SentencePiece with 32k vocabulary
- **Training Objective**: Next token prediction (causal language modeling)
- **Optimizer**: AdamW with learning rate scheduling
- **Hardware**: Trained on consumer GPU with gradient accumulation

### **📁 Model Files**

```
huggingface/
├── config.json              # Model configuration
├── generation_config.json   # Generation parameters
├── pytorch_model.bin        # Model weights (161MB)
├── tokenizer_config.json    # Tokenizer configuration
├── tokenizer.model          # SentencePiece tokenizer
└── load_hf_model.py         # Loading script
```

## 🚀 Usage

### **Loading with Hugging Face Transformers**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "path/to/huggingface"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
prompt = "The history of artificial intelligence"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### **Using the Custom Loader**

```python
from load_hf_model import load_openllm_model

# Load the model using our custom loader
model, tokenizer = load_openllm_model("path/to/huggingface")

# Generate text
prompt = "Explain quantum computing in simple terms"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=150,
    temperature=0.8,
    top_p=0.9
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### **Inference Server**

```bash
# Start the FastAPI inference server
python core/src/inference_server.py \
    --model_path exports/huggingface-7k/huggingface \
    --port 8000

# Make API calls
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "The future of renewable energy",
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

## 📈 Performance

### **Training Metrics**

- **Final Loss**: ~2.1 (cross-entropy)
- **Training Time**: ~7 hours on consumer GPU
- **Memory Usage**: ~2GB VRAM during training
- **Inference Speed**: ~50 tokens/second on CPU, ~200 tokens/second on GPU

### **Model Capabilities**

- **Text Generation**: Coherent paragraph generation
- **Question Answering**: Basic factual responses
- **Summarization**: Short text summarization
- **Language Understanding**: Context-aware responses

## 🔧 Configuration

### **Generation Parameters**

```json
{
  "max_length": 512,
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_k": 40,
  "top_p": 0.9,
  "do_sample": true,
  "pad_token_id": 0,
  "eos_token_id": 1,
  "bos_token_id": 2
}
```

### **Model Architecture**

```json
{
  "vocab_size": 32000,
  "n_layer": 6,
  "n_head": 8,
  "n_embd": 512,
  "block_size": 1024,
  "dropout": 0.1,
  "bias": true
}
```

## 🧪 Testing

### **Quick Test**

```python
# Test the model with a simple prompt
test_prompt = "Hello, how are you today?"
inputs = tokenizer(test_prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=20,
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Input: {test_prompt}")
print(f"Output: {response}")
```

## 📋 Limitations

- **Context Length**: Limited to 1,024 tokens
- **Training Data**: Only Wikipedia passages (limited domain)
- **Model Size**: Small model with limited reasoning capabilities
- **Bias**: May inherit biases from training data
- **Factual Accuracy**: Not guaranteed for current events

## 🔄 Model Comparison

| Model | Parameters | Training Steps | Context Length | Use Case |
|-------|------------|----------------|----------------|----------|
| Small 4K | 35.8M | 4,000 | 1,024 | Basic text generation |
| Small 6K | 35.8M | 6,000 | 1,024 | Improved coherence |
| **Small 7K** | **35.8M** | **7,000** | **1,024** | **Extended training** |

## 📄 License

This model is dual-licensed:
- **Open Source**: GNU General Public License v3.0
- **Commercial**: Commercial License (contact for details)

See `LICENSE` and `docs/LICENSES.md` for full license information.

## 🤝 Contributing

We welcome contributions to improve the model! Please see:
- `docs/CONTRIBUTING.md` for contribution guidelines
- `docs/CODE_OF_CONDUCT.md` for community standards

## 📞 Support

For questions, issues, or commercial licensing:
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check `docs/` directory
- **Commercial License**: Contact for enterprise use

---

**Author**: Louis Chua Bean Chong  
**Project**: OpenLLM - Open Source Large Language Model  
**Version**: 0.1.0  
**Last Updated**: 2024
