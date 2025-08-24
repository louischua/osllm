---
language:
- en
license:
- gpl-3.0
- other
tags:
- text-generation
- pytorch
- causal-lm
- openllm
- gpt
- language-model
datasets:
- squad
metrics:
- perplexity
- loss
pipeline_tag: text-generation
model-index:
- name: OpenLLM Small Extended 10k
  results:
  - task:
      type: text-generation
    dataset:
      type: squad
      name: SQUAD
    metrics:
      - type: loss
        value: 5.22
      - type: perplexity
        value: 184.5
---

# OpenLLM Small Extended 10k

This is the OpenLLM small model trained for 10,000 steps on the SQUAD dataset.

## Model Details

- **Model Type**: GPT-style transformer (decoder-only)
- **Training Steps**: 10,000
- **Parameters**: 35.8M
- **Vocabulary Size**: 32,000
- **Context Length**: 1,024 tokens
- **Architecture**: 6 layers, 8 attention heads, 512 embedding dimension

## Training Information

- **Dataset**: SQUAD (Stanford Question Answering Dataset)
- **Training Data**: ~41k Wikipedia passages
- **Tokenizer**: SentencePiece BPE with 32k vocabulary
- **Optimizer**: AdamW
- **Learning Rate**: 3e-4
- **Batch Size**: 4 (with gradient accumulation)

## Performance

- **Final Loss**: ~5.22
- **Inference Speed**: ~8.3 tokens/second (CPU)
- **Memory Usage**: ~143MB for inference

## Usage

### Using the Model

This model uses a custom configuration format and requires the OpenLLM framework to load properly.

```python
# Load using the OpenLLM framework
from core.src.model import GPTModel
import json
import torch

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Create model instance
model = GPTModel(config["model_config"])

# Load trained weights
model.load_state_dict(torch.load("pytorch_model.bin", map_location="cpu"))

# Load tokenizer
import sentencepiece as spm
tokenizer = spm.SentencePieceProcessor()
tokenizer.load("tokenizer.model")

# Generate text
prompt = "The future of artificial intelligence"
tokens = tokenizer.encode(prompt)
inputs = torch.tensor([tokens], dtype=torch.long)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=100,
        temperature=0.7
    )

generated_text = tokenizer.decode(outputs[0].tolist())
print(generated_text)
```

### Using the Custom Loader

```python
from load_hf_model import load_model_and_tokenizer

# Load model using custom loader
model, tokenizer = load_model_and_tokenizer("lemms/openllm-small-extended-10k")

# Generate text
prompt = "The history of machine learning"
tokens = tokenizer.encode(prompt)
inputs = torch.tensor([tokens], dtype=torch.long)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=100,
        temperature=0.7
    )

print(tokenizer.decode(outputs[0].tolist()))
```

## Model Architecture

This model follows the standard GPT architecture:

- **Token Embeddings**: Maps token IDs to dense vectors
- **Positional Embeddings**: Adds position information
- **Transformer Blocks**: 6 layers with multi-head attention and feed-forward networks
- **Layer Normalization**: Pre-norm placement for training stability
- **Output Head**: Linear projection to vocabulary for next-token prediction

## Training Details

The model was trained using:
- **Framework**: PyTorch
- **Hardware**: CPU training with gradient accumulation
- **Regularization**: Dropout (0.1), weight decay
- **Optimization**: AdamW with cosine learning rate scheduling
- **Gradient Clipping**: 1.0

## Limitations

- This is a small model (35.8M parameters) with limited capacity
- Training was done on CPU, which limited the training steps
- Model quality is basic and suitable for educational/research purposes
- Not suitable for production use without further training

## License

This model is dual-licensed:
- **Open Source**: GPLv3 License
- **Commercial**: Commercial License available

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{openllm2024,
  title={OpenLLM: Open Source Large Language Model Framework},
  author={Louis Chua Bean Chong},
  year={2024},
  url={https://github.com/louischua/openllm}
}
```

## Model Card

- **Developed by**: Louis Chua Bean Chong
- **Model type**: Language Model
- **Language(s)**: English
- **License**: GPLv3 / Commercial
- **Finetuned from model**: Trained from scratch
- **Training data**: SQUAD dataset
- **Training procedure**: Supervised learning
- **Evaluation results**: Basic text generation capability

## Related Models

- [lemms/openllm-small-extended-4k](https://huggingface.co/lemms/openllm-small-extended-4k)
- [lemms/openllm-small-extended-6k](https://huggingface.co/lemms/openllm-small-extended-6k)
- [lemms/openllm-small-extended-7k](https://huggingface.co/lemms/openllm-small-extended-7k)
- [lemms/openllm-small-extended-8k](https://huggingface.co/lemms/openllm-small-extended-8k)
- [lemms/openllm-small-extended-9k](https://huggingface.co/lemms/openllm-small-extended-9k)
