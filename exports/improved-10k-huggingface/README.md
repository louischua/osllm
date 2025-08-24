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
- name: OpenLLM Small Extended 10k Improved
  results:
  - task:
      type: text-generation
    dataset:
      type: squad
      name: SQUAD
    metrics:
      - type: loss
        value: 5.1774
      - type: perplexity
        value: 177.23
---

# OpenLLM Small Extended 10k Improved

This is an improved version of the OpenLLM Small model trained for 10,000 steps using the enhanced training process with proper checkpoint saving and validation monitoring.

## Model Details

- **Model Type**: GPT-style language model
- **Architecture**: Transformer decoder-only
- **Parameters**: 35.8M
- **Training Steps**: 10,000 (resumed from 9k model)
- **Training Time**: 21.57 hours
- **Final Loss**: 5.1774
- **Final Perplexity**: 177.23
- **Best Validation Loss**: 5.4179

## Training Process

This model was trained using the improved training process that includes:
- ✅ Proper checkpoint saving with full metadata
- ✅ Best checkpoint tracking
- ✅ Validation monitoring
- ✅ Early stopping mechanism
- ✅ Complete training logs

## Usage

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

## Training Configuration

- **Learning Rate**: 3e-4
- **Batch Size**: 4
- **Gradient Accumulation Steps**: 4
- **Max Steps**: 10,000
- **Warmup Steps**: 100
- **Weight Decay**: 0.01
- **Sequence Length**: 512

## Model Performance

This improved 10k model maintains the same high performance as the 9k model while having proper checkpoint format and complete training metadata.

## License

This model is licensed under the GNU General Public License v3.0.
