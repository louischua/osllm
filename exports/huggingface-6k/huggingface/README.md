---
language:
- en
license:
- gpl-3.0
- other
tags:
- text-generation
- language-model
- gpt
- transformer
- open-source
- squad
- wikipedia
datasets:
- squad
metrics:
- perplexity
- text-generation-quality
library_name: transformers
pipeline_tag: text-generation
model-index:
- name: OpenLLM Small Extended 6k
  results:
  - task:
      type: text-generation
    dataset:
      type: squad
      name: SQUAD Wikipedia Passages
    metrics:
      - type: perplexity
        value: 816.04
      - type: training_loss
        value: 5.4302
---

# OpenLLM Small Extended 6k

This is the OpenLLM Small Extended model trained for 6,000 steps on Wikipedia passages from the SQUAD dataset.

## Model Details

- **Model Type:** GPT-style Transformer
- **Architecture:** Small (35.8M parameters)
- **Training Steps:** 6,000
- **Training Data:** ~41k Wikipedia passages from SQUAD dataset
- **Tokenizer:** SentencePiece BPE (32k vocabulary)
- **License:** GPL-3.0 (Open Source) / Commercial License available

## Model Performance

- **Final Training Loss:** 5.4302
- **Model Parameters:** 35,823,616
- **Context Length:** 512 tokens
- **Training Hardware:** CPU/GPU compatible

## Usage

### Using Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "lemms/openllm-small-extended-6k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
prompt = "The history of artificial intelligence"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=50,
        temperature=0.7,
        top_k=40,
        do_sample=True
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Using the Custom Loader

```python
# Use the provided load_hf_model.py script
from load_hf_model import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer()
# ... rest of usage
```

## Training Details

This model was trained using the OpenLLM training pipeline:

1. **Data Preparation:** SQUAD dataset processing (~41k passages)
2. **Tokenizer Training:** SentencePiece BPE with 32k vocabulary
3. **Model Training:** GPT-style transformer for 6,000 steps
4. **Evaluation:** Perplexity and text generation quality assessment

## Model Architecture

- **Layers:** 12 transformer layers
- **Attention Heads:** 12
- **Hidden Size:** 768
- **Intermediate Size:** 3072
- **Activation:** GELU
- **Layer Norm:** Pre-norm

## Limitations

- **Training Data:** Limited to Wikipedia passages
- **Context Length:** 512 tokens maximum
- **Model Size:** Small model with 35.8M parameters
- **Performance:** Basic text generation capabilities

## License

This model is dual-licensed:
- **Open Source:** GPL-3.0 for research and community use
- **Commercial:** Commercial license available for enterprise use

For commercial licensing, contact: louischua@gmail.com

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{openllm2024,
  title={OpenLLM: Open Source Large Language Model},
  author={Louis Chua Bean Chong},
  year={2024},
  url={https://github.com/louischua/openllm}
}
```

## Links

- **Repository:** https://github.com/louischua/openllm
- **Documentation:** https://github.com/louischua/openllm/docs
- **Training Pipeline:** https://github.com/louischua/openllm/docs/training_pipeline.md
