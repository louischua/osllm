#!/usr/bin/env python3
"""
Export Improved 10k Model to Hugging Face Format

This script exports the new improved 10k model to Hugging Face format
with proper metadata and configuration.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import json
import os
import torch
from pathlib import Path

def main():
    """Export the improved 10k model to Hugging Face format."""
    
    print("🚀 Exporting Improved 10k Model to Hugging Face Format")
    print("=" * 60)
    
    # Paths
    model_dir = Path("models/small-extended-10k-improved")
    export_dir = Path("exports/improved-10k-huggingface")
    
    # Check if model exists
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return False
    
    # Load the final model checkpoint
    final_model_path = model_dir / "final_model.pt"
    if not final_model_path.exists():
        print(f"❌ Final model not found: {final_model_path}")
        return False
    
    print(f"✅ Loading model from: {final_model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(final_model_path, map_location='cpu')
    
    # Extract model state dict
    model_state_dict = checkpoint["model_state_dict"]
    
    # Save model weights as pytorch_model.bin
    model_path = export_dir / "pytorch_model.bin"
    torch.save(model_state_dict, model_path)
    print(f"✅ Model weights saved: {model_path}")
    
    # Create config.json for Hugging Face
    config = {
        "model_config": {
            "model_name": "OpenLLM-Small-10k-Improved",
            "model_size": "small",
            "vocab_size": 32000,
            "n_layer": 6,
            "n_head": 8,
            "n_embd": 512,
            "block_size": 1024,
            "dropout": 0.1,
            "bias": False,
            "training_info": {
                "step": 10000,
                "best_loss": checkpoint["best_loss"],
                "model_type": "gpt-small-improved"
            }
        },
        "tokenizer_config": {
            "type": "sentencepiece",
            "vocab_size": 32000,
            "model_file": "tokenizer.model"
        },
        "training_config": {
            "learning_rate": 3e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_steps": 10000,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "training_time_hours": 21.57,
            "final_perplexity": 177.23,
            "best_validation_loss": checkpoint.get("best_validation_loss", float('inf'))
        }
    }
    
    # Save config
    config_path = export_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved: {config_path}")
    
    # Create README.md with YAML metadata
    readme_content = """---
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
"""
    
    readme_path = export_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✅ README saved: {readme_path}")
    
    # Copy training logs
    training_log_src = model_dir / "training_log.json"
    training_log_dst = export_dir / "training_log.json"
    if training_log_src.exists():
        import shutil
        shutil.copy2(training_log_src, training_log_dst)
        print(f"✅ Training log copied: {training_log_dst}")
    
    # Show file sizes
    print(f"\n📊 Export Summary:")
    for file_path in export_dir.glob("*"):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  {file_path.name}: {size_mb:.1f} MB")
    
    print(f"\n✅ Export completed successfully!")
    print(f"📁 Export directory: {export_dir}")
    print(f"📋 Ready to upload to Hugging Face")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
