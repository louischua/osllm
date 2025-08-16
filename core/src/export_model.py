#!/usr/bin/env python3
# Copyright (C) 2024 Louis Chua Bean Chong
#
# This file is part of OpenLLM.
#
# OpenLLM is dual-licensed:
# 1. For open source use: GNU General Public License v3.0
# 2. For commercial use: Commercial License (contact for details)
#
# See LICENSE and docs/LICENSES.md for full license information.

"""
OpenLLM Model Export Script

This script implements Step 6 of the training pipeline: Model Export & Deployment.
It exports trained OpenLLM models to various formats for production inference.

Supported Formats:
- PyTorch native format (for Python inference)
- Hugging Face format (for ecosystem compatibility)
- ONNX format (for optimized cross-platform inference)

Usage:
    # PyTorch format
    python core/src/export_model.py \
        --model_dir models/small-extended-4k \
        --format pytorch \
        --output_dir exports/pytorch/

    # Hugging Face format
    python core/src/export_model.py \
        --model_dir models/small-extended-4k \
        --format huggingface \
        --output_dir exports/huggingface/

    # ONNX format
    python core/src/export_model.py \
        --model_dir models/small-extended-4k \
        --format onnx \
        --output_dir exports/onnx/ \
        --optimize_for_inference

Author: Louis Chua Bean Chong
License: GPLv3
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import sentencepiece as spm
import torch
import torch.nn as nn

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPTConfig, GPTModel, create_model


class ModelExporter:
    """
    Comprehensive model exporter for OpenLLM models.

    Handles export to multiple formats including PyTorch, Hugging Face,
    and ONNX for different deployment scenarios.
    """

    def __init__(self, model_dir: str, output_dir: str):
        """
        Initialize the model exporter.

        Args:
            model_dir: Directory containing trained model checkpoints
            output_dir: Base directory for exported models
        """
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model and metadata
        self.model, self.config, self.training_info = self._load_model()
        self.tokenizer_path = self._find_tokenizer()

        print(f"🔧 ModelExporter initialized")
        print(f"  Model: {self.config.model_name}")
        print(f"  Parameters: {self.model.get_num_params():,}")
        print(f"  Output directory: {output_dir}")

    def _load_model(self):
        """Load model from checkpoint directory."""
        # Find best model checkpoint
        best_model_path = self.model_dir / "best_model.pt"
        if not best_model_path.exists():
            # Look for latest checkpoint
            checkpoints = list(self.model_dir.glob("checkpoint_step_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"No model checkpoints found in {self.model_dir}")

            # Get latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
            best_model_path = latest_checkpoint

        print(f"📂 Loading model from {best_model_path}")

        # Load checkpoint
        checkpoint = torch.load(best_model_path, map_location="cpu")

        # Determine model size from config
        config_dict = checkpoint.get("config", {})
        n_layer = config_dict.get("n_layer", 12)

        if n_layer <= 6:
            model_size = "small"
        elif n_layer <= 12:
            model_size = "medium"
        else:
            model_size = "large"

        # Create and load model
        model = create_model(model_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()  # Set to evaluation mode

        # Extract training info
        training_info = {
            "step": checkpoint.get("step", 0),
            "best_loss": checkpoint.get("best_loss", 0.0),
            "model_size": model_size,
        }

        return model, model.config, training_info

    def _find_tokenizer(self):
        """Find tokenizer path."""
        # Try multiple possible locations
        possible_paths = [
            self.model_dir.parent / "tokenizer" / "tokenizer.model",
            Path("data/tokenizer/tokenizer.model"),
            self.model_dir / "tokenizer.model",
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        raise FileNotFoundError("Tokenizer not found in expected locations")

    def export_pytorch(self) -> str:
        """
        Export model in PyTorch native format.

        Returns:
            Path to exported model directory
        """
        output_path = self.output_dir / "pytorch"
        output_path.mkdir(parents=True, exist_ok=True)

        print("🔄 Exporting to PyTorch format...")

        # Save model state dict
        model_path = output_path / "model.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.__dict__,
                "training_info": self.training_info,
            },
            model_path,
        )

        # Save configuration
        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "model_config": self.config.__dict__,
                    "training_info": self.training_info,
                    "export_format": "pytorch",
                },
                f,
                indent=2,
            )

        # Copy tokenizer
        tokenizer_out = output_path / "tokenizer.model"
        shutil.copy2(self.tokenizer_path, tokenizer_out)

        # Create loading script
        self._create_pytorch_loader(output_path)

        print(f"✅ PyTorch export completed: {output_path}")
        return str(output_path)

    def export_huggingface(self) -> str:
        """
        Export model in Hugging Face compatible format.

        Returns:
            Path to exported model directory
        """
        output_path = self.output_dir / "huggingface"
        output_path.mkdir(parents=True, exist_ok=True)

        print("🔄 Exporting to Hugging Face format...")

        # Save model weights in HF format
        model_path = output_path / "pytorch_model.bin"
        torch.save(self.model.state_dict(), model_path)

        # Create HF-compatible config
        hf_config = {
            "architectures": ["GPTModel"],
            "model_type": "gpt",
            "vocab_size": self.config.vocab_size,
            "n_layer": self.config.n_layer,
            "n_head": self.config.n_head,
            "n_embd": self.config.n_embd,
            "block_size": self.config.block_size,
            "dropout": self.config.dropout,
            "bias": self.config.bias,
            "torch_dtype": "float32",
            "transformers_version": "4.0.0",
            "openllm_version": "0.1.0",
            "training_steps": self.training_info["step"],
            "model_size": self.training_info["model_size"],
        }

        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(hf_config, f, indent=2)

        # Copy tokenizer with HF naming
        shutil.copy2(self.tokenizer_path, output_path / "tokenizer.model")

        # Create tokenizer config
        tokenizer_config = {
            "tokenizer_class": "SentencePieceTokenizer",
            "model_max_length": self.config.block_size,
            "vocab_size": self.config.vocab_size,
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
        }

        with open(output_path / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        # Create generation config
        generation_config = {
            "max_length": 512,
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
        }

        with open(output_path / "generation_config.json", "w") as f:
            json.dump(generation_config, f, indent=2)

        # Create HF loading script
        self._create_hf_loader(output_path)

        print(f"✅ Hugging Face export completed: {output_path}")
        return str(output_path)

    def export_onnx(self, optimize_for_inference: bool = False) -> str:
        """
        Export model to ONNX format for optimized inference.

        Args:
            optimize_for_inference: Whether to apply ONNX optimizations

        Returns:
            Path to exported ONNX model
        """
        try:
            import onnx
            import onnxruntime
        except ImportError:
            raise ImportError("ONNX export requires: pip install onnx onnxruntime")

        output_path = self.output_dir / "onnx"
        output_path.mkdir(parents=True, exist_ok=True)

        print("🔄 Exporting to ONNX format...")

        # Prepare model for export
        self.model.eval()

        # Create dummy input for tracing
        batch_size = 1
        seq_len = 64  # Use shorter sequence for compatibility
        dummy_input = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        # Export to ONNX
        onnx_path = output_path / "model.onnx"

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
        )

        # Verify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        # Apply optimizations if requested
        if optimize_for_inference:
            self._optimize_onnx_model(onnx_path)

        # Save metadata
        metadata = {
            "model_config": self.config.__dict__,
            "training_info": self.training_info,
            "export_format": "onnx",
            "input_shape": [batch_size, seq_len],
            "input_names": ["input_ids"],
            "output_names": ["logits"],
            "optimized": optimize_for_inference,
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Copy tokenizer
        shutil.copy2(self.tokenizer_path, output_path / "tokenizer.model")

        # Create ONNX inference script
        self._create_onnx_inference(output_path)

        print(f"✅ ONNX export completed: {onnx_path}")
        return str(onnx_path)

    def _optimize_onnx_model(self, onnx_path: Path):
        """Apply ONNX optimizations for inference."""
        try:
            from onnxruntime.tools import optimizer

            print("🔧 Applying ONNX optimizations...")

            # Create optimized model
            optimized_path = onnx_path.parent / "model_optimized.onnx"

            # Apply graph optimizations
            optimizer.optimize_model(
                str(onnx_path),
                str(optimized_path),
                optimization_level=optimizer.OptimizationLevel.ORT_ENABLE_ALL,
            )

            # Replace original with optimized
            shutil.move(str(optimized_path), str(onnx_path))

            print("✅ ONNX optimizations applied")

        except ImportError:
            print("⚠️  ONNX optimization requires onnxruntime-tools")
        except Exception as e:
            print(f"⚠️  ONNX optimization failed: {e}")

    def _create_pytorch_loader(self, output_path: Path):
        """Create PyTorch model loader script."""
        loader_script = '''#!/usr/bin/env python3
"""
PyTorch Model Loader for OpenLLM

Usage:
    from load_model import load_model, generate_text
    
    model, tokenizer, config = load_model(".")
    text = generate_text(model, tokenizer, "Hello world", max_length=50)
    print(text)
"""

import torch
import json
import sentencepiece as spm
from pathlib import Path

def load_model(model_dir="."):
    """Load OpenLLM model from PyTorch export."""
    model_dir = Path(model_dir)
    
    # Load config
    with open(model_dir / "config.json", 'r') as f:
        config_data = json.load(f)
    
    model_config = config_data['model_config']
    
    # Recreate model architecture (you'll need to have the model.py file)
    # This is a simplified loader - in practice you'd import your GPTModel class
    print(f"Model config: {model_config}")
    print("Note: You need to import and create the actual model class")
    
    # Load model state
    checkpoint = torch.load(model_dir / "model.pt", map_location='cpu')
    
    # Load tokenizer
    tokenizer = smp.SentencePieceProcessor()
    tokenizer.load(str(model_dir / "tokenizer.model"))
    
    return None, tokenizer, model_config  # Placeholder

def generate_text(model, tokenizer, prompt, max_length=100):
    """Generate text using the loaded model."""
    # Implement text generation
    return f"Generated text for: {prompt}"

if __name__ == "__main__":
    model, tokenizer, config = load_model()
    print(f"Model loaded with {config.get('vocab_size', 'unknown')} vocabulary size")
'''

        with open(output_path / "load_model.py", "w") as f:
            f.write(loader_script)

    def _create_hf_loader(self, output_path: Path):
        """Create Hugging Face model loader script."""
        loader_script = '''#!/usr/bin/env python3
"""
Hugging Face Compatible Loader for OpenLLM

Usage:
    # Using transformers library (if you implement custom model class)
    # from transformers import AutoModel, AutoTokenizer
    # model = AutoModel.from_pretrained(".")
    # tokenizer = AutoTokenizer.from_pretrained(".")
    
    # Manual loading
    from load_hf_model import load_model_manual
    model, tokenizer = load_model_manual(".")
"""

import torch
import json
import sentencepiece as smp
from pathlib import Path

def load_model_manual(model_dir="."):
    """Manually load model in HF format."""
    model_dir = Path(model_dir)
    
    # Load config
    with open(model_dir / "config.json", 'r') as f:
        config = json.load(f)
    
    # Load model weights
    state_dict = torch.load(model_dir / "pytorch_model.bin", map_location='cpu')
    
    # Load tokenizer
    tokenizer = smp.SentencePieceProcessor()
    tokenizer.load(str(model_dir / "tokenizer.model"))
    
    print(f"Loaded model: {config['model_type']} with {config['n_layer']} layers")
    print(f"Vocabulary size: {config['vocab_size']}")
    
    return state_dict, tokenizer

if __name__ == "__main__":
    state_dict, tokenizer = load_model_manual()
    print(f"Model weights loaded: {len(state_dict)} parameters")
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size()}")
'''

        with open(output_path / "load_hf_model.py", "w") as f:
            f.write(loader_script)

    def _create_onnx_inference(self, output_path: Path):
        """Create ONNX inference script."""
        inference_script = '''#!/usr/bin/env python3
"""
ONNX Inference for OpenLLM

Usage:
    from onnx_inference import ONNXInference
    
    inference = ONNXInference(".")
    output = inference.generate("Hello world", max_length=50)
    print(output)
"""

import numpy as np
import json
import sentencepiece as smp
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("Install onnxruntime: pip install onnxruntime")
    ort = None

class ONNXInference:
    def __init__(self, model_dir="."):
        if ort is None:
            raise ImportError("onnxruntime not available")
        
        model_dir = Path(model_dir)
        
        # Load ONNX model
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"))
        
        # Load metadata
        with open(model_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load tokenizer
        self.tokenizer = smp.SentencePieceProcessor()
        self.tokenizer.load(str(model_dir / "tokenizer.model"))
        
        print(f"ONNX model loaded: {self.metadata['model_config']['model_name']}")
    
    def predict(self, input_ids):
        """Run inference on input token IDs."""
        # Prepare input
        input_data = {"input_ids": input_ids.astype(np.int64)}
        
        # Run inference
        outputs = self.session.run(None, input_data)
        return outputs[0]  # logits
    
    def generate(self, prompt, max_length=50, temperature=0.7):
        """Generate text from prompt."""
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt)
        input_ids = np.array([tokens], dtype=np.int64)
        
        # Simple greedy generation (can be improved)
        generated = tokens.copy()
        
        for _ in range(max_length):
            if len(generated) >= 512:  # Max sequence length
                break
            
            # Get current input (last 64 tokens to fit ONNX model)
            current_input = np.array([generated[-64:]], dtype=np.int64)
            
            # Predict next token
            logits = self.predict(current_input)
            next_token_logits = logits[0, -1, :]  # Last position
            
            # Apply temperature and sample
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
                next_token = np.random.choice(len(probs), p=probs)
            else:
                next_token = np.argmax(next_token_logits)
            
            generated.append(int(next_token))
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated[len(tokens):])
        return generated_text

if __name__ == "__main__":
    inference = ONNXInference()
    result = inference.generate("The future of AI is", max_length=30)
    print(f"Generated: {result}")
'''

        with open(output_path / "onnx_inference.py", "w") as f:
            f.write(inference_script)

    def export_all_formats(self, optimize_onnx: bool = False) -> Dict[str, str]:
        """
        Export model to all supported formats.

        Args:
            optimize_onnx: Whether to optimize ONNX model

        Returns:
            Dictionary mapping format names to export paths
        """
        results = {}

        print("🚀 Exporting to all formats...")

        try:
            results["pytorch"] = self.export_pytorch()
        except Exception as e:
            print(f"❌ PyTorch export failed: {e}")

        try:
            results["huggingface"] = self.export_huggingface()
        except Exception as e:
            print(f"❌ Hugging Face export failed: {e}")

        try:
            results["onnx"] = self.export_onnx(optimize_onnx)
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")

        # Create summary
        summary = {
            "export_timestamp": torch.datetime.now().isoformat(),
            "model_info": {
                "name": self.config.model_name,
                "parameters": self.model.get_num_params(),
                "training_steps": self.training_info["step"],
                "best_loss": self.training_info["best_loss"],
            },
            "exports": results,
        }

        with open(self.output_dir / "export_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Export summary saved: {self.output_dir / 'export_summary.json'}")

        return results


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(
        description="Export OpenLLM models to various formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to PyTorch format
  python core/src/export_model.py \\
    --model_dir models/small-extended-4k \\
    --format pytorch \\
    --output_dir exports/pytorch/
  
  # Export to Hugging Face format
  python core/src/export_model.py \\
    --model_dir models/small-extended-4k \\
    --format huggingface \\
    --output_dir exports/huggingface/
  
  # Export to ONNX with optimizations
  python core/src/export_model.py \\
    --model_dir models/small-extended-4k \\
    --format onnx \\
    --output_dir exports/onnx/ \\
    --optimize_for_inference
  
  # Export to all formats
  python core/src/export_model.py \\
    --model_dir models/small-extended-4k \\
    --format all \\
    --output_dir exports/
        """,
    )

    parser.add_argument(
        "--model_dir", required=True, help="Directory containing trained model checkpoints"
    )

    parser.add_argument(
        "--format",
        choices=["pytorch", "huggingface", "onnx", "all"],
        required=True,
        help="Export format",
    )

    parser.add_argument("--output_dir", required=True, help="Output directory for exported models")

    parser.add_argument(
        "--optimize_for_inference",
        action="store_true",
        help="Apply optimizations for inference (ONNX only)",
    )

    args = parser.parse_args()

    print("📦 OpenLLM Model Export")
    print("=" * 50)

    try:
        # Create exporter
        exporter = ModelExporter(args.model_dir, args.output_dir)

        # Export based on format
        if args.format == "pytorch":
            result = exporter.export_pytorch()
            print(f"\n✅ PyTorch export completed: {result}")

        elif args.format == "huggingface":
            result = exporter.export_huggingface()
            print(f"\n✅ Hugging Face export completed: {result}")

        elif args.format == "onnx":
            result = exporter.export_onnx(args.optimize_for_inference)
            print(f"\n✅ ONNX export completed: {result}")

        elif args.format == "all":
            results = exporter.export_all_formats(args.optimize_for_inference)
            print(f"\n✅ All formats exported:")
            for fmt, path in results.items():
                print(f"  {fmt}: {path}")

        print(f"\n🎉 Export completed successfully!")

    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
