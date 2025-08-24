#!/usr/bin/env python3
"""
OpenLLM Real Models App - Final working version with correct attribute naming
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPTConfig:
    """GPT model configuration"""

    def __init__(
        self,
        vocab_size=32000,
        n_layer=6,
        n_head=8,
        n_embd=512,
        block_size=1024,
        dropout=0.1,
        bias=False,
        **kwargs,
    ):
        # Accept any additional kwargs to handle extra config fields
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias


class GPT(nn.Module):
    """GPT-style transformer model - EXACT architecture matching the saved model"""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Create the transformer module with the exact naming convention
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        # Language model head - Use bias=False to match saved models
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def generate(
        self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, do_sample=True
    ):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class Block(nn.Module):
    """Transformer block with self-attention and feed-forward layers"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking - FINAL WORKING VERSION"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_bias = config.bias  # Use different name for the boolean flag

        # REGISTER THE ATTENTION BIAS as a buffer (not parameter) to match saved model
        # This is actually an attention mask, not a learnable bias
        if config.bias:
            # Create a causal attention mask buffer
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            mask = mask.view(1, 1, config.block_size, config.block_size)
            self.register_buffer("bias", mask)  # This matches the saved model's 'bias' key
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        B, T, C = x.size()

        # Calculate query, key, values for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal self-attention using the bias mask
        if self.bias is not None:
            # Use the causal mask
            attn_mask = self.bias[:, :, :T, :T]
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            # Use built-in causal attention
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Multi-layer perceptron"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class RealOpenLLMInference:
    """Real OpenLLM inference engine using actual trained models"""

    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.current_model = None

        # Real model configurations from Hugging Face
        self.model_configs = {
            "openllm-small-extended-4k": {
                "name": "OpenLLM Small (4k steps)",
                "description": "Real model trained for 4,000 steps - Early training stage",
                "hf_repo": "lemms/openllm-small-extended-4k",
                "training_steps": 4000,
                "parameters": "35.8M",
            },
            "openllm-small-extended-6k": {
                "name": "OpenLLM Small (6k steps)",
                "description": "Real model trained for 6,000 steps - Improved coherence (Perplexity: 816.040)",
                "hf_repo": "lemms/openllm-small-extended-6k",
                "training_steps": 6000,
                "parameters": "35.8M",
            },
            "openllm-small-extended-7k": {
                "name": "OpenLLM Small (7k steps)",
                "description": "Real model trained for 7,000 steps - Enhanced quality (Loss: 2.100, Perplexity: 8.200)",
                "hf_repo": "lemms/openllm-small-extended-7k",
                "training_steps": 7000,
                "parameters": "35.8M",
            },
            "openllm-small-extended-8k": {
                "name": "OpenLLM Small (8k steps)",
                "description": "Real model trained for 8,000 steps - Sophisticated understanding",
                "hf_repo": "lemms/openllm-small-extended-8k",
                "training_steps": 8000,
                "parameters": "35.8M",
            },
            "openllm-small-extended-9k": {
                "name": "OpenLLM Small (9k steps)",
                "description": "Real model trained for 9,000 steps - Best performing model",
                "hf_repo": "lemms/openllm-small-extended-9k",
                "training_steps": 9000,
                "parameters": "35.8M",
            },
            "openllm-small-extended-10k": {
                "name": "OpenLLM Small (10k steps)",
                "description": "Real model trained for 10,000 steps - Latest extended training",
                "hf_repo": "lemms/openllm-small-extended-10k",
                "training_steps": 10000,
                "parameters": "35.8M",
            },
            "openllm-small-extended-10k-improved": {
                "name": "OpenLLM Small (10k steps - Improved)",
                "description": "Real model trained for 10,000 steps with improved training process - Proper checkpoint format",
                "hf_repo": "lemms/openllm-small-extended-10k-improved",
                "training_steps": 10000,
                "parameters": "35.8M",
            },
        }

        logger.info("🚀 Real OpenLLM Inference Engine initialized")

    def load_model_from_hf(self, model_id: str) -> bool:
        """Load a real model from Hugging Face"""
        try:
            config = self.model_configs.get(model_id)
            if not config:
                logger.error(f"❌ Unknown model ID: {model_id}")
                return False

            logger.info(f"📥 Loading real model from HF: {config['hf_repo']}")

            # Download model from Hugging Face
            local_dir = snapshot_download(
                repo_id=config["hf_repo"],
                repo_type="model",
                local_dir=f"temp_{model_id}",
                allow_patterns=["*.pt", "*.json", "*.model", "*.bin"],
            )

            logger.info(f"✅ Downloaded model to: {local_dir}")

            # Load model and tokenizer
            success = self._load_model_and_tokenizer(local_dir, model_id)
            if success:
                self.current_model = model_id
                logger.info(f"✅ Successfully loaded real model: {model_id}")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"❌ Failed to load real model from HF {model_id}: {e}")
            return False

    def _load_model_and_tokenizer(self, model_dir: str, model_id: str) -> bool:
        """Load model and tokenizer from local directory"""
        try:
            model_path = Path(model_dir)

            # Load model configuration
            config_file = model_path / "config.json"
            if config_file.exists():
                with open(config_file, "r") as f:
                    config_data = json.load(f)

                logger.info(f"📋 Config data keys: {list(config_data.keys())}")

                # Handle different config structures
                if "model_config" in config_data:
                    # Extract model_config section
                    model_config_data = config_data["model_config"]
                else:
                    # Use the entire config as model config
                    model_config_data = config_data

                # Create GPTConfig with only the expected parameters
                expected_params = {
                    "vocab_size",
                    "n_layer",
                    "n_head",
                    "n_embd",
                    "block_size",
                    "dropout",
                    "bias",
                }

                config_kwargs = {}
                for key, value in model_config_data.items():
                    if key in expected_params:
                        config_kwargs[key] = value

                logger.info(f"🔧 Using config parameters: {config_kwargs}")
                model_config = GPTConfig(**config_kwargs)
            else:
                # Default configuration for OpenLLM small models
                model_config = GPTConfig(
                    vocab_size=32000,
                    n_layer=6,
                    n_head=8,
                    n_embd=512,
                    block_size=1024,
                    dropout=0.1,
                    bias=False,
                )

            # Load model weights
            model_file = model_path / "best_model.pt"
            if not model_file.exists():
                model_file = model_path / "model.pt"
            if not model_file.exists():
                model_file = model_path / "pytorch_model.bin"

            if model_file.exists():
                logger.info(f"📦 Loading model from: {model_file}")
                model = GPT(model_config)
                checkpoint = torch.load(model_file, map_location="cpu")

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if "model_state_dict" in checkpoint:
                        # Extract the actual model weights
                        state_dict = checkpoint["model_state_dict"]
                        logger.info(f"📋 Loading from model_state_dict with {len(state_dict)} keys")
                    elif "model" in checkpoint:
                        state_dict = checkpoint["model"]
                        logger.info(f"📋 Loading from model with {len(state_dict)} keys")
                    else:
                        # Try to load directly as state dict
                        state_dict = checkpoint
                        logger.info(f"📋 Loading direct state dict with {len(state_dict)} keys")
                else:
                    # Direct state dict
                    state_dict = checkpoint
                    logger.info(f"📋 Loading direct state dict with {len(state_dict)} keys")

                # Load the state dict
                model.load_state_dict(state_dict)
                model.eval()
                self.models[model_id] = model
                logger.info(f"✅ Model loaded successfully")
            else:
                logger.error(f"❌ Model file not found in {model_dir}")
                logger.error(f"   Available files: {list(model_path.glob('*'))}")
                return False

            # Load tokenizer
            tokenizer_file = model_path / "tokenizer.model"
            if tokenizer_file.exists():
                tokenizer = spm.SentencePieceProcessor()
                tokenizer.load(str(tokenizer_file))
                self.tokenizers[model_id] = tokenizer
                logger.info(f"✅ Tokenizer loaded successfully")
            else:
                logger.error(f"❌ Tokenizer file not found in {model_dir}")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model and tokenizer: {e}")
            import traceback

            logger.error(f"📋 Full traceback: {traceback.format_exc()}")
            return False

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """Generate text using the loaded real model"""
        if not self.current_model or self.current_model not in self.models:
            return "❌ No model loaded. Please select a model first."

        try:
            model = self.models[self.current_model]
            tokenizer = self.tokenizers[self.current_model]

            # Tokenize input
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long)

            logger.info(f"🎯 Generating text with prompt: '{prompt[:50]}...'")
            logger.info(
                f"📊 Parameters: max_length={max_length}, temperature={temperature}, top_k={top_k}, top_p={top_p}"
            )

            # Generate text
            with torch.no_grad():
                output_ids = model.generate(
                    input_tensor,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                )

            # Decode output
            generated_text = tokenizer.decode(output_ids[0].tolist())

            # Remove the input prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt) :].strip()

            logger.info(f"✅ Generated text: '{generated_text[:100]}...'")
            return generated_text

        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}"
            logger.error(error_msg)
            import traceback

            logger.error(f"📋 Full traceback: {traceback.format_exc()}")
            return error_msg


# Initialize the real inference engine
inference_engine = RealOpenLLMInference()


def load_model_info(model_id: str) -> str:
    """Get information about a specific model"""
    config = inference_engine.model_configs.get(model_id)
    if config:
        return f"**{config['name']}**\n\n{config['description']}\n\n**Parameters:** {config['parameters']}\n**Training Steps:** {config['training_steps']:,}"
    return "❌ Model not found"


def generate_text_interface(
    model_id: str, prompt: str, max_length: int, temperature: float, top_k: int, top_p: float
) -> str:
    """Gradio interface function for text generation"""
    try:
        # Load model if not already loaded
        if model_id not in inference_engine.models:
            logger.info(f"🔄 Loading real model: {model_id}")
            success = inference_engine.load_model_from_hf(model_id)
            if not success:
                return f"❌ Failed to load real model: {model_id}"

        # Generate text
        result = inference_engine.generate_text(
            prompt=prompt, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p
        )

        return result

    except Exception as e:
        error_msg = f"❌ Error in generation interface: {str(e)}"
        logger.error(error_msg)
        return error_msg


# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(title="🚀 OpenLLM Real Models Space", theme=gr.themes.Soft()) as interface:
        # Header
        gr.Markdown(
            """
        # 🚀 OpenLLM Real Models Space
        
        Welcome to the OpenLLM Real Models Space! This interface uses **actual trained models** from Hugging Face.
        
        ## 🎯 Real Trained Models
        
        We provide **5 different real models** with varying training steps:
        
        | Model | Training Steps | Parameters | Performance |
        |-------|---------------|------------|-------------|
        | **4k Model** | 4,000 | 35.8M | Early training stage |
        | **6k Model** | 6,000 | 35.8M | Improved coherence (Perplexity: 816.040) |
        | **7k Model** | 7,000 | 35.8M | Enhanced quality (Loss: 2.100, Perplexity: 8.200) |
        | **8k Model** | 8,000 | 35.8M | Sophisticated understanding |
        | **9k Model** | 9,000 | 35.8M | Best performing model |
        | **10k Model** | 10,000 | 35.8M | Latest extended training |
        
        **These are real GPT-style transformer models trained on Wikipedia passages from the SQuAD dataset.**
        
        ---
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=list(inference_engine.model_configs.keys()),
                    value="openllm-small-extended-10k",
                    label="🎯 Select Model",
                    info="Choose the real trained model to use",
                )

                # Model information display
                model_info = gr.Markdown(
                    value=load_model_info("openllm-small-extended-10k"), label="📋 Model Information"
                )

                # Update model info when selection changes
                model_dropdown.change(
                    fn=load_model_info, inputs=[model_dropdown], outputs=[model_info]
                )

            with gr.Column(scale=2):
                # Input prompt
                prompt_input = gr.Textbox(
                    lines=5,
                    label="📝 Input Prompt",
                    placeholder="Enter your text prompt here...",
                    info="The text that will be used as input for generation",
                )

                # Generation parameters
                with gr.Row():
                    max_length = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="📏 Max Length",
                        info="Maximum number of tokens to generate",
                    )

                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="🌡️ Temperature",
                        info="Controls randomness (higher = more random)",
                    )

                with gr.Row():
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="🔝 Top-K",
                        info="Number of highest probability tokens to consider",
                    )

                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        label="📊 Top-P",
                        info="Nucleus sampling parameter",
                    )

                # Generate button
                generate_btn = gr.Button("🚀 Generate Text", variant="primary", size="lg")

        # Output
        output_text = gr.Textbox(
            lines=10, label="🎯 Generated Text", info="The generated text will appear here"
        )

        # Connect the generate button
        generate_btn.click(
            fn=generate_text_interface,
            inputs=[model_dropdown, prompt_input, max_length, temperature, top_k, top_p],
            outputs=[output_text],
        )

        # Footer
        gr.Markdown(
            """
        ---
        
        ## 🔧 Technical Details
        
        - **Architecture**: GPT-style transformer decoder
        - **Model Size**: Small (6 layers, 8 heads, 512 embedding dim)
        - **Vocabulary**: 32k tokens (SentencePiece BPE)
        - **Training Data**: Wikipedia passages from SQuAD dataset
        - **Framework**: PyTorch with real trained models
        - **Gradio Version**: 4.44.1 (latest)
        
        **These models generate actual text based on their training on Wikipedia content.**
        
        **Model Sources:**
        - [4k Model](https://huggingface.co/lemms/openllm-small-extended-4k)
        - [6k Model](https://huggingface.co/lemms/openllm-small-extended-6k)
        - [7k Model](https://huggingface.co/lemms/openllm-small-extended-7k)
        - [8k Model](https://huggingface.co/lemms/openllm-small-extended-8k)
        - [9k Model](https://huggingface.co/lemms/openllm-small-extended-9k)
        - [10k Model](https://huggingface.co/lemms/openllm-small-extended-10k)
        """
        )

    return interface


# Create and launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
