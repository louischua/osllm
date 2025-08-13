#!/usr/bin/env python3
"""
Hugging Face Hub Training Configuration

This script provides the training configuration for running OpenLLM training
on Hugging Face Hub's training infrastructure.

Author: Louis Chua Bean Chong
License: GPL-3.0
"""

import os
import sys
import torch
from pathlib import Path
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# Add core/src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'src'))

from model import create_model, GPTConfig
from data_loader import TextDataLoader

def get_training_args():
    """
    Get training arguments for Hugging Face Hub training.
    
    Returns:
        TrainingArguments: Configuration for HF training
    """
    
    return TrainingArguments(
        # Output directory
        output_dir="./results",
        
        # Training parameters
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        
        # Learning rate
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        
        # Optimization
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        
        # Logging and saving
        logging_steps=50,
        save_steps=200,
        eval_steps=100,
        save_total_limit=3,
        
        # Evaluation
        evaluation_strategy="steps",
        
        # Model saving
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # HF Hub integration
        push_to_hub=True,
        hub_model_id="lemms/openllm-small-extended-8k",
        hub_strategy="end",
        
        # Training settings
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        
        # Report to HF Hub
        report_to=["tensorboard"],
        
        # Logging
        logging_dir="./logs",
        log_level="info",
        
        # Seed for reproducibility
        seed=42,
        
        # Data collation
        dataloader_num_workers=0,
        
        # Mixed precision
        fp16=False,  # Set to True if using GPU
        
        # Gradient clipping
        max_grad_norm=1.0,
        
        # Early stopping
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
    )

def prepare_dataset_for_hf():
    """
    Prepare the dataset for Hugging Face training format.
    
    Returns:
        Dataset: HF-compatible dataset
    """
    
    # Load our data loader
    data_loader = TextDataLoader(
        data_file="data/clean/training_data.txt",
        tokenizer_dir="data/tokenizer/",
        block_size=1024,
        batch_size=4
    )
    
    # Convert to HF format
    texts = []
    for batch in data_loader.get_batches():
        # Decode tokens back to text for HF processing
        for sequence in batch:
            text = data_loader.tokenizer.decode(sequence.tolist())
            texts.append({"text": text})
    
    return Dataset.from_list(texts)

def tokenize_function(examples, tokenizer):
    """
    Tokenize the dataset for training.
    
    Args:
        examples: Dataset examples
        tokenizer: HF tokenizer
        
    Returns:
        dict: Tokenized examples
    """
    
    # Tokenize the texts
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=1024,
        return_tensors="pt"
    )
    
    # Set labels to input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def setup_hf_training():
    """
    Set up Hugging Face training configuration.
    
    Returns:
        tuple: (model, tokenizer, dataset, training_args)
    """
    
    print("🔧 Setting up Hugging Face Hub training...")
    
    # Load or create model
    model_path = "models/small-extended-7k"
    if os.path.exists(model_path):
        print(f"📥 Loading existing model from {model_path}")
        # Load our custom model
        model = create_model("small")
        checkpoint = torch.load(os.path.join(model_path, "best_model.pt"), map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("🏗️ Creating new model")
        model = create_model("small")
    
    # Load tokenizer
    tokenizer_path = "data/tokenizer/"
    if os.path.exists(os.path.join(tokenizer_path, "tokenizer.model")):
        print(f"📥 Loading tokenizer from {tokenizer_path}")
        # We'll need to convert our SentencePiece tokenizer to HF format
        # For now, we'll use a basic tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print("❌ Tokenizer not found")
        return None
    
    # Prepare dataset
    print("📂 Preparing dataset...")
    dataset = prepare_dataset_for_hf()
    
    # Tokenize dataset
    print("🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Get training arguments
    training_args = get_training_args()
    
    return model, tokenizer, tokenized_dataset, training_args

def create_hf_training_script():
    """
    Create a training script for Hugging Face Hub.
    """
    
    script_content = '''#!/usr/bin/env python3
"""
Hugging Face Hub Training Script for OpenLLM

This script is designed to run on Hugging Face Hub's training infrastructure.
It continues training the 7k model for 1000 additional steps.

Author: Louis Chua Bean Chong
License: GPL-3.0
"""

import os
import sys
import torch
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# Add our model code
sys.path.append("/workspace/core/src")

from model import create_model
from data_loader import TextDataLoader

def main():
    """Main training function for HF Hub."""
    
    print("🚀 Starting OpenLLM training on Hugging Face Hub")
    
    # Configuration
    model_size = "small"
    data_file = "/workspace/data/clean/training_data.txt"
    tokenizer_dir = "/workspace/data/tokenizer/"
    input_model_dir = "/workspace/models/small-extended-7k"
    output_dir = "/workspace/models/small-extended-8k"
    
    # Load model
    model = create_model(model_size)
    
    # Load checkpoint if available
    if os.path.exists(os.path.join(input_model_dir, "best_model.pt")):
        checkpoint = torch.load(
            os.path.join(input_model_dir, "best_model.pt"), 
            map_location='cpu'
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded 7k model checkpoint")
    
    # Load data
    data_loader = TextDataLoader(
        data_file=data_file,
        tokenizer_dir=tokenizer_dir,
        block_size=1024,
        batch_size=4
    )
    
    # Prepare dataset
    texts = []
    for batch in data_loader.get_batches():
        for sequence in batch:
            text = data_loader.tokenizer.decode(sequence.tolist())
            texts.append({"text": text})
    
    dataset = Dataset.from_list(texts)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=0,
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
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset.select(range(min(100, len(dataset)))),
        tokenizer=data_loader.tokenizer,
    )
    
    # Start training
    print("🔥 Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    print("✅ Training completed!")

if __name__ == "__main__":
    main()
'''
    
    with open("hf_training_script.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created HF training script: hf_training_script.py")

if __name__ == "__main__":
    create_hf_training_script()
    print("\n📋 Next steps:")
    print("1. Upload this script to Hugging Face Hub")
    print("2. Configure training job with:")
    print("   - Model: lemms/openllm-small-extended-7k")
    print("   - Dataset: Your training data")
    print("   - Compute: CPU or GPU")
    print("   - Duration: ~1-2 hours")
    print("3. Monitor training progress on HF Hub")
