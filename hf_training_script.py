#!/usr/bin/env python3
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
    
    print("Starting OpenLLM training on Hugging Face Hub")
    
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
        print("Loaded 7k model checkpoint")
    
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
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    print("Training completed!")

if __name__ == "__main__":
    main()
