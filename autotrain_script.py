#!/usr/bin/env python3
"""
AutoTrain Script for OpenLLM 7k to 8k Training

This script runs on Hugging Face AutoTrain infrastructure.
"""

import os
import sys
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset

def main():
    """Main training function for AutoTrain."""
    
    print("🚀 Starting OpenLLM training on Hugging Face AutoTrain")
    
    # Configuration
    model_name = "lemms/openllm-small-extended-7k"
    dataset_name = os.getenv("DATASET_NAME", "your-username/openllm-training-data")
    output_dir = "./results"
    
    # Load model and tokenizer
    print(f"📥 Loading model from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"📂 Loading dataset from {dataset_name}")
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"⚠️ Could not load dataset: {e}")
        print("📝 Creating sample dataset for testing...")
        
        # Create sample data for testing
        sample_texts = [
            "The future of artificial intelligence is promising.",
            "Machine learning models continue to improve.",
            "Natural language processing has advanced significantly.",
            "Deep learning has revolutionized many fields.",
            "Open source AI projects are growing rapidly."
        ] * 100  # Repeat to create more data
        
        dataset = Dataset.from_dict({"text": sample_texts})
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=1024,
            return_tensors="pt"
        )
    
    # Tokenize dataset
    print("🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling
    )
    
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
        max_steps=1000,  # Train for exactly 1000 steps
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset.select(range(min(100, len(tokenized_dataset)))),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    print("🔥 Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    print("✅ Training completed!")
    print(f"📤 Model pushed to: lemms/openllm-small-extended-8k")

if __name__ == "__main__":
    main()
