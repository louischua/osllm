#!/usr/bin/env python3
"""
Script to continue training the OpenLLM 7k model for an additional 1000 steps

This script loads the existing 7k model checkpoint and continues training
for 1000 more steps to create an 8k model.

Author: Louis Chua Bean Chong
License: GPL-3.0
"""

import os
import sys
import torch
from pathlib import Path

# Add core/src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'src'))

from train_model import ModelTrainer
from model import create_model
from data_loader import TextDataLoader

def train_7k_to_8k():
    """
    Continue training the 7k model for 1000 additional steps to reach 8k.
    
    This function:
    1. Loads the existing 7k model checkpoint
    2. Continues training for 1000 more steps
    3. Saves the new 8k model
    """
    
    print("🚀 Continuing OpenLLM 7k Model Training to 8k Steps")
    print("=" * 60)
    
    # Configuration
    model_size = "small"
    data_file = "data/clean/training_data.txt"
    tokenizer_dir = "data/tokenizer/"
    input_model_dir = "models/small-extended-7k"
    output_model_dir = "models/small-extended-8k"
    max_steps = 1000  # Additional steps to train
    start_step = 7000  # Starting from step 7000
    
    print(f"📊 Training Configuration:")
    print(f"  Model Size: {model_size}")
    print(f"  Data File: {data_file}")
    print(f"  Tokenizer Dir: {tokenizer_dir}")
    print(f"  Input Model: {input_model_dir}")
    print(f"  Output Model: {output_model_dir}")
    print(f"  Start Step: {start_step}")
    print(f"  Additional Steps: {max_steps}")
    print(f"  Target Step: {start_step + max_steps}")
    
    # Check if input model exists
    if not os.path.exists(input_model_dir):
        print(f"❌ Input model directory not found: {input_model_dir}")
        return False
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False
    
    # Check if tokenizer exists
    if not os.path.exists(tokenizer_dir):
        print(f"❌ Tokenizer directory not found: {tokenizer_dir}")
        return False
    
    try:
        # Create output directory
        os.makedirs(output_model_dir, exist_ok=True)
        
        # Initialize data loader
        print(f"📂 Loading training data...")
        data_loader = TextDataLoader(
            data_file=data_file,
            tokenizer_dir=tokenizer_dir,
            block_size=1024,
            batch_size=4
        )
        
        # Create model
        print(f"🏗️ Creating model architecture...")
        model = create_model(model_size)
        
        # Load existing checkpoint
        checkpoint_path = os.path.join(input_model_dir, "best_model.pt")
        print(f"📥 Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize trainer with continued training settings
        print(f"🎯 Initializing trainer for continued training...")
        trainer = ModelTrainer(
            model=model,
            data_loader=data_loader,
            output_dir=output_model_dir,
            device="cpu",  # Use CPU for compatibility
            learning_rate=3e-4,
            weight_decay=0.01,
            warmup_steps=0,  # No warmup needed for continued training
            max_steps=max_steps,
            gradient_accumulation_steps=4,
            gradient_clipping=1.0,
            save_every=200,  # Save every 200 steps
            eval_every=100,  # Evaluate every 100 steps
            log_every=50     # Log every 50 steps
        )
        
        # Set the starting step
        trainer.step = start_step
        trainer.best_loss = checkpoint.get('best_loss', float('inf'))
        
        # Continue training
        print(f"🔥 Starting continued training from step {start_step}...")
        print(f"🎯 Target: {start_step + max_steps} steps")
        
        trainer.train()
        
        print(f"✅ Training completed successfully!")
        print(f"📁 Model saved to: {output_model_dir}")
        print(f"🎯 Final step: {start_step + max_steps}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to execute the continued training process.
    """
    
    print("🎯 OpenLLM 7k to 8k Training Script")
    print("=" * 50)
    
    # Check dependencies
    try:
        import torch
        print(f"✅ PyTorch found: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found!")
        print("   Please install it with: pip install torch")
        return False
    
    # Execute training
    success = train_7k_to_8k()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Export the 8k model to Hugging Face format")
        print("   2. Push the 8k model to Hugging Face Hub")
        print("   3. Update documentation with 8k model information")
        print("   4. Test the 8k model performance")
    else:
        print("\n💥 Training failed!")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if the 7k model exists")
        print("   2. Verify training data is available")
        print("   3. Ensure sufficient disk space")
        print("   4. Check system resources")
        sys.exit(1)

if __name__ == "__main__":
    main()
