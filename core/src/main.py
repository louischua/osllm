#!/usr/bin/env python3
"""
OpenLLM - Main CLI Entry Point

This module provides a unified command-line interface for all OpenLLM operations
including data preparation, tokenizer training, model training, and inference.

Usage:
    python core/src/main.py <command> [options]

Available Commands:
    prepare-data    Download and prepare training data from SQUAD dataset
    train-tokenizer Train a SentencePiece tokenizer on the prepared data
    test-model      Test and validate model architecture
    train-model     Train the language model
    inference       Run model inference (coming soon)
    evaluate        Evaluate model performance (coming soon)

Examples:
    # Full pipeline
    python core/src/main.py prepare-data
    python core/src/main.py train-tokenizer --vocab-size 32000
    python core/src/main.py test-model --model-size small
    python core/src/main.py train-model --model-size small --output-dir models/my-model
    
    # Help for specific commands
    python core/src/main.py train-model --help
"""

import argparse
import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from download_and_prepare import prepare_training_data
    from train_tokenizer import train_sentencepiece_tokenizer, validate_input_file, count_training_sentences, save_huggingface_config, test_tokenizer
    from test_model import ModelTester
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the correct directory.")
    sys.exit(1)


def cmd_prepare_data(args):
    """Execute data preparation command."""
    print("🗂️  Starting data preparation...")
    print(f"Output path: {args.output}")
    print(f"Minimum words per passage: {args.min_words}")
    
    try:
        prepare_training_data(
            output_path=args.output,
            min_words=args.min_words
        )
        print("✅ Data preparation completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False


def cmd_train_tokenizer(args):
    """Execute tokenizer training command."""
    print("🔤 Starting tokenizer training...")
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Vocabulary size: {args.vocab_size:,}")
    print(f"Model type: {args.model_type}")
    
    try:
        # Step 1: Validate input
        validate_input_file(args.input)
        
        # Step 2: Count training data
        sentence_count = count_training_sentences(args.input)
        
        # Step 3: Train tokenizer
        config = train_sentencepiece_tokenizer(
            input_path=args.input,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            max_sentence_length=args.max_sentence_length,
        )
        
        # Step 4: Save Hugging Face config
        save_huggingface_config(args.output_dir, config)
        
        # Step 5: Test tokenizer (unless skipped)
        if not args.no_test:
            model_path = os.path.join(args.output_dir, "tokenizer.model")
            test_tokenizer(model_path)
        
        print("✅ Tokenizer training completed successfully!")
        print(f"📁 Output: {args.output_dir}")
        print(f"📊 Vocabulary size: {config['vocab_size']:,}")
        print(f"📄 Training sentences: {sentence_count:,}")
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer training failed: {e}")
        return False


def cmd_train_model(args):
    """Execute model training command."""
    print("🏗️  Starting model training...")
    
    try:
        from train_model import ModelTrainer, create_model
        from data_loader import TextDataLoader
        import torch
        import os
        
        # Determine device
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        
        print(f"Device: {device}")
        
        # Create model
        print(f"Creating {args.model_size} model...")
        model = create_model(args.model_size)
        
        # Create data loader
        print("Setting up data loader...")
        tokenizer_path = os.path.join(args.tokenizer_dir, "tokenizer.model")
        
        if not os.path.exists(tokenizer_path):
            print(f"❌ Tokenizer not found at {tokenizer_path}")
            print("Please run: python core/src/main.py train-tokenizer --input data/clean/training_data.txt")
            return False
        
        data_loader = TextDataLoader(
            data_file=args.data_file,
            tokenizer_path=tokenizer_path,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # Get data statistics
        data_stats = data_loader.get_data_stats()
        
        # Create trainer
        print("Setting up trainer...")
        trainer = ModelTrainer(
            model=model,
            data_loader=data_loader,
            output_dir=args.output_dir,
            device=device,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_every=args.save_every
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer._load_checkpoint(args.resume)
        
        # Start training
        trainer.train()
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cmd_inference(args):
    """Execute model inference command (placeholder)."""
    print("🚀 Inference functionality coming soon!")
    print("This will implement:")
    print("  • Text generation")
    print("  • REST API server")
    print("  • Batch processing")
    print("  • ONNX model support")
    return False


def cmd_evaluate(args):
    """Execute model evaluation command (placeholder)."""
    print("📊 Evaluation functionality coming soon!")
    print("This will implement:")
    print("  • Perplexity calculation")
    print("  • Downstream task evaluation")
    print("  • Benchmark comparisons")
    print("  • Performance metrics")
    return False


def cmd_test_model(args):
    """Execute model testing command."""
    print("🧪 Testing model architecture...")
    
    try:
        # Initialize model tester
        tester = ModelTester(device=args.device)
        
        if args.all_sizes:
            # Test all model sizes
            test_sizes = ["small", "medium", "large"]
            all_success = True
            
            for size in test_sizes:
                print(f"\n{'='*20} Testing {size.upper()} Model {'='*20}")
                results = tester.run_comprehensive_test(size)
                
                if not results["initialization"]["success"]:
                    all_success = False
                    print(f"❌ {size.upper()} model failed initialization")
                else:
                    print(f"✓ {size.upper()} model passed all tests")
            
            return all_success
        else:
            # Test single model size
            results = tester.run_comprehensive_test(args.model_size)
            
            if args.save_results:
                import json
                with open(args.save_results, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Results saved to {args.save_results}")
            
            return results["initialization"]["success"]
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        return False


def create_parser():
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="OpenLLM - Open Source Large Language Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare training data from SQUAD dataset
  python core/src/main.py prepare-data --output data/clean/training_data.txt
  
  # Train tokenizer with custom settings
  python core/src/main.py train-tokenizer \\
    --input data/clean/training_data.txt \\
    --vocab-size 32000 \\
    --output-dir data/tokenizer/
  
  # Get help for specific commands
  python core/src/main.py train-tokenizer --help
        """
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="OpenLLM v0.1.0"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True
    )
    
    # Data preparation command
    parser_data = subparsers.add_parser(
        "prepare-data",
        help="Download and prepare training data from SQUAD dataset",
        description="Downloads SQUAD v1.1 and v2.0 datasets, extracts Wikipedia passages, and prepares clean training text."
    )
    parser_data.add_argument(
        "--output",
        default="data/clean/training_data.txt",
        help="Output path for cleaned training data (default: data/clean/training_data.txt)"
    )
    parser_data.add_argument(
        "--min-words",
        type=int,
        default=10,
        help="Minimum number of words per passage (default: 10)"
    )
    parser_data.set_defaults(func=cmd_prepare_data)
    
    # Tokenizer training command
    parser_tokenizer = subparsers.add_parser(
        "train-tokenizer",
        help="Train a SentencePiece tokenizer on prepared data",
        description="Trains a BPE or Unigram tokenizer using SentencePiece on the prepared training text."
    )
    parser_tokenizer.add_argument(
        "--input",
        required=True,
        help="Path to training text file"
    )
    parser_tokenizer.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000)"
    )
    parser_tokenizer.add_argument(
        "--model-type",
        choices=["bpe", "unigram"],
        default="bpe",
        help="Tokenization algorithm (default: bpe)"
    )
    parser_tokenizer.add_argument(
        "--output-dir",
        default="data/tokenizer/",
        help="Output directory for tokenizer files (default: data/tokenizer/)"
    )
    parser_tokenizer.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Character coverage (default: 0.9995)"
    )
    parser_tokenizer.add_argument(
        "--max-sentence-length",
        type=int,
        default=4192,
        help="Maximum sentence length (default: 4192)"
    )
    parser_tokenizer.add_argument(
        "--no-test",
        action="store_true",
        help="Skip tokenizer testing after training"
    )
    parser_tokenizer.set_defaults(func=cmd_train_tokenizer)
    
    # Model testing command
    parser_test = subparsers.add_parser(
        "test-model",
        help="Test and validate model architecture",
        description="Test model initialization, forward pass, memory usage, and tokenizer integration."
    )
    parser_test.add_argument(
        "--model-size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size to test (default: medium)"
    )
    parser_test.add_argument(
        "--all-sizes",
        action="store_true",
        help="Test all model sizes"
    )
    parser_test.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for testing (default: auto)"
    )
    parser_test.add_argument(
        "--save-results",
        help="Save test results to JSON file"
    )
    parser_test.set_defaults(func=cmd_test_model)
    
    # Model training command
    parser_model = subparsers.add_parser(
        "train-model",
        help="Train the language model",
        description="Train a GPT-style transformer language model on tokenized text."
    )
    parser_model.add_argument(
        "--model-size",
        choices=["small", "medium", "large"],
        default="small",
        help="Model size to train (default: small)"
    )
    parser_model.add_argument(
        "--tokenizer-dir",
        default="data/tokenizer/",
        help="Path to trained tokenizer directory (default: data/tokenizer/)"
    )
    parser_model.add_argument(
        "--data-file",
        default="data/clean/training_data.txt",
        help="Path to training text file (default: data/clean/training_data.txt)"
    )
    parser_model.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for model checkpoints"
    )
    parser_model.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for training (default: 512)"
    )
    parser_model.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4, reduce for low memory)"
    )
    parser_model.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser_model.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum training steps (default: 10000)"
    )
    parser_model.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Warmup steps (default: 1000)"
    )
    parser_model.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser_model.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Training device (default: auto)"
    )
    parser_model.add_argument(
        "--resume",
        help="Path to checkpoint to resume training from"
    )
    parser_model.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (default: 1000)"
    )
    parser_model.set_defaults(func=cmd_train_model)
    
    # Inference command (placeholder)
    parser_inference = subparsers.add_parser(
        "inference",
        help="Run model inference (coming soon)",
        description="Generate text using a trained model."
    )
    parser_inference.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model"
    )
    parser_inference.add_argument(
        "--prompt",
        required=True,
        help="Input text prompt"
    )
    parser_inference.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum generation length"
    )
    parser_inference.set_defaults(func=cmd_inference)
    
    # Evaluation command (placeholder)
    parser_eval = subparsers.add_parser(
        "evaluate",
        help="Evaluate model performance (coming soon)",
        description="Evaluate model on various benchmarks and metrics."
    )
    parser_eval.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model"
    )
    parser_eval.add_argument(
        "--eval-data",
        help="Path to evaluation dataset"
    )
    parser_eval.add_argument(
        "--metrics",
        nargs="+",
        default=["perplexity"],
        help="Metrics to compute"
    )
    parser_eval.set_defaults(func=cmd_evaluate)
    
    return parser


def main():
    """Main entry point for the OpenLLM CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    print("🚀 OpenLLM - Open Source Large Language Model")
    print("=" * 60)
    
    # Execute the selected command
    success = args.func(args)
    
    # Exit with appropriate code
    if success:
        print("\n🎉 Command completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Command failed or not implemented yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
