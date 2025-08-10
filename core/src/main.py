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
    """
    Execute model inference command.
    
    This function implements text generation using trained OpenLLM models.
    It supports multiple model formats and provides flexible generation options.
    
    Args:
        args: Namespace containing CLI arguments including:
            - model_path: Path to trained model directory
            - prompt: Input text prompt for generation
            - max_length: Maximum number of tokens to generate
            - temperature: Sampling temperature (0.1-2.0)
            - format: Model format (auto-detect by default)
    
    Returns:
        bool: True if inference succeeded, False otherwise
        
    Implementation Details:
        - Auto-detects model format (PyTorch, Hugging Face, ONNX)
        - Uses inference_server.py's OpenLLMInference class for generation
        - Supports configurable generation parameters
        - Handles errors gracefully with informative messages
    """
    print("🚀 OpenLLM Model Inference")
    print("=" * 40)
    
    try:
        # Import inference functionality
        # We import here to avoid circular imports and handle missing dependencies
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from inference_server import OpenLLMInference
        
        # Validate model path exists
        # Early validation prevents confusing error messages later
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"❌ Model path not found: {args.model_path}")
            print("   Please check the path and try again.")
            return False
        
        # Initialize inference engine
        # This handles model loading and format detection automatically
        print(f"📂 Loading model from: {args.model_path}")
        inference_engine = OpenLLMInference(
            model_path=str(model_path),
            model_format=getattr(args, 'format', 'auto')  # Default to auto-detection
        )
        
        # Prepare generation parameters
        # These parameters control the quality and style of generated text
        generation_params = {
            'max_length': args.max_length,
            'temperature': getattr(args, 'temperature', 0.7),  # Default temperature
            'top_k': getattr(args, 'top_k', 40),              # Default top-k
            'top_p': getattr(args, 'top_p', 0.9),             # Default nucleus sampling
            'num_return_sequences': getattr(args, 'num_sequences', 1)  # Default single sequence
        }
        
        print(f"💭 Generating text for prompt: '{args.prompt}'")
        print(f"⚙️  Parameters: max_length={generation_params['max_length']}, "
              f"temperature={generation_params['temperature']}")
        
        # Generate text using the inference engine
        # This is the core functionality that produces the output
        import time
        start_time = time.time()
        
        generated_texts = inference_engine.generate(
            prompt=args.prompt,
            **generation_params
        )
        
        generation_time = time.time() - start_time
        
        # Display results with formatting
        # Clear presentation helps users understand the output
        print(f"\n✨ Generated Text:")
        print("-" * 50)
        
        for i, text in enumerate(generated_texts, 1):
            if len(generated_texts) > 1:
                print(f"\n[Sequence {i}]")
            print(text)
        
        print("-" * 50)
        print(f"⏱️  Generation time: {generation_time:.2f} seconds")
        print(f"📊 Tokens generated: ~{len(generated_texts[0].split())}")
        print(f"🎯 Model: {inference_engine.config.get('model_name', 'OpenLLM')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies for inference: {e}")
        print("   Please install: pip install fastapi uvicorn")
        return False
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cmd_evaluate(args):
    """
    Execute model evaluation command.
    
    This function implements comprehensive model evaluation including intrinsic
    metrics (perplexity) and downstream task performance assessment.
    
    Args:
        args: Namespace containing CLI arguments including:
            - model_path: Path to trained model directory
            - eval_data: Path to evaluation dataset (optional)
            - metrics: Comma-separated list of metrics to compute
            - output_dir: Directory to save evaluation results
            - format: Model format (auto-detect by default)
    
    Returns:
        bool: True if evaluation succeeded, False otherwise
        
    Implementation Details:
        - Uses evaluate_model.py's ModelEvaluator class for comprehensive testing
        - Computes perplexity on held-out data if provided
        - Runs downstream task evaluation (reading comprehension, sentiment, etc.)
        - Generates detailed evaluation report with metrics and examples
        - Saves results to JSON file for further analysis
    """
    print("📊 OpenLLM Model Evaluation")
    print("=" * 40)
    
    try:
        # Import evaluation functionality
        # We import here to avoid circular imports and handle missing dependencies
        import sys
        import os
        import json
        from pathlib import Path
        
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from evaluate_model import ModelEvaluator
        
        # Validate model path exists
        # Early validation prevents confusing error messages later
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"❌ Model path not found: {args.model_path}")
            print("   Please check the path and try again.")
            return False
        
        # Determine output directory for results
        # Create output directory if it doesn't exist
        output_dir = Path(getattr(args, 'output_dir', 'evaluation_results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse requested metrics
        # Default to comprehensive evaluation if not specified
        requested_metrics = getattr(args, 'metrics', 'perplexity,generation,downstream').split(',')
        requested_metrics = [m.strip() for m in requested_metrics]
        
        print(f"📂 Loading model from: {args.model_path}")
        print(f"📋 Requested metrics: {', '.join(requested_metrics)}")
        print(f"💾 Results will be saved to: {output_dir}")
        
        # Initialize model evaluator
        # This handles model loading and tokenizer setup
        evaluator = ModelEvaluator(
            model_dir=str(model_path),
            tokenizer_path=getattr(args, 'tokenizer_path', None)  # Auto-detect if not provided
        )
        
        # Prepare evaluation results container
        # This will store all evaluation metrics and examples
        evaluation_results = {
            'model_info': {
                'model_path': str(model_path),
                'model_name': evaluator.config.get('model_name', 'OpenLLM'),
                'parameters': evaluator.model.get_num_params(),
                'evaluation_time': None
            },
            'metrics': {},
            'examples': {},
            'summary': {}
        }
        
        import time
        start_time = time.time()
        
        # 1. Perplexity Evaluation
        # This measures how well the model predicts the next token
        if 'perplexity' in requested_metrics:
            print(f"\n🔍 Computing perplexity...")
            
            eval_data_path = getattr(args, 'eval_data', None)
            if eval_data_path and Path(eval_data_path).exists():
                # Use provided evaluation data
                perplexity_result = evaluator.evaluate_perplexity(eval_data_path)
            else:
                # Use a subset of training data for perplexity calculation
                print("   No eval data provided, using default test set")
                perplexity_result = evaluator.evaluate_perplexity()
            
            evaluation_results['metrics']['perplexity'] = perplexity_result
            
            print(f"   ✅ Perplexity: {perplexity_result.get('perplexity', 'N/A'):.2f}")
            print(f"   📊 Loss: {perplexity_result.get('loss', 'N/A'):.4f}")
        
        # 2. Text Generation Quality Assessment
        # This evaluates the coherence and quality of generated text
        if 'generation' in requested_metrics:
            print(f"\n✍️  Evaluating text generation quality...")
            
            generation_result = evaluator.evaluate_text_generation()
            evaluation_results['metrics']['generation'] = generation_result
            evaluation_results['examples']['generation'] = generation_result.get('examples', [])
            
            print(f"   ✅ Average quality score: {generation_result.get('average_quality', 'N/A'):.2f}")
            print(f"   📝 Generated {len(generation_result.get('examples', []))} examples")
        
        # 3. Downstream Task Evaluation
        # This tests specific capabilities like reading comprehension
        if 'downstream' in requested_metrics:
            print(f"\n🎯 Evaluating downstream tasks...")
            
            downstream_result = evaluator.evaluate_downstream_tasks()
            evaluation_results['metrics']['downstream'] = downstream_result
            evaluation_results['examples']['downstream'] = {
                task: result.get('examples', [])
                for task, result in downstream_result.items()
            }
            
            # Display summary of downstream results
            for task_name, task_result in downstream_result.items():
                accuracy = task_result.get('accuracy', 0) * 100
                print(f"   ✅ {task_name.replace('_', ' ').title()}: {accuracy:.1f}%")
        
        # Calculate total evaluation time
        evaluation_time = time.time() - start_time
        evaluation_results['model_info']['evaluation_time'] = evaluation_time
        
        # Generate evaluation summary
        # This provides a high-level overview of model performance
        summary = {
            'overall_score': 0.0,  # Will be calculated based on available metrics
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Calculate overall score based on available metrics
        scores = []
        
        if 'perplexity' in evaluation_results['metrics']:
            ppl = evaluation_results['metrics']['perplexity'].get('perplexity', float('inf'))
            # Convert perplexity to 0-100 score (lower perplexity is better)
            ppl_score = max(0, 100 - (ppl - 10) * 5)  # Rough conversion
            scores.append(ppl_score)
            
            if ppl < 15:
                summary['strengths'].append("Good language modeling (low perplexity)")
            else:
                summary['weaknesses'].append("High perplexity indicates poor language modeling")
        
        if 'generation' in evaluation_results['metrics']:
            gen_score = evaluation_results['metrics']['generation'].get('average_quality', 0) * 100
            scores.append(gen_score)
            
            if gen_score > 70:
                summary['strengths'].append("High-quality text generation")
            else:
                summary['weaknesses'].append("Text generation needs improvement")
        
        if 'downstream' in evaluation_results['metrics']:
            downstream_scores = []
            for task_result in evaluation_results['metrics']['downstream'].values():
                downstream_scores.append(task_result.get('accuracy', 0) * 100)
            
            if downstream_scores:
                avg_downstream = sum(downstream_scores) / len(downstream_scores)
                scores.append(avg_downstream)
                
                if avg_downstream > 50:
                    summary['strengths'].append("Good performance on downstream tasks")
                else:
                    summary['weaknesses'].append("Poor downstream task performance")
        
        # Calculate overall score
        if scores:
            summary['overall_score'] = sum(scores) / len(scores)
        
        # Add recommendations based on performance
        if summary['overall_score'] < 40:
            summary['recommendations'].extend([
                "Consider training for more steps",
                "Verify training data quality",
                "Check model architecture and hyperparameters"
            ])
        elif summary['overall_score'] < 70:
            summary['recommendations'].extend([
                "Model shows promise - consider extended training",
                "Fine-tune on specific downstream tasks"
            ])
        else:
            summary['recommendations'].append("Model performs well - ready for deployment")
        
        evaluation_results['summary'] = summary
        
        # Save detailed results to file
        # This allows for further analysis and comparison between models
        results_file = output_dir / f"evaluation_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Display comprehensive results summary
        print(f"\n" + "=" * 60)
        print(f"📊 EVALUATION SUMMARY")
        print(f"=" * 60)
        print(f"🎯 Overall Score: {summary['overall_score']:.1f}/100")
        print(f"⏱️  Evaluation Time: {evaluation_time:.1f} seconds")
        
        if summary['strengths']:
            print(f"\n✅ Strengths:")
            for strength in summary['strengths']:
                print(f"   • {strength}")
        
        if summary['weaknesses']:
            print(f"\n⚠️  Areas for Improvement:")
            for weakness in summary['weaknesses']:
                print(f"   • {weakness}")
        
        if summary['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in summary['recommendations']:
                print(f"   • {rec}")
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        print(f"🎉 Evaluation completed successfully!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies for evaluation: {e}")
        print("   Please check that all required packages are installed.")
        return False
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
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

    # --- Optional: Enterprise module integration ---
    # Load enterprise-only CLI commands if an external module is available.
    # This preserves the core's open-source nature while allowing private
    # extensions to register additional commands without modifying core code.
    try:
        from enterprise_integration import load_enterprise_cli

        if load_enterprise_cli(subparsers):
            print("🧩 Enterprise extensions detected and loaded")
        else:
            # No enterprise plugin found (normal for open-source-only usage)
            pass
    except Exception:
        # Never fail core CLI due to enterprise integration issues
        pass
    
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
