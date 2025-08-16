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
OpenLLM Model Evaluation Script

This script implements comprehensive evaluation for trained OpenLLM models,
including intrinsic evaluation (perplexity, loss) and text generation quality
assessment as specified in Step 5 of the training pipeline.

Usage:
    python core/src/evaluate_model.py \
        --model_dir models/openllm-medium \
        --eval_data data/clean/validation_data.txt \
        --metrics perplexity,loss

Features:
- Perplexity calculation on held-out data
- Text generation quality assessment
- Multiple evaluation metrics
- Comprehensive quality benchmarks
- JSON output for downstream analysis

Author: Louis Chua Bean Chong
License: GPLv3
"""

import argparse
import json
import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
import sentencepiece as smp

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPTModel, create_model
from data_loader import TextDataLoader


class ModelEvaluator:
    """
    Comprehensive evaluator for OpenLLM models.

    Implements intrinsic evaluation metrics and text generation quality
    assessment following the training pipeline specifications.
    """

    def __init__(self, model: GPTModel, tokenizer_path: str, device: str = "cpu"):
        """
        Initialize the model evaluator.

        Args:
            model: Trained GPT model
            tokenizer_path: Path to tokenizer model file
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device

        # Load tokenizer
        self.tokenizer = smp.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)

        print(f"🔧 ModelEvaluator initialized")
        print(f"  Device: {device}")
        print(f"  Model parameters: {model.get_num_params():,}")
        print(f"  Vocabulary size: {self.tokenizer.vocab_size():,}")

    def evaluate_perplexity(
        self, eval_data: List[str], max_seq_len: int = 512, batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Calculate perplexity on evaluation data.

        Args:
            eval_data: List of text passages for evaluation
            max_seq_len: Maximum sequence length for evaluation
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with loss and perplexity metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_sequences = 0

        print(f"📊 Calculating perplexity on {len(eval_data)} passages...")

        with torch.no_grad():
            for i, text in enumerate(eval_data):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(eval_data)} passages")

                # Tokenize text
                tokens = self.tokenizer.encode(text)
                if len(tokens) < 2:
                    continue

                # Truncate if too long
                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]

                # Create input and target tensors
                input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=self.device)
                target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=self.device)

                # Forward pass
                logits, loss = self.model(input_ids, target_ids)

                # Accumulate loss
                seq_length = len(tokens) - 1
                total_loss += loss.item() * seq_length
                total_tokens += seq_length
                num_sequences += 1

        # Calculate metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = math.exp(min(avg_loss, 10))  # Cap to prevent overflow

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "num_sequences": num_sequences,
        }

    def evaluate_text_generation(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        num_samples: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate text generation quality.

        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            num_samples: Number of samples per prompt

        Returns:
            List of generation results with quality metrics
        """
        self.model.eval()
        results = []

        print(f"✍️  Evaluating text generation on {len(prompts)} prompts...")

        with torch.no_grad():
            for prompt in prompts:
                prompt_results = []

                for sample_idx in range(num_samples):
                    # Tokenize prompt
                    input_ids = self.tokenizer.encode(prompt)
                    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

                    start_time = time.time()

                    # Generate text
                    output = self.model.generate(
                        input_tensor,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        top_k=top_k,
                    )

                    generation_time = time.time() - start_time

                    # Decode output
                    generated_ids = output[0].tolist()
                    full_text = self.tokenizer.decode(generated_ids)
                    generated_text = self.tokenizer.decode(generated_ids[len(input_ids) :])

                    # Calculate quality metrics
                    quality_metrics = self._assess_generation_quality(generated_text)

                    prompt_results.append(
                        {
                            "prompt": prompt,
                            "generated_text": generated_text,
                            "full_text": full_text,
                            "generation_time": generation_time,
                            "tokens_generated": len(generated_ids) - len(input_ids),
                            "tokens_per_second": (len(generated_ids) - len(input_ids))
                            / generation_time,
                            "quality_metrics": quality_metrics,
                        }
                    )

                results.extend(prompt_results)

        return results

    def _assess_generation_quality(self, text: str) -> Dict[str, float]:
        """
        Assess basic quality metrics for generated text.

        Args:
            text: Generated text to assess

        Returns:
            Dictionary of quality metrics
        """
        if not text.strip():
            return {
                "length": 0,
                "avg_word_length": 0,
                "repetition_rate": 1.0,
                "coherence_score": 0.0,
            }

        words = text.split()

        # Basic metrics
        length = len(words)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        # Repetition rate (simple n-gram repetition)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        unique_bigrams = len(set(bigrams))
        repetition_rate = 1 - (unique_bigrams / len(bigrams) if bigrams else 0)

        # Simple coherence score (based on sentence structure)
        sentences = text.split(".")
        valid_sentences = [s for s in sentences if len(s.strip().split()) > 3]
        coherence_score = len(valid_sentences) / len(sentences) if sentences else 0

        return {
            "length": length,
            "avg_word_length": avg_word_length,
            "repetition_rate": repetition_rate,
            "coherence_score": coherence_score,
        }

    def evaluate_downstream_tasks(self) -> Dict[str, Any]:
        """
        Evaluate model performance on downstream tasks.

        This function implements basic downstream task evaluation including:
        - Reading comprehension (simplified SQUAD-style)
        - Sentiment analysis (few-shot)
        - Common sense reasoning

        Returns:
            Dictionary of downstream task results
        """
        results = {}

        # 1. Reading Comprehension (Simplified SQUAD-style)
        results["reading_comprehension"] = self._evaluate_reading_comprehension()

        # 2. Sentiment Analysis (Few-shot learning)
        results["sentiment_analysis"] = self._evaluate_sentiment_analysis()

        # 3. Common Sense Reasoning
        results["reasoning"] = self._evaluate_reasoning()

        # 4. Text Completion Quality
        results["text_completion"] = self._evaluate_text_completion()

        return results

    def _evaluate_reading_comprehension(self) -> Dict[str, Any]:
        """Simplified reading comprehension evaluation."""
        # Sample reading comprehension tasks
        tasks = [
            {
                "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
                "question": "Who is the Eiffel Tower named after?",
                "expected": "Gustave Eiffel",
            },
            {
                "context": "Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991.",
                "question": "When was Python first released?",
                "expected": "1991",
            },
            {
                "context": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                "question": "What is machine learning a subset of?",
                "expected": "artificial intelligence",
            },
        ]

        correct = 0
        total = len(tasks)

        for task in tasks:
            prompt = f"Context: {task['context']}\nQuestion: {task['question']}\nAnswer:"

            # Generate answer
            input_ids = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

            with torch.no_grad():
                output = self.model.generate(input_tensor, max_new_tokens=20, temperature=0.1)

            generated_ids = output[0].tolist()
            answer = self.tokenizer.decode(generated_ids[len(input_ids) :]).strip().lower()

            # Simple substring matching
            if task["expected"].lower() in answer:
                correct += 1

        return {
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
            "score": correct / total,
        }

    def _evaluate_sentiment_analysis(self) -> Dict[str, Any]:
        """Few-shot sentiment analysis evaluation."""
        # Few-shot examples
        examples = "Examples:\nText: 'I love this movie!' Sentiment: Positive\nText: 'This is terrible.' Sentiment: Negative\nText: 'It was okay.' Sentiment: Neutral\n\n"

        # Test cases
        test_cases = [
            {"text": "This is amazing!", "expected": "positive"},
            {"text": "I hate this.", "expected": "negative"},
            {"text": "This is wonderful.", "expected": "positive"},
            {"text": "This is awful.", "expected": "negative"},
            {"text": "It was fine.", "expected": "neutral"},
        ]

        correct = 0
        total = len(test_cases)

        for case in test_cases:
            prompt = f"{examples}Text: '{case['text']}' Sentiment:"

            # Generate sentiment
            input_ids = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

            with torch.no_grad():
                output = self.model.generate(input_tensor, max_new_tokens=5, temperature=0.1)

            generated_ids = output[0].tolist()
            sentiment = self.tokenizer.decode(generated_ids[len(input_ids) :]).strip().lower()

            # Check if expected sentiment is in the generated response
            if case["expected"] in sentiment:
                correct += 1

        return {
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
            "score": correct / total,
        }

    def _evaluate_reasoning(self) -> Dict[str, Any]:
        """Simple reasoning evaluation."""
        # Basic reasoning tasks
        tasks = [
            {
                "question": "If all birds can fly and a penguin is a bird, can a penguin fly?",
                "expected": "no",  # This tests if model knows real-world facts
            },
            {
                "question": "If it is raining outside, should you take an umbrella?",
                "expected": "yes",
            },
            {"question": "What comes after Monday?", "expected": "tuesday"},
            {"question": "Is the sun larger than the earth?", "expected": "yes"},
        ]

        correct = 0
        total = len(tasks)

        for task in tasks:
            prompt = f"Question: {task['question']}\nAnswer:"

            # Generate answer
            input_ids = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

            with torch.no_grad():
                output = self.model.generate(input_tensor, max_new_tokens=10, temperature=0.1)

            generated_ids = output[0].tolist()
            answer = self.tokenizer.decode(generated_ids[len(input_ids) :]).strip().lower()

            # Check if expected answer is in the response
            if task["expected"] in answer:
                correct += 1

        return {
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
            "score": correct / total,
        }

    def _evaluate_text_completion(self) -> Dict[str, Any]:
        """Evaluate text completion quality."""
        # Common phrases that should be completed predictably
        completions = [
            {"prompt": "The capital of France is", "expected_word": "paris"},
            {"prompt": "Two plus two equals", "expected_word": "four"},
            {"prompt": "The largest planet in our solar system is", "expected_word": "jupiter"},
            {"prompt": "Water boils at", "expected_word": "100"},
        ]

        correct = 0
        total = len(completions)

        for completion in completions:
            # Generate completion
            input_ids = self.tokenizer.encode(completion["prompt"])
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

            with torch.no_grad():
                output = self.model.generate(input_tensor, max_new_tokens=5, temperature=0.1)

            generated_ids = output[0].tolist()
            generated_text = self.tokenizer.decode(generated_ids[len(input_ids) :]).strip().lower()

            # Check if expected word appears in completion
            if completion["expected_word"] in generated_text:
                correct += 1

        return {
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
            "score": correct / total,
        }

    def run_comprehensive_evaluation(
        self, eval_data_path: str, metrics: List[str] = None, generation_prompts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive model evaluation.

        Args:
            eval_data_path: Path to evaluation text file
            metrics: List of metrics to compute
            generation_prompts: Prompts for text generation evaluation

        Returns:
            Complete evaluation results
        """
        if metrics is None:
            metrics = ["perplexity", "loss", "generation"]

        if generation_prompts is None:
            generation_prompts = [
                "The history of artificial intelligence",
                "Machine learning algorithms",
                "The future of technology",
                "In a world where",
                "Scientists have discovered",
            ]

        results = {
            "model_info": {
                "parameters": self.model.get_num_params(),
                "device": self.device,
                "vocab_size": self.tokenizer.vocab_size(),
            },
            "evaluation_timestamp": time.time(),
        }

        # Load evaluation data
        print(f"📂 Loading evaluation data from {eval_data_path}")
        if os.path.exists(eval_data_path):
            with open(eval_data_path, "r", encoding="utf-8") as f:
                eval_texts = [line.strip() for line in f if line.strip()]
        else:
            print(f"⚠️  Evaluation file not found, using sample texts")
            eval_texts = [
                "Artificial intelligence is a rapidly growing field of computer science.",
                "Machine learning algorithms can learn patterns from data automatically.",
                "Natural language processing helps computers understand human language.",
                "Deep learning uses neural networks with multiple layers for complex tasks.",
                "The development of large language models has transformed AI applications.",
            ]

        # Intrinsic evaluation
        if "perplexity" in metrics or "loss" in metrics:
            perplexity_results = self.evaluate_perplexity(eval_texts)
            results["intrinsic_evaluation"] = perplexity_results

        # Text generation evaluation
        if "generation" in metrics:
            generation_results = self.evaluate_text_generation(generation_prompts)
            results["generation_evaluation"] = {
                "results": generation_results,
                "summary": self._summarize_generation_results(generation_results),
            }

        # Downstream tasks (placeholder)
        results["downstream_evaluation"] = self.evaluate_downstream_tasks()

        # Overall quality assessment
        results["quality_assessment"] = self._assess_overall_quality(results)

        return results

    def _summarize_generation_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Summarize text generation results."""
        if not results:
            return {}

        total_time = sum(r["generation_time"] for r in results)
        total_tokens = sum(r["tokens_generated"] for r in results)

        quality_metrics = [r["quality_metrics"] for r in results]

        return {
            "avg_generation_time": total_time / len(results),
            "avg_tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
            "avg_length": sum(q["length"] for q in quality_metrics) / len(quality_metrics),
            "avg_repetition_rate": sum(q["repetition_rate"] for q in quality_metrics)
            / len(quality_metrics),
            "avg_coherence_score": sum(q["coherence_score"] for q in quality_metrics)
            / len(quality_metrics),
        }

    def _assess_overall_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall model quality based on evaluation results."""
        assessment = {"quality_level": "unknown", "recommendations": []}

        # Check intrinsic metrics
        if "intrinsic_evaluation" in results:
            perplexity = results["intrinsic_evaluation"].get("perplexity", float("inf"))

            if perplexity < 12:
                assessment["quality_level"] = "good"
                assessment["recommendations"].append("Model shows good perplexity scores")
            elif perplexity < 50:
                assessment["quality_level"] = "fair"
                assessment["recommendations"].append(
                    "Model shows fair performance, could benefit from more training"
                )
            else:
                assessment["quality_level"] = "poor"
                assessment["recommendations"].append(
                    "Model needs significant more training or data improvements"
                )

        # Check generation quality
        if "generation_evaluation" in results:
            summary = results["generation_evaluation"].get("summary", {})
            repetition_rate = summary.get("avg_repetition_rate", 1.0)
            coherence_score = summary.get("avg_coherence_score", 0.0)

            if repetition_rate > 0.7:
                assessment["recommendations"].append(
                    "High repetition rate - consider training longer or adjusting data"
                )
            if coherence_score < 0.3:
                assessment["recommendations"].append(
                    "Low coherence - model may need more training steps"
                )

        return assessment


def load_model_from_directory(model_dir: str, device: str = "cpu") -> Tuple[GPTModel, str]:
    """
    Load model from directory containing checkpoints.

    Args:
        model_dir: Directory containing model files
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer_path)
    """
    model_dir = Path(model_dir)

    # Find best model checkpoint
    best_model_path = model_dir / "best_model.pt"
    if not best_model_path.exists():
        # Look for latest checkpoint
        checkpoints = list(model_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No model checkpoints found in {model_dir}")

        # Get latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
        best_model_path = latest_checkpoint

    print(f"📂 Loading model from {best_model_path}")

    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location=device)

    # Determine model size from config
    config = checkpoint.get("config", {})
    n_layer = config.get("n_layer", 12)

    if n_layer <= 6:
        model_size = "small"
    elif n_layer <= 12:
        model_size = "medium"
    else:
        model_size = "large"

    # Create and load model
    model = create_model(model_size)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"✅ Model loaded successfully ({model_size}, {model.get_num_params():,} parameters)")

    # Find tokenizer
    tokenizer_path = model_dir.parent / "tokenizer" / "tokenizer.model"
    if not tokenizer_path.exists():
        tokenizer_path = Path("data/tokenizer/tokenizer.model")

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    return model, str(tokenizer_path)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate OpenLLM model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python core/src/evaluate_model.py \\
    --model_dir models/small-extended-4k \\
    --eval_data data/clean/training_data.txt
  
  # Specific metrics
  python core/src/evaluate_model.py \\
    --model_dir models/small-extended-4k \\
    --metrics perplexity,generation \\
    --output results.json
        """,
    )

    parser.add_argument("--model_dir", required=True, help="Directory containing trained model")

    parser.add_argument(
        "--eval_data", help="Path to evaluation text file (default: use sample texts)"
    )

    parser.add_argument(
        "--metrics",
        default="perplexity,loss,generation",
        help="Comma-separated list of metrics to evaluate (default: perplexity,loss,generation)",
    )

    parser.add_argument("--output", help="Output JSON file for results (default: print to console)")

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for evaluation (default: auto)",
    )

    parser.add_argument(
        "--generation_prompts", help="File containing prompts for text generation evaluation"
    )

    args = parser.parse_args()

    print("📊 OpenLLM Model Evaluation")
    print("=" * 50)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    try:
        # Load model
        model, tokenizer_path = load_model_from_directory(args.model_dir, device)

        # Create evaluator
        evaluator = ModelEvaluator(model, tokenizer_path, device)

        # Parse metrics
        metrics = [m.strip() for m in args.metrics.split(",")]

        # Load generation prompts if specified
        generation_prompts = None
        if args.generation_prompts and os.path.exists(args.generation_prompts):
            with open(args.generation_prompts, "r", encoding="utf-8") as f:
                generation_prompts = [line.strip() for line in f if line.strip()]

        # Run evaluation
        eval_data_path = args.eval_data or "data/clean/training_data.txt"
        results = evaluator.run_comprehensive_evaluation(
            eval_data_path, metrics, generation_prompts
        )

        # Output results
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
        else:
            print(f"\n📊 Evaluation Results:")
            print("=" * 50)

            # Print key metrics
            if "intrinsic_evaluation" in results:
                intrinsic = results["intrinsic_evaluation"]
                print(f"📈 Intrinsic Metrics:")
                print(f"  Loss: {intrinsic['loss']:.4f}")
                print(f"  Perplexity: {intrinsic['perplexity']:.2f}")
                print(f"  Sequences evaluated: {intrinsic['num_sequences']:,}")

            if "generation_evaluation" in results:
                gen_summary = results["generation_evaluation"]["summary"]
                print(f"\n✍️  Generation Quality:")
                print(
                    f"  Avg generation speed: {gen_summary['avg_tokens_per_second']:.1f} tokens/sec"
                )
                print(f"  Avg text length: {gen_summary['avg_length']:.1f} words")
                print(f"  Repetition rate: {gen_summary['avg_repetition_rate']:.3f}")
                print(f"  Coherence score: {gen_summary['avg_coherence_score']:.3f}")

            # Quality assessment
            if "quality_assessment" in results:
                assessment = results["quality_assessment"]
                print(f"\n🎯 Overall Assessment:")
                print(f"  Quality Level: {assessment['quality_level'].upper()}")
                for rec in assessment["recommendations"]:
                    print(f"  • {rec}")

        print(f"\n🎉 Evaluation completed successfully!")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
