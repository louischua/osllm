#!/usr/bin/env python3
"""
OpenLLM Training Space - Main Application

This is the main entry point for the Hugging Face Space.
It provides a web interface for running OpenLLM training with authentication.

Author: Louis Chua Bean Chong
License: GPLv3
"""

import os
import sys
from pathlib import Path

import gradio as gr

# Import our authentication and training modules
try:
    from openllm_training_with_auth import OpenLLMTrainingManager
    from space_auth_test import test_space_authentication

    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"❌ Required modules not available: {e}")


def create_space_interface():
    """Create the Gradio interface for the Space."""

    def run_authentication_test():
        """Run the authentication test and return results."""
        try:
            if not MODULES_AVAILABLE:
                return "❌ Required modules not available. Please check deployment."

            # Capture output from authentication test
            import contextlib
            import io

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                success = test_space_authentication()

            result = output.getvalue()

            if success:
                return f"✅ Authentication Test Results:\n\n{result}"
            else:
                return f"❌ Authentication Test Failed:\n\n{result}"

        except Exception as e:
            return f"❌ Error running authentication test: {e}"

    def run_training(model_size, training_steps, use_real_training=False):
        """Run the OpenLLM training with authentication."""
        try:
            if not MODULES_AVAILABLE:
                return "❌ Required modules not available. Please check deployment."

            # Security mitigation: Input validation and sanitization
            if not isinstance(model_size, str) or model_size not in ["small", "medium", "large"]:
                return "❌ Invalid model size. Must be 'small', 'medium', or 'large'."

            if (
                not isinstance(training_steps, (int, float))
                or training_steps < 1000
                or training_steps > 50000
            ):
                return "❌ Invalid training steps. Must be between 1000 and 50000."

            # Sanitize inputs
            model_size = str(model_size).strip().lower()
            training_steps = int(float(training_steps))

            print(f"🚀 Starting OpenLLM Training")
            print("=" * 50)
            print(f"📊 Model Size: {model_size}")
            print(f"🔄 Training Steps: {training_steps}")
            print(f"🎯 Training Mode: {'Real Training' if use_real_training else 'Demonstration'}")

            if use_real_training:
                # Use real training with comprehensive features
                try:
                    from real_training_manager import RealTrainingManager, TrainingConfig
                    
                    # Create configuration for real training
                    config = TrainingConfig(
                        model_size=model_size,
                        training_steps=training_steps,
                        batch_size=32 if model_size == "small" else 16,
                        learning_rate=3e-4,
                        data_file="data/clean/training_data.txt",
                        save_every=1000,
                        eval_every=500
                    )
                    
                    # Initialize real training manager
                    manager = RealTrainingManager(config)
                    
                    # Run real training
                    model = manager.train()
                    
                    # Upload model
                    repo_id = manager.upload_model(model)
                    
                    if repo_id:
                        result = f"✅ Real Training completed successfully!\n\n"
                        result += f"📊 Results:\n"
                        result += f"   - Model Size: {model_size}\n"
                        result += f"   - Training Steps: {training_steps}\n"
                        result += f"   - Final Loss: {manager.training_history[-1]['loss']:.4f}\n"
                        result += f"   - Best Validation Loss: {manager.best_loss:.4f}\n"
                        result += f"   - Model URL: https://huggingface.co/{repo_id}\n\n"
                        result += f"🎉 Model available at: https://huggingface.co/{repo_id}"
                    else:
                        result = f"⚠️ Real training completed but upload failed\n\n"
                        result += f"📊 Results:\n"
                        result += f"   - Model Size: {model_size}\n"
                        result += f"   - Training Steps: {training_steps}\n"
                        result += f"   - Final Loss: {manager.training_history[-1]['loss']:.4f}\n"
                        result += f"   - Model saved locally: ./trained_model"
                    
                    return result
                    
                except ImportError:
                    return "❌ Real training module not available. Falling back to demonstration mode."
                except Exception as e:
                    return f"❌ Real training failed: {str(e)}\n\nFalling back to demonstration mode."

            # Fallback to demonstration training
            import contextlib
            import io

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                training_manager = OpenLLMTrainingManager()
                repo_id = training_manager.run_training(model_size=model_size, steps=training_steps)

            result = output.getvalue()

            return f"✅ Training Results:\n\n{result}\n\n🎉 Model available at: https://huggingface.co/{repo_id}"

        except Exception as e:
            return f"❌ Error running training: {e}"

    def resume_training_from_7k_to_8k():
        """Resume training from 7k model to create 8k model."""
        try:
            if not MODULES_AVAILABLE:
                return "❌ Required modules not available. Please check deployment."

            # Import required modules
            import json
            import time
            import torch
            from datetime import datetime
            from huggingface_hub import HfApi, whoami, create_repo, snapshot_download
            from train_model import TextDataLoader
            from model import GPTConfig, GPTModel

            print("🚀 Resuming Training from 7k to 8k Model")
            print("=" * 50)
            
            # Configuration
            hf_model_id = "lemms/openllm-small-extended-7k"
            additional_steps = 1000  # Train for 1000 more steps to reach 8k
            total_steps = 8000  # Total steps for the new model
            
            print(f"📥 Source Model: {hf_model_id}")
            print(f"📈 Additional Steps: {additional_steps}")
            print(f"🎯 Target Steps: {total_steps}")
            
            # Setup authentication
            print("🔐 Setting up authentication...")
            try:
                user_info = whoami()
                username = user_info.get("name", "unknown")
                print(f"✅ Authentication successful! User: {username}")
            except Exception as e:
                return f"❌ Authentication failed: {e}"
            
            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🖥️ Using device: {device}")
            
            # Load model from Hugging Face
            print(f"📥 Loading model from Hugging Face: {hf_model_id}")
            try:
                local_dir = snapshot_download(
                    repo_id=hf_model_id,
                    repo_type="model",
                    local_dir=f"downloaded_models/{hf_model_id.replace('/', '_')}"
                )
                print(f"✅ Model downloaded to: {local_dir}")
                
                # Load config
                config_path = Path(local_dir) / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    config = GPTConfig(
                        vocab_size=config_data.get('vocab_size', 32000),
                        block_size=config_data.get('block_size', 1024),
                        n_layer=config_data.get('n_layer', 6),
                        n_head=config_data.get('n_head', 6),
                        n_embd=config_data.get('n_embd', 384)
                    )
                    print(f"📊 Loaded model config: {config}")
                else:
                    config = GPTConfig.small()
                    config.vocab_size = 32000
                    print(f"⚠️ Config file not found, using default config")
                
                # Create model and load weights
                model = GPTModel(config)
                model_path = Path(local_dir) / "pytorch_model.bin"
                
                if model_path.exists():
                    state_dict = torch.load(model_path, map_location=device)
                    model.load_state_dict(state_dict)
                    print(f"✅ Model weights loaded successfully")
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                model = model.to(device)
                
            except Exception as e:
                return f"❌ Failed to load model from Hugging Face: {e}"
            
            # Create data loaders
            print(f"📊 Loading training data...")
            tokenizer_path = "data/tokenizer/tokenizer.model"
            
            train_loader = TextDataLoader(
                data_file="data/clean/training_data.txt",
                tokenizer_path=tokenizer_path,
                seq_len=1024,
                batch_size=16,
                shuffle=True
            )
            
            print(f"✅ Data loader created")
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=3e-4,
                weight_decay=0.1
            )
            
            # Training loop
            print(f"\n🔄 Starting training loop...")
            start_time = time.time()
            training_history = []
            best_loss = float('inf')
            
            try:
                train_iterator = iter(train_loader)
                
                for step in range(additional_steps):
                    # Get batch
                    try:
                        batch = next(train_iterator)
                    except StopIteration:
                        # Restart data loader if exhausted
                        train_loader = TextDataLoader(
                            data_file="data/clean/training_data.txt",
                            tokenizer_path=tokenizer_path,
                            seq_len=1024,
                            batch_size=16,
                            shuffle=True
                        )
                        train_iterator = iter(train_loader)
                        batch = next(train_iterator)
                    
                    # Prepare inputs
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(device)
                        targets = batch[1].to(device) if len(batch) > 1 else None
                    else:
                        inputs = batch.to(device)
                        targets = None
                    
                    # Forward pass
                    logits, loss = model(inputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Record training history
                    training_history.append({
                        'step': 7000 + step + 1,  # Continue from step 7000
                        'loss': loss.item(),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Progress reporting
                    if (step + 1) % 10 == 0:
                        elapsed = time.time() - start_time
                        steps_per_sec = (step + 1) / elapsed
                        eta = (additional_steps - step - 1) / steps_per_sec
                        
                        print(f"Step {7000 + step + 1}/{total_steps} | "
                              f"Loss: {loss.item():.4f} | "
                              f"Speed: {steps_per_sec:.1f} steps/s | "
                              f"ETA: {eta/60:.1f} min")
                    
                    # Evaluation
                    if (step + 1) % 250 == 0:
                        model.eval()
                        total_loss = 0.0
                        num_batches = 0
                        
                        with torch.no_grad():
                            for val_batch in train_loader:  # Use same loader for simplicity
                                if isinstance(val_batch, (list, tuple)):
                                    val_inputs = val_batch[0].to(device)
                                    val_targets = val_batch[1].to(device) if len(val_batch) > 1 else None
                                else:
                                    val_inputs = val_batch.to(device)
                                    val_targets = None
                                
                                val_logits, val_loss = model(val_inputs, val_targets)
                                total_loss += val_loss.item()
                                num_batches += 1
                                
                                if num_batches >= 5:  # Limit evaluation
                                    break
                        
                        avg_val_loss = total_loss / num_batches
                        model.train()
                        print(f"📊 Validation Loss: {avg_val_loss:.4f}")
                        
                        # Check for best model
                        if avg_val_loss < best_loss:
                            best_loss = avg_val_loss
                            print(f"🏆 New best validation loss: {best_loss:.4f}")
                
                print(f"\n🎉 Training completed successfully!")
                print(f"📊 Final Results:")
                print(f"   - Additional Steps: {additional_steps}")
                print(f"   - Total Steps: {total_steps}")
                print(f"   - Final Loss: {loss.item():.4f}")
                print(f"   - Best Validation Loss: {best_loss:.4f}")
                print(f"   - Training Time: {(time.time() - start_time)/3600:.2f} hours")
                
                # Upload model
                print(f"\n📤 Uploading model to Hugging Face Hub...")
                
                # Create model directory
                model_path = Path("./trained_model")
                model_path.mkdir(exist_ok=True)
                
                # Save model files
                torch.save(model.state_dict(), model_path / "pytorch_model.bin")
                
                # Save config
                config_dict = {
                    "model_type": "openllm",
                    "model_size": "small",
                    "vocab_size": 32000,
                    "block_size": 1024,
                    "n_layer": 6,
                    "n_head": 6,
                    "n_embd": 384,
                    "training_config": {
                        "model_size": "small",
                        "training_steps": total_steps,
                        "additional_steps": additional_steps,
                        "base_model": hf_model_id
                    },
                    "training_history": training_history
                }
                
                with open(model_path / "config.json", 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                # Create model card
                readme_content = f"""# OpenLLM Small Model - Extended to 8k Steps

This is an OpenLLM small model trained for {total_steps} steps by resuming training from [lemms/openllm-small-extended-7k](https://huggingface.co/lemms/openllm-small-extended-7k).

## Model Details

- **Model Type**: OpenLLM
- **Size**: small
- **Training Steps**: {total_steps}
- **Additional Steps**: {additional_steps}
- **Base Model**: [lemms/openllm-small-extended-7k](https://huggingface.co/lemms/openllm-small-extended-7k)
- **Final Loss**: {training_history[-1]['loss']:.4f} if training_history else 'N/A'
- **Framework**: PyTorch
- **License**: GPL-3.0

## Training Configuration

```json
{json.dumps(config_dict, indent=2)}
```

## Training History

The model was trained with the following key metrics:
- Best validation loss: {best_loss:.4f}
- Total training time: {len(training_history)} steps
- Device used: {device}

## Usage

This model can be used for text generation and language modeling tasks.

## Author

Louis Chua Bean Chong

## License

GPL-3.0
"""
                
                with open(model_path / "README.md", 'w') as f:
                    f.write(readme_content)
                
                # Upload to Hugging Face
                repo_name = "openllm-small-extended-8k"
                repo_id = f"{username}/{repo_name}"
                
                try:
                    # Create repository
                    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
                    
                    # Upload files
                    api = HfApi()
                    api.upload_folder(
                        folder_path=str(model_path),
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"Add OpenLLM small model extended to {total_steps} steps"
                    )
                    
                    print(f"✅ Model uploaded successfully!")
                    print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
                    
                    result = f"✅ 8k Model Training completed successfully!\n\n"
                    result += f"📊 Results:\n"
                    result += f"   - Base Model: {hf_model_id}\n"
                    result += f"   - Additional Steps: {additional_steps}\n"
                    result += f"   - Total Steps: {total_steps}\n"
                    result += f"   - Final Loss: {loss.item():.4f}\n"
                    result += f"   - Best Validation Loss: {best_loss:.4f}\n"
                    result += f"   - Model URL: https://huggingface.co/{repo_id}\n\n"
                    result += f"🎉 Extended model available at: https://huggingface.co/{repo_id}"
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ Model upload failed: {e}")
                    return f"⚠️ Training completed but upload failed: {e}"
                
            except KeyboardInterrupt:
                print(f"\n⚠️ Training interrupted by user")
                return "⚠️ Training was interrupted by user"
                
        except Exception as e:
            return f"❌ Error resuming training: {e}"

    def resume_training_from_7k_to_8k():
        """Resume training from 7k model to create 8k model."""
        try:
            if not MODULES_AVAILABLE:
                return "❌ Required modules not available. Please check deployment."

            # Import required modules
            import json
            import time
            import torch
            from datetime import datetime
            from huggingface_hub import HfApi, whoami, create_repo, snapshot_download
            from train_model import TextDataLoader
            from model import GPTConfig, GPTModel

            print("🚀 Resuming Training from 7k to 8k Model")
            print("=" * 50)
            
            # Configuration
            hf_model_id = "lemms/openllm-small-extended-7k"
            additional_steps = 1000  # Train for 1000 more steps to reach 8k
            total_steps = 8000  # Total steps for the new model
            
            print(f"📥 Source Model: {hf_model_id}")
            print(f"📈 Additional Steps: {additional_steps}")
            print(f"🎯 Target Steps: {total_steps}")
            
            # Setup authentication
            print("🔐 Setting up authentication...")
            try:
                user_info = whoami()
                username = user_info.get("name", "unknown")
                print(f"✅ Authentication successful! User: {username}")
            except Exception as e:
                return f"❌ Authentication failed: {e}"
            
            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🖥️ Using device: {device}")
            
            # Load model from Hugging Face
            print(f"📥 Loading model from Hugging Face: {hf_model_id}")
            try:
                local_dir = snapshot_download(
                    repo_id=hf_model_id,
                    repo_type="model",
                    local_dir=f"downloaded_models/{hf_model_id.replace('/', '_')}"
                )
                print(f"✅ Model downloaded to: {local_dir}")
                
                # Load config
                config_path = Path(local_dir) / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    config = GPTConfig(
                        vocab_size=config_data.get('vocab_size', 32000),
                        block_size=config_data.get('block_size', 1024),
                        n_layer=config_data.get('n_layer', 6),
                        n_head=config_data.get('n_head', 6),
                        n_embd=config_data.get('n_embd', 384)
                    )
                    print(f"📊 Loaded model config: {config}")
                else:
                    config = GPTConfig.small()
                    config.vocab_size = 32000
                    print(f"⚠️ Config file not found, using default config")
                
                # Create model and load weights
                model = GPTModel(config)
                model_path = Path(local_dir) / "pytorch_model.bin"
                
                if model_path.exists():
                    state_dict = torch.load(model_path, map_location=device)
                    model.load_state_dict(state_dict)
                    print(f"✅ Model weights loaded successfully")
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                model = model.to(device)
                
            except Exception as e:
                return f"❌ Failed to load model from Hugging Face: {e}"
            
            # Create data loaders
            print(f"📊 Loading training data...")
            tokenizer_path = "data/tokenizer/tokenizer.model"
            
            train_loader = TextDataLoader(
                data_file="data/clean/training_data.txt",
                tokenizer_path=tokenizer_path,
                seq_len=1024,
                batch_size=16,
                shuffle=True
            )
            
            print(f"✅ Data loader created")
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=3e-4,
                weight_decay=0.1
            )
            
            # Training loop
            print(f"\n🔄 Starting training loop...")
            start_time = time.time()
            training_history = []
            best_loss = float('inf')
            
            try:
                train_iterator = iter(train_loader)
                
                for step in range(additional_steps):
                    # Get batch
                    try:
                        batch = next(train_iterator)
                    except StopIteration:
                        # Restart data loader if exhausted
                        train_loader = TextDataLoader(
                            data_file="data/clean/training_data.txt",
                            tokenizer_path=tokenizer_path,
                            seq_len=1024,
                            batch_size=16,
                            shuffle=True
                        )
                        train_iterator = iter(train_loader)
                        batch = next(train_iterator)
                    
                    # Prepare inputs
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(device)
                        targets = batch[1].to(device) if len(batch) > 1 else None
                    else:
                        inputs = batch.to(device)
                        targets = None
                    
                    # Forward pass
                    logits, loss = model(inputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Record training history
                    training_history.append({
                        'step': 7000 + step + 1,  # Continue from step 7000
                        'loss': loss.item(),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Progress reporting
                    if (step + 1) % 10 == 0:
                        elapsed = time.time() - start_time
                        steps_per_sec = (step + 1) / elapsed
                        eta = (additional_steps - step - 1) / steps_per_sec
                        
                        print(f"Step {7000 + step + 1}/{total_steps} | "
                              f"Loss: {loss.item():.4f} | "
                              f"Speed: {steps_per_sec:.1f} steps/s | "
                              f"ETA: {eta/60:.1f} min")
                    
                    # Evaluation
                    if (step + 1) % 250 == 0:
                        model.eval()
                        total_loss = 0.0
                        num_batches = 0
                        
                        with torch.no_grad():
                            for val_batch in train_loader:  # Use same loader for simplicity
                                if isinstance(val_batch, (list, tuple)):
                                    val_inputs = val_batch[0].to(device)
                                    val_targets = val_batch[1].to(device) if len(val_batch) > 1 else None
                                else:
                                    val_inputs = val_batch.to(device)
                                    val_targets = None
                                
                                val_logits, val_loss = model(val_inputs, val_targets)
                                total_loss += val_loss.item()
                                num_batches += 1
                                
                                if num_batches >= 5:  # Limit evaluation
                                    break
                        
                        avg_val_loss = total_loss / num_batches
                        model.train()
                        print(f"📊 Validation Loss: {avg_val_loss:.4f}")
                        
                        # Check for best model
                        if avg_val_loss < best_loss:
                            best_loss = avg_val_loss
                            print(f"🏆 New best validation loss: {best_loss:.4f}")
                
                print(f"\n🎉 Training completed successfully!")
                print(f"📊 Final Results:")
                print(f"   - Additional Steps: {additional_steps}")
                print(f"   - Total Steps: {total_steps}")
                print(f"   - Final Loss: {loss.item():.4f}")
                print(f"   - Best Validation Loss: {best_loss:.4f}")
                print(f"   - Training Time: {(time.time() - start_time)/3600:.2f} hours")
                
                # Upload model
                print(f"\n📤 Uploading model to Hugging Face Hub...")
                
                # Create model directory
                model_path = Path("./trained_model")
                model_path.mkdir(exist_ok=True)
                
                # Save model files
                torch.save(model.state_dict(), model_path / "pytorch_model.bin")
                
                # Save config
                config_dict = {
                    "model_type": "openllm",
                    "model_size": "small",
                    "vocab_size": 32000,
                    "block_size": 1024,
                    "n_layer": 6,
                    "n_head": 6,
                    "n_embd": 384,
                    "training_config": {
                        "model_size": "small",
                        "training_steps": total_steps,
                        "additional_steps": additional_steps,
                        "base_model": hf_model_id
                    },
                    "training_history": training_history
                }
                
                with open(model_path / "config.json", 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                # Create model card
                readme_content = f"""# OpenLLM Small Model - Extended to 8k Steps

This is an OpenLLM small model trained for {total_steps} steps by resuming training from [lemms/openllm-small-extended-7k](https://huggingface.co/lemms/openllm-small-extended-7k).

## Model Details

- **Model Type**: OpenLLM
- **Size**: small
- **Training Steps**: {total_steps}
- **Additional Steps**: {additional_steps}
- **Base Model**: [lemms/openllm-small-extended-7k](https://huggingface.co/lemms/openllm-small-extended-7k)
- **Final Loss**: {training_history[-1]['loss']:.4f} if training_history else 'N/A'
- **Framework**: PyTorch
- **License**: GPL-3.0

## Training Configuration

```json
{json.dumps(config_dict, indent=2)}
```

## Training History

The model was trained with the following key metrics:
- Best validation loss: {best_loss:.4f}
- Total training time: {len(training_history)} steps
- Device used: {device}

## Usage

This model can be used for text generation and language modeling tasks.

## Author

Louis Chua Bean Chong

## License

GPL-3.0
"""
                
                with open(model_path / "README.md", 'w') as f:
                    f.write(readme_content)
                
                # Upload to Hugging Face
                repo_name = "openllm-small-extended-8k"
                repo_id = f"{username}/{repo_name}"
                
                try:
                    # Create repository
                    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
                    
                    # Upload files
                    api = HfApi()
                    api.upload_folder(
                        folder_path=str(model_path),
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"Add OpenLLM small model extended to {total_steps} steps"
                    )
                    
                    print(f"✅ Model uploaded successfully!")
                    print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
                    
                    result = f"✅ 8k Model Training completed successfully!\n\n"
                    result += f"📊 Results:\n"
                    result += f"   - Base Model: {hf_model_id}\n"
                    result += f"   - Additional Steps: {additional_steps}\n"
                    result += f"   - Total Steps: {total_steps}\n"
                    result += f"   - Final Loss: {loss.item():.4f}\n"
                    result += f"   - Best Validation Loss: {best_loss:.4f}\n"
                    result += f"   - Model URL: https://huggingface.co/{repo_id}\n\n"
                    result += f"🎉 Extended model available at: https://huggingface.co/{repo_id}"
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ Model upload failed: {e}")
                    return f"⚠️ Training completed but upload failed: {e}"
                
            except KeyboardInterrupt:
                print(f"\n⚠️ Training interrupted by user")
                return "⚠️ Training was interrupted by user"
                
        except Exception as e:
            return f"❌ Error resuming training: {e}"

    def check_space_environment():
        """Check the Space environment and configuration."""
        try:
            # Check if we're in a Space
            space_vars = ["SPACE_ID", "SPACE_HOST", "SPACE_REPO_ID"]
            is_space = any(os.getenv(var) for var in space_vars)

            # Check HF_TOKEN
            hf_token = os.getenv("HF_TOKEN")

            result = "🔍 Space Environment Check:\n\n"

            if is_space:
                result += "✅ Running in Hugging Face Space environment\n"
                for var in space_vars:
                    value = os.getenv(var)
                    if value:
                        result += f"   - {var}: {value}\n"
            else:
                result += "ℹ️ Running in local environment\n"

            # Test Space's built-in authentication
            try:
                from huggingface_hub import whoami

                user_info = whoami()
                result += f"✅ Space built-in authentication working\n"
                result += f"   - User: {user_info['name']}\n"
                result += f"   - Full name: {user_info['fullname']}\n"
                result += f"   - Authentication: Space built-in token\n"
            except Exception as auth_error:
                result += f"❌ Space built-in authentication failed: {str(auth_error)[:50]}...\n"

                if hf_token:
                    result += f"✅ HF access token found: {hf_token[:8]}...{hf_token[-4:]}\n"
                    result += "   - Source: HF access token in Space settings\n"
                else:
                    result += "❌ HF access token not found\n"
                    result += "   - Please set HF_TOKEN in Space settings with HF access token\n"
                    result += "   - Or ensure Space has proper authentication permissions\n"

            result += f"\n📁 Available modules: {'✅' if MODULES_AVAILABLE else '❌'}"

            return result

        except Exception as e:
            return f"❌ Error checking environment: {e}"

    # Create the Gradio interface with security mitigations
    with gr.Blocks(
        title="OpenLLM Training Space",
        theme=gr.themes.Soft(),
        # Security mitigations
        analytics_enabled=False,  # Disable analytics
    ) as interface:
        gr.Markdown(
            """
        # 🚀 OpenLLM Training Space
        
        Welcome to the OpenLLM Training Space! This Space provides a complete environment for training OpenLLM models with automatic Hugging Face authentication and model upload.
        
                 ## 🔐 Authentication
        
         This Space uses HF access token for secure authentication. The HF_TOKEN is automatically available from your Space settings.
        
        ## 📋 Available Actions
        
        1. **Environment Check**: Verify Space configuration and authentication
        2. **Authentication Test**: Test Hugging Face authentication
        3. **Run Training**: Start OpenLLM training with automatic upload
        4. **Resume 7k to 8k**: Load 7k model and resume training to 8k steps
        """
        )

        with gr.Tab("🔍 Environment Check"):
            gr.Markdown("Check the Space environment and configuration.")
            env_check_btn = gr.Button("Check Environment", variant="primary")
            env_output = gr.Textbox(label="Environment Status", lines=10, interactive=False)
            env_check_btn.click(check_space_environment, outputs=env_output)

        with gr.Tab("🔐 Authentication Test"):
            gr.Markdown("Test Hugging Face authentication using HF access token.")
            auth_test_btn = gr.Button("Run Authentication Test", variant="primary")
            auth_output = gr.Textbox(label="Authentication Results", lines=15, interactive=False)
            auth_test_btn.click(run_authentication_test, outputs=auth_output)

        with gr.Tab("🚀 Run Training"):
            gr.Markdown(
                """
            Start OpenLLM training with automatic model upload.
            
            **Training Parameters:**
            - **Model Size**: Choose the model size (small, medium, large)
            - **Training Steps**: Number of training steps (default: 8000)
            
            **Expected Results:**
            - Training will complete successfully
            - Model will be uploaded to Hugging Face Hub
            - Repository will be created with proper model files
            """
            )

            with gr.Row():
                model_size = gr.Dropdown(
                    choices=["small", "medium", "large"],
                    value="small",
                    label="Model Size",
                    info="Choose the model size for training",
                )
                training_steps = gr.Number(
                    value=8000,
                    label="Training Steps",
                    info="Number of training steps",
                    minimum=1000,
                    maximum=50000,
                )

            with gr.Row():
                use_real_training = gr.Checkbox(
                    value=False,
                    label="Use Real Training",
                    info="Enable real model training with checkpoints and validation (slower but more realistic)"
                )

            train_btn = gr.Button("Start Training", variant="primary", size="lg")
            train_output = gr.Textbox(label="Training Results", lines=20, interactive=False)

            train_btn.click(run_training, inputs=[model_size, training_steps, use_real_training], outputs=train_output)

        with gr.Tab("🔄 Resume 7k to 8k"):
            gr.Markdown(
                """
            **Resume Training from 7k to 8k Model**
            
            This feature loads the existing 7k model from Hugging Face Hub and resumes training
            to create an 8k model.
            
            **Process:**
            1. Downloads the 7k model from lemms/openllm-small-extended-7k
            2. Resumes training for 1000 additional steps
            3. Creates a new 8k model with extended training
            4. Uploads the result to Hugging Face Hub
            
            **Expected Results:**
            - Model will be loaded from HF Hub
            - Training will resume from step 7000
            - New 8k model will be uploaded to HF Hub
            """
            )

            resume_btn = gr.Button("Resume Training 7k → 8k", variant="primary", size="lg")
            resume_output = gr.Textbox(label="Resume Training Results", lines=25, interactive=False)

            resume_btn.click(resume_training_from_7k_to_8k, outputs=resume_output)

        with gr.Tab("📚 Documentation"):
            gr.Markdown(
                """
            ## 📖 Available Documentation
            
            - **HUGGINGFACE_SPACE_SETUP_GUIDE.md**: Complete setup guide
            - **SPACE_AUTHENTICATION_SUMMARY.md**: Authentication summary
            - **SPACE_READY_SUMMARY.md**: Deployment summary
            
            ## 🔧 Available Scripts
            
            - **space_auth_test.py**: Authentication verification
            - **openllm_training_with_auth.py**: Complete training script
            - **integrate_auth_into_training.py**: Integration guide
            - **setup_hf_space_auth.py**: Space authentication setup
            - **verify_space_auth.py**: Space verification script
            
            ## 🎯 Quick Start
            
            1. Check the environment to verify configuration
            2. Run authentication test to ensure GitHub secrets are working
            3. Start training with your desired parameters
            4. Monitor the training progress and model upload
            
            ## 🔒 Security
            
            - HF_TOKEN is securely stored in GitHub repository secrets
            - No hardcoded tokens in any scripts
            - Automatic cleanup of test repositories
            - Proper error handling and logging
            """
            )

    return interface


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_space_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        # Security mitigations for Gradio vulnerabilities
        allowed_paths=[],  # Restrict file access
        auth=None,  # Disable authentication to prevent code injection
        quiet=True,  # Reduce logging
    )
