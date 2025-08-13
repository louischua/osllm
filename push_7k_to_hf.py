#!/usr/bin/env python3
"""
Script to push the OpenLLM 7k model to Hugging Face Hub

This script uploads the OpenLLM Small Extended 7K model to the Hugging Face Hub
for easy access and distribution to the community.

Author: Louis Chua Bean Chong
License: GPL-3.0
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

def push_7k_model_to_hf():
    """
    Push the OpenLLM 7k model to Hugging Face Hub.
    
    This function handles the complete upload process including:
    - Repository creation/verification
    - Model file upload
    - Metadata and documentation upload
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Model details for the 7k version
    model_name = "openllm-small-extended-7k"
    repo_id = f"lemms/{model_name}"
    model_path = "exports/huggingface-7k/huggingface"
    
    print(f"🚀 Pushing OpenLLM 7k model to Hugging Face Hub")
    print(f"Repository: {repo_id}")
    print(f"Model path: {model_path}")
    print("=" * 60)
    
    # Validate model directory exists
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        print(f"   Please run the export script first:")
        print(f"   python core/src/export_model.py --model_dir models/small-extended-7k --format huggingface --output_dir exports/huggingface-7k")
        return False
    
    # Check for required files
    required_files = [
        "config.json",
        "pytorch_model.bin", 
        "tokenizer.model",
        "tokenizer_config.json",
        "generation_config.json",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    try:
        # Initialize Hugging Face API
        print("🔐 Initializing Hugging Face API...")
        api = HfApi()
        
        # Create repository (will fail if it already exists, which is fine)
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=False,
                exist_ok=True
            )
            print(f"✅ Repository created/verified: {repo_id}")
        except Exception as e:
            print(f"⚠️ Repository creation warning: {e}")
        
        # Upload the model files
        print(f"📤 Uploading model files...")
        print(f"   This may take a few minutes for the large model file...")
        
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add OpenLLM Small Extended 7k model",
            commit_description="""
            OpenLLM Small Extended model trained for 7,000 steps.
            
            Model Specifications:
            - Architecture: GPT-style transformer
            - Parameters: 35,823,616 (35.8M)
            - Layers: 6 transformer layers
            - Heads: 8 attention heads
            - Embedding Dimension: 512
            - Vocabulary Size: 32,000 tokens
            - Context Length: 1,024 tokens
            - Training Steps: 7,000
            
            Training Details:
            - Dataset: Wikipedia passages from SQuAD dataset (~41k passages)
            - Tokenization: SentencePiece with 32k vocabulary
            - Training Objective: Next token prediction (causal language modeling)
            - Optimizer: AdamW with learning rate scheduling
            
            Usage:
            ```python
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            tokenizer = AutoTokenizer.from_pretrained("lemms/openllm-small-extended-7k")
            model = AutoModelForCausalLM.from_pretrained("lemms/openllm-small-extended-7k")
            
            # Generate text
            prompt = "The history of artificial intelligence"
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(inputs.input_ids, max_new_tokens=100)
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            ```
            
            License: GPL-3.0 / Commercial available
            For more details, see: https://github.com/louischua/openllm
            """
        )
        
        print(f"✅ Model successfully uploaded to: https://huggingface.co/{repo_id}")
        print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
        print(f"📊 Model size: ~161MB (pytorch_model.bin)")
        print(f"🎯 Training steps: 7,000")
        print(f"📈 Parameters: 35,823,616")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to push model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to execute the model upload process.
    
    This function provides user feedback and handles the upload workflow
    with proper error handling and status reporting.
    """
    
    print("🎯 OpenLLM 7k Model Upload Script")
    print("=" * 50)
    
    # Check if huggingface_hub is available
    try:
        import huggingface_hub
        print(f"✅ Hugging Face Hub library found: {huggingface_hub.__version__}")
    except ImportError:
        print("❌ Hugging Face Hub library not found!")
        print("   Please install it with: pip install huggingface_hub")
        return False
    
    # Execute the upload
    success = push_7k_model_to_hf()
    
    if success:
        print("\n🎉 Model push completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Visit the model page to verify upload")
        print("   2. Test the model with the provided examples")
        print("   3. Share with the community")
        print("   4. Monitor usage and feedback")
    else:
        print("\n💥 Model push failed!")
        print("\n🔧 Troubleshooting:")
        print("   1. Check your Hugging Face authentication")
        print("   2. Verify the model export was successful")
        print("   3. Ensure all required files are present")
        print("   4. Check internet connection")
        sys.exit(1)

if __name__ == "__main__":
    main()
