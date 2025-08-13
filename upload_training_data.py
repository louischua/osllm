#!/usr/bin/env python3
"""
Upload Training Data to Hugging Face Hub

This script uploads the OpenLLM training data to Hugging Face Hub
as a dataset for use with AutoTrain.

Author: Louis Chua Bean Chong
License: GPL-3.0
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from datasets import Dataset

def upload_training_data():
    """Upload training data to Hugging Face Hub."""
    
    print("🚀 Uploading training data to Hugging Face Hub...")
    
    # Configuration
    dataset_name = "lemms/openllm-training-data"
    data_file = "data/clean/training_data.txt"
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"❌ Training data file not found: {data_file}")
        print("📝 Please ensure the training data exists before uploading.")
        return False
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Create dataset repository
        print(f"📂 Creating dataset repository: {dataset_name}")
        try:
            create_repo(
                repo_id=dataset_name,
                repo_type="dataset",
                exist_ok=True
            )
        except Exception as e:
            print(f"⚠️ Repository creation warning: {e}")
        
        # Read training data
        print("📖 Reading training data...")
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Clean and prepare data
        texts = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Filter out empty or very short lines
                texts.append({"text": line})
        
        print(f"📊 Prepared {len(texts)} training examples")
        
        # Create dataset
        dataset = Dataset.from_list(texts)
        
        # Upload dataset
        print("📤 Uploading dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            dataset_name,
            private=False,  # Make it public
            token=None  # Use default token
        )
        
        print("✅ Training data uploaded successfully!")
        print(f"🔗 Dataset URL: https://huggingface.co/datasets/{dataset_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading training data: {e}")
        return False

def create_dataset_info():
    """Create dataset information file."""
    
    info_content = {
        "description": "OpenLLM training data for language model training",
        "license": "GPL-3.0",
        "homepage": "https://github.com/louischua/openllm",
        "repository": "https://github.com/louischua/openllm",
        "citation": "@misc{openllm2024,\n  title={OpenLLM: Open Source Large Language Model},\n  author={Louis Chua Bean Chong},\n  year={2024},\n  url={https://github.com/louischua/openllm}\n}",
        "tags": [
            "language-modeling",
            "text-generation",
            "open-source",
            "training-data"
        ],
        "task_categories": ["text-generation"],
        "language": ["en"],
        "size_categories": ["10K<n<100K"],
        "source_datasets": ["squad"],
        "dataset_info": {
            "features": {
                "text": {
                    "dtype": "string",
                    "_type": "Value"
                }
            },
            "num_rows": 0  # Will be filled automatically
        }
    }
    
    with open("dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info_content, f, indent=2)
    
    print("✅ Created dataset_info.json")

def main():
    """Main function."""
    
    print("🔧 Setting up training data upload...")
    
    # Create dataset info
    create_dataset_info()
    
    # Upload training data
    success = upload_training_data()
    
    if success:
        print("\n🎉 Training data setup complete!")
        print("\n📋 Next steps for AutoTrain:")
        print("1. Go to https://huggingface.co/autotrain")
        print("2. Create a new Text Generation project")
        print("3. Use dataset: lemms/openllm-training-data")
        print("4. Use model: lemms/openllm-small-extended-7k")
        print("5. Configure training parameters")
        print("6. Start training!")
    else:
        print("\n❌ Training data setup failed.")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()
