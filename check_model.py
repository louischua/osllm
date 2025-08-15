#!/usr/bin/env python3
"""
Check if the 8k model exists on Hugging Face
"""

from huggingface_hub import HfApi

def check_model():
    api = HfApi()
    repo_id = "lemms/openllm-small-extended-8k"
    
    print(f"🔍 Checking if model exists: {repo_id}")
    
    try:
        model_info = api.model_info(repo_id)
        print(f"✅ Model exists on Hugging Face!")
        print(f"   - Repository: {repo_id}")
        print(f"   - Last modified: {model_info.lastModified}")
        print(f"   - Files: {len(model_info.siblings)} files")
        
        print(f"\n📁 Files in repository:")
        for file in model_info.siblings:
            print(f"   - {file.rfilename} ({file.size} bytes)")
        
        return True
    except Exception as e:
        print(f"❌ Model not found on Hugging Face: {e}")
        return False

if __name__ == "__main__":
    check_model()
