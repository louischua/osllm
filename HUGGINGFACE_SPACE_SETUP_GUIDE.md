# 🚀 Hugging Face Space Setup Guide for OpenLLM Training (GitHub Secrets)

This guide will help you set up proper authentication for Hugging Face Spaces using GitHub secrets so that your OpenLLM training and model uploads work correctly.

## 🎯 Overview

The issue you encountered was that training completed successfully in Hugging Face Spaces, but the model upload failed due to authentication problems. This guide will ensure that future training runs in Spaces will have proper authentication using GitHub secrets and successful uploads.

## 🔧 Step-by-Step Setup

### Step 1: Get Your Hugging Face Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "OpenLLM Space Training")
4. Select "Write" role for full access
5. Copy the generated token

### Step 2: Set Up GitHub Repository Secrets

1. Go to your GitHub repository:
   ```
   https://github.com/your-username/your-repo
   ```

2. Click on the "Settings" tab

3. In the left sidebar, click "Secrets and variables" → "Actions"

4. Click "New repository secret"

5. Add a new secret:
   - **Name**: `HF_TOKEN`
   - **Value**: Your Hugging Face token from Step 1

6. Click "Add secret"

**Note**: Hugging Face Spaces automatically have access to GitHub repository secrets, so you don't need to set them separately in the Space.

### Step 3: Verify Authentication in Your Space

Add this code to your Space to verify authentication is working:

```python
# Add this to your Space's main script or run it separately
import os
from huggingface_hub import HfApi, whoami

def verify_space_auth():
    """Verify authentication is working in the Space using GitHub secrets."""
    print("🔐 Verifying Space Authentication (GitHub Secrets)")
    
    # Check if HF_TOKEN is set (from GitHub secrets)
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found in Space environment")
        print("   - Please set HF_TOKEN in your GitHub repository secrets")
        print("   - Go to GitHub repository → Settings → Secrets and variables → Actions")
        return False
    
    try:
        # Test authentication
        from huggingface_hub import login
        login(token=token)
        
        user_info = whoami()
        username = user_info["name"]
        
        print(f"✅ Authentication successful!")
        print(f"   - Username: {username}")
        print(f"   - Token: {token[:8]}...{token[-4:]}")
        print(f"   - Source: GitHub secrets")
        
        # Test API access
        api = HfApi()
        print(f"✅ API access working")
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

# Run verification
if __name__ == "__main__":
    verify_space_auth()
```

### Step 4: Update Your Training Script

Modify your training script to include proper authentication using GitHub secrets:

```python
import os
from huggingface_hub import HfApi, login, create_repo
import json

class SpaceTrainingManager:
    """Manages training and upload in Hugging Face Spaces using GitHub secrets."""
    
    def __init__(self):
        self.api = None
        self.username = None
        self.setup_authentication()
    
    def setup_authentication(self):
        """Set up authentication for the Space using GitHub secrets."""
        try:
            # Get token from GitHub secrets (automatically available in Space)
            token = os.getenv("HF_TOKEN")
            if not token:
                raise ValueError("HF_TOKEN not found in Space environment. Please set it in GitHub repository secrets.")
            
            # Login
            login(token=token)
            
            # Initialize API
            self.api = HfApi()
            user_info = whoami()
            self.username = user_info["name"]
            
            print(f"✅ Space authentication successful: {self.username}")
            print(f"   - Source: GitHub secrets")
            
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            raise
    
    def upload_model(self, model_dir: str, model_size: str = "small", steps: int = 8000):
        """Upload the trained model to Hugging Face Hub."""
        try:
            # Create repository name
            repo_name = f"openllm-{model_size}-extended-{steps//1000}k"
            repo_id = f"{self.username}/{repo_name}"
            
            print(f"📤 Uploading model to {repo_id}")
            
            # Create repository
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=False
            )
            
            # Create model configuration
            self.create_model_config(model_dir, model_size)
            
            # Create model card
            self.create_model_card(model_dir, repo_id, model_size, steps)
            
            # Upload all files
            self.api.upload_folder(
                folder_path=model_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add OpenLLM {model_size} model ({steps} steps)"
            )
            
            print(f"✅ Model uploaded successfully!")
            print(f"   - Repository: https://huggingface.co/{repo_id}")
            
            return repo_id
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            raise
    
    def create_model_config(self, model_dir: str, model_size: str):
        """Create Hugging Face compatible configuration."""
        config = {
            "architectures": ["GPTModel"],
            "model_type": "gpt",
            "vocab_size": 32000,
            "n_positions": 2048,
            "n_embd": 768 if model_size == "small" else 1024 if model_size == "medium" else 1280,
            "n_layer": 12 if model_size == "small" else 24 if model_size == "medium" else 32,
            "n_head": 12 if model_size == "small" else 16 if model_size == "medium" else 20,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "unk_token_id": 3,
            "transformers_version": "4.35.0",
            "use_cache": True
        }
        
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    def create_model_card(self, model_dir: str, repo_id: str, model_size: str, steps: int):
        """Create model card (README.md)."""
        model_card = f"""# OpenLLM {model_size.capitalize()} Model ({steps} steps)

This is a trained OpenLLM {model_size} model with extended training.

## Model Details

- **Model Type**: GPT-style decoder-only transformer
- **Architecture**: Custom OpenLLM implementation
- **Training Data**: SQUAD dataset (Wikipedia passages)
- **Vocabulary Size**: 32,000 tokens
- **Sequence Length**: 2,048 tokens
- **Model Size**: {model_size.capitalize()}
- **Training Steps**: {steps:,}

## Usage

This model can be used with the OpenLLM framework for text generation and language modeling tasks.

## Training

The model was trained using the OpenLLM training pipeline with:
- SentencePiece tokenization
- Custom GPT architecture
- SQUAD dataset for training
- Extended training for improved performance

## License

This model is released under the GNU General Public License v3.0.

## Repository

This model is hosted on Hugging Face Hub: https://huggingface.co/{repo_id}
"""
        
        readme_path = os.path.join(model_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card)

# Usage in your training script
def main():
    # Initialize training manager
    training_manager = SpaceTrainingManager()
    
    # Your training code here...
    # ... (training logic) ...
    
    # After training completes, upload the model
    model_dir = "./openllm-trained"  # Your model directory
    repo_id = training_manager.upload_model(model_dir, "small", 8000)
    
    print(f"🎉 Training and upload completed!")
    print(f"   - Model available at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
```

### Step 5: Test the Setup

Run the authentication verification script in your Space to ensure everything is working:

```python
# Add this to your Space to test
from setup_hf_space_auth import HuggingFaceSpaceAuthSetup

def test_space_setup():
    """Test the Space authentication setup with GitHub secrets."""
    auth_setup = HuggingFaceSpaceAuthSetup()
    
    if auth_setup.setup_space_authentication():
        print("✅ Space authentication working")
        
        # Test repository creation
        if auth_setup.test_repository_creation():
            print("✅ Repository creation working")
        
        # Test model upload
        if auth_setup.test_model_upload():
            print("✅ Model upload working")
        
        print("🎉 All tests passed! Ready for training.")
    else:
        print("❌ Authentication setup failed")

# Run the test
test_space_setup()
```

## 🔍 Troubleshooting

### Common Issues

1. **"HF_TOKEN not found"**
   - **Solution**: Make sure you've added the HF_TOKEN secret in your GitHub repository secrets
   - **Check**: Go to GitHub repository → Settings → Secrets and variables → Actions

2. **"401 Unauthorized"**
   - **Solution**: Verify your token has "Write" permissions
   - **Check**: Go to https://huggingface.co/settings/tokens and ensure the token has "Write" role

3. **"Repository creation failed"**
   - **Solution**: Check if the repository name is unique
   - **Check**: Ensure you have permission to create repositories

4. **"Upload failed"**
   - **Solution**: Check Space logs for detailed error messages
   - **Check**: Verify network connectivity and file permissions

5. **"GitHub secrets not accessible"**
   - **Solution**: Ensure your Space is connected to the GitHub repository
   - **Check**: Verify the Space is created from the GitHub repository

### Verification Steps

1. **Check Space Environment**:
   ```python
   import os
   print("Space Environment Variables:")
   for var in ["SPACE_ID", "SPACE_HOST", "HF_TOKEN"]:
       value = os.getenv(var)
       print(f"  {var}: {'✅ Set' if value else '❌ Not set'}")
   ```

2. **Test Authentication**:
   ```python
   from huggingface_hub import whoami
   try:
       user_info = whoami()
       print(f"✅ Authenticated as: {user_info['name']}")
   except Exception as e:
       print(f"❌ Authentication failed: {e}")
   ```

3. **Test Repository Creation**:
   ```python
   from huggingface_hub import create_repo, delete_repo
   try:
       repo_id = "lemms/test-repo"
       create_repo(repo_id, repo_type="model", private=True)
       print("✅ Repository creation working")
       delete_repo(repo_id, repo_type="model")
   except Exception as e:
       print(f"❌ Repository creation failed: {e}")
   ```

## 📋 Complete Workflow

1. **Set up GitHub repository secrets** with your HF_TOKEN
2. **Verify authentication** using the test script
3. **Run your training** with the updated training manager
4. **Monitor upload progress** in the Space logs
5. **Verify the model** appears on Hugging Face Hub

## 🎯 Expected Results

After successful setup, you should see:

```
✅ Running in Hugging Face Space environment
✅ HF_TOKEN found: hf_xxxx...xxxx
   - Source: GitHub secrets
✅ Authentication successful!
   - Username: lemms
✅ API access working

🧪 Testing Repository Creation
🔄 Creating test repository: lemms/test-openllm-verification
✅ Repository created successfully
🔄 Cleaning up test repository...
✅ Repository deleted

🎉 All verification tests passed!
   - Authentication: ✅ Working
   - Repository Creation: ✅ Working
   - GitHub Secrets Integration: ✅ Working
   - Ready for training and model uploads!

📤 Uploading model to lemms/openllm-small-extended-8k
✅ Model uploaded successfully!
   - Repository: https://huggingface.co/lemms/openllm-small-extended-8k
```

Your model will then be available at: `https://huggingface.co/lemms/openllm-small-extended-8k`

## 🔒 Security Notes

- **Token Security**: The HF_TOKEN is stored securely in GitHub repository secrets
- **Repository Access**: Only you can access your model repositories
- **Cleanup**: Test repositories are automatically deleted after testing
- **Monitoring**: Check Space logs for any authentication issues
- **GitHub Integration**: Secrets are automatically available in connected Spaces

## 🚀 Benefits of GitHub Secrets

1. **Centralized Management**: All secrets managed in one place
2. **Automatic Access**: Spaces automatically have access to repository secrets
3. **Version Control**: Secrets are tied to your repository
4. **Security**: GitHub provides secure secret management
5. **Easy Updates**: Update secrets without touching Space settings

---

**Next Steps**: Once you've set up the GitHub repository secrets, you can re-run your training and the model upload should work correctly!
