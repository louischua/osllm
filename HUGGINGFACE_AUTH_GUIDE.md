# 🔐 Hugging Face Authentication Setup Guide

This guide will help you fix the Hugging Face authentication issues that are preventing model uploads in the OpenLLM training pipeline.

## 🚨 Problem Summary

The error you're seeing:
```
⚠️ Repository creation warning: 401 Client Error: Unauthorized
Invalid username or password.
❌ Failed to save/upload model: 401 Client Error
Repository Not Found for url: https://huggingface.co/api/models/lemms/openllm-small-extended-8k/preupload/main
```

This indicates that the training pipeline is trying to upload your model to Hugging Face Hub but doesn't have proper authentication credentials.

## 🛠️ Solution Options

### Option 1: Quick Fix (Recommended)

Use the provided authentication setup script:

```bash
# Run the authentication setup script
python setup_hf_auth.py --test-upload

# Then fix your existing upload
python fix_training_upload.py --model-dir ./openllm-trained
```

### Option 2: Manual Setup

#### Step 1: Get a Hugging Face Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "OpenLLM Training")
4. Select "Write" role for full access
5. Copy the generated token

#### Step 2: Set Up Authentication

Choose one of these methods:

**Method A: Environment Variable (Recommended)**
```bash
# Linux/macOS
export HUGGING_FACE_HUB_TOKEN=your_token_here

# Windows PowerShell
$env:HUGGING_FACE_HUB_TOKEN="your_token_here"

# Windows Command Prompt
set HUGGING_FACE_HUB_TOKEN=your_token_here
```

**Method B: CLI Login**
```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub>=0.34.0

# Login via CLI
huggingface-cli login
# Enter your token when prompted
```

**Method C: Python Script**
```python
from huggingface_hub import login
login(token="your_token_here")
```

#### Step 3: Test Authentication

```python
from huggingface_hub import whoami
user_info = whoami()
print(f"Authenticated as: {user_info['name']}")
```

### Option 3: Fix Existing Upload

If you already have a trained model that failed to upload:

```bash
# Use the fix script
python fix_training_upload.py --model-dir ./openllm-trained --repo-name my-openllm-model
```

## 🔧 Detailed Troubleshooting

### Common Issues

#### 1. "Invalid username or password"
- **Cause**: Token is invalid, expired, or has wrong permissions
- **Solution**: Generate a new token with "Write" role

#### 2. "Repository Not Found"
- **Cause**: Repository doesn't exist and can't be created due to authentication
- **Solution**: Ensure proper authentication before attempting upload

#### 3. "401 Unauthorized"
- **Cause**: No authentication credentials or invalid credentials
- **Solution**: Set up proper authentication using one of the methods above

#### 4. "Rate limit exceeded"
- **Cause**: Too many API requests
- **Solution**: Wait a few minutes and try again

### Verification Steps

1. **Check Authentication Status**:
   ```python
   from huggingface_hub import whoami
   try:
       user_info = whoami()
       print(f"✅ Authenticated as: {user_info['name']}")
   except Exception as e:
       print(f"❌ Not authenticated: {e}")
   ```

2. **Test Repository Creation**:
   ```python
   from huggingface_hub import create_repo, delete_repo
   
   # Create test repository
   repo_id = f"{username}/test-repo"
   create_repo(repo_id=repo_id, repo_type="model", private=True)
   print(f"✅ Repository created: {repo_id}")
   
   # Clean up
   delete_repo(repo_id=repo_id, repo_type="model")
   ```

3. **Test File Upload**:
   ```python
   from huggingface_hub import HfApi
   
   api = HfApi()
   api.upload_file(
       path_or_fileobj="test.txt",
       path_in_repo="test.txt",
       repo_id=f"{username}/test-repo",
       repo_type="model"
   )
   ```

## 📋 Complete Setup Process

### For New Training Runs

1. **Set up authentication before training**:
   ```bash
   python setup_hf_auth.py --test-upload --save-config
   ```

2. **Run your training**:
   ```bash
   # Your existing training command
   python your_training_script.py
   ```

3. **Verify upload worked**:
   - Check the training output for successful upload messages
   - Visit your model on Hugging Face Hub

### For Existing Failed Uploads

1. **Fix the upload**:
   ```bash
   python fix_training_upload.py --model-dir ./openllm-trained
   ```

2. **Verify the fix**:
   - Check that your model appears on Hugging Face Hub
   - Test downloading the model

## 🔒 Security Best Practices

1. **Token Security**:
   - Never commit tokens to version control
   - Use environment variables for production
   - Rotate tokens regularly
   - Use minimal required permissions

2. **Repository Management**:
   - Use descriptive repository names
   - Add proper model cards and documentation
   - Set appropriate visibility (public/private)

3. **Access Control**:
   - Only give tokens to trusted applications
   - Monitor token usage
   - Revoke unused tokens

## 📚 Additional Resources

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [Authentication Guide](https://huggingface.co/docs/hub/security-tokens)
- [Model Upload Guide](https://huggingface.co/docs/hub/models-uploading)
- [API Reference](https://huggingface.co/docs/huggingface_hub/index)

## 🆘 Getting Help

If you're still having issues:

1. **Check the troubleshooting section above**
2. **Verify your token permissions**
3. **Test with a simple upload first**
4. **Check network connectivity**
5. **Review Hugging Face Hub status**

## 🎯 Expected Results

After successful setup, you should see:

```
✅ Authentication successful!
   - Username: your_username
   - Method: token

✅ Repository created: your_username/openllm-small-extended-8k

✅ Model uploaded successfully!
   - Repository: https://huggingface.co/your_username/openllm-small-extended-8k
   - Model files: X files
```

Your model will then be available at: `https://huggingface.co/your_username/openllm-small-extended-8k`

---

**Note**: This guide follows the OpenLLM project's open source philosophy and uses only open source tools and libraries for authentication and model management.
