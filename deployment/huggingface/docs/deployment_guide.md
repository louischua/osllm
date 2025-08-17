# 🚀 Complete Deployment Guide: GitHub to Hugging Face Space

## 📋 Overview

This guide explains the complete flow from GitHub repository to Hugging Face Space deployment, ensuring your OpenLLM training scripts are automatically deployed with proper authentication.

## 🔄 Complete Flow

```
1. GitHub Repository (Scripts) 
   ↓
2. GitHub Actions (Automatic Deployment)
   ↓
3. Hugging Face Space (Web Interface)
   ↓
4. Training with Authentication
   ↓
5. Model Upload to Hugging Face Hub
```

## 🛠️ Setup Steps

### Step 1: GitHub Repository Setup

1. **Create/Clone Repository**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Add All Files**
   ```bash
   # Add all the scripts and configuration files
   git add .
   git commit -m "Add OpenLLM training scripts and GitHub Actions workflow"
   git push origin main
   ```

### Step 2: GitHub Secrets Configuration

1. **Go to Repository Settings**
   - Navigate to your GitHub repository
   - Click on "Settings" tab
   - Click on "Secrets and variables" → "Actions"

2. **Add Required Secrets**
   - **`HF_TOKEN`**: Your Hugging Face token
     - Get from: https://huggingface.co/settings/tokens
     - Make sure it has "Write" permissions
   - **`SPACE_ID`**: Your Hugging Face Space ID
     - Format: `your-username/your-space-name`
     - Example: `lemms/openllm-training-space`

### Step 3: Hugging Face Space Setup

1. **Create Space**
   - Go to: https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Gradio" as SDK
   - Set Space name (e.g., `openllm-training-space`)
   - Make it public or private as needed

2. **Connect to GitHub**
   - In Space settings, ensure it's connected to your GitHub repository
   - This allows GitHub secrets to be available in the Space

## 🔧 Files Deployed by GitHub Actions

When you push to GitHub, the following files are automatically deployed to your Space:

### Core Application Files
- **`app.py`**: Main Space web interface
- **`requirements.txt`**: Python dependencies
- **`space_auth_test.py`**: Authentication verification
- **`openllm_training_with_auth.py`**: Complete training script

### Integration Files
- **`integrate_auth_into_training.py`**: Integration guide
- **`setup_hf_space_auth.py`**: Space authentication setup
- **`verify_space_auth.py`**: Space verification script

### Documentation Files
- **`HUGGINGFACE_SPACE_SETUP_GUIDE.md`**: Setup guide
- **`SPACE_AUTHENTICATION_SUMMARY.md`**: Authentication summary
- **`SPACE_READY_SUMMARY.md`**: Deployment summary

## 🚀 GitHub Actions Workflow

The `.github/workflows/deploy-to-space.yml` workflow:

1. **Triggers**: Push to main/master branch or manual dispatch
2. **Environment**: Ubuntu latest with Python 3.10
3. **Dependencies**: Installs huggingface_hub
4. **Deployment**: Uploads all files to Space
5. **Verification**: Lists deployed files for confirmation

### Workflow Steps:
```yaml
1. Checkout repository
2. Set up Python 3.10
3. Install dependencies
4. Deploy files to Space
5. Verify deployment
```

## 🎯 Expected Results

### After GitHub Push:
```
✅ GitHub Actions workflow completed
✅ Files deployed to Hugging Face Space
✅ Space web interface accessible
```

### In Hugging Face Space:
```
✅ Web interface with training options
✅ Authentication working with GitHub secrets
✅ Training pipeline ready
✅ Model upload functionality ready
```

## 🔍 Verification Steps

### 1. Check GitHub Actions
- Go to your repository → "Actions" tab
- Verify workflow completed successfully
- Check for any error messages

### 2. Check Space Deployment
- Go to your Space URL
- Verify web interface loads
- Check that all scripts are available

### 3. Test Authentication
- In Space, click "Environment Check"
- Verify HF_TOKEN is found
- Run authentication test

### 4. Test Training
- Select model size and training steps
- Start training
- Monitor progress and upload

## 🆘 Troubleshooting

### GitHub Actions Issues

**Problem**: Workflow fails
**Solutions**:
1. Check if secrets are set correctly
2. Verify Space ID format
3. Check workflow logs for specific errors

**Problem**: Files not deployed
**Solutions**:
1. Verify file paths in workflow
2. Check Space permissions
3. Ensure HF_TOKEN has write access

### Space Issues

**Problem**: Authentication fails
**Solutions**:
1. Verify HF_TOKEN is set in GitHub secrets
2. Check token has "Write" permissions
3. Ensure Space is connected to GitHub repository

**Problem**: Scripts not available
**Solutions**:
1. Check GitHub Actions deployment logs
2. Verify files were uploaded to Space
3. Check Space file browser

### Training Issues

**Problem**: Training fails
**Solutions**:
1. Run authentication test first
2. Check model parameters
3. Verify training data availability
4. Check Space logs for errors

## 📊 Monitoring and Logs

### GitHub Actions Logs
- Repository → Actions → Workflow runs
- Check individual step logs
- Look for error messages

### Space Logs
- Space → Settings → Logs
- Check for authentication errors
- Monitor training progress

### Model Upload Verification
- Check Hugging Face Hub for uploaded models
- Verify repository creation
- Check model files are complete

## 🎉 Success Criteria

Your deployment is successful when:

- ✅ **GitHub Actions**: Workflow completes without errors
- ✅ **Space Interface**: Web UI loads and is functional
- ✅ **Authentication**: HF_TOKEN is found and working
- ✅ **Training**: Can start and complete training
- ✅ **Upload**: Model uploads to Hugging Face Hub
- ✅ **Repository**: Model repository is created with files

## 🔄 Continuous Deployment

Once set up, the flow becomes automatic:

1. **Make changes** to scripts in GitHub
2. **Push to main branch**
3. **GitHub Actions automatically deploys** to Space
4. **Space is updated** with new scripts
5. **Ready for next training run**

## 📚 Additional Resources

- **Setup Guide**: `HUGGINGFACE_SPACE_SETUP_GUIDE.md`
- **Authentication Summary**: `SPACE_AUTHENTICATION_SUMMARY.md`
- **Space Ready Summary**: `SPACE_READY_SUMMARY.md`
- **GitHub Actions Documentation**: https://docs.github.com/en/actions

---

**Status**: 🚀 **Ready for Complete Deployment** - Follow this guide to set up the full GitHub to Space deployment flow!
