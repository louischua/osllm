# 🚀 GitHub Actions Setup Guide

## 📋 Overview

This guide explains how to set up GitHub Actions to automatically deploy both OpenLLM spaces to Hugging Face:

1. **[Inference Space](https://huggingface.co/spaces/lemms/llm)** - Model testing and comparison
2. **[Training Space](https://huggingface.co/spaces/lemms/openllm)** - Live model training

## 🔐 Required GitHub Secrets

To enable automatic deployment, you need to configure the following secrets in your GitHub repository:

### **1. HF_TOKEN**
**Purpose**: Hugging Face authentication token for uploading files

**How to get it**:
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "OpenLLM GitHub Actions")
4. Select "Write" permissions
5. Copy the generated token

**Value**: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### **2. SPACE_ID**
**Purpose**: Identifier for the inference space (lemms/llm)

**Value**: `lemms/llm`

**Note**: The training space ID (`lemms/openllm`) is hardcoded in the workflow and doesn't require a separate secret.

## ⚙️ Setting Up GitHub Secrets

### **Step 1: Access Repository Settings**
1. Go to your GitHub repository
2. Click on "Settings" tab
3. In the left sidebar, click "Secrets and variables" → "Actions"

### **Step 2: Add Secrets**
Click "New repository secret" and add each secret:

#### **Secret 1: HF_TOKEN**
- **Name**: `HF_TOKEN`
- **Value**: Your Hugging Face token (e.g., `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)

#### **Secret 2: SPACE_ID**
- **Name**: `SPACE_ID`
- **Value**: `lemms/llm`

### **Step 3: Verify Secrets**
After adding all secrets, you should see:
- ✅ `HF_TOKEN` (hidden)
- ✅ `SPACE_ID` (hidden)

## 🔄 How the Deployment Works

### **Workflow Structure**
The GitHub Actions workflow consists of several jobs:

1. **🔍 Validate Deployment Files**
   - Checks that all required files exist
   - Validates file contents and Python syntax
   - Ensures both spaces have their required files

2. **🚀 Deploy Inference Space (lemms/llm)**
   - Uploads `llm/app.py` and `llm/requirements.txt`
   - Deploys to the inference space

3. **🚀 Deploy Training Space (lemms/openllm)**
   - Uploads `training_space/app.py`, `training_space/requirements.txt`, and `training_space/README.md`
   - Deploys to the training space

4. **🔍 Verify Deployment**
   - Checks that all files were uploaded successfully
   - Verifies both spaces are accessible
   - Provides deployment summary

### **Trigger Conditions**
The workflow runs automatically when:
- ✅ Code is pushed to `main` or `master` branch
- ✅ Pull requests are created/updated
- ✅ Manual trigger via GitHub Actions UI

## 📁 Required File Structure

For the deployment to work, your repository must have this structure:

```
openllm/
├── .github/workflows/deploy-to-space.yml  # GitHub Actions workflow
├── llm/                                   # Inference Space files
│   ├── app.py
│   └── requirements.txt
├── training_space/                        # Training Space files
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
└── docs/
    └── GITHUB_ACTIONS_SETUP.md           # This file
```

## 🚀 Testing the Deployment

### **Manual Trigger**
1. Go to your GitHub repository
2. Click "Actions" tab
3. Select "Deploy to Hugging Face Spaces" workflow
4. Click "Run workflow" → "Run workflow"

### **Automatic Trigger**
1. Make changes to your code
2. Commit and push to `main` branch
3. GitHub Actions will automatically run

### **Monitoring Deployment**
1. Go to "Actions" tab in your repository
2. Click on the running workflow
3. Monitor each job's progress
4. Check logs for any errors

## 🔍 Troubleshooting

### **Common Issues**

#### **1. "HF_TOKEN not found"**
**Solution**: Ensure you've added the `HF_TOKEN` secret with the correct value

#### **2. "SPACE_ID not found"**
**Solution**: Verify the SPACE_ID secret is set correctly

#### **3. "File not found" errors**
**Solution**: Ensure all required files exist in the correct locations

#### **4. "Permission denied" errors**
**Solution**: Check that your Hugging Face token has "Write" permissions

#### **5. "Space does not exist" errors**
**Solution**: Create the spaces on Hugging Face first:
- [Create Inference Space](https://huggingface.co/new-space?owner=lemms&space=llm)
- [Create Training Space](https://huggingface.co/new-space?owner=lemms&space=openllm)

### **Debugging Steps**
1. **Check Secrets**: Verify HF_TOKEN and SPACE_ID are set correctly
2. **Check Files**: Ensure all required files exist
3. **Check Permissions**: Verify Hugging Face token permissions
4. **Check Spaces**: Ensure both spaces exist on Hugging Face
5. **Check Logs**: Review GitHub Actions logs for specific errors

## 📊 Deployment Status

### **Success Indicators**
- ✅ All validation jobs pass
- ✅ Both deployment jobs complete successfully
- ✅ Verification job confirms files are uploaded
- ✅ Deployment summary shows both spaces are ready

### **Expected Output**
```
🎉 Deployment Summary
==================
✅ Inference Space (lemms/llm): Deployed and verified
✅ Training Space (lemms/openllm): Deployed and verified

🌐 Spaces are available at:
  - Inference: https://huggingface.co/spaces/lemms/llm
  - Training: https://huggingface.co/spaces/lemms/openllm

⏳ Spaces will take a few minutes to build and become available.
```

## 🔗 Related Resources

- **[GitHub Actions Documentation](https://docs.github.com/en/actions)**
- **[Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)**
- **[Hugging Face Hub Python Library](https://huggingface.co/docs/huggingface_hub)**
- **[OpenLLM Project Documentation](README.md)**

## 📞 Support

If you encounter issues with the deployment:

1. **Check the troubleshooting section above**
2. **Review GitHub Actions logs for specific errors**
3. **Verify all secrets are configured correctly**
4. **Ensure both spaces exist on Hugging Face**
5. **Contact the maintainer if issues persist**

---

**With this setup, your OpenLLM spaces will be automatically deployed every time you push changes to the main branch! 🚀**
