# 🚀 OpenLLM Training Space

This repository contains scripts and configuration for automatically deploying OpenLLM training to Hugging Face Spaces with proper authentication.

## 🔄 Deployment Flow

```
GitHub Repository → GitHub Actions → Hugging Face Space
```

1. **Scripts are uploaded to GitHub** (this repository)
2. **GitHub Actions automatically deploys** scripts to Hugging Face Space
3. **Space runs training** with authentication from GitHub secrets
4. **Model is uploaded** to Hugging Face Hub automatically

## 📁 Repository Structure

```
├── .github/workflows/
│   └── deploy-to-space.yml          # GitHub Actions workflow
├── space_auth_test.py               # Authentication verification
├── openllm_training_with_auth.py    # Complete training script
├── integrate_auth_into_training.py  # Integration guide
├── setup_hf_space_auth.py           # Space authentication setup
├── verify_space_auth.py             # Space verification script
├── app.py                           # Main Space application
├── requirements.txt                 # Space dependencies
├── HUGGINGFACE_SPACE_SETUP_GUIDE.md # Setup guide
├── SPACE_AUTHENTICATION_SUMMARY.md  # Authentication summary
├── SPACE_READY_SUMMARY.md          # Deployment summary
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Set Up GitHub Secrets

In your GitHub repository, go to **Settings → Secrets and variables → Actions** and add:

- **`HF_TOKEN`**: Your Hugging Face token (get from https://huggingface.co/settings/tokens)
- **`SPACE_ID`**: Your Hugging Face Space ID (e.g., `your-username/your-space-name`)

### 2. Push to GitHub

When you push to the `main` or `master` branch, GitHub Actions will automatically:

1. Deploy all scripts to your Hugging Face Space
2. Verify the deployment
3. Make the Space ready for training

### 3. Use the Space

Once deployed, your Space will have:

- **Web Interface**: Access via the Space URL
- **Authentication**: Automatic using GitHub secrets
- **Training**: Complete OpenLLM training pipeline
- **Upload**: Automatic model upload to Hugging Face Hub

## 🔧 GitHub Actions Workflow

The `.github/workflows/deploy-to-space.yml` workflow:

1. **Triggers on**: Push to main/master branch or manual dispatch
2. **Installs**: Python and dependencies
3. **Deploys**: All scripts to Hugging Face Space
4. **Verifies**: Deployment success

## 📋 Available Scripts

### Core Scripts
- **`space_auth_test.py`**: Test Hugging Face authentication
- **`openllm_training_with_auth.py`**: Complete training with upload
- **`app.py`**: Main Space web interface

### Integration Scripts
- **`integrate_auth_into_training.py`**: Guide for existing code
- **`setup_hf_space_auth.py`**: Space authentication setup
- **`verify_space_auth.py`**: Space verification

### Documentation
- **`HUGGINGFACE_SPACE_SETUP_GUIDE.md`**: Complete setup guide
- **`SPACE_AUTHENTICATION_SUMMARY.md`**: Authentication summary
- **`SPACE_READY_SUMMARY.md`**: Deployment summary

## 🎯 Expected Results

After successful deployment:

1. **Space Interface**: Web UI with training options
2. **Authentication**: Working with GitHub secrets
3. **Training**: Complete OpenLLM training pipeline
4. **Model Upload**: Automatic upload to Hugging Face Hub
5. **Repository**: Created at `your-username/openllm-*-extended-*k`

## 🔒 Security

- **HF_TOKEN**: Stored securely in GitHub repository secrets
- **No Hardcoded Tokens**: All authentication uses environment variables
- **Automatic Cleanup**: Test repositories are cleaned up
- **Error Handling**: Proper error handling and logging

## 🆘 Troubleshooting

### GitHub Actions Issues
1. Check if secrets are set correctly
2. Verify Space ID format
3. Check workflow logs for errors

### Space Issues
1. Verify HF_TOKEN has "Write" permissions
2. Check Space logs for authentication errors
3. Ensure Space is connected to GitHub repository

### Training Issues
1. Run authentication test first
2. Check model parameters
3. Verify training data availability

## 📚 Documentation

- **Setup Guide**: `HUGGINGFACE_SPACE_SETUP_GUIDE.md`
- **Authentication Summary**: `SPACE_AUTHENTICATION_SUMMARY.md`
- **Deployment Summary**: `SPACE_READY_SUMMARY.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the deployment
5. Submit a pull request

## 📄 License

This project is licensed under the GNU General Public License v3.0.

## 🎉 Success Criteria

Your deployment is successful when:

- ✅ GitHub Actions workflow completes successfully
- ✅ Scripts are deployed to Hugging Face Space
- ✅ Space web interface is accessible
- ✅ Authentication test passes
- ✅ Training can be started and completed
- ✅ Model is uploaded to Hugging Face Hub

---

**Status**: 🚀 **Ready for Deployment** - Push to GitHub to automatically deploy to your Hugging Face Space!
