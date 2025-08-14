# Split-Repository Structure for OpenLLM

## 🏗️ **Overview**

This document describes the split-repository architecture that keeps the main GitHub repository and Hugging Face Space in sync while maintaining separate model repositories.

## 📁 **Repository Structure**

### **1. Main GitHub Repository** (`louischua/openllm`)
- **Core functionality**: Model architecture, training pipeline, evaluation
- **Documentation**: Comprehensive guides and examples
- **Tests**: Complete test suite
- **Data**: Training data and configurations
- **Exports**: Model export functionality

### **2. Hugging Face Space** (`lemms/openllm`)
- **Training infrastructure**: Copied from main repo
- **UI**: Gradio interface for training
- **Scripts**: Training and data upload scripts
- **Configs**: Model configurations

### **3. Model Repositories** (Separate HF Repos)
- `lemms/openllm-small-extended-7k` - 7k step model
- `lemms/openllm-small-extended-8k` - 8k step model  
- `lemms/openllm-training-data` - Training dataset

## 🔄 **Sync Strategy**

### **Automated Sync (GitHub Actions)**
- Triggers on changes to core functionality
- Copies relevant files to HF Space
- Maintains version consistency

### **Manual Sync Process**
1. Update core code in main repo
2. Run sync script
3. Push to HF Space
4. Update model repositories

## 🛠️ **Setup Instructions**

### **Step 1: Configure GitHub Repository**
1. Add HF_TOKEN secret in GitHub settings
2. Enable GitHub Actions
3. Sync workflow runs automatically

### **Step 2: Setup HF Space**
1. Clone HF Space: `git clone https://huggingface.co/spaces/lemms/openllm`
2. Copy files from main repo to training/ directory
3. Add HF Space specific files (app.py, requirements.txt)
4. Push to HF Space

### **Step 3: Configure Model Repositories**
1. Create model repos on HF Hub
2. Upload training data
3. Configure model distribution

## 📋 **File Synchronization**

### **Synced Files**
- `core/src/*.py` → `training/*.py`
- `configs/*.json` → `configs/*.json`

### **HF Space Specific**
- `app.py` - Gradio interface
- `requirements.txt` - Dependencies
- `README.md` - Space documentation

### **Not Synced**
- `data/` - Training data
- `tests/` - Test suite
- `docs/` - Documentation
- `exports/` - Model exports

## 🚀 **Benefits**

- **Separation of Concerns**: Each repo has specific purpose
- **Scalability**: Independent scaling per component
- **Maintainability**: Clear ownership and versioning
- **Automation**: Reduced manual sync work

## 📞 **Support**

For questions: Open an issue or email louischua@gmail.com

---

**Author**: Louis Chua Bean Chong  
**License**: GPL-3.0
