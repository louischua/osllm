# Split-Repository Structure for OpenLLM

## 🏗️ **Architecture Overview**

This document provides a comprehensive guide to the split-repository architecture designed for the OpenLLM project. This architecture maintains a clean separation of concerns while ensuring seamless synchronization between the main GitHub repository and the Hugging Face Space environment.

The split-repository approach addresses several key challenges in modern machine learning development:
- **Centralized Development**: All core development happens in the main GitHub repository
- **Distributed Training**: Training infrastructure is deployed in Hugging Face Spaces
- **Model Distribution**: Trained models are stored in separate, dedicated repositories
- **Automated Synchronization**: Changes are automatically propagated between repositories
- **Scalable Infrastructure**: Each component can scale independently

## 📁 **Repository Structure and Responsibilities**

### **1. Main GitHub Repository** (`louischua/openllm`)

The main repository serves as the central hub for all OpenLLM development and contains:

#### **Core Functionality**
- **Model Architecture**: Complete implementation of the transformer-based language model
- **Training Pipeline**: End-to-end training workflow with data loading, training, and evaluation
- **Tokenizer Training**: SentencePiece tokenizer training and configuration
- **Data Processing**: Data loading, preprocessing, and validation utilities
- **Model Evaluation**: Comprehensive evaluation metrics and benchmarking tools
- **Export Functionality**: Model export to various formats (PyTorch, Hugging Face, ONNX)

#### **Development Infrastructure**
- **Documentation**: Comprehensive guides, tutorials, and API documentation
- **Test Suite**: Unit tests, integration tests, and performance benchmarks
- **Configuration Files**: Model configurations for different sizes and use cases
- **Training Data**: Processed datasets and data preparation scripts
- **CI/CD Pipeline**: Automated testing, building, and deployment workflows

#### **Project Management**
- **Issue Tracking**: Bug reports, feature requests, and development planning
- **Release Management**: Version control and release notes
- **Community Guidelines**: Contributing guidelines and code of conduct
- **Licensing**: Dual licensing (GPL-3.0 and Commercial) documentation

### **2. Hugging Face Space** (`lemms/openllm`)

The Hugging Face Space provides the training infrastructure and user interface:

#### **Training Infrastructure**
- **Core Training Modules**: Copied from the main repository for consistency
- **Training Scripts**: Specialized scripts for HF Space environment
- **Configuration Management**: Model and training configurations
- **Data Upload Tools**: Utilities for uploading training data to HF Hub

#### **User Interface**
- **Gradio Application**: Web-based interface for training configuration and monitoring
- **Real-time Monitoring**: Progress tracking and status updates
- **Interactive Controls**: Start/stop training and parameter adjustment
- **Documentation**: In-app instructions and help resources

#### **Integration Features**
- **HF Hub Integration**: Direct connection to Hugging Face Hub for model distribution
- **Resource Management**: Memory and compute resource optimization
- **Error Handling**: Comprehensive error reporting and recovery
- **Logging**: Detailed training logs and debugging information

### **3. Model Repositories** (Separate HF Repos)

Dedicated repositories for storing and distributing trained models:

#### **Model Storage**
- **Versioned Models**: Different training iterations (7k, 8k, etc.)
- **Model Metadata**: Training parameters, performance metrics, and usage instructions
- **Model Cards**: Comprehensive documentation for each model version
- **Download Links**: Direct access to model files and configurations

#### **Dataset Repositories**
- **Training Data**: Processed datasets used for model training
- **Data Documentation**: Dataset descriptions, statistics, and usage guidelines
- **Version Control**: Tracked changes and improvements to training data
- **Access Control**: Public datasets for community use

## 🔄 **Synchronization Strategy**

### **Automated Sync (GitHub Actions)**

The automated synchronization system ensures that changes in the main repository are automatically propagated to the HF Space:

#### **Trigger Conditions**
- **Core Code Changes**: Updates to training, model, or evaluation code
- **Configuration Updates**: Changes to model configurations or hyperparameters
- **Documentation Updates**: Improvements to training guides and instructions
- **Workflow Changes**: Updates to the sync process itself

#### **Sync Process**
1. **File Detection**: Identify changed files that affect training infrastructure
2. **Dependency Resolution**: Ensure all required files are included
3. **File Copying**: Copy relevant files to the HF Space structure
4. **Configuration Generation**: Create Space-specific configuration files
5. **Validation**: Verify that the sync was successful and complete

#### **Error Handling**
- **Authentication Issues**: Handle HF token expiration or permission problems
- **Network Failures**: Retry mechanisms for temporary connectivity issues
- **File Conflicts**: Resolution strategies for conflicting file versions
- **Rollback Capability**: Ability to revert to previous working state

### **Manual Sync Process**

For situations requiring manual intervention or testing:

#### **Step-by-Step Process**
1. **Update Core Code**: Make changes in the main GitHub repository
2. **Test Locally**: Verify changes work correctly in local environment
3. **Run Sync Script**: Execute the synchronization script manually
4. **Verify HF Space**: Check that changes are properly applied
5. **Update Models**: Push any new models to separate repositories

#### **Verification Steps**
- **File Integrity**: Ensure all files are copied correctly
- **Dependency Compatibility**: Verify requirements.txt is up-to-date
- **Functionality Testing**: Test training functionality in HF Space
- **Documentation Updates**: Update any relevant documentation

## 🛠️ **Setup and Configuration**

### **Step 1: Configure GitHub Repository**

#### **Repository Settings**
1. **Enable GitHub Actions**: Ensure Actions are enabled in repository settings
2. **Set Up Secrets**: Add HF_TOKEN secret for Hugging Face Hub access
3. **Configure Permissions**: Set appropriate permissions for Actions workflow
4. **Enable Issues**: Configure issue templates and labels for development tracking

#### **Required Secrets**
- **HF_TOKEN**: Hugging Face Hub access token with write permissions
- **GITHUB_TOKEN**: Default token for repository access (automatically provided)

#### **Workflow Configuration**
- **Trigger Paths**: Configure which file changes trigger synchronization
- **Branch Protection**: Protect main branch from direct pushes
- **Review Requirements**: Require code review for significant changes
- **Status Checks**: Ensure sync workflow passes before merging

### **Step 2: Setup Hugging Face Space**

#### **Space Creation**
1. **Clone HF Space**: Download the Space repository to local machine
2. **Copy Core Files**: Transfer essential files from main repository
3. **Add Space-Specific Files**: Include UI components and Space configuration
4. **Push to HF Hub**: Upload the complete Space structure

#### **File Organization**
```
lemms/openllm/
├── training/           # Core training modules
│   ├── model.py       # Model architecture
│   ├── train_model.py # Training pipeline
│   └── ...
├── scripts/           # Training and utility scripts
├── configs/           # Model configurations
├── app.py            # Gradio interface
├── requirements.txt  # Dependencies
└── README.md         # Space documentation
```

#### **Configuration Files**
- **requirements.txt**: Python dependencies for the Space
- **README.md**: Space metadata and documentation
- **app.py**: Main Gradio application entry point
- **.gitignore**: Exclude unnecessary files from version control

### **Step 3: Configure Model Repositories**

#### **Repository Creation**
1. **Create Model Repos**: Set up separate repositories for different model versions
2. **Configure Access**: Set appropriate visibility and access permissions
3. **Add Documentation**: Include model cards and usage instructions
4. **Set Up CI/CD**: Configure automatic model upload and testing

#### **Model Organization**
- **Versioning Strategy**: Use semantic versioning or step-based naming
- **Metadata Management**: Track training parameters and performance metrics
- **Distribution**: Provide multiple download formats and access methods
- **Documentation**: Comprehensive model cards with usage examples

## 📋 **File Synchronization Rules**

### **Files Synced from GitHub to HF Space**

| GitHub Path | HF Space Path | Purpose | Sync Frequency |
|-------------|---------------|---------|----------------|
| `core/src/model.py` | `training/model.py` | Model architecture | On change |
| `core/src/train_model.py` | `training/train_model.py` | Training pipeline | On change |
| `core/src/train_tokenizer.py` | `training/train_tokenizer.py` | Tokenizer training | On change |
| `core/src/data_loader.py` | `training/data_loader.py` | Data loading | On change |
| `core/src/evaluate_model.py` | `training/evaluate_model.py` | Model evaluation | On change |
| `configs/*.json` | `configs/*.json` | Model configurations | On change |

### **HF Space Specific Files**

| File | Purpose | Creation Method |
|------|---------|----------------|
| `app.py` | Gradio interface | Manual creation |
| `requirements.txt` | Dependencies | Auto-generated |
| `README.md` | Space documentation | Manual creation |
| `scripts/upload_training_data.py` | Data upload script | Manual creation |

### **Files NOT Synced**

| File | Reason for Exclusion |
|------|---------------------|
| `data/` | Training data stays in main repo for version control |
| `tests/` | Test suite not needed in HF Space environment |
| `docs/` | Documentation not required for training interface |
| `exports/` | Model exports stay in main repo for distribution |
| `.github/` | GitHub-specific workflows not applicable to HF Space |

## 🚀 **Benefits and Advantages**

### **Separation of Concerns**
- **Clear Ownership**: Each repository has a specific, well-defined purpose
- **Independent Development**: Teams can work on different components simultaneously
- **Focused Maintenance**: Issues and updates are isolated to relevant repositories
- **Specialized Tools**: Each environment can use tools optimized for its purpose

### **Scalability and Performance**
- **Independent Scaling**: Each component can scale based on its specific needs
- **Resource Optimization**: HF Space provides dedicated compute resources for training
- **Parallel Processing**: Multiple training runs can be executed simultaneously
- **Load Distribution**: Training load is distributed across HF infrastructure

### **Maintainability and Reliability**
- **Automated Sync**: Reduces manual work and human error
- **Version Control**: Proper versioning for each component
- **Rollback Capability**: Easy recovery from failed deployments
- **Testing Isolation**: Each component can be tested independently

### **Community and Collaboration**
- **Open Development**: Main repository remains open for community contributions
- **Focused Training**: HF Space provides dedicated environment for training
- **Model Sharing**: Easy distribution of trained models through HF Hub
- **Documentation**: Comprehensive guides for each component

## 🔧 **Maintenance and Operations**

### **Daily Operations**

#### **Development Workflow**
1. **Code Development**: Work in main GitHub repository
2. **Testing**: Run tests locally and in CI environment
3. **Code Review**: Submit pull requests for review
4. **Merge and Deploy**: Merge changes trigger automatic sync
5. **Verification**: Verify changes are properly applied in HF Space

#### **Training Operations**
1. **Configuration**: Set up training parameters in HF Space
2. **Data Upload**: Upload training data to HF Hub
3. **Training Execution**: Start training through UI or terminal
4. **Monitoring**: Monitor progress and handle any issues
5. **Model Distribution**: Push trained models to separate repositories

### **Version Management**

#### **Core Repository Versioning**
- **Semantic Versioning**: Use semantic versioning for releases
- **Release Notes**: Comprehensive release notes for each version
- **Backward Compatibility**: Maintain compatibility between versions
- **Migration Guides**: Provide guides for upgrading between versions

#### **Model Versioning**
- **Step-Based Naming**: Use training steps for model identification
- **Performance Tracking**: Track performance metrics across versions
- **Model Cards**: Comprehensive documentation for each model
- **Deprecation Policy**: Clear policy for deprecated models

### **Troubleshooting and Support**

#### **Common Issues**
- **Sync Failures**: Check HF token and permissions
- **Build Issues**: Verify requirements.txt compatibility
- **Training Errors**: Review HF Space logs and resources
- **Model Upload Issues**: Check HF Hub permissions and quotas

#### **Support Resources**
- **Documentation**: Comprehensive guides and troubleshooting
- **Community Support**: GitHub issues and discussions
- **Direct Contact**: Email support for urgent issues
- **Status Monitoring**: Real-time status of all components

## 📞 **Support and Contact**

For questions, issues, or contributions related to this split-repository structure:

### **Support Channels**
- **GitHub Issues**: [https://github.com/louischua/openllm/issues](https://github.com/louischua/openllm/issues)
- **GitHub Discussions**: [https://github.com/louischua/openllm/discussions](https://github.com/louischua/openllm/discussions)
- **Email Support**: louischua@gmail.com

### **Documentation Resources**
- **Main Project**: [https://github.com/louischua/openllm](https://github.com/louischua/openllm)
- **Training Guide**: [docs/training_pipeline.md](docs/training_pipeline.md)
- **Quick Start**: [README.md#getting-started](README.md#getting-started)

### **Community Guidelines**
- **Contributing**: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
- **Code of Conduct**: [docs/CODE_OF_CONDUCT.md](docs/CODE_OF_CONDUCT.md)
- **License Information**: [LICENSE](LICENSE)

---

**Author**: Louis Chua Bean Chong  
**Project**: OpenLLM - Open Source Large Language Model  
**License**: GPL-3.0  
**Last Updated**: 2024  
**Version**: 1.0.0
