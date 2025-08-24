# 🧹 OpenLLM v0.1.0 Deduplication Plan

## Overview
This document outlines the deduplication strategy to clean up the codebase for version 0.1.0 release.

## 🎯 Goals
- Remove duplicate files and redundant code
- Organize project structure for production readiness
- Maintain only essential files for v0.1.0
- Improve maintainability and reduce confusion

## 📋 Deduplication Tasks

### 1. Root Directory Cleanup
**Files to Remove:**
- Multiple deployment scripts (keep only the latest working version)
- Duplicate app.py files
- Temporary test files
- Old performance reports

**Files to Keep:**
- `README.md` (main project documentation)
- `requirements.txt` (dependencies)
- `LICENSE` (GPLv3 license)
- `pyproject.toml` (project configuration)
- `.gitignore` (version control)
- `.cursorrules` (development rules)

### 2. LLM Directory Cleanup
**Files to Remove:**
- Multiple app variants (keep only the latest working version)
- Duplicate deployment scripts
- Test files

**Files to Keep:**
- `app.py` (latest working version)
- `README.md` (space documentation)
- `requirements.txt` (space dependencies)

### 3. Exports Directory Cleanup
**Directories to Remove:**
- Old model exports (keep only the latest versions)
- Duplicate Hugging Face exports

**Directories to Keep:**
- `improved-10k-huggingface/` (latest improved model)
- `huggingface-10k/` (original 10k model for comparison)

### 4. Core Directory
**Status:** ✅ Keep as is - this is the main framework

### 5. Models Directory
**Status:** ✅ Keep as is - contains trained models

### 6. Documentation Cleanup
**Files to Keep:**
- `docs/V0.1.0_RELEASE_SUMMARY.md`
- `docs/roadmap.md`
- `docs/TRAINING_IMPROVEMENTS.md`
- `docs/PROJECT_STRUCTURE_OPTIMIZATION_SUMMARY.md`

**Files to Remove:**
- Old performance reports
- Duplicate documentation

### 7. Scripts and Utilities
**Files to Keep:**
- `train_new_10k_from_9k.py` (training orchestration)
- `export_improved_10k_to_hf.py` (model export)

**Files to Remove:**
- Multiple deployment scripts
- Duplicate training scripts

## 🗂️ Final Directory Structure

```
openllm-v0.1.0/
├── README.md                    # Main project documentation
├── LICENSE                      # GPLv3 license
├── requirements.txt             # Dependencies
├── pyproject.toml              # Project configuration
├── .gitignore                  # Version control
├── .cursorrules                # Development rules
├── core/                       # Core framework
│   └── src/
│       ├── model.py
│       ├── train_model_improved.py
│       └── inference_server.py
├── llm/                        # Hugging Face Space
│   ├── app.py                  # Latest working version
│   ├── README.md
│   └── requirements.txt
├── models/                     # Trained models
│   ├── small-extended-9k/
│   └── small-extended-10k-improved/
├── exports/                    # Model exports
│   ├── improved-10k-huggingface/
│   └── huggingface-10k/
├── docs/                       # Documentation
│   ├── V0.1.0_RELEASE_SUMMARY.md
│   ├── roadmap.md
│   ├── TRAINING_IMPROVEMENTS.md
│   └── PROJECT_STRUCTURE_OPTIMIZATION_SUMMARY.md
├── data/                       # Training data
├── tests/                      # Test suite
├── scripts/                    # Utility scripts
│   ├── train_new_10k_from_9k.py
│   └── export_improved_10k_to_hf.py
└── .github/                    # GitHub workflows
```

## ✅ Success Criteria
- [ ] No duplicate files in root directory
- [ ] Single working app.py in llm/ directory
- [ ] Clean exports directory with only latest models
- [ ] Organized documentation
- [ ] Removed all temporary and test files
- [ ] Maintained all essential functionality
- [ ] Ready for v0.1.0 release

## 🚀 Post-Deduplication Actions
1. Update version numbers to 0.1.0
2. Update documentation references
3. Test all functionality
4. Create release notes
5. Tag v0.1.0 release
