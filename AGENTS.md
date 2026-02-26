# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

OpenLLM is a pure Python/PyTorch framework for training and deploying GPT-style LLMs from scratch. No Docker, databases, or external infrastructure required.

### Services

| Service | Port | Command | Notes |
|---------|------|---------|-------|
| **FastAPI Inference Server** | 8000 | `python3 core/src/inference_server.py --model_path <path>` | Requires a trained model directory. For testing without a model, use `create_test_model()` from `core/src/inference_server.py`. |
| **Gradio Web UI** | 7860 | `cd llm && python3 app.py` | Downloads models from Hugging Face Hub on demand. First model load takes ~60s. |

### Running Tests

```
python3 -m pytest tests/test_simple.py tests/test_basic.py tests/test_model.py tests/test_inference.py -v --tb=short
```

- Tests in `test_training.py` that involve `TestDataLoader` require `data/tokenizer/tokenizer.model` which is not committed to the repo. These 4 tests fail by design unless you train a tokenizer first (`python3 core/src/main.py train-tokenizer`).
- All other tests pass without any external resources.
- The test suite uses `--cov=core` by default (configured in `pyproject.toml`).

### Lint and Formatting

Lint tools are configured in `pyproject.toml`. Key commands (matching CI):

- `black --check core/src/ tests/ llm/app.py`
- `isort --check-only core/src/ tests/ llm/app.py`
- `flake8 core/src/ tests/ llm/app.py --max-line-length=100 --extend-ignore=E203,W503`
- `ruff check core/src/ tests/ llm/app.py`
- `mypy core/src/ --ignore-missing-imports` (note: `pyproject.toml` sets `python_version = "3.8"` which mypy >= 1.x rejects; run with `--python-version 3.10` or ignore the config error)

### Non-obvious Caveats

- **PATH**: pip installs scripts to `~/.local/bin`, which may not be on PATH. Prefix commands with `export PATH="$HOME/.local/bin:$PATH"` or use `python3 -m <tool>`.
- **No `python` symlink**: The system has `python3` but not `python`. Use `python3` explicitly.
- **PYTHONPATH for core imports**: When running `core/src/` scripts directly, you may need `export PYTHONPATH="/workspace/core/src:$PYTHONPATH"` since modules import each other by bare name (e.g., `from model import ...`).
- **`unittest-mock` in test requirements**: `tests/requirements-test.txt` lists `unittest-mock>=1.0.1` which doesn't exist on PyPI. It's part of Python stdlib (`unittest.mock`). Skip it during install.
- **Empty data directories**: `data/tokenizer/` and `data/raw/` contain only `.gitkeep`. Training data must be prepared via `python3 core/src/main.py prepare-data` before data-dependent tests/features work.
