# OpenLLM Test Suite

This directory contains the comprehensive test suite for the OpenLLM project. The tests ensure code quality, functionality, and reliability across all components.

## 📋 Test Coverage

### **🧠 Model Tests** (`test_model.py`)
- **GPTConfig**: Configuration validation and parameter estimation
- **GPTModel**: Model initialization, forward pass, and parameter counting
- **CausalSelfAttention**: Attention mechanism and causal masking
- **Model Persistence**: Saving/loading models and checkpoints
- **Performance**: Memory usage and inference speed

### **🚀 Training Tests** (`test_training.py`)
- **DataLoader**: Data loading, preprocessing, and batch generation
- **TrainingConfig**: Training parameter validation
- **Training Loop**: Training steps, loss computation, and optimization
- **Model Evaluation**: Perplexity calculation and metrics
- **Integration**: End-to-end training simulation

### **🌐 Inference Tests** (`test_inference.py`)
- **Inference Server**: FastAPI server functionality and endpoints
- **Text Generation**: Generation parameters and validation
- **API Testing**: Request/response handling and error cases
- **Performance**: Inference speed and concurrent request handling
- **Reliability**: Server stability and error recovery

## 🚀 Quick Start

### **Install Test Dependencies**
```bash
pip install -r tests/requirements-test.txt
```

### **Run All Tests**
```bash
python tests/run_tests.py
```

### **Run Tests with Verbose Output**
```bash
python tests/run_tests.py --verbose
```

### **Run Tests with Coverage Report**
```bash
python tests/run_tests.py --coverage
```

### **Run Specific Test Module**
```bash
python tests/run_tests.py test_model
python tests/run_tests.py test_training
python tests/run_tests.py test_inference
```

### **Check Dependencies**
```bash
python tests/run_tests.py --check-deps
```

## 🧪 Running Individual Tests

### **Using Python's unittest**
```bash
# Run specific test file
python -m unittest tests.test_model

# Run specific test class
python -m unittest tests.test_model.TestGPTModel

# Run specific test method
python -m unittest tests.test_model.TestGPTModel.test_forward_pass_small_batch
```

### **Using pytest** (if installed)
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run tests matching pattern
pytest -k "test_forward_pass"

# Run with coverage
pytest --cov=core/src tests/
```

## 📊 Test Results

### **Expected Output**
```
✅ All required packages are available

============================================================
TEST SUMMARY
============================================================
Tests run: 45
Failures: 0
Errors: 0
Skipped: 0
Time taken: 12.34 seconds
✅ All tests passed!
```

### **Coverage Report** (with --coverage flag)
```
============================================================
COVERAGE REPORT
============================================================
Name                           Stmts   Miss  Cover
--------------------------------------------------
core/src/model.py                 200     15    93%
core/src/train_model.py           180     20    89%
core/src/inference_server.py      150     10    93%
core/src/data_loader.py           120     15    88%
--------------------------------------------------
TOTAL                             650     60    91%
```

## 🔧 Test Configuration

### **Environment Variables**
```bash
# Set test environment
export OPENLLM_TEST_ENV=test

# Enable debug output
export OPENLLM_DEBUG=1

# Set test data directory
export OPENLLM_TEST_DATA_DIR=/path/to/test/data
```

### **Test Data**
Tests use minimal synthetic data to avoid large file dependencies:
- **Model Tests**: Small synthetic tensors and configurations
- **Training Tests**: Sample text data (~5 lines)
- **Inference Tests**: Mocked model responses

## 🐛 Troubleshooting

### **Common Issues**

#### **Import Errors**
```bash
# Ensure you're in the project root
cd /path/to/openllm

# Add core/src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/core/src"
```

#### **Missing Dependencies**
```bash
# Install all test dependencies
pip install -r tests/requirements-test.txt

# Or install individually
pip install pytest coverage fastapi httpx
```

#### **CUDA/GPU Issues**
Tests are designed to run on CPU by default. For GPU testing:
```bash
# Set device to CPU explicitly
export CUDA_VISIBLE_DEVICES=""

# Or run specific GPU tests
python tests/run_tests.py test_model --gpu
```

### **Test Failures**

#### **Model Loading Failures**
- Ensure model files are in expected locations
- Check file permissions and paths
- Verify model configuration compatibility

#### **Memory Issues**
- Reduce batch sizes in test configurations
- Use smaller model configurations for testing
- Run tests on machine with sufficient RAM

#### **Timeout Issues**
- Increase timeout values for slow systems
- Run tests individually to isolate slow tests
- Use `--verbose` flag to see progress

## 📈 Continuous Integration

### **GitHub Actions**
Tests are automatically run on:
- **Push to main**: Full test suite
- **Pull requests**: Full test suite + coverage
- **Scheduled**: Daily regression testing

### **Local CI Setup**
```bash
# Run full CI locally
./scripts/run_ci.sh

# Or run individual CI steps
python tests/run_tests.py --coverage
flake8 core/src/
black --check core/src/
mypy core/src/
```

## 🎯 Test Quality Standards

### **Code Coverage**
- **Minimum**: 80% line coverage
- **Target**: 90% line coverage
- **Critical paths**: 100% coverage

### **Performance Benchmarks**
- **Model inference**: < 1 second for small model
- **API response**: < 5 seconds for generation
- **Memory usage**: < 1GB for small model

### **Reliability**
- **Test stability**: 99% pass rate
- **Flaky tests**: Zero tolerance
- **Error handling**: All error cases covered

## 🤝 Contributing Tests

### **Adding New Tests**
1. **Follow naming convention**: `test_*.py` for test files
2. **Use descriptive names**: `test_function_name_scenario`
3. **Include docstrings**: Explain what each test validates
4. **Add to appropriate module**: Group related tests together

### **Test Guidelines**
- **Keep tests fast**: Individual tests should complete in < 1 second
- **Use minimal data**: Synthetic data preferred over large files
- **Mock external dependencies**: Don't rely on network or file system
- **Test edge cases**: Include boundary conditions and error scenarios
- **Document assumptions**: Explain test setup and expected behavior

### **Example Test Structure**
```python
class TestNewFeature(unittest.TestCase):
    """Test cases for new feature."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPTConfig.small()
        self.model = GPTModel(self.config)
    
    def test_feature_basic_functionality(self):
        """Test basic functionality of new feature."""
        # Arrange
        input_data = create_test_data()
        
        # Act
        result = self.model.new_feature(input_data)
        
        # Assert
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, expected_shape)
    
    def test_feature_edge_case(self):
        """Test edge case handling."""
        # Test with empty input
        result = self.model.new_feature([])
        self.assertEqual(result, expected_empty_result)
```

## 📚 Additional Resources

- **[Testing Best Practices](https://docs.python.org/3/library/unittest.html)**
- **[pytest Documentation](https://docs.pytest.org/)**
- **[Coverage.py Documentation](https://coverage.readthedocs.io/)**
- **[FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)**

## 🆘 Support

For test-related issues:
1. **Check troubleshooting section** above
2. **Review test logs** for specific error messages
3. **Run tests individually** to isolate issues
4. **Create issue** with test output and environment details
