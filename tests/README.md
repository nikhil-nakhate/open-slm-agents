# Unit Tests for Open-SLM-Agents

This directory contains comprehensive unit tests for the open-slm-agents project, covering all major components in the `ops`, `modules`, and `data` directories.

## Test Structure

```
tests/
├── __init__.py
├── README.md
├── test_ops_config.py          # Tests for ops.config module
├── test_ops_tokenizer.py       # Tests for ops.tokenizer module
├── test_modules_activations.py # Tests for models.modules.activations
├── test_modules_embeddings.py  # Tests for models.modules.embeddings
├── test_modules_layer_norm.py  # Tests for models.modules.layer_norm
├── test_modules_losses.py      # Tests for models.modules.losses
├── test_modules_mha.py         # Tests for models.modules.mha
├── test_modules_transformer.py # Tests for models.modules.transformer
├── test_modules_build.py       # Tests for models.modules.build
├── test_data_dataset.py        # Tests for data.dataset
└── test_data_transforms.py     # Tests for data.transforms
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
# Using the test runner script
python run_tests.py

# Or directly with pytest
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Run only ops tests
pytest tests/test_ops_*.py -v

# Run only modules tests
pytest tests/test_modules_*.py -v

# Run only data tests
pytest tests/test_data_*.py -v

# Run specific test file
pytest tests/test_ops_config.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Coverage

The tests provide comprehensive coverage for:

### Ops Module Tests
- **Config Management**: YAML loading, hierarchical configs, freeze flags
- **Tokenizer**: Simple char, regex word, tiktoken, and HuggingFace tokenizers

### Modules Tests
- **Activations**: GELU, SiLU, ReLU activation functions
- **Embeddings**: Token, positional, and combined embeddings
- **Layer Normalization**: Custom LayerNorm implementation
- **Losses**: Cross-entropy loss wrapper
- **Multi-Head Attention**: Self-attention mechanism with causal masking
- **Transformer**: Complete transformer blocks and full transformer model
- **Build Functions**: Module construction and configuration

### Data Tests
- **Datasets**: Text file and SFT JSON datasets
- **Transforms**: Alpaca instruction formatting and text pair transforms
- **Collate Functions**: Batching and padding for different data types

## Test Features

- **Comprehensive Coverage**: Tests cover all public methods and edge cases
- **Mocking**: External dependencies are properly mocked
- **Edge Cases**: Tests include boundary conditions and error cases
- **Differentiability**: Tests verify gradient flow through neural networks
- **Shape Preservation**: Tests ensure tensor shapes are maintained
- **Configuration**: Tests verify proper handling of various config options

## Test Quality

- **Isolated Tests**: Each test is independent and can run in any order
- **Deterministic**: Tests produce consistent results across runs
- **Fast Execution**: Tests run quickly without external dependencies
- **Clear Assertions**: Tests have clear, descriptive assertions
- **Error Messages**: Tests provide helpful error messages when they fail

## Adding New Tests

When adding new functionality, follow these guidelines:

1. **Test File Naming**: Use `test_<module_name>.py` format
2. **Test Class Naming**: Use `Test<ClassName>` format
3. **Test Method Naming**: Use `test_<functionality>` format
4. **Docstrings**: Include docstrings explaining what each test does
5. **Edge Cases**: Test boundary conditions and error cases
6. **Mocking**: Mock external dependencies appropriately
7. **Assertions**: Use specific, meaningful assertions

## Example Test Structure

```python
class TestExampleClass:
    """Test the ExampleClass class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        obj = ExampleClass()
        assert obj.param == expected_value

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        obj = ExampleClass(custom_param=value)
        assert obj.custom_param == value

    def test_method_functionality(self):
        """Test that method works correctly."""
        obj = ExampleClass()
        result = obj.method(input_data)
        assert result == expected_output

    def test_method_edge_cases(self):
        """Test method with edge cases."""
        obj = ExampleClass()
        with pytest.raises(ValueError):
            obj.method(invalid_input)
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines and should:
- Complete within a reasonable time frame
- Not require external network access
- Not depend on specific hardware
- Produce consistent results across environments

