# Test Suite for pyterrier-generative

This directory contains comprehensive tests for the pyterrier-generative library.

## Test Structure

```
tests/
├── __init__.py
├── README.md                    # This file
├── test_algorithms.py          # Algorithm implementation tests
├── test_base_ranker.py         # GenerativeRanker base class tests
├── test_standard_ranker.py     # StandardRanker and variant tests
├── test_lit5.py                # LiT5-specific tests
├── test_variants.py            # Variants metaclass tests
└── test_integration.py         # End-to-end integration tests
```

## Test Categories

### Unit Tests

**test_algorithms.py** - Tests for ranking algorithms
- RankedList data structure
- Window iteration utilities
- Algorithm implementations (sliding_window, single_window, setwise, tdpart)
- Algorithm enum
- Batching support in algorithms

**test_base_ranker.py** - Tests for GenerativeRanker base class
- Initialization and configuration
- Prompt construction (Jinja2 templates and callables)
- Output parsing
- Window ranking methods
- Transform pipeline
- Batching infrastructure

**test_standard_ranker.py** - Tests for StandardRanker
- Variant system (RankZephyr, RankVicuna, RankGPT, etc.)
- Backend auto-detection
- Model configuration
- Convenience functions

**test_lit5.py** - Tests for LiT5 model
- LiT5Backend initialization
- FiD architecture integration
- LiT5-specific batching
- Transform functionality

**test_variants.py** - Tests for Variants metaclass
- Metaclass functionality
- Variant creation and instantiation
- Docstring generation
- Error handling

### Integration Tests

**test_integration.py** - End-to-end integration tests
- Full ranking pipeline
- Multiple queries and algorithms
- Batching efficiency
- Different prompt formats
- Edge cases
- Parameter combinations

## Running Tests

### Run all tests
```bash
uv run pytest
```

### Run specific test file
```bash
uv run pytest tests/test_algorithms.py
```

### Run specific test class
```bash
uv run pytest tests/test_algorithms.py::TestSlidingWindow
```

### Run specific test
```bash
uv run pytest tests/test_algorithms.py::TestSlidingWindow::test_basic_sliding_window
```

### Run with coverage
```bash
uv run pytest --cov=pyterrier_generative --cov-report=html
```

### Run only fast tests (exclude integration tests)
```bash
uv run pytest -m "not integration"
```

### Run only integration tests
```bash
uv run pytest -m integration
```

### Run with verbose output
```bash
uv run pytest -v
```

### Run with detailed output for failures
```bash
uv run pytest -vv --tb=long
```

## Test Markers

Tests are marked with custom markers for selective execution:

- `@pytest.mark.integration` - Integration tests that may download models
- `@pytest.mark.slow` - Tests that take significant time
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.api` - Tests requiring API keys

## Test Philosophy

### No Mocking

These tests use **real implementations** without mocking wherever possible:

1. **Deterministic backends** instead of mocked backends
2. **Simple rankers** that return predictable results
3. **Real data structures** (pandas DataFrames, numpy arrays)
4. **Actual algorithm execution**

This approach ensures tests validate real behavior rather than mocked interactions.

### Test Fixtures

Common test data is provided via pytest fixtures:

- `sample_data` - Small DataFrame with 10 documents
- `multi_query_data` - DataFrame with multiple queries
- `simple_backend` - Deterministic backend for testing
- `small_dataframe` - DataFrame with 5 documents

### Parameterized Tests

Many tests use `@pytest.mark.parametrize` to test multiple configurations:

```python
@pytest.mark.parametrize("window_size,stride", [
    (5, 3),
    (10, 5),
    (20, 10),
])
def test_window_stride_combinations(window_size, stride):
    ...
```

## Test Coverage

Current test coverage focuses on:

✅ **Core functionality**
- All algorithms (sliding_window, single_window, setwise, tdpart)
- Prompt construction (Jinja2 and callable)
- Output parsing
- Transform pipeline
- Batching infrastructure

✅ **Model variants**
- StandardRanker variants
- RankZephyr, RankVicuna, RankGPT aliases
- LiT5 model
- Variants metaclass

✅ **Edge cases**
- Empty inputs
- Single document
- Large datasets
- Missing or malformed data

✅ **Integration scenarios**
- Multiple queries
- Different algorithms
- Batching efficiency
- Parameter combinations

## Writing New Tests

### Test Naming Convention

- Test files: `test_<module>.py`
- Test classes: `Test<Feature>`
- Test methods: `test_<specific_behavior>`

### Example Test Structure

```python
class TestMyFeature:
    """Test MyFeature functionality."""

    def test_basic_usage(self):
        """Test basic usage of MyFeature."""
        # Arrange
        feature = MyFeature(param=value)

        # Act
        result = feature.do_something()

        # Assert
        assert result == expected_value

    def test_edge_case(self):
        """Test edge case handling."""
        # Test implementation
        ...
```

### Using Fixtures

```python
@pytest.fixture
def my_data():
    """Provide test data."""
    return create_test_data()

def test_with_fixture(my_data):
    """Test using fixture."""
    result = process(my_data)
    assert result is not None
```

### Parameterized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("a", 1),
    ("b", 2),
    ("c", 3),
])
def test_multiple_cases(input, expected):
    """Test multiple input/output pairs."""
    assert transform(input) == expected
```

## Dependencies

Test dependencies are specified in `requirements-dev.txt`:

- pytest
- pytest-cov (coverage reporting)
- pytest-subtests (subtests support)
- pytest-json-report (JSON reporting)

## Continuous Integration

Tests should be run in CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    uv run pytest --cov=pyterrier_generative --cov-report=xml
```

## Troubleshooting

### Import Errors

If tests fail with import errors:
```bash
uv pip install -e .
uv run pytest
```

### Model Download Failures

Integration tests marked with `@pytest.mark.integration` may require model downloads. Skip them with:
```bash
uv run pytest -m "not integration"
```

### Missing Dependencies

Install all test dependencies:
```bash
uv pip install -r requirements-dev.txt
```

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all existing tests pass
3. Add integration tests for major features
4. Update this README if adding new test categories
5. Maintain >80% code coverage
