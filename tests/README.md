# Tests for MoralHazard Package

This directory contains tests for the `moralhazard` package.

## Test Structure

- `test_basic.py` - Basic functionality tests including:
  - Problem creation and validation
  - Solver functionality with both "a_hat" and "iterative" solvers
  - **Edge case tests for `a_hat` parameter:**
    - Single element `a_hat` array (e.g., `[0.0]`)
    - Two element `a_hat` array (e.g., `[0.0, 0.0]`)
  - Error handling for invalid inputs
  - `SolveResults` dataclass validation

## Running Tests

### Option 1: Using pytest directly
```bash
# Install dev dependencies first
uv add --dev pytest pytest-cov

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_basic.py -v

# Run with coverage
python -m pytest tests/ --cov=moralhazard --cov-report=html
```

### Option 2: Using the test runner script
```bash
python run_tests.py
```

### Option 3: Using uv
```bash
uv run pytest tests/ -v
```

## Test Coverage

The tests cover:

1. **Basic Functionality**
   - Valid configuration creation
   - Invalid configuration error handling
   - Problem parameter validation

2. **Solver Tests**
   - `a_hat` solver with single and multiple elements
   - Iterative solver
   - Parameter validation for each solver type

3. **Edge Cases**
   - Single element `a_hat` arrays (requested specifically)
   - Two element `a_hat` arrays
   - Invalid input handling

4. **Result Validation**
   - `SolveResults` structure verification
   - Immutability of results
   - Multiplier and constraint shapes

## Adding New Tests

To add new tests:

1. Create a new test file following the naming convention `test_*.py`
2. Import the necessary modules and classes
3. Create test classes that inherit from nothing (pytest style)
4. Use descriptive test method names starting with `test_`
5. Use fixtures for common setup code

Example:
```python
import pytest
from moralhazard import MoralHazardProblem

class TestNewFeature:
    def test_new_functionality(self):
        # Test implementation here
        assert True
```
