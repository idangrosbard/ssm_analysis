---
description: Testing Patterns and Procedures
inclusion: manual
---

# Experiment Testing Framework

This document provides a comprehensive guide to the experiment testing infrastructure in the SSM Analysis project, covering test patterns, baseline generation, and validation workflows.

## Overview

The experiment testing framework provides robust validation of the complete experimental pipeline through:

- **Baseline Generation**: Creating reference outputs from experiment runs
- **Granular Testing**: Fine-grained validation of individual experiment components
- **Recovery Testing**: Validating experiment interruption and resumption
- **Configuration Testing**: Ensuring experiment configurations are valid

## Test Architecture

### Test Types

#### 1. Baseline Builder (`baseline_builder.py`)

**Purpose**: Generate and manage test baselines

```python
# Generate new baseline
python tests/src/experiments/baseline_builder.py

# Validate existing baseline
python tests/src/experiments/baseline_builder.py --validate-after-build

# Selective baseline generation
python tests/src/experiments/baseline_builder.py --models '["mamba1", "gpt2"]' --experiments '["evaluate_model"]'
```

**Key Features**:

- **Incremental Generation**: Resume from existing outputs
- **Selective Testing**: Target specific models/experiments
- **Validation Pipeline**: Automatic integrity checking
- **Comparison Tools**: Diff between baseline versions

#### 2. Granular Tests (`test_full_pipeline_granular.py`)

**Purpose**: Fine-grained validation of experiment outputs

```bash
# Run all granular tests
pytest tests/src/experiments/test_full_pipeline_granular.py

# Test specific model
pytest tests/src/experiments/test_full_pipeline_granular.py::TestEvaluateModel::test_evaluate_model_output[gpt2-355M]

# Test specific experiment type
pytest tests/src/experiments/test_full_pipeline_granular.py::TestHeatmap
```

#### 3. Recovery Tests (`test_info_flow.py`)

**Purpose**: Validate experiment interruption and recovery

```python
def test_info_flow_recovery(tmp_path: Path):
    """Test that info flow can save and recover from intermediate results correctly."""
```

## Best Practices

### 1. Test Organization

- **Isolation**: Each test uses temporary directories
- **Determinism**: Fixed seeds and controlled environments
- **Granularity**: Separate tests for different experiment aspects

### 4. Baseline Management

- **Versioning**: Track baseline changes with git
- **Validation**: Always validate after generation
- **Documentation**: Document expected baseline structure
