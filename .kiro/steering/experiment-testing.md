---
description: Experiment Testing Patterns and Procedures
inclusion: manual
---


# Experiment Testing Patterns and Procedures

## Testing Framework

### Testing Structure
- **Location**: `tests/`
- **Framework**: Pytest 8.3.4
- **Purpose**: Testing patterns and procedures for experiments

### Testing Patterns
- **UNIT tests** for individual components
- **INTEGRATION tests** for experiment workflows
- **END-TO-END tests** for complete pipelines

## Testing Procedures

### Test Organization
- **FOLLOW pytest conventions** for test structure
- **USE appropriate fixtures** for test setup
- **MAINTAIN test isolation** between test cases

### Testing Rules
- **TEST all critical functionality**
- **USE mocking** for expensive operations
- **MAINTAIN test reproducibility**

## Critical Testing Rules

### Testing Standards
- **COMPREHENSIVE test coverage** for critical components
- **PROPER test isolation** and cleanup
- **CONSISTENT testing patterns** across modules

### Test Execution
```bash
pytest -vv                            # Run all tests
pytest tests/specific_test.py         # Run specific test
```

**Reference**: docs/experiment-testing.md for complete testing patterns
