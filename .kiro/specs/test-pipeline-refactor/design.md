# Design Document

## Overview

This design refactors the existing `test_full_pipeline.py` into a builder/tester pattern while maintaining simplicity and backward compatibility. The refactoring separates concerns into two main components: a configuration builder for creating test configurations and a tester for executing and validating tests.

## Architecture

### High-Level Structure

```
tests/src/experiments/
├── baseline_builder.py             # Baseline generation (replaces current script)
└── runners/                        # Individual runner tests
    ├── test_evaluate_model.py      # EvaluateModel runner tests
    ├── test_heatmap.py             # Heatmap runner tests
    └── test_info_flow.py           # InfoFlow runner tests
```

### Component Responsibilities

1. **baseline_builder.py**: Generates baselines exactly as the current script does
2. **runners/test_*.py**: Individual test files for each runner type that compare against baselines

## Components and Interfaces

### BaselineBuilder

```python
class BaselineBuilder:
    """Generates baseline test data exactly as the current script does."""
    
    def __init__(self, test_base_path: Path):
        self.test_base_path = test_base_path
    
    def clean_and_generate_base_test_data(self) -> None:
        """Clean test directory and generate base test data."""
    
    def run_baseline_experiments(
        self, 
        normalizing_outputs: bool = True,
        with_plotting: bool = True
    ) -> None:
        """Run all experiments using TEST_MODEL_CONFIGS to generate baseline results."""
    
    def build_baseline(self, params: CreateBaselineParams) -> None:
        """Main method to build complete baseline."""
    
    def get_config_for_model(
        self, 
        model_arch: MODEL_ARCH, 
        model_size: str, 
        with_plotting: bool = True
    ) -> FullPipelineRunner:
        """Get configuration for specific model - used by both baseline and tests."""
```

### Test Configuration Management

```python
# baseline_builder.py - Contains all test parameters
TEST_MODEL_CONFIGS = [
    (MODEL_ARCH.MAMBA1, "130M"),
    (MODEL_ARCH.MAMBA2, "130M"),
    (MODEL_ARCH.GPT2, "355M"),
    (MODEL_ARCH.LLAMA3_2, "1B"),
    (MODEL_ARCH.QWEN2, "0.5B"),
    (MODEL_ARCH.QWEN2_5, "0.5B"),
]

ORIGINAL_IDS = {
    SPLIT.TRAIN1: [53, 59, 74, 90, 93],
    SPLIT.TRAIN2: [10594, 6410, 140, 148, 159, 182],
}

def get_test_full_pipeline_config_per_model_arch(
    code_version_name: str, 
    model_arch: MODEL_ARCH, 
    model_size: str, 
    with_plotting: bool
) -> FullPipelineRunner:
    """Get configuration for specific model architecture - shared by baseline and tests."""

# runners/test_*.py - Import parameters from baseline_builder
from ..baseline_builder import TEST_MODEL_CONFIGS, get_test_full_pipeline_config_per_model_arch

def get_baseline_model_configs() -> list[tuple[MODEL_ARCH, str]]:
    """Get model configurations from baseline builder."""
    return TEST_MODEL_CONFIGS

class TestEvaluateModelRunner:
    """Tests for EvaluateModel runner comparing against baseline."""
    
    def setup_method(self) -> None:
        """Set up temporary test directory."""
    
    def teardown_method(self) -> None:
        """Clean up temporary test directory."""
    
    @pytest.mark.parametrize("model_arch,model_size", get_baseline_model_configs())
    def test_evaluate_model(
        self, 
        model_arch: MODEL_ARCH, 
        model_size: str
    ) -> None:
        """Test model evaluation for specific model against baseline."""
    
    def compare_results_with_tolerance(
        self, 
        actual_results: dict, 
        baseline_results: dict,
        tolerance: float = 1e-6
    ) -> None:
        """Compare results with numerical tolerance."""
```

## Data Models

### Configuration Data

The existing data structures remain unchanged:

- `InputParams`: Input data parameters  
- `MetadataParams`: Metadata parameters
- `CreateBaselineParams`: Baseline creation parameters

### Test Data Management

```python
@dataclass
class TestEnvironment:
    """Test environment configuration."""
    test_base_path: Path
    original_ids: dict[SPLIT, list[TPromptOriginalIndex]]
    paths_config: PathsConfig
```

## Error Handling

### Test Execution Errors

1. **Configuration Errors**: Clear error messages for invalid model configurations
2. **Data Setup Errors**: Graceful handling of test data generation failures
3. **Experiment Execution Errors**: Detailed error reporting for failed experiments
4. **Validation Errors**: Specific information about which validations failed

### Tolerance Handling

```python
def compare_with_tolerance(actual: float, expected: float, tolerance: float) -> bool:
    """Compare numerical values with tolerance for cross-system compatibility."""
    return abs(actual - expected) <= tolerance
```

## Testing Strategy

### Unit Tests

1. **Builder Tests**: Test configuration generation for different model architectures
2. **Runner Tests**: Test individual experiment execution
3. **Validation Tests**: Test result comparison with tolerance

### Integration Tests

1. **Full Pipeline Tests**: End-to-end testing of complete pipeline
2. **Baseline Generation Tests**: Test baseline creation and git tracking
3. **Cross-System Tests**: Test tolerance handling across different systems

### Test Organization

```python
# runners/test_evaluate_model.py
class TestEvaluateModelRunner:
    @pytest.mark.parametrize("model_arch,model_size", get_baseline_model_configs())
    def test_evaluate_model(self, model_arch: MODEL_ARCH, model_size: str):
        """Test model evaluation against baseline."""

# runners/test_heatmap.py  
class TestHeatmapRunner:
    @pytest.mark.parametrize("model_arch,model_size", get_baseline_model_configs())
    def test_heatmap_generation(self, model_arch: MODEL_ARCH, model_size: str):
        """Test heatmap generation against baseline."""

# runners/test_info_flow.py
class TestInfoFlowRunner:
    @pytest.mark.parametrize("model_arch,model_size", get_baseline_model_configs())
    def test_info_flow_analysis(self, model_arch: MODEL_ARCH, model_size: str):
        """Test info flow analysis against baseline."""
```

## Implementation Details

### Baseline Generation

The baseline builder replicates current script functionality:

```python
# baseline_builder.py
@pyrallis.wrap()
def main(params: CreateBaselineParams):
    """Generate baseline test data."""
    builder = BaselineBuilder(TEST_BASE_PATH)
    builder.build_baseline(params)

if __name__ == "__main__":
    main()
```

### Monkey Patching Strategy

Centralize monkey patching in the test runner:

```python
class FullPipelineTestRunner:
    def _setup_monkey_patches(self, normalizing_outputs: bool) -> pytest.MonkeyPatch:
        """Set up all required monkey patches."""
        mp = pytest.MonkeyPatch()
        mp.setattr(PATHS_PROJECT_DIR_PATH, self.test_base_path)
        mp.setattr(INFO_FLOW_PRINT_INTERVAL_PATH, 1)
        if normalizing_outputs:
            mp.setattr(GET_COMMIT_HASH_PATH, lambda *args, **kwargs: "test_commit_hash")
            mp.setattr(CREATE_RUN_ID_PATH, lambda *args, **kwargs: "test_run_id")
        return mp
```

### Architecture-Specific Configurations

Handle model-specific configurations in the builder:

```python
def build_config_for_arch(self, model_arch: MODEL_ARCH, model_size: str, with_plotting: bool) -> FullPipelineRunner:
    """Build architecture-specific configuration."""
    base_config = self.build_config(model_arch, model_size, with_plotting)
    
    if model_arch in [MODEL_ARCH.QWEN2_5, MODEL_ARCH.QWEN2]:
        return base_config.modify(
            variant_params=base_config.variant_params.modify(
                knockout_map={
                    TokenType.last: [
                        (TokenType.last, FeatureCategory.ALL),
                        (TokenType.first, FeatureCategory.ALL),
                        (TokenType.subject, FeatureCategory.ALL),
                    ],
                },
                info_flow_window_size=TWindowSize(3),
                heatmap_window_size=TWindowSize(3),
            )
        )
    
    return base_config
```

## Migration Strategy

### Phase 1: Create Baseline Builder

1. Create `baseline_builder.py` with current script functionality
2. Extract baseline generation logic into `BaselineBuilder` class
3. Maintain same CLI interface and behavior

### Phase 2: Create Runner Test Structure

1. Create `runners/` subdirectory
2. Create individual test files for each runner type
3. Implement test classes with setup/teardown methods

### Phase 3: Implement Individual Tests

1. Add tests for each model architecture per runner
2. Implement result comparison with tolerance
3. Add temporary directory management for tests

### Phase 4: Remove Original File

1. Delete original `test_full_pipeline.py`
2. Update any references to point to new structure
3. Verify all functionality is preserved

## Notes

The current implementation is using monkey patch to patch some global definitions.
It may make things a bit more complicated and we need to make sure that the monkey patching is done correctly.
We need to make sure that the tests only donwloading the subset of data that they need.
We need to make sure that the tests are not overwriting the original data.
