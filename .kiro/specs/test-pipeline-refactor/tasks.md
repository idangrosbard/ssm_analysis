# Implementation Plan

- [x] 1. Create baseline_builder.py with centralized test parameters
  - Extract all test configuration constants from current test_full_pipeline.py (ORIGINAL_IDS, TEST_MODEL_CONFIGS)
  - Create BaselineBuilder class with baseline generation functionality
  - Include clean_and_generate_base_test_data functionality that loads full dataset and filters by ORIGINAL_IDS
  - Maintain same CLI interface and behavior as current script
  - make sure that it only download and tests with the subset, and not the whole data
  - _Requirements: 3.1, 3.2, 3.3, 6.1, 6.2, 6.3, 6.4_

- [x] 2. Create runners directory structure and test utilities
  - Create tests/src/experiments/runners/ directory
  - Add __init__.py file for proper Python package structure
  - Create shared test utilities module for path management and data copying
  - _Requirements: 4.1, 4.2, 5.1, 5.2_

- [x] 2.1. Design path management strategy for tests
  - Create utility to copy baseline test data (filtered dataset) from baseline location to temporary directory
  - Implement context manager for PROJECT_DIR patching during test execution
  - Design baseline result loading that uses original (unpatched) baseline path
  - Ensure tests can access both temp directory for execution and baseline for comparison
  - Handle case where full dataset needs to be loaded and filtered by ORIGINAL_IDS for each test
  - _Requirements: 2.2, 4.3, 5.1_

- [x] 3. Implement test_evaluate_model.py runner test
  - Create TestEvaluateModelRunner class with setup/teardown methods
  - Import test parameters (ORIGINAL_IDS, TEST_MODEL_CONFIGS) from baseline_builder.py
  - Generate filtered test dataset in temporary directory using ORIGINAL_IDS for each test
  - Copy relevant dependency data for each model from baseline to temp directory
  - Patch PROJECT_DIR to point to temp directory during test execution
  - Run EvaluateModel with compute_dependencies to ensure all required data is available
  - Load baseline results from original baseline location (unpatched path)
  - Compare test results against baseline using tolerance for numerical values
  - _Requirements: 2.1, 2.2, 4.1, 4.3, 4.4_

- [x] 4. Implement test_heatmap.py runner test
  - Create TestHeatmapRunner class with setup/teardown methods
  - Import test parameters (ORIGINAL_IDS, TEST_MODEL_CONFIGS) from baseline_builder.py
  - Generate filtered test dataset in temporary directory using ORIGINAL_IDS for each test
  - Copy relevant dependency data for each model from baseline to temp directory
  - Patch PROJECT_DIR to point to temp directory during test execution
  - Run Heatmap with compute_dependencies to ensure all required data is available
  - Load baseline results from original baseline location (unpatched path)
  - Compare test results against baseline using tolerance for numerical values
  - _Requirements: 2.1, 2.2, 4.1, 4.3, 4.4_

- [x] 5. Implement test_info_flow.py runner test
  - Create TestInfoFlowRunner class with setup/teardown methods
  - Import test parameters (ORIGINAL_IDS, TEST_MODEL_CONFIGS) from baseline_builder.py
  - Generate filtered test dataset in temporary directory using ORIGINAL_IDS for each test
  - Copy relevant dependency data for each model from baseline to temp directory
  - Patch PROJECT_DIR to point to temp directory during test execution
  - Run InfoFlow with compute_dependencies to ensure all required data is available
  - Load baseline results from original baseline location (unpatched path)
  - Compare test results against baseline using tolerance for numerical values
  - _Requirements: 2.1, 2.2, 4.1, 4.3, 4.4_

- [x] 6. Add shared test utilities for result comparison and path management
  - Create utility functions for comparing serialized results with numerical tolerance
  - Implement temporary directory management that recreates filtered dataset in temp directory
  - Add utility to copy model-specific dependency data from baseline to temp directory
  - Add baseline result loading functionality from original baseline location (not patched path)
  - Create comparison logic that handles nested dictionaries and arrays with tolerance
  - Implement dataset filtering utility that loads full counter_fact dataset and filters by ORIGINAL_IDS
  - Design path patching strategy that allows baseline loading from original location
  - _Requirements: 2.2, 4.3, 5.1, 5.3_

- [x] 7. Remove original test_full_pipeline.py file
  - Delete the original test_full_pipeline.py file
  - Verify all functionality is preserved in new structure
  - Update any references to point to baseline_builder.py
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 8. Test and validate the refactored implementation
  - Run baseline generation to ensure identical behavior
  - Execute all individual runner tests
  - Verify parameterized tests work correctly with tolerance
  - Confirm git tracking of baselines works as expected
  - _Requirements: 2.3, 3.1, 4.1, 4.2, 4.3, 4.4, 6.1, 6.2, 6.3, 6.4_
