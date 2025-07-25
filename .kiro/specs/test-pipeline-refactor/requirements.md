# Requirements Document

## Introduction

This feature refactors the existing `tests/src/experiments/test_full_pipeline.py` file into a builder/tester pattern to improve code organization, maintainability, and reusability. The current file mixes configuration building, test data generation, experiment execution, and baseline management responsibilities.

## Requirements

### Requirement 1

**User Story:** As a developer, I want a dedicated test configuration builder, so that I can easily create and customize full pipeline test configurations without mixing concerns.

#### Acceptance Criteria

1. WHEN I need to create a test configuration THEN the system SHALL provide a dedicated builder class that handles configuration creation
2. WHEN I specify model architecture and parameters THEN the builder SHALL generate appropriate FullPipelineRunner configurations
3. WHEN I need different configurations for different model architectures THEN the builder SHALL handle architecture-specific customizations
4. WHEN I want to reuse configuration logic THEN the builder SHALL provide a clean, composable interface

### Requirement 2

**User Story:** As a developer, I want a separate test execution and validation system, so that I can run tests independently from configuration building.

#### Acceptance Criteria

1. WHEN I want to run full pipeline tests THEN the system SHALL provide a dedicated tester class that handles test execution
2. WHEN I need to validate test results THEN the tester SHALL provide result validation capabilities
3. WHEN I want to run tests with different configurations THEN the tester SHALL accept configurations from the builder
4. WHEN tests fail THEN the tester SHALL provide clear error reporting and diagnostics

### Requirement 3

**User Story:** As a developer, I want baseline management that tracks changes in git, so that I can monitor and validate changes to test baselines over time.

#### Acceptance Criteria

1. WHEN I run baseline generation THEN the system SHALL create baselines that are tracked in git
2. WHEN baselines change THEN I SHALL be able to see the differences in git history
3. WHEN I need to rebuild baselines THEN the system SHALL provide a simple script interface
4. WHEN baselines are updated THEN they SHALL be stored in a format suitable for version control

### Requirement 4

**User Story:** As a developer, I want granular, fast tests for individual experiments and models, so that I can quickly validate that changes don't break functionality.

#### Acceptance Criteria

1. WHEN I run individual model tests THEN they SHALL execute quickly and validate specific model behavior
2. WHEN I run experiment-specific tests THEN they SHALL test individual experiment types in isolation
3. WHEN comparing results across systems THEN the tests SHALL use tolerance for logits to handle small numerical differences
4. WHEN tests fail THEN they SHALL provide specific information about which model or experiment failed

### Requirement 5

**User Story:** As a developer, I want a simple refactoring approach that doesn't create excessive new objects, so that the code remains maintainable and easy to understand.

#### Acceptance Criteria

1. WHEN I examine the refactored code THEN it SHALL use a minimal number of new classes and objects
2. WHEN I need to understand the code THEN the structure SHALL be straightforward and intuitive
3. WHEN I modify the code THEN the changes SHALL be localized and predictable
4. WHEN the system runs THEN it SHALL maintain the same performance characteristics as the original
### Requirement 6

**User Story:** As a developer, I want the baseline generation to maintain the same CLI interface and behavior, so that existing workflows continue to work.

#### Acceptance Criteria

1. WHEN I run the baseline builder script THEN it SHALL behave identically to the current script
2. WHEN baseline generation completes THEN the output SHALL be identical to current implementation
3. WHEN I use the same CLI parameters THEN the results SHALL be the same
4. WHEN baselines are generated THEN they SHALL be stored in the same location and format
