# Implementation Plan

- [x] 1. Set up steering directory structure and organize existing files
  - Create subdirectories in `.kiro/steering/` for logical organization
  - Move existing steering files to appropriate subdirectories
  - Update any references to moved steering files
  - _Requirements: 4.1, 4.2_

- [x] 1.1 Create steering subdirectories
  - Create `.kiro/steering/core/` directory for core infrastructure steerings
  - Create `.kiro/steering/experiments/` directory for experiment-related steerings
  - Create `.kiro/steering/data/` directory for data processing steerings
  - Create `.kiro/steering/analysis/` directory for analysis and visualization steerings
  - Create `.kiro/steering/ui/` directory for user interface steerings
  - Create `.kiro/steering/setup/` directory for environment setup steerings
  - Create `.kiro/steering/meta/` directory for meta-documentation steerings
  - _Requirements: 4.1_

- [x] 1.2 Move existing steering files to subdirectories
  - Move `core-modules.md` to `.kiro/steering/core/core-modules.md`
  - Move `infrastructure.md` to `.kiro/steering/core/infrastructure.md`
  - Move `utilities-and-patterns.md` to `.kiro/steering/core/utilities-and-patterns.md`
  - Move `experiment-runners.md` to `.kiro/steering/experiments/experiment-runners.md`
  - Move `knockout-mechanisms.md` to `.kiro/steering/experiments/knockout-mechanisms.md`
  - Move `experiment-testing.md` to `.kiro/steering/experiments/experiment-testing.md`
  - Move `prompt-filteration.md` to `.kiro/steering/experiments/prompt-filteration.md`
  - Move `data-relationships-interface.md` to `.kiro/steering/data/data-relationships-interface.md`
  - Move `dataset-processing-flow.md` to `.kiro/steering/data/dataset-processing-flow.md`
  - Move `analysis-and-plotting.md` to `.kiro/steering/analysis/analysis-and-plotting.md`
  - Move `streamlit-infrastructure.md` to `.kiro/steering/ui/streamlit-infrastructure.md`
  - Move `streamlit-app-pages.md` to `.kiro/steering/ui/streamlit-app-pages.md`
  - Move `streamlit-app-components.md` to `.kiro/steering/ui/streamlit-app-components.md`
  - Move `setup-and-environment.md` to `.kiro/steering/setup/setup-and-environment.md`
  - _Requirements: 4.1, 4.2_

- [x] 2. Migrate content from docs to steering files
  - Copy complete content from each doc file to its corresponding steering file
  - Preserve all technical details, code examples, and implementation patterns
  - Maintain critical warnings and verification checklists
  - _Requirements: 2.1, 2.2, 4.3_

- [x] 2.1 Migrate core infrastructure steering content
  - Copy content from `docs/core-modules.md` to `.kiro/steering/core/core-modules.md`
  - Copy content from `docs/infrastructure.md` to `.kiro/steering/core/infrastructure.md`
  - Copy content from `docs/utilities-and-patterns.md` to `.kiro/steering/core/utilities-and-patterns.md`
  - Ensure all 3-file coordination patterns and base class rules are preserved
  - _Requirements: 2.1, 2.2_

- [x] 2.2 Migrate experiment-related steering content
  - Copy content from `docs/experiment-runners.md` to `.kiro/steering/experiments/experiment-runners.md`
  - Copy content from `docs/knockout-mechanisms.md` to `.kiro/steering/experiments/knockout-mechanisms.md`
  - Copy content from `docs/experiment-testing.md` to `.kiro/steering/experiments/experiment-testing.md`
  - Copy content from `docs/prompt-filteration.md` to `.kiro/steering/experiments/prompt-filteration.md`
  - Preserve all runner patterns and dependency management rules
  - _Requirements: 2.1, 2.2_

- [x] 2.3 Migrate data processing steering content
  - Copy content from `docs/data-relationships-interface.md` to `.kiro/steering/data/data-relationships-interface.md`
  - Copy content from `docs/dataset-processing-flow.md` to `.kiro/steering/data/dataset-processing-flow.md`
  - Preserve all data object interfaces and processing flow patterns
  - _Requirements: 2.1, 2.2_

- [x] 2.4 Migrate analysis and UI steering content
  - Copy content from `docs/analysis-and-plotting.md` to `.kiro/steering/analysis/analysis-and-plotting.md`
  - Copy content from `docs/streamlit-infrastructure.md` to `.kiro/steering/ui/streamlit-infrastructure.md`
  - Copy content from `docs/streamlit-app-pages.md` to `.kiro/steering/ui/streamlit-app-pages.md`
  - Copy content from `docs/streamlit-app-components.md` to `.kiro/steering/ui/streamlit-app-components.md`
  - Preserve all visualization patterns and UI component architecture
  - _Requirements: 2.1, 2.2_

- [x] 2.5 Migrate setup steering content
  - Copy content from `docs/setup-and-environment.md` to `.kiro/steering/setup/setup-and-environment.md`
  - Preserve all package management rules and environment setup procedures
  - _Requirements: 2.1, 2.2_

- [x] 3. Update front-matter configuration for all steering files
  - Add proper YAML front-matter to all migrated steering files
  - Use consistent format with description and inclusion settings
  - Ensure manual inclusion for all non-navigation steering files
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 3.1 Add front-matter to core steering files
  - Add front-matter to `.kiro/steering/core/core-modules.md` with description "Core Module Coordination Patterns" and inclusion: manual
  - Add front-matter to `.kiro/steering/core/infrastructure.md` with description "Infrastructure Base Classes and Patterns" and inclusion: manual
  - Add front-matter to `.kiro/steering/core/utilities-and-patterns.md` with description "Code Organization Standards" and inclusion: manual
  - _Requirements: 5.1, 5.3, 5.4_

- [x] 3.2 Add front-matter to experiment steering files
  - Add front-matter to `.kiro/steering/experiments/experiment-runners.md` with description "Experiment Runner Types and Implementation Patterns" and inclusion: manual
  - Add front-matter to `.kiro/steering/experiments/knockout-mechanisms.md` with description "Model Intervention and Knockout Implementations" and inclusion: manual
  - Add front-matter to `.kiro/steering/experiments/experiment-testing.md` with description "Testing Patterns and Procedures" and inclusion: manual
  - Add front-matter to `.kiro/steering/experiments/prompt-filteration.md` with description "Prompt Filtering Logic and Patterns" and inclusion: manual
  - _Requirements: 5.1, 5.3, 5.4_

- [x] 3.3 Add front-matter to data, analysis, UI, and setup steering files
  - Add front-matter to data steering files with appropriate descriptions and inclusion: manual
  - Add front-matter to analysis steering files with appropriate descriptions and inclusion: manual
  - Add front-matter to UI steering files with appropriate descriptions and inclusion: manual
  - Add front-matter to setup steering files with appropriate descriptions and inclusion: manual
  - _Requirements: 5.1, 5.3, 5.4_

- [x] 4. Update cross-references in all steering files
  - Replace all `[docs/filename.md](filename.md)` references with steering context format
  - Update cross-reference sections to reflect steering-based system
  - Maintain logical grouping by category while using steering references
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 4.1 Update cross-references in core steering files
  - Update all cross-references in core steering files to use `#steering-name` format
  - Replace docs references with steering context keys
  - Maintain descriptive text explaining what referenced steerings contain
  - _Requirements: 6.1, 6.2_

- [x] 4.2 Update cross-references in experiment steering files
  - Update all cross-references in experiment steering files to use steering format
  - Replace docs references with appropriate steering context keys
  - Preserve logical relationships between experiment components
  - _Requirements: 6.1, 6.2_

- [x] 4.3 Update cross-references in remaining steering files
  - Update cross-references in data, analysis, UI, and setup steering files
  - Ensure all references point to valid steering files in subdirectories
  - Maintain cross-reference integrity across the entire steering system
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 5. Create meta-steering for steering guidelines
  - Write comprehensive steering guidelines document
  - Define characteristics of effective steering files
  - Provide guidance for future steering development
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [x] 5.1 Write steering principles and characteristics
  - Document that steerings should be short but information-dense
  - Explain that steerings describe concepts and modules with associated rules
  - Define requirements for describing connections to other modules
  - Specify guidelines for minimal code with expansion instructions
  - _Requirements: 7.2, 7.3, 7.4, 7.5_

- [x] 5.2 Create steering creation guidelines
  - Provide instructions on how LLMs can expand context or get examples
  - Document front-matter requirements and formatting standards
  - Explain manual inclusion system and context key usage
  - Create template for new steering file creation
  - _Requirements: 7.6, 7.7_

- [x] 5.3 Create steering refinement and maintenance guidelines
  - Write meta-steering document on how to refine and update existing steering files
  - Document process for applying new steering principles to existing files
  - Create guidelines for evaluating steering effectiveness and identifying improvement areas
  - Provide framework for systematic steering review and enhancement
  - Include instructions for maintaining consistency when steering standards evolve
  - _Requirements: 7.1, 7.6_

- [x] 6. Update master navigation steering file
  - Replace all docs references with steering context keys
  - Update task-based navigation to reference steering files in subdirectories
  - Modify available modules list to reflect steering organization
  - Ensure inclusion: always configuration is maintained
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 3.3, 3.4, 5.2_

- [x] 6.1 Update navigation references and task categories
  - Replace all `docs/filename.md` references with `#steering-name` format
  - Update task-based navigation sections to use steering context keys
  - Modify quick navigation sections to reference steering files in subdirectories
  - Update available documentation modules list to reflect new steering organization
  - _Requirements: 1.4, 3.1, 3.2, 3.3_

- [x] 6.2 Update navigation guidance and instructions
  - Modify instructions to guide LLMs to use steering context keys instead of file paths
  - Update critical documentation rules to reflect steering-based approach
  - Ensure navigation provides clear guidance on which steering files to read for different task types
  - Maintain inclusion: always configuration for master navigation
  - _Requirements: 1.1, 1.2, 1.3, 5.2_

- [x] 7. Review and verify migration completeness
  - Verify all content from docs is present in steering files
  - Check cross-reference integrity across all steering files
  - Test navigation functionality with sample task scenarios
  - Confirm front-matter consistency across all files
  - _Requirements: 2.3, 4.4, 5.4, 6.4_

- [x] 7.1 Content completeness verification
  - Compare each doc file with its corresponding steering file to ensure all content is migrated
  - Verify technical details, code examples, and critical warnings are preserved
  - Check that all cross-reference sections are updated appropriately
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 7.2 Cross-reference and navigation testing
  - Verify all steering context keys resolve to valid steering files
  - Test task-based navigation scenarios to ensure appropriate steering files are referenced
  - Check that cross-references maintain logical grouping and descriptive context
  - Confirm master navigation provides clear guidance for different task types
  - _Requirements: 1.1, 1.2, 6.1, 6.2, 6.3, 6.4_
