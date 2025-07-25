# Requirements Document

## Introduction

This feature involves migrating the existing `docs/` directory structure into proper Kiro steering files within `.kiro/steering/`. The goal is to transform the documentation system from a traditional docs approach into a context-aware steering system that guides LLM agents to read the appropriate steering files based on their specific tasks. The master `docs-navigation.md` steering file will serve as the gateway that directs agents to additional relevant steering files.

## Requirements

### Requirement 1

**User Story:** As an LLM agent working on SSM Analysis project tasks, I want to be guided by a master steering file that directs me to read additional relevant steering files, so that I can access the most appropriate context for my specific task type.

#### Acceptance Criteria

1. WHEN an LLM agent starts working on any task THEN the system SHALL always include the master `docs-navigation.md` steering file with `inclusion: always`
2. WHEN the master steering file is read THEN it SHALL provide clear guidance on which additional steering files to read based on task type
3. WHEN task-based navigation is provided THEN it SHALL categorize tasks into New Development, Experiment Development, Model Analysis, and UI Development
4. WHEN steering file references are made THEN they SHALL use the format `#core-modules` to enable manual inclusion of specific steering files

### Requirement 2

**User Story:** As an LLM agent, I want each steering file to contain the complete context from its corresponding doc file, so that I have all necessary information without needing to reference external documentation.

#### Acceptance Criteria

1. WHEN a steering file is created THEN it SHALL contain all content from its corresponding doc file
2. WHEN content is migrated THEN it SHALL preserve all technical details, code examples, and cross-references
3. WHEN steering files are created THEN they SHALL have proper front-matter with `inclusion: manual` for conditional inclusion
4. WHEN cross-references exist THEN they SHALL be updated to reference other steering files instead of doc files

### Requirement 3

**User Story:** As an LLM agent, I want the master navigation steering to replace references to docs with references to steering files, so that the system uses a consistent steering-based approach.

#### Acceptance Criteria

1. WHEN the master navigation is updated THEN it SHALL replace all `docs/filename.md` references with `#filename` steering references
2. WHEN task-based navigation is provided THEN it SHALL reference steering files using the manual inclusion format
3. WHEN the navigation describes available modules THEN it SHALL list steering files instead of documentation files
4. WHEN guidance is provided THEN it SHALL instruct agents to use steering context keys like `#core-modules` instead of file paths

### Requirement 4

**User Story:** As a developer, I want the docs directory to be preserved during migration, so that I can verify the migration was successful before removing the original files.

#### Acceptance Criteria

1. WHEN migration occurs THEN the original `docs/` directory SHALL remain untouched
2. WHEN steering files are updated THEN they SHALL be modified in place within `.kiro/steering/`
3. WHEN content is migrated THEN it SHALL be a copy operation, not a move operation
4. WHEN migration is complete THEN both docs and steering versions SHALL exist for verification

### Requirement 5

**User Story:** As an LLM agent, I want steering files to have consistent front-matter configuration, so that the inclusion system works properly across all steering files.

#### Acceptance Criteria

1. WHEN steering files are created THEN they SHALL have front-matter with `description` and `inclusion: manual`
2. WHEN the master navigation steering is configured THEN it SHALL have `inclusion: always`
3. WHEN front-matter is added THEN it SHALL follow the established YAML format with proper delimiters
4. WHEN descriptions are provided THEN they SHALL be concise and descriptive of the steering file's purpose

### Requirement 6

**User Story:** As an LLM agent, I want cross-references within steering files to point to other steering files, so that the navigation system is self-contained within the steering directory.

#### Acceptance Criteria

1. WHEN cross-references are migrated THEN they SHALL be updated from `[docs/filename.md](filename.md)` format to steering context format
2. WHEN steering files reference each other THEN they SHALL use descriptive text that explains what the referenced steering contains
3. WHEN cross-reference sections exist THEN they SHALL be updated to reflect the steering-based system
4. WHEN references are made THEN they SHALL maintain the logical grouping by category (Core, Infrastructure, Development, etc.)

### Requirement 7

**User Story:** As a developer or LLM agent, I want a meta-steering file that explains how to create good steering files, so that I can understand the principles and standards for effective steering content.

#### Acceptance Criteria

1. WHEN a meta-steering file is created THEN it SHALL define the characteristics of good steering files
2. WHEN steering principles are documented THEN they SHALL specify that steerings should be short but encode a lot of information
3. WHEN steering content guidelines are provided THEN they SHALL specify that steerings describe concepts and modules with associated rules
4. WHEN steering structure is defined THEN it SHALL require descriptions of connections to other modules
5. WHEN code inclusion guidelines are provided THEN they SHALL specify minimal code with references for expansion
6. WHEN steering files are created THEN they SHALL include instructions on how LLMs can expand context or get examples
7. WHEN the meta-steering is configured THEN it SHALL have `inclusion: manual` and be referenceable as `#steering-guidelines`
