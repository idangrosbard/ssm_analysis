# Design Document

## Overview

The docs-to-steering migration system will transform the existing documentation structure into a context-aware steering system. The design centers around a master gateway steering file that guides LLM agents to read appropriate additional steering files based on their task context. This creates a more intelligent and targeted approach to providing context compared to traditional documentation.

## Architecture

### Master Gateway Pattern

The `docs-navigation.md` steering file serves as the central entry point with `inclusion: always`, ensuring every LLM interaction begins with navigation guidance. This file will:

- Provide task-based navigation categories (New Development, Experiment Development, Model Analysis, UI Development)
- Reference other steering files using the `#steering-name` format for manual inclusion
- Replace all `docs/` references with steering context keys
- Maintain the quick navigation structure but adapted for steering consumption

### Steering Directory Organization

To improve manageability, steering files will be organized into logical subdirectories with an additional top-level grouping:

```
.kiro/steering/
├── docs-navigation.md       # Master gateway (root level)
├── docs/                    # Documentation-related steerings
│   ├── core/                # Core infrastructure and coordination
│   │   ├── core-modules.md
│   │   ├── infrastructure.md
│   │   └── utilities-and-patterns.md
│   ├── experiments/         # Experiment-related steerings
│   │   ├── experiment-runners.md
│   │   ├── knockout-mechanisms.md
│   │   ├── experiment-testing.md
│   │   └── prompt-filteration.md
│   ├── data/                # Data processing and interfaces
│   │   ├── data-relationships-interface.md
│   │   └── dataset-processing-flow.md
│   ├── analysis/            # Analysis and visualization
│   │   └── analysis-and-plotting.md
│   ├── ui/                  # User interface components
│   │   ├── streamlit-infrastructure.md
│   │   ├── streamlit-app-pages.md
│   │   └── streamlit-app-components.md
│   ├── setup/               # Environment and setup
│   │   └── setup-and-environment.md
│   └── meta/                # Meta-documentation about steerings
│       └── steering-guidelines.md
├── product.md              # Existing product steering
├── structure.md            # Existing structure steering
└── tech.md                 # Existing tech steering
```

### Content Migration Strategy

Each existing doc file will have its complete content migrated to its corresponding steering file while preserving:

- All technical details and implementation patterns
- Code examples and file references
- Cross-reference relationships (updated to steering format)
- Critical warnings and verification checklists
- Step-by-step procedures and examples

### Front-Matter Configuration System

All steering files will use consistent YAML front-matter:

```yaml
---
description: [Concise description of steering purpose]
inclusion: manual
---
```

The master navigation will use `inclusion: always` to ensure it's always loaded.

## Components and Interfaces

### File Mapping Structure

| Doc File | New Steering Location | Description |
|----------|----------------------|-------------|
| `docs/README.md` | `.kiro/steering/docs-navigation.md` | Master gateway (root level) |
| `docs/core-modules.md` | `.kiro/steering/docs/core/core-modules.md` | Core coordination patterns |
| `docs/infrastructure.md` | `.kiro/steering/docs/core/infrastructure.md` | Base classes and patterns |
| `docs/utilities-and-patterns.md` | `.kiro/steering/docs/core/utilities-and-patterns.md` | Code organization |
| `docs/experiment-runners.md` | `.kiro/steering/docs/experiments/experiment-runners.md` | Runner implementation patterns |
| `docs/knockout-mechanisms.md` | `.kiro/steering/docs/experiments/knockout-mechanisms.md` | Model intervention |
| `docs/experiment-testing.md` | `.kiro/steering/docs/experiments/experiment-testing.md` | Testing patterns |
| `docs/prompt-filteration.md` | `.kiro/steering/docs/experiments/prompt-filteration.md` | Prompt filtering |
| `docs/data-relationships-interface.md` | `.kiro/steering/docs/data/data-relationships-interface.md` | Data interfaces |
| `docs/dataset-processing-flow.md` | `.kiro/steering/docs/data/dataset-processing-flow.md` | Data processing |
| `docs/analysis-and-plotting.md` | `.kiro/steering/docs/analysis/analysis-and-plotting.md` | Visualization components |
| `docs/streamlit-infrastructure.md` | `.kiro/steering/docs/ui/streamlit-infrastructure.md` | UI architecture |
| `docs/streamlit-app-pages.md` | `.kiro/steering/docs/ui/streamlit-app-pages.md` | Page navigation |
| `docs/streamlit-app-components.md` | `.kiro/steering/docs/ui/streamlit-app-components.md` | UI components |
| `docs/setup-and-environment.md` | `.kiro/steering/docs/setup/setup-and-environment.md` | Package management |
| New file | `.kiro/steering/docs/meta/steering-guidelines.md` | Meta-steering for best practices |

### Meta-Steering Creation

A new `steering-guidelines.md` file will be created to document steering best practices:

- Short but information-dense content
- Concept and module descriptions with rules
- Connection descriptions to other modules
- Minimal code with expansion instructions
- Guidelines for LLM context expansion

## Content Transformation Patterns

### Cross-Reference Updates

When migrating content, cross-references need to be updated from documentation format to steering format:

- **Old format**: `[docs/filename.md](filename.md)` or `See [docs/filename.md](filename.md) for details`
- **New format**: `#steering-name` or contextual references like "Use #core-modules for coordination patterns"

### Front-Matter Standardization

All steering files will use consistent YAML front-matter structure:

```yaml
---
description: [Concise description of steering purpose]
inclusion: manual
---
```

### Navigation Reference Updates

The master navigation will need updates to reference steering files using the manual inclusion format rather than file paths.

## Quality Assurance

### Content Verification

- **Completeness Check**: Ensure all content from docs is present in steering files
- **Cross-Reference Accuracy**: Verify all steering references are valid and resolvable
- **Front-Matter Consistency**: Check YAML front-matter is properly formatted across all files
- **Navigation Functionality**: Confirm task-based navigation guides to correct steering files

### Migration Safety

- **Preserve Originals**: Keep docs directory intact during migration for reference
- **Incremental Approach**: Migrate and verify one steering file at a time
- **Review Process**: Have content reviewed before considering migration complete

## Migration Approach

### Phase 1: Directory Structure Setup
- Create subdirectories in `.kiro/steering/` (core/, experiments/, data/, analysis/, ui/, setup/, meta/)
- Move existing steering files to appropriate subdirectories
- Update any existing references to moved steering files

### Phase 2: Content Migration
- Copy content from each doc file to its corresponding steering file location
- Add proper front-matter to all steering files
- Update cross-references from docs format to steering format

### Phase 3: Navigation Update
- Update master `docs-navigation.md` to reference steering files in subdirectories
- Replace all `docs/` references with `#steering-name` format
- Update task-based navigation to use steering context keys

### Phase 4: Meta-Steering Creation
- Create `steering-guidelines.md` with best practices for creating effective steering files
- Document steering principles (short but information-dense, rules-focused, etc.)
- Provide guidance for future steering development

### Phase 5: Review and Refinement
- Review all migrated content for completeness and accuracy
- Verify cross-reference integrity across all steering files
- Test navigation functionality with sample task scena
