---
description: Documentation Navigation Guide
inclusion: always
---

# Documentation Navigation Guide

## How to Use README.md and /docs for Task Context

When working on tasks in this SSM Analysis project, always start by consulting the documentation to understand the relevant context and patterns.

### Primary Navigation Flow

**Use [docs/README.md](docs/README.md)** - LLM Navigation Guide that provides structured access to all documentation modules
**Navigate to specific docs based on task type** using the Quick Navigation by Use Case section

This documentation is designed for LLM agents to quickly understand project structure and navigate to relevant documentation modules.

### Quick Task-Based Navigation

#### For New Development
1. **docs/core-modules.md** - Critical coordination patterns
2. **docs/infrastructure.md** - Base class understanding
3. **docs/setup-and-environment.md** - Environment setup

#### For Experiment Development
1. **docs/experiment-runners.md** - Runner patterns and dependencies
2. **docs/data-relationships-interface.md** - Data object interactions
3. **docs/dataset-processing-flow.md** - Data processing patterns
4. **docs/analysis-and-plotting.md** - Visualization coordination

#### For Model Analysis
1. **docs/infrastructure.md** - ModelInterface patterns
2. **docs/knockout-mechanisms.md** - Model intervention patterns
3. **docs/experiment-runners.md** - Analysis workflows

#### For UI Development
1. **docs/streamlit-infrastructure.md** - UI component architecture
2. **docs/streamlit-app-pages.md** - Navigation patterns
3. **docs/streamlit-app-components.md** - Component implementations

### Critical Documentation Rules

- **ALWAYS start with docs/README.md** for project context
- **Use task-based navigation** to find relevant documentation quickly
- **Follow cross-references** between documentation modules
- **Check file references** for specific implementation details
- **Consult multiple docs** for complex tasks spanning multiple areas

### Available Documentation Modules

- **analysis-and-plotting.md** - Visualization components and configuration
- **core-modules.md** - Critical 3-file coordination pattern
- **data-relationships-interface.md** - Data object interfaces and relationships
- **dataset-processing-flow.md** - Raw data processing flow
- **experiment-runners.md** - All 4 runner types and patterns
- **experiment-testing.md** - Testing patterns and procedures
- **infrastructure.md** - Base classes and dependency management
- **knockout-mechanisms.md** - Model intervention implementations
- **prompt-filteration.md** - Prompt filtering logic and patterns
- **setup-and-environment.md** - Package management and reproducibility
- **streamlit-app-components.md** - UI component implementations
- **streamlit-app-pages.md** - Page architecture and navigation
- **streamlit-infrastructure.md** - Component-based UI architecture
- **utilities-and-patterns.md** - Code organization standards
