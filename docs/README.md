# SSM Analysis Project - LLM Navigation Guide

## Project Context

**Purpose**: Research project for analyzing Mamba State-Space Models using knockout methodology
**Technology**: Python 3.12, PyTorch, Mamba-SSM, UV package manager
**Core Functionality**: Factual information flow analysis in language models
**Repository Goal**: Fully reproducible research environment

## Architecture Overview

- **Infrastructure Layer**: BaseRunner, BasePromptFilteration, ModelInterface
- **Experiment Layer**: 4 specialized runners (evaluate_model, info_flow, heatmap, full_pipeline)
- **Data Layer**: Object-oriented interfaces (DataReqs, ResultBank, FulfilledReqs)
- **Core Coordination**: 3-file coordination pattern (consts.py, names.py, types.py)
- **Analysis Layer**: Plotting with configuration management
- **UI Layer**: Streamlit component-based architecture

## Documentation Structure and Editing Guidelines

### Documentation Purpose and Style

**Target Audience**: LLM agents and developers working on the SSM Analysis project
**Style**: Concise, rule-based content with clear file references instead of verbose explanations
**Length**: Each document should be under 200 lines with bullet-based organization
**Focus**: What LLMs need to know to understand and work with the codebase

### Documentation Categories

#### Core Infrastructure (Critical - Start Here)
- **[Core Modules Coordination](core-modules.md)** - Critical 3-file coordination pattern
  - **File**: `src/core/consts.py`, `src/core/names.py`, `src/core/types.py`
  - **Purpose**: Centralized constants, names, and types with strict coordination
  - **Expected Content**: Rules for 3-file coordination, step-by-step procedures, correct/incorrect patterns
  - **Editing Guidelines**: Focus on coordination rules, avoid verbose explanations, emphasize critical warnings

- **[Infrastructure Patterns](infrastructure.md)** - Base classes and dependency management
  - **File**: `src/experiments/infrastructure/base_runner.py`, `src/experiments/infrastructure/base_prompt_filteration.py`
  - **Purpose**: BaseRunner, BasePromptFilteration, ModelInterface patterns
  - **Expected Content**: Base class implementation rules, inheritance patterns, dependency management
  - **Editing Guidelines**: Focus on base class patterns, show implementation examples, emphasize critical rules

#### Experimental Framework
- **[Experiment Runners](experiment-runners.md)** - All 4 runner types and implementation patterns
  - **Files**: `src/experiments/runners/evaluate_model.py`, `src/experiments/runners/info_flow.py`, `src/experiments/runners/heatmap.py`, `src/experiments/runners/full_pipeline.py`
  - **Purpose**: Runner types, dependencies, and file locations
  - **Expected Content**: Runner types and purposes, implementation rules, output file patterns
  - **Editing Guidelines**: Focus on runner architecture, dependency patterns, file organization

- **[Data Relationships Interface](data-relationships-interface.md)** - Object-oriented interfaces for files, dataframes, and data model transitions
  - **File**: `src/data_ingestion/data_defs/data_defs.py`
  - **Purpose**: All data interface classes and their relationships
  - **Expected Content**: Data object hierarchy, interface patterns, transition methods
  - **Editing Guidelines**: Focus on object relationships, interface patterns, data flow

- **[Dataset Processing Flow](dataset-processing-flow.md)** - Raw data processing flow from loading to experiment usage
  - **Files**: `src/data_ingestion/datasets/download_dataset.py`, `src/data_ingestion/datasets/splitting.py`
  - **Purpose**: Data flow from raw datasets through processing to experiments
  - **Expected Content**: Data flow overview, key components, critical rules
  - **Editing Guidelines**: Focus on data flow patterns, processing steps, consistency rules

- **[Analysis and Plotting](analysis-and-plotting.md)** - Visualization components and configuration
  - **Files**: `src/analysis/plots/`, `src/analysis/experiment_results/`
  - **Purpose**: Plot types, configuration objects, and coordination rules
  - **Expected Content**: Plot types and configurations, coordination patterns
  - **Editing Guidelines**: Focus on plot types, configuration management, coordination rules

#### Development and Setup
- **[Setup and Environment](setup-and-environment.md)** - Package management and reproducibility
  - **Files**: `scripts/`, requirements files
  - **Purpose**: UV package manager usage and configuration patterns
  - **Expected Content**: Package management rules, environment setup, reproducibility requirements
  - **Editing Guidelines**: Focus on setup procedures, package management rules, environment standards

- **[Utilities and Patterns](utilities-and-patterns.md)** - Code organization standards
  - **File**: `src/utils/`
  - **Purpose**: Utility organization rules and common patterns
  - **Expected Content**: Directory structure, utility placement rules, common patterns
  - **Editing Guidelines**: Focus on organization rules, utility patterns, best practices

#### Model Analysis
- **[Knockout Mechanisms](knockout-mechanisms.md)** - Model intervention and hook-based implementations
  - **Files**: `src/experiments/knockout/`, `src/experiments/infrastructure/model_interface.py`
  - **Purpose**: Knockout patterns and model intervention
  - **Expected Content**: Knockout implementation methods, model interface patterns, usage examples
  - **Editing Guidelines**: Focus on implementation methods, interface patterns, usage examples

#### UI Development
- **[Streamlit Infrastructure](streamlit-infrastructure.md)** - Component-based architecture
  - **File**: `src/app/`
  - **Purpose**: UI component patterns and base classes
  - **Expected Content**: Base classes, component architecture, session state management
  - **Editing Guidelines**: Focus on component architecture, base classes, session state patterns

- **[Streamlit App Pages](streamlit-app-pages.md)** - Navigation system and page interfaces
  - **Purpose**: Page architecture and navigation patterns
  - **Expected Content**: Page structure, navigation patterns, component integration
  - **Editing Guidelines**: Focus on page architecture, navigation patterns, component integration

- **[Streamlit App Components](streamlit-app-components.md)** - Component implementations
  - **Purpose**: Reusable UI components for scientific computing interface
  - **Expected Content**: Component categories, implementation patterns, integration guidelines
  - **Editing Guidelines**: Focus on component patterns, integration guidelines, best practices

## Quick Navigation by Use Case

### For New Development
1. **[Core Modules Coordination](core-modules.md)** - Critical coordination patterns
2. **[Infrastructure Patterns](infrastructure.md)** - Base class understanding
3. **[Setup and Environment](setup-and-environment.md)** - Environment setup

### For Experiment Development
1. **[Experiment Runners](experiment-runners.md)** - Runner patterns and dependencies
2. **[Data Relationships Interface](data-relationships-interface.md)** - Data object interactions
3. **[Dataset Processing Flow](dataset-processing-flow.md)** - Data processing and flow patterns
4. **[Analysis and Plotting](analysis-and-plotting.md)** - Visualization coordination

### For Model Analysis
1. **[Infrastructure Patterns](infrastructure.md)** - ModelInterface patterns
2. **[Knockout Mechanisms](knockout-mechanisms.md)** - Model intervention patterns
3. **[Experiment Runners](experiment-runners.md)** - Analysis workflows
4. **[Dataset Processing Flow](dataset-processing-flow.md)** - Data processing for model analysis

### For UI Development
1. **[Infrastructure Patterns](infrastructure.md)** - Base patterns
2. **[Streamlit Infrastructure](streamlit-infrastructure.md)** - UI component architecture
3. **[Streamlit App Pages](streamlit-app-pages.md)** - Navigation patterns

## Documentation Editing Standards

### Content Requirements
- **Concise**: Each document should be under 200 lines
- **Rule-based**: Focus on rules and patterns rather than verbose explanations
- **File-focused**: Include specific file references and line numbers where appropriate
- **Cross-referenced**: Each document should have comprehensive cross-references to related modules

### Structure Requirements
- **Clear headings**: Use descriptive headings that indicate content purpose
- **Bullet organization**: Use bullet points for rules, patterns, and procedures
- **Code examples**: Include relevant code examples with file references
- **Critical warnings**: Highlight important rules and potential pitfalls

### Cross-Reference Requirements
- **Comprehensive**: Each document should reference all related documentation modules
- **Consistent format**: Use "See [docs/filename.md](filename.md) for description" format
- **Descriptive**: Each cross-reference should explain what the referenced document contains
- **Logical grouping**: Group cross-references by category (Core, Infrastructure, Development, etc.)

### File Reference Standards
- **Specific**: Reference specific files and line ranges when possible
- **Consistent**: Use consistent file reference patterns throughout
- **Descriptive**: Explain what each file contains and its purpose
- **Updated**: Keep file references current with actual codebase structure

## Critical Rules

- **ALWAYS use UV package manager** instead of pip
- **NEVER hardcode constants** outside `src/core/` module
- **ALWAYS follow 3-file coordination** pattern for core modules
- **NEVER create direct dependencies** between runners - use dependency system
- **ALWAYS inherit from BaseRunner** for experiment implementations
- **NEVER access experiment outputs directly** - use ResultBank objects

## File Reference Pattern

- **Code Examples**: Replace with file references like `See: src/core/consts.py:45-67`
- **Implementation Details**: Reference specific files and line ranges
- **Cross-References**: Link to related documentation modules

## Related Resources

- **[Main README](../README.md)** - Project overview and installation
- **[shrimp-rules.md](../shrimp-rules.md)** - High-level development guidelines
- **[Project Structure](../src/)** - Source code organization

---

*This documentation is designed for LLM agents to quickly understand project structure and navigate to relevant documentation modules. Each document focuses on rules, patterns, and file references rather than verbose explanations.*
