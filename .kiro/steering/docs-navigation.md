---
description: Steering Navigation Guide
inclusion: always
---

# Steering Navigation Guide

## How to Use Steering Files for Task Context

When working on tasks in this SSM Analysis project, always start by reading this navigation guide to understand which additional steering files to include based on your specific task type.

### Primary Navigation Flow

This steering navigation guide provides structured access to all steering modules organized by task type. Use the task-based navigation sections below to identify which steering files to include using the `#steering-name` format for manual inclusion.

### Quick Task-Based Navigation

#### For New Development

1. **#core-modules** - Critical 3-file coordination patterns
2. **#infrastructure** - Base class understanding and patterns
3. **#setup-and-environment** - Environment setup and package management

#### For Experiment Development

1. **#experiment-runners** - Runner patterns and dependencies
2. **#data-relationships-interface** - Data object interactions
3. **#dataset-processing-flow** - Data processing patterns
4. **#analysis-and-plotting** - Visualization coordination

#### For Model Analysis

1. **#infrastructure** - ModelInterface patterns and base classes
2. **#knockout-mechanisms** - Model intervention patterns
3. **#experiment-runners** - Analysis workflows and dependencies

#### For UI Development

1. **#streamlit-infrastructure** - UI component architecture
2. **#streamlit-app-pages** - Navigation patterns and page structure
3. **#streamlit-app-components** - Component implementations

### Critical Steering Rules

- **ALWAYS start with this navigation guide** for task context
- **Use task-based navigation** to identify relevant steering files
- **Include steering files using #steering-name format** for manual inclusion
- **Follow cross-references** between steering files for complete guidance
- **Consult multiple steerings** for complex tasks spanning multiple areas

### Available Steering Modules

#### Core Infrastructure

- **#core-modules** - Critical 3-file coordination pattern
- **#infrastructure** - Base classes and dependency management
- **#utilities-and-patterns** - Code organization standards

#### Experiment Framework

- **#experiment-runners** - All 4 runner types and patterns
- **#knockout-mechanisms** - Model intervention implementations
- **#experiment-testing** - Testing patterns and procedures
- **#prompt-filteration** - Prompt filtering logic and patterns

#### Data Management

- **#data-relationships-interface** - Data object interfaces and relationships
- **#dataset-processing-flow** - Raw data processing flow

#### Analysis and Visualization

- **#analysis-and-plotting** - Visualization components and configuration

#### User Interface

- **#streamlit-infrastructure** - Component-based UI architecture
- **#streamlit-app-pages** - Page architecture and navigation
- **#streamlit-app-components** - UI component implementations

#### Environment and Setup

- **#setup-and-environment** - Package management and reproducibility

#### Meta-Documentation

- **#steering-guidelines** - Steering creation and best practices
- **#steering-refinement** - Steering maintenance and refinement procedures
