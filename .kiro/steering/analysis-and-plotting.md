---
description: Analysis and Plotting Components
inclusion: manual
---


# Analysis and Plotting Components

## Plotting Architecture

### Plot Components

- **File**: `src/analysis/plots/`
- **Purpose**: Visualization components and configuration
- **Usage**: Standardized plotting for experiment results

### Experiment Results Analysis

- **File**: `src/analysis/experiment_results/`
- **Purpose**: Result handling and analysis utilities
- **Usage**: Process and analyze experiment outputs

## Plotting Patterns

### Configuration Management

- **USE configuration objects** for plot settings
- **MAINTAIN consistent styling** across all plots
- **FOLLOW standardized plot types** for different data

### Coordination Rules

- **COORDINATE with experiment runners** for data access
- **USE ResultBank objects** for accessing experiment data
- **FOLLOW plotting configuration patterns**

## Key Components

### Plot Types

- **Heatmaps** for layer-by-layer visualization
- **Information flow plots** for knockout analysis
- **Performance metrics** for model evaluation

### Configuration Objects

- **Plot settings** centralized in configuration
- **Consistent styling** across all visualizations
- **Reusable components** for common plot types

## Critical Rules

### Plotting Rules

- **USE configuration objects** for all plot settings
- **MAINTAIN consistency** across visualization types
- **COORDINATE with data interfaces** for data access

### File Organization

- `src/analysis/plots/` - Plotting components
- `src/analysis/experiment_results/` - Result analysis
- `src/analysis/prompt_filterations.py` - Prompt filtering logic

**Reference**: docs/analysis-and-plotting.md for complete patterns and examples
