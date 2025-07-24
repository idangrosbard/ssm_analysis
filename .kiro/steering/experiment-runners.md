---
description: Experiment Runner Types and Implementation Patterns
inclusion: manual
---


# Experiment Runner Types and Implementation Patterns

## Runner Types and Purposes

### EvaluateModelRunner (Foundation)
- **File**: `src/experiments/runners/evaluate_model.py`
- **Purpose**: Model performance evaluation on datasets
- **Output**: CSV files with accuracy metrics, confidence scores
- **Usage**: Foundation for all other experiments

### InfoFlowRunner (Information Flow Analysis)
- **File**: `src/experiments/runners/info_flow.py`
- **Purpose**: Information flow analysis using knockout methodology
- **Output**: JSON files with hit/miss data, probability differences
- **Dependencies**: Requires `EvaluateModelRunner` results

### HeatmapRunner (Layer-by-Layer Visualization)
- **File**: `src/experiments/runners/heatmap.py`
- **Purpose**: Layer-by-layer probability visualization
- **Output**: HDF5 files with probability matrices per prompt
- **Dependencies**: Requires `EvaluateModelRunner` results

### FullPipelineRunner (Orchestrated Workflows)
- **File**: `src/experiments/runners/full_pipeline.py`
- **Purpose**: Orchestrates complete experimental workflows
- **Dependencies**: Coordinates heatmap and info_flow runners

## Implementation Rules

### Core Requirements
- **NEVER hardcode experiment dependencies** - declare in `get_runner_dependencies()`
- **ALWAYS use caching decorators** (`@lru_cache`, `@cached`) for expensive operations
- **MUST handle partial computation states** in `is_computed()` method

### Output File Extensions
- **EvaluateModelRunner**: `.csv` files
- **InfoFlowRunner**: `.json` files
- **HeatmapRunner**: `.h5` files

### Dependency Pattern
```python
def get_runner_dependencies(self) -> TDependencies:
    return {"evaluate_model": EvaluateModelRunner(...)}
```

**Reference**: docs/experiment-runners.md for complete patterns and examples
