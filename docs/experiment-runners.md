# Experiment Runners Architecture

Critical runner types, dependencies, and file locations for the experimental framework.

## Runner Types and Purposes

### EvaluateModelRunner - Foundation Runner

**File**: `src/experiments/runners/evaluate_model.py`

**Purpose**: Model performance evaluation on datasets
**Output**: CSV files with accuracy metrics, confidence scores, target rankings
**Usage**: Foundation for all other experiments - provides model correctness data

### InfoFlowRunner - Information Flow Analysis

**File**: `src/experiments/runners/info_flow.py`

**Purpose**: Information flow analysis using knockout methodology
**Output**: JSON files with hit/miss data, probability differences per layer
**Dependencies**: Requires `EvaluateModelRunner` results

### HeatmapRunner - Layer-by-Layer Visualization

**File**: `src/experiments/runners/heatmap.py`

**Purpose**: Layer-by-layer probability visualization
**Output**: HDF5 files with probability matrices per prompt
**Dependencies**: Requires `EvaluateModelRunner` results

### FullPipelineRunner - Orchestrated Workflows

**File**: `src/experiments/runners/full_pipeline.py`

**Purpose**: Orchestrates complete experimental workflows
**Dependencies**: Coordinates heatmap and info_flow runners

## Runner Implementation Rules

### Core Requirements

**NEVER hardcode experiment dependencies** - declare them in `get_runner_dependencies()`

```python
# CORRECT: Use dependency system
def get_runner_dependencies(self) -> TDependencies:
    return {"evaluate_model": EvaluateModelRunner(...)}

# INCORRECT: Direct dependency
def _compute_impl(self) -> None:
    evaluate_runner = EvaluateModelRunner(...)  # Don't do this
```

**ALWAYS use caching decorators** (`@lru_cache`, `@cached`) for expensive operations

```python
from functools import lru_cache
from cachetools import cached, LRUCache

@lru_cache(maxsize=100)
def expensive_operation(self, param):
    # Expensive computation here
    pass
```

**MUST handle partial computation states** in `is_computed()` method

```python
def is_computed(self) -> bool:
    if not self.output_path.exists():
        return False
    
    # For incremental computation, check if all required data is present
    if hasattr(self, 'get_remaining_prompt_original_indices'):
        remaining = self.get_remaining_prompt_original_indices()
        return len(remaining) == 0
    
    return True
```

### Output File Extensions

**USE proper output file extensions**: `.csv` for evaluate_model, `.json` for info_flow, `.h5` for heatmap

```python
# EvaluateModelRunner
output_path = self.variation_paths.outputs_path / "outputs.csv"

# InfoFlowRunner  
output_path = self.variation_paths.outputs_path / "info_flow.json"

# HeatmapRunner
output_path = self.variation_paths.outputs_path / "heatmaps.h5"
```

## Cross-References

- **Infrastructure Patterns**: See [docs/infrastructure.md](infrastructure.md) for base classes and infrastructure coordination
- **Data Interfaces**: See [docs/data-relationships-interface.md](data-relationships-interface.md) for object-oriented data interaction patterns
- **Analysis and Plotting**: See [docs/analysis-and-plotting.md](analysis-and-plotting.md) for visualization and plotting coordination
- **Core Modules**: See [docs/core-modules.md](core-modules.md) for critical 3-file coordination patterns

## Critical Warnings

⚠️ **Breaking the runner dependency patterns will cause cascading failures throughout the project**

⚠️ **Always test changes thoroughly - runner errors propagate to all dependent experiments**

⚠️ **When in doubt, follow the established patterns exactly - don't improvise**

⚠️ **Runner changes require coordination with the entire development team** 
