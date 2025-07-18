# Experiment Runners Architecture

This document covers the four core experiment runners that form the backbone of the SSM Analysis Project's experimental framework. Each runner serves a specific purpose in the research pipeline and follows consistent implementation patterns.

## Table of Contents

- [Runner Types and Purposes](#runner-types-and-purposes)
  - [EvaluateModelRunner - Foundation Runner](#evaluatemodelrunner---foundation-runner)
  - [InfoFlowRunner - Information Flow Analysis](#infoflowrunner---information-flow-analysis)
  - [HeatmapRunner - Layer-by-Layer Visualization](#heatmaprunner---layer-by-layer-visualization)
  - [FullPipelineRunner - Orchestrated Workflows](#fullpipelinerunner---orchestrated-workflows)
- [Runner Implementation Rules](#runner-implementation-rules)
- [Dependency Management](#dependency-management)
- [Output Formats and File Extensions](#output-formats-and-file-extensions)
- [Caching and Performance Patterns](#caching-and-performance-patterns)
- [Implementation Examples](#implementation-examples)

## Runner Types and Purposes

### EvaluateModelRunner - Foundation Runner

**File**: `src/experiments/runners/evaluate_model.py`

#### Purpose and Usage

- **Purpose**: Model performance evaluation on datasets
- **Output**: CSV files with accuracy metrics, confidence scores, target rankings
- **Usage**: Foundation for all other experiments - provides model correctness data
- **Key fields**: `MODEL_CORRECT`, `TARGET_RANK`, `TARGET_PROBS`, `MODEL_TOP_OUTPUTS`

#### Implementation Pattern

```python
@dataclass(frozen=True)
class EvaluateModelParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = field(init=False, default=ExperimentName.evaluate_model)
    drop_subject: bool = False
    drop_subj_last_token: bool = False
    with_3_dots: bool = False
    new_max_tokens: int = 5
    top_k_tokens: int = 5

class EvaluateModelRunner(BaseRunner[EvaluateModelParams]):
    """Configuration for model evaluation."""
    
    @property
    def output_result_path(self) -> Path:
        return self.variation_paths.outputs_path / "outputs.csv"
    
    def get_outputs(self) -> TPromptDataFlat:
        return _get_output_path(self)
    
    def _compute_impl(self) -> None:
        run(self)
    
    def is_computed(self) -> bool:
        return self.output_result_path.exists()
    
    def get_runner_dependencies(self):
        return self.input_params.filteration.contextualize(self).get_dependencies()
```

#### Key Features

- **Batch Processing**: Processes prompts in batches for efficiency
- **Token-Level Analysis**: Provides detailed token-level predictions and rankings
- **Model Interface Integration**: Uses the unified model interface for consistent behavior
- **Caching**: Implements LRU caching for expensive operations

### InfoFlowRunner - Information Flow Analysis

**File**: `src/experiments/runners/info_flow.py`

#### Purpose and Usage

- **Purpose**: Information flow analysis using knockout methodology
- **Output**: JSON files with hit/miss data, probability differences per layer
- **Dependencies**: Requires `EvaluateModelRunner` results
- **Key parameters**: `source`, `target`, `feature_category`, `window_size`
- **ALWAYS use specific `TokenType` and `FeatureCategory` combinations**

#### Implementation Pattern

```python
@dataclass(frozen=True)
class InfoFlowParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = field(init=False, default=ExperimentName.info_flow)
    window_size: TWindowSize
    source: TokenType
    feature_category: FeatureCategory
    target: TokenType
    subset_layers: Optional[TWindowLayerStartIndex] = None

class InfoFlowRunner(BaseRunner[InfoFlowParams]):
    """Configuration for information flow analysis."""
    
    @property
    def output_file(self) -> JSONInfoFlowFile:
        return JSONInfoFlowFile(self.variation_paths.outputs_path / "info_flow.json")
    
    def get_outputs(self) -> TInfoFlowOutput:
        return self.output_file.load_to_info_flow_output()
    
    def _compute_impl(self) -> None:
        run(self)
    
    def is_computed(self) -> bool:
        return self.output_file.path.exists()
    
    def get_runner_dependencies(self) -> InfoFlowDependencies:
        return InfoFlowDependencies(
            evaluate_model=EvaluateModelRunner.init_from_runner(
                self,
                variant_params=EvaluateModelParams(
                    model_arch=self.variant_params.model_arch,
                    model_size=self.variant_params.model_size,
                ),
            ),
        )
```

#### Key Features

- **Knockout Methodology**: Implements attention knockout for information flow analysis
- **Layer-by-Layer Analysis**: Tracks information flow across all model layers
- **Statistical Analysis**: Provides confidence intervals and statistical measures
- **Incremental Computation**: Supports partial computation and recovery

### HeatmapRunner - Layer-by-Layer Visualization

**File**: `src/experiments/runners/heatmap.py`

#### Purpose and Usage

- **Purpose**: Layer-by-layer probability visualization
- **Output**: HDF5 files with probability matrices per prompt
- **Dependencies**: Requires `EvaluateModelRunner` results
- **Usage**: Single prompt analysis across model layers

#### Implementation Pattern

```python
@dataclass(frozen=True)
class HeatmapParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = field(init=False, default=ExperimentName.heatmap)
    window_size: TWindowSize

class HeatmapRunner(BaseRunner[HeatmapParams]):
    """Configuration for heatmap generation."""
    
    @property
    def output_hdf5_path(self) -> HDF5HeatmapFile:
        return HDF5HeatmapFile(self.variation_paths.outputs_path / "heatmaps.h5")
    
    def get_outputs(self) -> HeatmapExperimentOutput:
        if not self.output_hdf5_path.path.exists():
            return {}
        return self.output_hdf5_path.get_prompt_idx_heatmaps(
            self.input_params.filteration.contextualize(self).get_prompt_ids()
        )
    
    def _compute_impl(self) -> None:
        run(self)
    
    def is_computed(self) -> bool:
        if not self.output_hdf5_path.path.exists():
            return False
        existing_prompts = self.output_hdf5_path.get_existing_prompt_idx()
        return all(
            idx in existing_prompts 
            for idx in self.input_params.filteration.contextualize(self).get_prompt_ids()
        )
    
    def get_runner_dependencies(self) -> HeatmapDependencies:
        return HeatmapDependencies(
            evaluate_model=EvaluateModelRunner.init_from_runner(
                self,
                variant_params=EvaluateModelParams(
                    model_arch=self.variant_params.model_arch,
                    model_size=self.variant_params.model_size,
                ),
            ),
        )
```

#### Key Features

- **HDF5 Storage**: Uses HDF5 format for efficient storage of large probability matrices
- **Incremental Computation**: Supports partial computation and recovery
- **Visualization Integration**: Integrates with plotting infrastructure for publication-quality figures
- **Window-Based Analysis**: Supports sliding window analysis across layers

### FullPipelineRunner - Orchestrated Workflows

**File**: `src/experiments/runners/full_pipeline.py`

#### Purpose and Usage

- **Purpose**: Orchestrates complete experimental workflows
- **Dependencies**: Coordinates heatmap and info_flow runners
- **Usage**: End-to-end experiment execution with plotting

#### Implementation Pattern

```python
@dataclass(frozen=True)
class FullPipelineParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = field(init=False, default=ExperimentName.full_pipeline)
    knockout_map: dict[TokenType, list[tuple[TokenType, FeatureCategory]]]
    info_flow_window_size: TWindowSize
    heatmap_window_size: TWindowSize
    heatmap_prompts: BasePromptFilteration
    with_plotting: bool = False
    enforce_no_missing_outputs: bool = True
    with_generation: bool = True

class FullPipelineRunner(BaseRunner):
    """Configuration for the full experiment pipeline."""
    
    def get_outputs(self) -> dict:
        return {}
    
    def _compute_impl(self) -> None:
        main_local(self)
    
    def get_runner_dependencies(self) -> FullPipelineDependencies:
        info_flow_deps: dict[TokenType, dict[tuple[TokenType, FeatureCategory], InfoFlowRunner]] = {}
        for target_token, source in self.variant_params.knockout_map.items():
            info_flow_deps[target_token] = {}
            for source_token, feature_category in source:
                config = InfoFlowRunner.init_from_runner(
                    runner=self,
                    variant_params=InfoFlowParams(
                        model_arch=self.variant_params.model_arch,
                        model_size=self.variant_params.model_size,
                        window_size=self.variant_params.info_flow_window_size,
                        source=source_token,
                        feature_category=feature_category,
                        target=target_token,
                    ),
                )
                if not config.variant_params.should_skip_task():
                    info_flow_deps[target_token][(source_token, feature_category)] = config

        return FullPipelineDependencies(
            heatmap=HeatmapRunner.init_from_runner(
                runner=self,
                variant_params=HeatmapParams(
                    model_arch=self.variant_params.model_arch,
                    model_size=self.variant_params.model_size,
                    window_size=self.variant_params.heatmap_window_size,
                ),
                input_params=InputParams(
                    filteration=self.variant_params.heatmap_prompts,
                ),
            ),
            info_flow=info_flow_deps,
        )
    
    def is_computed(self) -> bool:
        return not self.variant_params.with_plotting
```

#### Key Features

- **Workflow Orchestration**: Coordinates multiple experiments in sequence
- **Plotting Integration**: Generates publication-quality figures automatically
- **Configuration Management**: Maintains consistent configuration across all steps
- **Error Handling**: Provides robust error handling for complex workflows

## Runner Implementation Rules

### Core Requirements

**NEVER hardcode experiment dependencies** - declare them in `get_runner_dependencies()`

```python
# CORRECT: Use dependency system
def get_runner_dependencies(self) -> TDependencies:
    return {
        "evaluate_model": EvaluateModelRunner(...)
    }

# INCORRECT: Direct dependency
def _compute_impl(self) -> None:
    evaluate_runner = EvaluateModelRunner(...)  # Don't do this
    evaluate_runner.run(with_dependencies=True)
```

**ALWAYS use caching decorators** (`@lru_cache`, `@cached`) for expensive operations

```python
from functools import lru_cache
from cachetools import cached, LRUCache

@lru_cache(maxsize=100)
def expensive_operation(self, param):
    # Expensive computation here
    pass

@cached(LRUCache(maxsize=30))
def cached_operation(self, param):
    # Another expensive operation
    pass
```

**MUST handle partial computation states** in `is_computed()` method

```python
def is_computed(self) -> bool:
    # Check if output file exists
    if not self.output_path.exists():
        return False
    
    # For incremental computation, check if all required data is present
    if hasattr(self, 'get_remaining_prompt_original_indices'):
        remaining = self.get_remaining_prompt_original_indices()
        return len(remaining) == 0
    
    return True
```

**ALWAYS provide atomic file operations** for concurrent execution safety

```python
from src.utils.file_system import atomic_write

def save_results(self, data):
    # Use atomic write for thread safety
    atomic_write(self.output_path, json.dumps(data))
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

## Dependency Management

### Dependency Declaration

Each runner declares its dependencies in the `get_runner_dependencies()` method:

```python
def get_runner_dependencies(self) -> TDependencies:
    return {
        "evaluate_model": EvaluateModelRunner.init_from_runner(
            self,
            variant_params=EvaluateModelParams(
                model_arch=self.variant_params.model_arch,
                model_size=self.variant_params.model_size,
            ),
        ),
        # Additional dependencies...
    }
```

### Dependency Checking

Runners check dependencies before computation:

```python
def run(self, with_dependencies: bool) -> None:
    if with_dependencies:
        # Check if dependencies are computed
        if not self.dependencies_are_computed():
            # Compute dependencies first
            self.compute_dependencies()
    
    # Run the experiment
    if not self.is_computed():
        self._compute_impl()
```

### Complex Dependencies

For complex workflows like FullPipelineRunner:

```python
def get_runner_dependencies(self) -> FullPipelineDependencies:
    info_flow_deps: dict[TokenType, dict[tuple[TokenType, FeatureCategory], InfoFlowRunner]] = {}
    
    # Create multiple InfoFlowRunner instances for different configurations
    for target_token, source in self.variant_params.knockout_map.items():
        info_flow_deps[target_token] = {}
        for source_token, feature_category in source:
            config = InfoFlowRunner.init_from_runner(
                runner=self,
                variant_params=InfoFlowParams(...)
            )
            if not config.variant_params.should_skip_task():
                info_flow_deps[target_token][(source_token, feature_category)] = config
    
    return FullPipelineDependencies(
        heatmap=HeatmapRunner.init_from_runner(...),
        info_flow=info_flow_deps,
    )
```

## Output Formats and File Extensions

### CSV Output (EvaluateModelRunner)

```python
# Output structure
{
    "MODEL_CORRECT": bool,
    "TARGET_RANK": int,
    "TARGET_PROBS": float,
    "MODEL_TOP_OUTPUTS": list[str],
    "MODEL_GENERATION": str,
    "TARGET_TOKENS": list[str]
}
```

### JSON Output (InfoFlowRunner)

```python
# Output structure
{
    "metadata": {
        "layers_amount": int,
        "banned_prompts": dict[int, str]
    },
    "data": {
        "prompt_id": {
            "layer_id": {
                "hit": bool,
                "true_probs": float,
                "diffs": float
            }
        }
    }
}
```

### HDF5 Output (HeatmapRunner)

```python
# HDF5 file structure
{
    "prompt_id": numpy.ndarray  # Probability matrix for each prompt
}
```

## Caching and Performance Patterns

### LRU Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def _get_output_path(evaluate_runner: EvaluateModelRunner) -> TPromptDataFlat:
    # Expensive file loading operation
    df = pd.read_csv(evaluate_runner.output_result_path, index_col=False)
    # Process and return
    return TPromptDataFlat(df)
```

### TTL Caching

```python
from cachetools import TTLCache

TTL_info_flow_output_cache = TTLCache(maxsize=10, ttl=60)

@cached(TTL_info_flow_output_cache)
def load_to_info_flow_output(self, prompt_idx_subset=None, layer_idx_subset=None):
    # Expensive JSON loading with 60-second TTL
    pass
```

### Cache Management

```python
# Clear caches when data changes
def save(self, data: InfoFlowFileContent) -> None:
    self.statistics_path.unlink(missing_ok=True)
    atomic_write(self.path, orjson.dumps(sanitize(data), option=orjson.OPT_INDENT_2))
    STATISTICS_CACHE.clear()  # Clear related caches
    OUTPUTS_CACHE.clear()
```

## Implementation Examples

### Complete Runner Example

```python
from src.experiments.infrastructure.base_runner import BaseRunner, BaseVariantParams
from src.core.names import ExperimentName
from src.core.types import MODEL_ARCH, MODEL_SIZE

@dataclass(frozen=True)
class MyExperimentParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = ExperimentName.my_experiment
    model_arch: MODEL_ARCH
    model_size: MODEL_SIZE
    # Additional parameters

class MyExperimentRunner(BaseRunner[MyExperimentParams]):
    def _compute_impl(self) -> None:
        # Create experiment directory
        self.create_experiment_dir()
        
        # Access model interface
        model_interface = self.variant_params.get_model_interface()
        
        # Your computation logic here
        # ...
        
    def is_computed(self) -> bool:
        output_file = self.variation_paths.output_dir / "results.json"
        return output_file.exists()
        
    def get_outputs(self) -> Any:
        # Load and return results
        output_file = self.variation_paths.output_dir / "results.json"
        with open(output_file, 'r') as f:
            return json.load(f)
            
    def get_runner_dependencies(self) -> TDependencies:
        return {
            "evaluate_model": EvaluateModelRunner.init_from_runner(
                self,
                variant_params=EvaluateModelParams(
                    model_arch=self.variant_params.model_arch,
                    model_size=self.variant_params.model_size
                ),
            )
        }
```

### Incremental Computation Example

```python
def get_remaining_prompt_original_indices(self):
    """Return the list of prompt indices that need to be computed."""
    if not self.output_hdf5_path.path.exists() or self.metadata_params.overwrite_existing_outputs:
        return self.input_params.filteration.contextualize(self).get_prompt_ids()

    existing_prompts = self.output_hdf5_path.get_existing_prompt_idx()
    return [
        idx
        for idx in self.input_params.filteration.contextualize(self).get_prompt_ids()
        if idx not in existing_prompts
    ]

def is_computed(self) -> bool:
    """Check if all required prompt heatmaps exist in the HDF5 file."""
    if not self.output_hdf5_path.path.exists():
        return False

    existing_prompts = self.output_hdf5_path.get_existing_prompt_idx()
    return all(
        idx in existing_prompts 
        for idx in self.input_params.filteration.contextualize(self).get_prompt_ids()
    )
```

## Cross-References

- **[Infrastructure Patterns](infrastructure.md)** - Base classes and infrastructure coordination
- **[Data Interfaces](data-interfaces.md)** - Object-oriented data interaction patterns
- **[Analysis and Plotting](analysis-and-plotting.md)** - Visualization and plotting coordination
- **[Core Modules Coordination](core-modules.md)** - Critical 3-file coordination patterns

---

*This experiment runners documentation provides the foundation for all experimental work. Understanding these patterns is essential for developing new experiments and maintaining consistency across the research pipeline.* 
