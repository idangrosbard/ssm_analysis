# Infrastructure Patterns and Base Classes

This document covers the foundational infrastructure components that form the backbone of the SSM Analysis Project. Understanding these patterns is critical for all experiment development and model analysis work.

## Table of Contents

- [Base Infrastructure Components](#base-infrastructure-components)
  - [BaseRunner Implementation](#baserunner-implementation)
  - [BasePromptFilteration System](#basepromptfilteration-system)
- [Model Interface Patterns](#model-interface-patterns)
  - [ModelInterface Abstract Class](#modelinterface-abstract-class)
  - [Architecture-Specific Implementations](#architecture-specific-implementations)
- [Infrastructure Coordination Rules](#infrastructure-coordination-rules)
- [Dependency Management System](#dependency-management-system)
- [Implementation Examples](#implementation-examples)

## Base Infrastructure Components

### BaseRunner Implementation

The `BaseRunner[_TVariantParams]` class is the foundation for all experiment implementations in the project.

#### Core Requirements

**ALWAYS inherit from `BaseRunner[_TVariantParams]` for all experiment implementations**

```python
from src.experiments.infrastructure.base_runner import BaseRunner, BaseVariantParams, InputParams, MetadataParams

@dataclass(frozen=True)
class MyVariantParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = ExperimentName.MY_EXPERIMENT
    # Add your specific variant parameters here

class MyExperimentRunner(BaseRunner[MyVariantParams]):
    # Implementation here
```

#### Abstract Methods Implementation

**MUST implement all abstract methods**: `_compute_impl()`, `is_computed()`, `get_outputs()`, `get_runner_dependencies()`

```python
class MyExperimentRunner(BaseRunner[MyVariantParams]):
    @abstractmethod
    def _compute_impl(self) -> None:
        """Main computation logic for the experiment."""
        # MUST call create_experiment_dir() before computation
        self.create_experiment_dir()
        # Your computation logic here
        
    @abstractmethod
    def is_computed(self) -> bool:
        """Check if experiment results already exist."""
        # Check if output files exist
        
    @abstractmethod
    def get_outputs(self) -> Any:
        """Return experiment outputs."""
        # Load and return results
        
    @abstractmethod
    def get_runner_dependencies(self) -> TDependencies:
        """Declare dependencies on other runners."""
        return {
            "evaluate_model": EvaluateModelRunner(...)
        }
```

#### Parameter Structure

**ALWAYS use `BaseVariantParams`, `InputParams`, and `MetadataParams` structure**

```python
@dataclass(frozen=True)
class MyVariantParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = ExperimentName.MY_EXPERIMENT
    model_arch: MODEL_ARCH
    model_size: TModelSize
    # Additional variant-specific parameters

@dataclass(frozen=True)
class MyInputParams(InputParams):
    filteration: BasePromptFilteration
    dataset_name: DatasetName = DatasetName.counter_fact
    # Additional input parameters

@dataclass(frozen=True)
class MyMetadataParams(MetadataParams):
    code_version: TCodeVersionName
    requested_batch_size: TBatchSize = TBatchSize(1)
    with_slurm: bool = False
    # Additional metadata parameters
```

#### Directory and Path Management

**MUST call `create_experiment_dir()` before computation in `_compute_impl()`**

```python
def _compute_impl(self) -> None:
    # Create experiment directory structure
    self.create_experiment_dir()
    
    # Access experiment paths
    output_dir = self.variation_paths.output_dir
    plots_dir = self.variation_paths.plots_path
    
    # Your computation logic here
```

### BasePromptFilteration System

The `BasePromptFilteration` class provides a sophisticated filtering system for experiment prompts.

#### Core Requirements

**ALWAYS inherit from `BasePromptFilteration` for filtering logic**

```python
from src.experiments.infrastructure.base_prompt_filteration import BasePromptFilteration

@dataclass(frozen=True)
class MyPromptFilteration(BasePromptFilteration):
    # Your filter parameters
    
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        """Return list of prompt IDs to include in experiment."""
        pass
        
    def get_dependencies(self) -> TDependencies:
        """Return dependencies on other runners."""
        return {}
        
    def display_name(self) -> str:
        """Return human-readable name for this filter."""
        return "My Filter"
```

#### Logical Operations

**USE logical operations**: `&` (AND), `|` (OR), `~` (NOT) for combining filters

```python
# AND operation - intersection of two filters
combined_filter = filter1 & filter2

# OR operation - union of two filters  
combined_filter = filter1 | filter2

# NOT operation - complement of a filter
inverted_filter = ~filter1

# Complex logical combinations
complex_filter = (filter1 & filter2) | (~filter3)
```

#### Filter Types

**USE `SelectivePromptFilteration` for specific prompt selections**

```python
from src.experiments.infrastructure.base_prompt_filteration import SelectivePromptFilteration

# Select specific prompt IDs
specific_filter = SelectivePromptFilteration(prompt_ids=(1, 5, 10, 15))
```

**USE `SamplePromptFilteration` for sampling with seeds**

```python
from src.experiments.infrastructure.base_prompt_filteration import SamplePromptFilteration

# Sample 100 prompts with seed 42
sampled_filter = SamplePromptFilteration(
    base_prompt_filteration=base_filter,
    sample_size=100,
    seed=42
)
```

**USE `LogicalPromptFilteration` for complex logical combinations**

```python
from src.experiments.infrastructure.base_prompt_filteration import LogicalPromptFilteration

# Create complex logical combinations
complex_filter = LogicalPromptFilteration.create_and([filter1, filter2, filter3])
```

#### Contextualization

**ALWAYS contextualize filters** using `contextualize(context)` before usage

```python
# Contextualize filter with experiment context
contextualized_filter = my_filter.contextualize(experiment_context)

# Check if filter has context
if contextualized_filter.has_context:
    # Use contextualized behavior
    pass
```

## Model Interface Patterns

### ModelInterface Abstract Class

The `ModelInterface` provides a unified interface for different model architectures.

#### Core Interface

```python
from src.experiments.infrastructure.model_interface import ModelInterface

class ModelInterface(ABC):
    def __init__(self, model_arch: MODEL_ARCH, model_size: TModelSize, 
                 device: Optional[TDevice] = None, tokenizer: Optional[TTokenizer] = None):
        """Initialize model with given architecture and size."""
        
    @abstractmethod
    def generate_logits(self, input_ids: torch.Tensor, 
                       num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
                       feature_category: FeatureCategory = FeatureCategory.ALL) -> torch.Tensor:
        """Generate logits with optional attention masking."""
        
    @abstractmethod
    def n_layers(self) -> int:
        """Return number of layers in the model."""
```

#### Usage Patterns

```python
# Get model interface for specific architecture
model_interface = get_model_interface(MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, MODEL_SIZE.SMALL))

# Setup model for experimentation
model_interface.setup(layers=[0, 1, 2, 3])  # Optional layer specification

# Generate logits with knockout
logits = model_interface.generate_logits(
    input_ids=input_tokens,
    num_to_masks={0: [(1, 2), (3, 4)]},  # Layer 0: block info from token 2->1, 4->3
    feature_category=FeatureCategory.SLOW_DECAY
)
```

### Architecture-Specific Implementations

#### Mamba Models

```python
class Mamba1Interface(ModelInterface):
    """Interface for Mamba-1 models with SSM interference hooks."""
    
    def setup(self, layers: Optional[Iterable[TLayerIndex]] = None):
        """Setup SSM interference hooks for specified layers."""
        
    def get_layer_moi(self, layer_i: int) -> torch.nn.Module:
        """Get mixer of interest for layer i."""
        
class Mamba2Interface(ModelInterface):
    """Interface for Mamba-2 models with enhanced SSM interference."""
```

#### Transformer Models

```python
class LlamaInterface(ModelInterface):
    """Interface for Llama models with attention knockout."""
    
    def setup(self, layers: Optional[Iterable[TLayerIndex]] = None):
        """Setup attention knockout for specified layers."""
        
class GPT2Interface(ModelInterface):
    """Interface for GPT-2 models with attention masking."""
```

## Infrastructure Coordination Rules

### Experiment Creation Rules

**When creating new experiment types**: Inherit from `BaseRunner` AND create corresponding prompt filteration if needed

```python
# 1. Create variant parameters
@dataclass(frozen=True)
class MyExperimentVariantParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = ExperimentName.MY_EXPERIMENT
    # Add experiment-specific parameters

# 2. Create runner implementation
class MyExperimentRunner(BaseRunner[MyExperimentVariantParams]):
    # Implement all abstract methods
    
# 3. Create corresponding prompt filteration if needed
@dataclass(frozen=True)
class MyExperimentFilteration(BasePromptFilteration):
    # Implement filtering logic for this experiment type
```

### Infrastructure Modification Rules

**When modifying base infrastructure**: Update ALL existing runners to maintain compatibility

```python
# When modifying BaseRunner:
# 1. Check all existing runners in src/experiments/runners/
# 2. Update any runners that depend on modified functionality
# 3. Test compatibility with existing experiments
```

### Dependency Management Rules

**NEVER create direct dependencies between runners** - use the dependency system

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

**ALWAYS check `dependencies_are_computed()` before execution**

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

## Dependency Management System

### Dependency Declaration

```python
def get_runner_dependencies(self) -> TDependencies:
    """Declare dependencies on other runners."""
    return {
        "evaluate_model": EvaluateModelRunner(
            variant_params=EvaluateModelVariantParams(...),
            input_params=self.input_params,
            metadata_params=self.metadata_params
        ),
        "info_flow": InfoFlowRunner(...)
    }
```

### Dependency Checking

```python
# Check if all dependencies are computed
if self.dependencies_are_computed():
    # Safe to proceed with computation
    self._compute_impl()
else:
    # Compute dependencies first
    self.compute_dependencies()
```

### Dependency Computation

```python
# Compute all dependencies recursively
self.compute_dependencies()

# Check for uncomputed dependencies
uncomputed = self.uncomputed_dependencies()
if uncomputed:
    print(f"Uncomputed dependencies: {uncomputed}")
```

## Implementation Examples

### Complete Runner Example

```python
from src.experiments.infrastructure.base_runner import BaseRunner, BaseVariantParams, InputParams, MetadataParams
from src.experiments.infrastructure.base_prompt_filteration import BasePromptFilteration
from src.core.names import ExperimentName
from src.core.types import MODEL_ARCH, MODEL_SIZE

@dataclass(frozen=True)
class MyExperimentVariantParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = ExperimentName.MY_EXPERIMENT
    model_arch: MODEL_ARCH
    model_size: MODEL_SIZE
    # Additional parameters

@dataclass(frozen=True)
class MyExperimentFilteration(BasePromptFilteration):
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        return [1, 2, 3, 4, 5]  # Example prompt IDs
        
    def get_dependencies(self) -> TDependencies:
        return {}
        
    def display_name(self) -> str:
        return "My Experiment Filter"

class MyExperimentRunner(BaseRunner[MyExperimentVariantParams]):
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
            "evaluate_model": EvaluateModelRunner(
                variant_params=EvaluateModelVariantParams(
                    model_arch=self.variant_params.model_arch,
                    model_size=self.variant_params.model_size
                ),
                input_params=self.input_params,
                metadata_params=self.metadata_params
            )
        }
```

### Filter Combination Example

```python
# Create base filters
specific_filter = SelectivePromptFilteration(prompt_ids=(1, 5, 10))
sampled_filter = SamplePromptFilteration(
    base_prompt_filteration=specific_filter,
    sample_size=2,
    seed=42
)

# Combine filters logically
combined_filter = specific_filter & sampled_filter
inverted_filter = ~combined_filter

# Use in experiment
runner = MyExperimentRunner(
    variant_params=variant_params,
    input_params=InputParams(filteration=combined_filter),
    metadata_params=metadata_params
)
```

## Cross-References

- **[Experiment Runners](experiment-runners.md)** - Detailed documentation of all runner types
- **[Data Interfaces](data-interfaces.md)** - Object-oriented data interaction patterns
- **Knockout Mechanisms** - Model intervention implementation details (coming soon)
- **[Core Modules Coordination](core-modules.md)** - Critical 3-file coordination patterns

---

*This infrastructure documentation provides the foundation for all experiment development. Understanding these patterns is essential for maintaining consistency and reliability across the project.* 
