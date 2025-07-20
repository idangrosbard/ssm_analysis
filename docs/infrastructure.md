# Infrastructure Patterns and Base Classes

Critical base class patterns and inheritance rules for experiment development.

## Base Infrastructure Components

### BaseRunner Implementation

**File**: `src/experiments/infrastructure/base_runner.py`

**ALWAYS inherit from `BaseRunner[_TVariantParams]` for all experiment implementations**

**MUST implement all abstract methods**: `_compute_impl()`, `is_computed()`, `get_outputs()`, `get_runner_dependencies()`

```python
from src.experiments.infrastructure.base_runner import BaseRunner, BaseVariantParams, InputParams, MetadataParams

@dataclass(frozen=True)
class MyVariantParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = ExperimentName.MY_EXPERIMENT
    model_arch: MODEL_ARCH
    model_size: TModelSize

class MyExperimentRunner(BaseRunner[MyVariantParams]):
    def _compute_impl(self) -> None:
        self.create_experiment_dir()  # MUST call before computation
        # Your computation logic here
        
    def is_computed(self) -> bool:
        # Check if output files exist
        pass
        
    def get_outputs(self) -> Any:
        # Load and return results
        pass
        
    def get_runner_dependencies(self) -> TDependencies:
        return {"evaluate_model": EvaluateModelRunner(...)}
```

### BasePromptFilteration System

**File**: `src/experiments/infrastructure/base_prompt_filteration.py`

**ALWAYS inherit from `BasePromptFilteration` for filtering logic**

```python
from src.experiments.infrastructure.base_prompt_filteration import BasePromptFilteration

@dataclass(frozen=True)
class MyPromptFilteration(BasePromptFilteration):
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        return [1, 2, 3, 4, 5]  # Example prompt IDs
        
    def get_dependencies(self) -> TDependencies:
        return {}
        
    def display_name(self) -> str:
        return "My Filter"
```

#### Logical Operations

**USE logical operations**: `&` (AND), `|` (OR), `~` (NOT)

```python
combined_filter = filter1 & filter2  # AND
combined_filter = filter1 | filter2  # OR
inverted_filter = ~filter1           # NOT
complex_filter = (filter1 & filter2) | (~filter3)
```

#### Filter Types

```python
# Specific prompt selections
specific_filter = SelectivePromptFilteration(prompt_ids=(1, 5, 10, 15))

# Sampling with seeds
sampled_filter = SamplePromptFilteration(base_filter, sample_size=100, seed=42)

# Complex logical combinations
complex_filter = LogicalPromptFilteration.create_and([filter1, filter2, filter3])
```

### ModelInterface Abstract Class

**File**: `src/experiments/infrastructure/model_interface.py`

```python
from src.experiments.infrastructure.model_interface import ModelInterface

class ModelInterface(ABC):
    @abstractmethod
    def generate_logits(self, input_ids: torch.Tensor, 
                       num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
                       feature_category: FeatureCategory = FeatureCategory.ALL) -> torch.Tensor:
        pass
        
    @abstractmethod
    def n_layers(self) -> int:
        pass
```

#### Usage Patterns

```python
# Get model interface
model_interface = get_model_interface(MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, MODEL_SIZE.SMALL))

# Setup and generate logits
model_interface.setup(layers=[0, 1, 2, 3])
logits = model_interface.generate_logits(
    input_ids=input_tokens,
    num_to_masks={0: [(1, 2), (3, 4)]},
    feature_category=FeatureCategory.SLOW_DECAY
)
```

## Infrastructure Coordination Rules

### Experiment Creation Rules

**When creating new experiment types**: Inherit from `BaseRunner` AND create corresponding prompt filteration if needed

```python
# 1. Create variant parameters
@dataclass(frozen=True)
class MyExperimentVariantParams(BaseVariantParams):
    experiment_name: ClassVar[ExperimentName] = ExperimentName.MY_EXPERIMENT

# 2. Create runner implementation
class MyExperimentRunner(BaseRunner[MyExperimentVariantParams]):
    # Implement all abstract methods
    
# 3. Create corresponding prompt filteration if needed
@dataclass(frozen=True)
class MyExperimentFilteration(BasePromptFilteration):
    # Implement filtering logic
```

### Dependency Management Rules

**NEVER create direct dependencies between runners** - use the dependency system

```python
# CORRECT: Use dependency system
def get_runner_dependencies(self) -> TDependencies:
    return {"evaluate_model": EvaluateModelRunner(...)}

# INCORRECT: Direct dependency
def _compute_impl(self) -> None:
    evaluate_runner = EvaluateModelRunner(...)  # Don't do this
```

**ALWAYS check `dependencies_are_computed()` before execution**

```python
def run(self, with_dependencies: bool) -> None:
    if with_dependencies and not self.dependencies_are_computed():
        self.compute_dependencies()
    
    if not self.is_computed():
        self._compute_impl()
```

## Cross-References

- **Experiment Runners**: See [docs/experiment-runners.md](experiment-runners.md) for how runners use infrastructure
- **Data Interfaces**: See [docs/data-relationships-interface.md](data-relationships-interface.md) for how data objects use infrastructure
- **Core Modules**: See [docs/core-modules.md](core-modules.md) for how infrastructure uses core constants
- **Analysis and Plotting**: See [docs/analysis-and-plotting.md](analysis-and-plotting.md) for how plotting uses infrastructure

## Critical Warnings

⚠️ **Breaking the base class inheritance patterns will cause cascading failures throughout the project**

⚠️ **Always test changes thoroughly - infrastructure errors propagate to all dependent experiments**

⚠️ **When in doubt, follow the established patterns exactly - don't improvise**

⚠️ **Infrastructure changes require coordination with the entire development team** 
