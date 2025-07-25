---
description: Prompt Filtering Logic and Patterns
inclusion: manual
---

# Prompt Filteration System

Controls which prompts are used in experiments through flexible filtering criteria.

## Core Architecture

**Base Classes**:
- `BasePromptFilteration` - Abstract interface for all filters
- `ProxyPromptFilteration` - Base for filters that delegate to others

**Key Features**:
- Context awareness with runner integration
- Logical operations (AND, OR, NOT) via operator overloading
- Dependency management for experiment coordination
- Immutable frozen dataclasses for thread safety

## Filter Types

### 1. AllPromptFilteration
Returns all prompts from a dataset/split.

### 2. SelectivePromptFilteration  
Returns specific prompt IDs for testing.

### 3. ModelCorrectPromptFilteration
Returns prompts where model gives correct answers.
- **MUST set model_arch_and_size=None for context-aware usage**
- Supports correctness levels: correct, top_5_correct, top_3_correct, top_2_to_5_correct
- Automatically creates EvaluateModelRunner dependency

### 4. AnyExistingCompletePromptFilteration
Returns prompts already processed by previous experiments.
- **MUST be contextualized with runner**
- Adapts to runner type (EvaluateModel, InfoFlow, Heatmap)

### 5. SamplePromptFilteration
Applies random sampling to any base filter.
- Use deterministic seeds for reproducibility

## Logical Operations

**Operator Overloading**:

```python
combined = filter1 & filter2 & filter3  # AND
union = filter1 | filter2               # OR  
negated = ~filter1.with_universe(all)   # NOT
```

**Factory Methods**:

```python
LogicalPromptFilteration.create_and([f1, f2, f3])
LogicalPromptFilteration.create_or([f1, f2])
LogicalPromptFilteration.create_not(f1, universe=all)
```

## Context System Rules

**ALWAYS contextualize filters in runners**:

```python
def get_runner_dependencies(self) -> TDependencies:
    filteration = self.input_params.filteration.contextualize(self)
    return filteration.get_dependencies()
```

**Context propagation is automatic in logical operations**:

```python
complex_filter = base_filter & other_filter
contextualized = complex_filter.contextualize(runner)  # Propagates to both
```

## Factory Pattern

Use `PromptFilterationFactory` for common patterns:
- `FilterationSource.current_model` - Model correctness from context
- `FilterationSource.existing_prompts` - Previously processed prompts
- `FilterationSource.preset` - Predefined configurations

## Common Patterns

**Multi-Model Consensus**:

```python
models = [MODEL_ARCH_AND_SIZE(...), ...]
consensus = LogicalPromptFilteration.create_and([
    ModelCorrectPromptFilteration(..., model_arch_and_size=m) 
    for m in models
])
```

**Progressive Filtering**:

```python
available = correct_prompts & existing_prompts
final = SamplePromptFilteration(available, sample_size=500, seed=42)
```

**Error Analysis**:

```python
wrong_prompts = (~correct_prompts).with_universe(all_prompts)
near_miss = top_5_correct & (~exact_correct).with_universe(all_prompts)
```

## Integration Rules

**ALWAYS use in InputParams**:

```python
@dataclass(frozen=True)
class InputParams(BaseParams):
    filteration: BasePromptFilteration
    dataset_name: DatasetName = DatasetName.counter_fact
```

**NEVER hardcode model info when context available**:

```python
# Good: Context-aware
ModelCorrectPromptFilteration(model_arch_and_size=None, ...)

# Bad: Hardcoded
ModelCorrectPromptFilteration(model_arch_and_size=MODEL_ARCH_AND_SIZE(...), ...)
```

## Performance Rules

- Results are cached within experiment runs
- Dependencies computed only when needed
- Use `dependencies_are_computed()` to check before running

## Cross-References

For implementation details:
- `src/analysis/prompt_filterations.py` - Base classes and implementations
- `src/analysis/experiment_results/prompt_filteration_factory.py` - Factory patterns
- `src/experiments/infrastructure/base_runner.py` - Runner integration patterns

For related patterns:
- Use #experiment-runners for dependency management
- Use #infrastructure for base class coordination
- Use #core-modules for constant definitions
