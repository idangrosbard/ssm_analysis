---
description: Infrastructure Base Classes and Patterns
inclusion: manual
---

# Infrastructure Base Classes and Patterns

## Base Class Requirements

### BaseRunner Pattern
- **ALWAYS inherit from `BaseRunner[_TVariantParams]`** for all experiment implementations
- **MUST implement all abstract methods**: `_compute_impl()`, `is_computed()`, `get_outputs()`, `get_runner_dependencies()`
- **MUST call `self.create_experiment_dir()`** before computation

### BasePromptFilteration Pattern
- **ALWAYS inherit from `BasePromptFilteration`** for filtering logic
- **USE logical operations**: `&` (AND), `|` (OR), `~` (NOT)
- **IMPLEMENT required methods**: `get_prompt_ids()`, `get_dependencies()`, `display_name()`

### ModelInterface Pattern
- **Abstract class for model interactions**
- **MUST implement**: `generate_logits()`, `n_layers()`
- **Use `get_model_interface()`** to obtain instances

## Dependency Management Rules

### Critical Rules
- **NEVER create direct dependencies between runners** - use dependency system
- **ALWAYS check `dependencies_are_computed()`** before execution
- **DECLARE dependencies in `get_runner_dependencies()`**

### Correct Dependency Pattern
```python
def get_runner_dependencies(self) -> TDependencies:
    return {"evaluate_model": EvaluateModelRunner(...)}
```

### Infrastructure Files
- `src/experiments/infrastructure/base_runner.py` - BaseRunner abstract class
- `src/experiments/infrastructure/base_prompt_filteration.py` - BasePromptFilteration system
- `src/experiments/infrastructure/model_interface.py` - ModelInterface abstract class

**Reference**: docs/infrastructure.md for complete patterns and examples
