# Core Module Coordination

This document covers the critical 3-file coordination pattern and multi-file update requirements for the core modules (`src/core/consts.py`, `src/core/names.py`, `src/core/types.py`). These coordination rules are **highest priority** - mistakes here break the entire project.

## Overview

The core modules form the foundation of the entire project architecture. All constants, names, enums, and type definitions are centralized in these three files, and they must be kept in perfect synchronization. Breaking this coordination pattern will cause cascading failures throughout the project.

## The 3-File Coordination Pattern

### File Responsibilities

#### `src/core/consts.py`
- **Purpose**: Centralized constant definitions and configuration values
- **Contains**: Path configurations, model mappings, environment variables, output keys
- **Usage**: Import constants for use throughout the project

#### `src/core/names.py`
- **Purpose**: Enum definitions and name constants
- **Contains**: Experiment names, dataset names, column names, parameter names
- **Usage**: Define standardized names and enums for type safety

#### `src/core/types.py`
- **Purpose**: Type definitions and aliases
- **Contains**: NewType definitions, TypeAlias declarations, complex type structures
- **Usage**: Provide type safety and clear interfaces

## Multi-File Update Requirements

### When Adding New Constants

**MUST update all three files in the correct sequence:**

1. **First**: Add enum/name to `src/core/names.py`
2. **Second**: Add type definition to `src/core/types.py` (if needed)
3. **Third**: Add constant value to `src/core/consts.py`

```python
# Example: Adding a new model architecture

# 1. Add to names.py
class MODEL_ARCH(StrEnum):
    NEW_ARCH = "new_arch"  # Add here first

# 2. Add to types.py (if needed)
TNewArchModel = TypeAlias = NewArchForCausalLM

# 3. Add to consts.py
MODEL_SIZES_PER_ARCH_TO_MODEL_ID: dict[MODEL_ARCH, dict[TModelSize, TModelID]] = {
    MODEL_ARCH.NEW_ARCH: {  # Reference from names.py
        TModelSize("1B"): TModelID("new-arch/1b-model"),  # Use types from types.py
    },
    # ... existing entries
}
```

### When Adding New Enums

**Follow the same sequence:**

1. **First**: Define enum in `src/core/names.py`
2. **Second**: Add any related types to `src/core/types.py`
3. **Third**: Use the enum in `src/core/consts.py`

```python
# Example: Adding a new experiment type

# 1. Add to names.py
class ExperimentName(StrEnum):
    new_experiment = "new_experiment"  # Add here first

# 2. Add to types.py (if needed)
TNewExperimentParams = NewType("TNewExperimentParams", dict)

# 3. Use in consts.py
BASE_OUTPUT_KEYS = {
    EXPERIMENT_NAME = OutputKey[ExperimentName]("experiment_name", key_display_name="")  # Reference from names.py
}
```

### When Adding New Types

**Coordinate with existing patterns:**

1. **First**: Define the type in `src/core/types.py`
2. **Second**: Add any related enums to `src/core/names.py`
3. **Third**: Use the type in `src/core/consts.py`

```python
# Example: Adding a new parameter type

# 1. Add to types.py
TNewParam = NewType("TNewParam", str)

# 2. Add to names.py (if needed)
class NewParamName(StrEnum):
    param_name = "param_name"

# 3. Use in consts.py
NEW_PARAM_CONFIG = {
    "default": TNewParam("default_value")  # Use type from types.py
}
```

## Core Module Rules

### **NEVER Do These Actions**

- **NEVER hardcode constants outside `src/core/` module**
- **NEVER modify only one core file when changes affect multiple files**
- **NEVER create constants in individual modules**
- **NEVER bypass the 3-file coordination pattern**

### **ALWAYS Do These Actions**

- **ALWAYS import constants from `src/core/consts.py` instead of defining locally**
- **ALWAYS use centralized parameter definitions from `src/core/names.py`**
- **ALWAYS check all three core files when modifying any parameter or constant**
- **ALWAYS follow the update sequence: names.py → types.py → consts.py**

## Step-by-Step Procedures

### Adding a New Model Architecture

1. **Update `src/core/names.py`**:
   ```python
   class MODEL_ARCH(StrEnum):
       NEW_ARCH = "new_arch"  # Add new architecture
   ```

2. **Update `src/core/types.py`**:
   ```python
   # Add model type if needed
   TNewArchModel: TypeAlias = NewArchForCausalLM
   ```

3. **Update `src/core/consts.py`**:
   ```python
   MODEL_SIZES_PER_ARCH_TO_MODEL_ID: dict[MODEL_ARCH, dict[TModelSize, TModelID]] = {
       MODEL_ARCH.NEW_ARCH: {
           TModelSize("1B"): TModelID("new-arch/1b-model"),
           TModelSize("7B"): TModelID("new-arch/7b-model"),
       },
       # ... existing entries
   }
   ```

### Adding a New Experiment Type

1. **Update `src/core/names.py`**:
   ```python
   class ExperimentName(StrEnum):
       new_experiment = "new_experiment"
   
   class NewExperimentVariantParam(StrEnum):
       new_param = "new_param"
   ```

2. **Update `src/core/types.py`**:
   ```python
   TNewExperimentParams = NewType("TNewExperimentParams", dict)
   ```

3. **Update `src/core/consts.py`**:
   ```python
   # Add any experiment-specific constants
   NEW_EXPERIMENT_CONFIG = {
       "default_param": "default_value"
   }
   ```

### Adding a New Dataset

1. **Update `src/core/names.py`**:
   ```python
   class DatasetName(StrEnum):
       new_dataset = "new_dataset"
   ```

2. **Update `src/core/types.py`**:
   ```python
   TNewDatasetID = NewType("TNewDatasetID", str)
   ```

3. **Update `src/core/consts.py`**:
   ```python
   DATASETS_IDS: dict[DatasetName, TDatasetID] = {
       DatasetName.new_dataset: TDatasetID("new-dataset-id"),
       # ... existing entries
   }
   ```

## Examples of Correct and Incorrect Patterns

### ✅ Correct Patterns

```python
# Correct: Import from core modules
from src.core.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID
from src.core.names import MODEL_ARCH, ExperimentName
from src.core.types import TModelSize, TModelID

# Correct: Use centralized definitions
model_id = MODEL_SIZES_PER_ARCH_TO_MODEL_ID[MODEL_ARCH.GPT2][TModelSize("small")]
```

```python
# Correct: Follow update sequence
# 1. Add to names.py first
class NewEnum(StrEnum):
    value = "value"

# 2. Add to types.py if needed
TNewType = NewType("TNewType", str)

# 3. Use in consts.py
NEW_CONSTANT = TNewType("value")
```

### ❌ Incorrect Patterns

```python
# WRONG: Hardcoding constants outside core
BATCH_SIZE = 32  # Should be in src/core/consts.py

# WRONG: Modifying only one file
# Only updating consts.py without updating names.py and types.py

# WRONG: Creating constants in individual modules
class LocalConfig:
    MODEL_PATH = "path/to/model"  # Should be in core modules
```

```python
# WRONG: Bypassing the coordination pattern
# Adding constants directly to consts.py without proper enum definitions

# WRONG: Inconsistent naming
# Using different naming conventions across files
```

## Cross-References and Dependencies

### Modules That Depend on Core Coordination

- **Experiment Runners**: See [docs/experiment-runners.md](experiment-runners.md) for how runners use core constants
- **Data Interfaces**: See [docs/data-interfaces.md](data-interfaces.md) for how data objects use core types
- **Infrastructure**: See [docs/infrastructure.md](infrastructure.md) for how base classes use core enums
- **Analysis and Plotting**: See [docs/analysis-and-plotting.md](analysis-and-plotting.md) for how plotting uses core constants

### Verification Checklist

When making changes to core modules, verify:

- [ ] All three files are updated in the correct sequence
- [ ] No constants are hardcoded outside `src/core/`
- [ ] All imports reference the core modules
- [ ] Type safety is maintained throughout
- [ ] No circular dependencies are created
- [ ] All related modules are updated to use new constants

## Best Practices

1. **Always follow the sequence**: names.py → types.py → consts.py
2. **Use type safety**: Leverage NewType and TypeAlias for clarity
3. **Maintain consistency**: Use the same naming patterns across all three files
4. **Test thoroughly**: Verify that all imports work correctly after changes
5. **Document changes**: Update related documentation when adding new constants
6. **Coordinate with team**: Core module changes affect the entire project

## Critical Warnings

⚠️ **Breaking the 3-file coordination pattern will cause cascading failures throughout the project**

⚠️ **Always test changes thoroughly - core module errors propagate to all dependent code**

⚠️ **When in doubt, follow the established patterns exactly - don't improvise**

⚠️ **Core module changes require coordination with the entire development team** 
