# Utilities and Patterns

This document covers utility organization standards, common patterns, and best practices for the `src/utils/` directory structure. These standards ensure code reusability, maintainability, and consistent utility implementation across the project.

## Overview

The utilities system provides a well-organized hierarchy of reusable functions and patterns that support the entire project architecture. All utilities follow strict organization standards to ensure they remain general-purpose, properly organized, and maintainable.

## Utility Organization Standards

### Directory Structure Hierarchy

```
src/utils/
├── __init__.py                    # Main utilities package
├── types_utils.py                 # Type checking and manipulation utilities
├── file_system.py                 # File and path operation utilities
├── json_utils.py                  # JSON handling and sanitization
├── pydantic_utils.py             # Pydantic integration utilities
├── PIL_utils.py                   # Image processing utilities
├── type_checking.py               # Type validation utilities
├── tests_utils.py                 # Testing support utilities
├── jsonable.py                    # JSON serialization utilities
├── infra/                         # Infrastructure-specific utilities
│   ├── __init__.py
│   ├── slurm.py                   # SLURM job management
│   ├── slurm_job_folder.py        # SLURM folder utilities
│   ├── output_path.py             # Output path management
│   ├── snapshot.py                # System snapshot utilities
│   ├── seed.py                    # Random seed management
│   ├── git.py                     # Git integration utilities
│   ├── experiment_helper.py        # Experiment support utilities
│   ├── data_object.py             # Data object utilities
│   └── image_utils.py             # Image processing utilities
└── streamlit/                     # Streamlit-specific utilities
    ├── __init__.py
    ├── helpers/                   # Core Streamlit helpers
    ├── components/                # Enhanced UI components
    ├── st_pydantic_v2/           # Pydantic integration
    └── ui_pydantic_v2/           # UI Pydantic utilities
```

### Utility Categories

#### General Utilities (`src/utils/`)
- **Purpose**: Broadly applicable utilities used across multiple modules
- **Examples**: Type checking, file operations, JSON handling, data manipulation
- **Rules**: Must be general-purpose, not tied to specific project features

#### Infrastructure Utilities (`src/utils/infra/`)
- **Purpose**: Utilities specific to infrastructure and system operations
- **Examples**: SLURM job management, output path handling, experiment coordination
- **Rules**: Can be project-specific but must follow infrastructure patterns

#### Streamlit Utilities (`src/utils/streamlit/`)
- **Purpose**: UI-specific utilities and components for Streamlit application
- **Examples**: Session state management, caching, component helpers
- **Rules**: Must follow Streamlit best practices and component patterns

## Utility Implementation Rules

### Core Principles

1. **ALWAYS place new general-purpose utilities in `src/utils/`**
2. **NEVER create utility functions outside `src/utils/` hierarchy**
3. **NEVER hardcode values in utility functions - use constants from `src/core/`**
4. **ALWAYS use existing utility patterns for file operations, type checking, and JSON handling**

### Implementation Standards

#### Type Safety and Generic Patterns

```python
# ✅ Correct: Generic type utilities
def subset_dict_by_keys(d: dict[_K, _V], keys: list[_K]) -> dict[_K, _V]:
    return {k: v for k, v in d.items() if k in keys}

# ✅ Correct: Type checking utilities
def get_enum_or_literal_options(typ: Any) -> list[str]:
    origin = get_origin(typ)
    args = get_args(typ)
    # Implementation with proper type handling
```

#### File Operation Patterns

```python
# ✅ Correct: Safe file operations
def atomic_write(path: Path, text: str | bytes) -> None:
    """Safely write text to a file using an atomic replace strategy."""
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp_file:
        tmp_file.write(text)
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(path)

# ✅ Correct: Path manipulation utilities
def fast_relative_to(path: Path, base_path: Path, allow_slow: bool = False) -> Path:
    """Get the relative path with performance optimization."""
    if allow_slow:
        return path.relative_to(base_path)
    else:
        base_parts = base_path.parts
        base_len = len(base_parts)
        path_parts = path.parts
        assert path_parts[:base_len] == base_parts
        return Path(*path_parts[base_len:])
```

#### JSON Handling Patterns

```python
# ✅ Correct: JSON sanitization utilities
def sanitize(obj: GeneralJsonObject) -> SanitizedJsonObject:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable):
        return [sanitize(x) for x in obj]
    else:
        return obj

# ✅ Correct: Dataclass JSON serialization
def json_dumps_dataclass(obj: Any, **kwargs) -> str:
    def dataclass_json_encoder(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Type {type(obj)} not serializable")
    
    return json.dumps(obj, default=dataclass_json_encoder, **kwargs)
```

### Constant Usage Patterns

#### Correct Constant Integration

```python
# ✅ Correct: Using constants from core modules
from src.core.consts import EXPERIMENT_CONFIGS
from src.core.names import ExperimentName

def create_experiment_config(exp_name: ExperimentName):
    return EXPERIMENT_CONFIGS[exp_name]
```

#### Incorrect Hardcoding

```python
# ❌ WRONG: Hardcoded values in utilities
def process_data(batch_size=32):  # Should use constant from src/core/consts.py
    pass

# ❌ WRONG: Project-specific values in general utilities
def get_model_path(model_name="gpt2"):  # Should be in specialized module
    pass
```

## Common Utility Patterns

### Type Checking and Validation

```python
# Enum and Literal handling
def get_enum_or_literal_options(typ: Any) -> list[str]:
    """Extract options from Enum or Literal types."""
    origin = get_origin(typ)
    args = get_args(typ)
    
    if origin is Literal:
        return [str(a) for a in args]
    elif isinstance(typ, type) and issubclass(typ, Enum):
        return [e.name if isinstance(e, StrEnum) else e.name for e in typ]
    elif origin is Union:
        values = []
        for arg in args:
            values += get_enum_or_literal_options(arg)
        return values
    return []

# Type-safe enum initialization
def init_str_enum_from_value(cls: Type[_T_STR_ENUM], value: str) -> _T_STR_ENUM:
    """Safely initialize StrEnum from string value."""
    assert value in str_enum_values(cls)
    return cast(_T_STR_ENUM, value)
```

### Data Structure Manipulation

```python
# Dictionary utilities
def subset_dict_by_condition(d: dict[_K, _V], condition: Callable[[_K, _V], bool]) -> dict[_K, _V]:
    """Filter dictionary by key-value condition."""
    return {k: v for k, v in d.items() if condition(k, v)}

def ommit_none(d: dict[Any, Optional[Any]]) -> dict[Any, Any]:
    """Remove None values from dictionary."""
    return {k: v for k, v in d.items() if v is not None}

# List utilities
def select_indexes_from_list(lst: list[_T], indexes: list[int]) -> list[_T]:
    """Select items from list by indexes."""
    return [lst[i] for i in indexes]

def get_list_indexes_of_set_values(lst: list[_T], values: set[_T]) -> list[int]:
    """Get indexes of list items that are in the given set."""
    return [i for i, v in enumerate(lst) if v in values]
```

### Context Management

```python
# Conditional context management
def conditional_context_manager(use_ctx: bool, ctx: ContextManager[None]) -> ContextManager[None]:
    """Returns context manager conditionally."""
    return ctx if use_ctx else contextlib.nullcontext()

# Usage example
with conditional_context_manager(debug_mode, logging_context()):
    # Code that may or may not need logging context
    pass
```

## Guidelines for Creating New Utilities

### When to Create New Utilities

1. **Repeated Patterns**: When the same code pattern appears in multiple modules
2. **Type Safety**: When you need type-safe operations that aren't available in standard library
3. **Project Standards**: When implementing project-specific patterns (use appropriate subdirectory)
4. **Performance**: When optimizing common operations across the project

### Where to Place New Utilities

1. **General utilities**: `src/utils/new_utility.py` or add to existing appropriate file
2. **Infrastructure utilities**: `src/utils/infra/new_infra_utility.py`
3. **Streamlit utilities**: `src/utils/streamlit/helpers/new_streamlit_utility.py`
4. **Specialized utilities**: Create new subdirectory under `src/utils/` if needed

### Utility Creation Checklist

- [ ] **General-purpose**: Utility should be reusable across different modules
- [ ] **Type-safe**: Use proper type hints and generic patterns
- [ ] **Constant-free**: Don't hardcode values, use constants from `src/core/`
- [ ] **Well-documented**: Include docstrings and usage examples
- [ ] **Tested**: Include unit tests for utility functions
- [ ] **Follows patterns**: Use existing utility patterns for similar operations

### Integration with Core Modules

```python
# ✅ Correct: Utility that integrates with core modules
from src.core.consts import DEFAULT_BATCH_SIZE
from src.core.names import ModelName

def create_model_config(model_name: ModelName, batch_size: int = None) -> dict:
    """Create model configuration using core constants."""
    return {
        "model": model_name,
        "batch_size": batch_size or DEFAULT_BATCH_SIZE,
        # Other configuration...
    }
```

## Cross-References

- **Core Modules**: See [Core Modules Coordination](core-modules.md) for constant usage patterns
- **Infrastructure**: See [Infrastructure Documentation](infrastructure.md) for base class patterns
- **Streamlit**: See [Infrastructure Patterns](infrastructure.md) for base patterns (Streamlit-specific documentation coming soon)
- **Data Interfaces**: See [Data Interfaces](data-interfaces.md) for data manipulation utilities

## Best Practices Summary

1. **Organization**: Always use the established `src/utils/` hierarchy
2. **Type Safety**: Use generics and proper type hints for all utilities
3. **Constants**: Never hardcode values, use constants from `src/core/`
4. **Patterns**: Follow existing utility patterns for similar operations
5. **Documentation**: Include clear docstrings and usage examples
6. **Testing**: Ensure utilities are properly tested and validated
7. **Integration**: Coordinate with core modules and infrastructure patterns

## Verification Checklist

- [ ] All utilities follow the established directory structure
- [ ] No hardcoded values in utility functions
- [ ] Proper type hints and generic patterns used
- [ ] Utilities are general-purpose and reusable
- [ ] Integration with core modules follows established patterns
- [ ] Documentation includes usage examples and cross-references
- [ ] Utilities support the project's scientific computing requirements 
