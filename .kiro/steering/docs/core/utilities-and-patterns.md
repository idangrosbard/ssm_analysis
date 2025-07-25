---
description: Code Organization Standards
inclusion: manual
---

# Utilities and Patterns

Utility organization standards and common patterns for `src/utils/` directory.
Focus on rules LLMs need to know for utility implementation.

## Directory Structure

```
src/utils/
├── __init__.py                    # Main utilities package
├── types_utils.py                 # Type checking and manipulation
├── file_system.py                 # File and path operations
├── json_utils.py                  # JSON handling and sanitization
├── pydantic_utils.py             # Pydantic integration
├── PIL_utils.py                   # Image processing
├── type_checking.py               # Type validation
├── tests_utils.py                 # Testing support
├── jsonable.py                    # JSON serialization
├── infra/                         # Infrastructure utilities
│   ├── slurm.py                   # SLURM job management
│   ├── slurm_job_folder.py        # SLURM folder utilities
│   ├── output_path.py             # Output path management
│   ├── snapshot.py                # System snapshot
│   ├── seed.py                    # Random seed management
│   ├── git.py                     # Git integration
│   ├── experiment_helper.py        # Experiment support
│   ├── data_object.py             # Data object utilities
│   └── image_utils.py             # Image processing
└── streamlit/                     # Streamlit utilities
    ├── helpers/                   # Core Streamlit helpers
    ├── components/                # Enhanced UI components
    ├── st_pydantic_v2/           # Pydantic integration
    └── ui_pydantic_v2/           # UI Pydantic utilities
```

## Core Rules

### Utility Placement Rules

1. **ALWAYS place new general-purpose utilities in `src/utils/`**
2. **NEVER create utility functions outside `src/utils/` hierarchy**
3. **NEVER hardcode values in utility functions - use constants from `src/core/`**
4. **ALWAYS use existing utility patterns for file operations, type checking, and JSON handling**

### Utility Categories

- **General utilities** (`src/utils/`): Broadly applicable across modules
- **Infrastructure utilities** (`src/utils/infra/`): System and infrastructure operations
- **Streamlit utilities** (`src/utils/streamlit/`): UI-specific utilities and components

## Common Patterns

### Type Safety Patterns

Use methods from `types_utils.py` for type safety and common datastructures operations, such as:
 `subset_dict_by_keys`, `get_enum_or_literal_options`,  `first_dict_key`, `ommit_none`, etc.

### When to Create New Utilities

1. **Repeated Patterns**: Same code pattern appears in multiple modules
2. **Type Safety**: Type-safe operations not available in standard library
3. **Project Standards**: Project-specific patterns (use appropriate subdirectory)
4. **Performance**: Optimizing common operations across project

### Where to Place New Utilities

1. **General utilities**: `src/utils/new_utility.py` or add to existing appropriate file
2. **Infrastructure utilities**: `src/utils/infra/new_infra_utility.py`
3. **Streamlit utilities**: `src/utils/streamlit/helpers/new_streamlit_utility.py`
4. **Specialized utilities**: Create new subdirectory under `src/utils/` if needed

### Utility Creation Checklist

- [ ] **General-purpose**: Reusable across different modules
- [ ] **Type-safe**: Proper type hints and generic patterns
- [ ] **Constant-free**: No hardcoded values, use constants from `src/core/`
- [ ] **Well-documented**: Docstrings and usage examples
- [ ] **Tested**: Unit tests for utility functions
- [ ] **Follows patterns**: Use existing utility patterns for similar operations

## Cross-References

- **Core Modules**: Use #core-modules for constant definitions used in utilities
- **Infrastructure**: Use #infrastructure for base class patterns
- **Setup and Environment**: Use #setup-and-environment for environment setup
- **Data Interfaces**: Use #data-relationships-interface for data object utilities
- **Analysis and Plotting**: Use #analysis-and-plotting for plotting utilities

## Best Practices Summary

1. **Organization**: Always use established `src/utils/` hierarchy
2. **Type Safety**: Use generics and proper type hints for all utilities
3. **Patterns**: Follow existing utility patterns for similar operations
4. **Documentation**: Include clear docstrings and usage examples
5. **Testing**: Ensure utilities are properly tested and validated
6. **Integration**: Coordinate with core modules and infrastructure patterns
