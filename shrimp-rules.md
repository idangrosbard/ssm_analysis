# SSM Analysis Project Development Guidelines

## Project Overview

- **Research project for analyzing Mamba State-Space Models using knockout methodology**
- **Technology stack: Python 3.12, PyTorch, Mamba-SSM, UV package manager**
- **Core functionality: Factual information flow analysis in language models**
- **Repository purpose: Fully reproducible research environment**

## High-Level Architecture Summary

This project follows a sophisticated modular architecture with strict coordination requirements:

- **Infrastructure Layer**: Base classes and dependency management (`BaseRunner`, `BasePromptFilteration`)
- **Experiment Layer**: Specialized runners for different analysis types (evaluate_model, info_flow, heatmap, full_pipeline)
- **Data Layer**: Object-oriented data interfaces and management (`DataReqs`, `ResultBank`, `FulfilledReqs`)
- **Analysis Layer**: Plotting and visualization components with configuration management
- **Core Coordination**: Centralized constants, names, and types with strict 3-file coordination
- **UI Layer**: Streamlit application with component-based architecture
- **Utilities**: Organized utility hierarchy supporting all layers

## Documentation Navigation

For detailed information on specific components, see the specialized documentation modules:

- **[Infrastructure Documentation](docs/infrastructure.md)** - Base classes, dependency management, and infrastructure coordination
- **[Experiment Runners](docs/experiment-runners.md)** - Runner types, implementation patterns, and experiment workflows
- **[Data Interfaces](docs/data-interfaces.md)** - Object-oriented data entities and interaction patterns
- **[Core Modules Coordination](docs/core-modules.md)** - Critical 3-file coordination pattern and update requirements
- **[Analysis and Plotting](docs/analysis-and-plotting.md)** - Visualization components and configuration management
- **[Setup and Environment](docs/setup-and-environment.md)** - Package management, configuration, and reproducibility
- **[Utilities and Patterns](docs/utilities-and-patterns.md)** - Utility organization standards and common patterns
- **[Streamlit Infrastructure](docs/streamlit-infrastructure.md)** - UI component architecture and patterns (coming soon)
- **[Streamlit App Pages](docs/streamlit-app-pages.md)** - Page organization and navigation system (coming soon)

## Priority Order for Implementation

1. **Core module coordination (highest priority)** - See [Core Modules Coordination](docs/core-modules.md)
2. **Infrastructure component compatibility** - See [Infrastructure Documentation](docs/infrastructure.md)
3. **Experiment structure compliance** - See [Experiment Runners](docs/experiment-runners.md)
4. **Data object consistency** - See [Data Interfaces](docs/data-interfaces.md)
5. **Utility organization standards** - See [Utilities and Patterns](docs/utilities-and-patterns.md)
6. **Documentation updates** - Maintain alignment with actual implementation
7. **Logging and output compliance** - Follow established patterns

## Cross-Cutting Coordination Rules

### Critical Multi-File Update Requirements

**When adding new experiment types or constants, you MUST update multiple coordinated files:**

1. **Core Module Updates**: Always update `src/core/consts.py`, `src/core/names.py`, AND `src/core/types.py` together
2. **Data Object Updates**: When modifying `src/data_ingestion/data_defs/data_defs.py`, update corresponding `src/analysis/experiment_results/` components
3. **Runner Updates**: When creating new runners, ensure proper dependency declarations and data object integration
4. **Documentation Updates**: When changing project structure, update README.md and relevant documentation modules

### Infrastructure Coordination

- **Base Class Compatibility**: All experiment runners MUST inherit from `BaseRunner` and implement all abstract methods
- **Dependency Management**: NEVER create direct dependencies between runners - use the established dependency system
- **Data Access**: NEVER access experiment outputs directly - use `ResultBank` and related objects
- **Prompt Filteration**: All filtering logic MUST inherit from `BasePromptFilteration` and use logical operations

### Package Management and Environment

- **UV Package Manager**: ALWAYS use `uv` command instead of `pip` for all package operations
- **Configuration Files**: ALL configuration files MUST be symlinked from home directory to repo files
- **Installation**: ALWAYS use `scripts/install.sh` for automated installation
- **Reproducibility**: Maintain Python 3.12 compatibility and CUDA requirements for mamba-ssm

## Conflict Resolution Rules

- **Core module consistency takes precedence over individual file optimization**
- **Infrastructure base class compatibility is non-negotiable**
- **Reproducibility requirements override convenience shortcuts**
- **UV package manager usage is non-negotiable**
- **Symlink configuration pattern must be preserved**

## Prohibited Actions

### **NEVER Do These Actions**

- **Use `pip` instead of `uv` for package management**
- **Hardcode constants outside `src/core/` module**
- **Create utilities outside `src/utils/` hierarchy**
- **Break symlink patterns for configuration files**
- **Modify only one core file when changes affect multiple files**
- **Create model-specific code in shared experiment files**
- **Skip README.md updates when changing project structure**
- **Use general development patterns that conflict with project-specific organization**
- **Bypass the BaseRunner dependency system**
- **Access experiment outputs without using ResultBank objects**
- **Create runners without proper prompt filteration support**
- **Modify data_defs.py objects without updating related experiment_results components**

### **NEVER Create These Patterns**

- **Direct configuration files instead of symlinks**
- **Scattered constant definitions across modules**
- **Mixed model architecture code in shared files**
- **Utility functions with hardcoded project-specific values**
- **Installation procedures that bypass provided scripts**
- **Experiment runners that don't inherit from BaseRunner**
- **Prompt filterations that don't follow the logical operation patterns**
- **Data access patterns that bypass the established object hierarchy**

## Quick Reference Examples

### ✅ Correct Implementation Patterns

```bash
# Correct package management
uv add transformers==4.50.3

# Correct constant usage
from src.core.consts import EXPERIMENT_CONFIGS
from src.core.names import ExperimentName
```

```python
# Correct runner implementation
@dataclass(frozen=True)
class NewExperimentRunner(BaseRunner[NewExperimentParams]):
    variant_params: NewExperimentParams
    
    def get_runner_dependencies(self):
        return {"evaluate_model": EvaluateModelRunner.init_from_runner(...)}
    
    def _compute_impl(self):
        self.create_experiment_dir()
        # Implementation here
    
    def is_computed(self):
        return self.output_path.exists()
```

```python
# Correct data object usage
result_bank = get_experiment_results_bank()
info_flow_results = result_bank.to_info_flow_results()
data_reqs = DataReqs.from_data_reqs_collection(collection)
```

### ❌ Incorrect Patterns

```bash
# WRONG: Using pip
pip install transformers

# WRONG: Hardcoded constants
BATCH_SIZE = 32  # Should be in src/core/consts.py
```

```python
# WRONG: Direct file access
data = pd.read_csv("outputs/results.csv")  # Use ResultBank instead

# WRONG: Bypassing dependency system
class BadRunner(BaseRunner):
    def _compute_impl(self):
        other_runner.run()  # Should use get_runner_dependencies()
```

### ✅ Correct Multi-File Update Sequence

**When adding new experiment type:**

1. Add enum to `src/core/names.py`
2. Add configuration to `src/core/consts.py`  
3. Add type definition to `src/core/types.py`
4. Create experiment runner in `src/experiments/runners/`
5. Update data objects in `src/data_ingestion/data_defs/data_defs.py`
6. Add result handling in `src/analysis/experiment_results/helpers.py`
7. Update README.md if needed

### ❌ Incorrect Single-File Updates

**NEVER modify only `consts.py` without checking coordination requirements with `names.py` and `types.py`**
**NEVER create new runners without updating corresponding data management objects**

## Development Workflow

1. **Start with Core Modules**: Always begin with the 3-file coordination pattern
2. **Follow Infrastructure Patterns**: Use established base classes and dependency management
3. **Coordinate Data Objects**: Ensure data interfaces are properly integrated
4. **Maintain Documentation**: Keep all documentation modules aligned with implementation
5. **Test Integration**: Verify that all components work together correctly

## Getting Help

- **For infrastructure questions**: See [Infrastructure Documentation](docs/infrastructure.md)
- **For experiment development**: See [Experiment Runners](docs/experiment-runners.md)
- **For data management**: See [Data Interfaces](docs/data-interfaces.md)
- **For core coordination**: See [Core Modules Coordination](docs/core-modules.md)
- **For visualization**: See [Analysis and Plotting](docs/analysis-and-plotting.md)
- **For environment setup**: See [Setup and Environment](docs/setup-and-environment.md)
- **For utility development**: See [Utilities and Patterns](docs/utilities-and-patterns.md)
- **For UI development**: See [Infrastructure Documentation](docs/infrastructure.md) for base patterns (Streamlit-specific documentation coming soon)
