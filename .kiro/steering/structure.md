---
inclusion: always
---
# Project Structure and Organization

## Root Directory Structure

```
ssm_analysis/
├── src/                    # Main source code
├── scripts/               # Utility and installation scripts
├── docs/                  # Comprehensive modular documentation
├── tests/                 # Test files
├── data/                  # Data directory (preprocessed and raw)
├── final_plots/           # Generated plots and figures
├── pyproject.toml         # Project configuration and dependencies
├── README.md              # Project overview and setup
└── shrimp-rules.md        # High-level development guidelines
```

## Source Code Architecture (`src/`)

### Core Module Coordination (Critical)

```
src/core/
├── names.py              # Enum definitions and name constants
├── types.py              # Type definitions and aliases
└── consts.py             # Centralized constant definitions
```

**CRITICAL**: These 3 files must be updated together in sequence: names.py → types.py → consts.py

### Infrastructure Layer

```
src/experiments/infrastructure/
├── base_runner.py                    # BaseRunner abstract class
├── base_prompt_filteration.py       # BasePromptFilteration system
└── model_interface.py               # ModelInterface abstract class
```

### Experiment Runners

```
src/experiments/runners/
├── evaluate_model.py      # Model evaluation experiments
├── info_flow.py          # Information flow analysis
├── heatmap.py            # Heatmap generation
└── full_pipeline.py      # Complete experimental pipeline
```

### Model-Specific Knockout Implementations

```
src/experiments/knockout/
├── gpt/                  # GPT-2 knockout implementations
├── llama/                # Llama/Mistral/Qwen knockout implementations
└── mamba/
    ├── mamba1/           # Mamba-1 specific knockout
    └── mamba2/           # Mamba-2 specific knockout
```

### Data Management

```
src/data_ingestion/
├── data_defs/
│   └── data_defs.py      # Data interface classes and relationships
├── datasets/
│   ├── download_dataset.py  # Raw data downloading
│   └── splitting.py         # Dataset splitting logic
└── preprocessing/        # Data preprocessing utilities
```

### Analysis and Visualization

```
src/analysis/
├── experiment_results/   # Result handling and analysis
├── plots/               # Plotting components and configuration
└── prompt_filterations.py  # Prompt filtering logic
```

### Streamlit Web Interface

```
src/app/
├── entry_point.py       # Main application entry
├── page/               # Individual page implementations
│   ├── p01_home.py     # Home page
│   ├── p02_results_bank.py  # Results management
│   └── ...             # Additional pages (p03-p09)
├── components/         # Reusable UI components
├── data_store.py       # Session state management
└── app_consts.py       # Application constants
```

### Utilities Hierarchy

```
src/utils/
├── infra/              # Infrastructure utilities
├── types_utils.py      # Type manipulation utilities
└── ...                 # Additional utility modules
```

## Configuration and Documentation

### Documentation Structure (`docs/`)

- **README.md** - Documentation navigation hub
- **core-modules.md** - Critical 3-file coordination pattern
- **infrastructure.md** - Base classes and dependency management
- **experiment-runners.md** - Runner implementation patterns
- **data-relationships-interface.md** - Data object interfaces
- **analysis-and-plotting.md** - Visualization components

### Configuration Files

- **pyproject.toml** - Main project configuration, dependencies, tool settings
- **uv.cuda.lock** / **uv.cpu.lock** - Reproducible dependency locks
- **.gitignore** - Git ignore patterns
- **.pre-commit-config.yaml** - Pre-commit hooks configuration

## Key Architectural Patterns

### 3-File Core Coordination

All constants, names, and types are centralized in `src/core/` with strict update sequence requirements.

### BaseRunner Pattern

All experiments inherit from `BaseRunner[ParamsType]` and implement:

- `get_runner_dependencies()` - Declare dependencies
- `_compute_impl()` - Main computation logic
- `is_computed()` - Check if results exist

### Data Object Hierarchy

- **DataReqs** - Data requirements specification
- **ResultBank** - Experiment result access
- **FulfilledReqs** - Fulfilled data requirements

### Model Interface Abstraction

Model-specific implementations through `ModelInterface` abstract class for consistent knockout operations across architectures.

## File Naming Conventions

### Experiment Files

- Runners: `{experiment_name}.py` in `src/experiments/runners/`
- Parameters: `{ExperimentName}VariantParams` dataclass
- Results: Corresponding classes in `src/analysis/experiment_results/`

### Streamlit Pages

- Format: `p{XX}_{page_name}.py` where XX is page order number
- Entry point: Always `src/app/entry_point.py`

### Data Files

- Raw data: `data/raw/{dataset_name}/`
- Preprocessed: `data/preprocessed/{dataset_name}/`
- Results: `results/{experiment_name}/`

## Critical Organization Rules

### NEVER Create These Patterns

- Constants outside `src/core/` module
- Utilities outside `src/utils/` hierarchy
- Model-specific code in shared experiment files
- Direct dependencies between runners

### ALWAYS Follow These Patterns

- Inherit from `BaseRunner` for experiments
- Use `ResultBank` for accessing experiment outputs
- Update core files in sequence: names.py → types.py → consts.py
- Place utilities in appropriate `src/utils/` subdirectories

### Directory Creation Rules

- Experiment outputs: Auto-created by `BaseRunner.create_experiment_dir()`
- Data directories: Auto-created by data ingestion processes
- Plot outputs: Auto-created by plotting components
