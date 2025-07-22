# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management

Always use `uv` (never `pip`) for all package operations:

```bash
# Install dependencies
uv sync

# Add new package
uv add package_name

# Install with optional dependencies
uv add -e .[typing,streamlit,dev]
```

### Installation

**Self-Contained Installation (Recommended)**

The installation script is completely self-contained and requires only Python 3.12:

```bash
# One-command installation (no admin permissions needed)
chmod +x scripts/install.sh
./scripts/install.sh
```

**Features:**
- ✅ **Auto-installs UV** if not present (no sudo required)
- ✅ **Detects system capabilities** (OS, architecture, CUDA)
- ✅ **Smart GPU/CPU detection** (5 different detection methods)
- ✅ **Comprehensive error handling** with recovery suggestions
- ✅ **System requirements check** (disk space, internet connectivity)
- ✅ **Graceful fallback** from GPU to CPU mode if needed

**Debug Mode:**
```bash
DEBUG=1 ./scripts/install.sh
```

**Manual Installation Options:**
```bash
# CPU-only installation (analysis/visualization only - no mamba-ssm)
uv sync --extra typing --extra streamlit --extra dev --extra cpu

# GPU installation with CUDA optimization (full functionality including mamba-ssm)
uv sync --extra typing --extra streamlit --extra dev --extra gpu
```

**Important**: mamba-ssm requires CUDA and will not work on CPU-only systems. The CPU installation provides all analysis, visualization, and development tools but excludes mamba model experiments.

### Code Quality

```bash
# Linting
ruff check src/ tests/

# Fix linting issues
ruff check --fix src/ tests/

# Type checking with mypy
mypy src/ tests/

# Type checking with pyright
pyright src/ tests/

# Run tests
pytest -vv
```

## Architecture Overview

### Core Module Coordination (CRITICAL)

The project follows a strict 3-file coordination pattern in `src/core/`:

1. **`src/core/names.py`**: Enum definitions and name constants
2. **`src/core/types.py`**: Type definitions and aliases  
3. **`src/core/consts.py`**: Centralized constant definitions

**NEVER modify only one core file** - changes must be coordinated across all three files in sequence: names.py → types.py → consts.py.

### Infrastructure Layer

- **Base Classes**: `BaseRunner`, `BasePromptFilteration` in `src/experiments/infrastructure/`
- **Dependency Management**: Runners declare dependencies via `get_runner_dependencies()`
- **Data Access**: Use `ResultBank` objects, never direct file access

### Experiment Runners

Located in `src/experiments/runners/`:

- `evaluate_model.py`: Model evaluation experiments
- `info_flow.py`: Information flow analysis
- `heatmap.py`: Heatmap generation
- `full_pipeline.py`: Complete experimental pipeline

All runners must inherit from `BaseRunner[ParamsType]` and implement:

- `get_runner_dependencies()`
- `_compute_impl()`
- `is_computed()`

### Data Management

- **Data Definitions**: `src/data_ingestion/data_defs/data_defs.py`
- **Experiment Results**: `src/analysis/experiment_results/`
- **Result Access**: Use `ResultBank`, `DataReqs`, `FulfilledReqs` objects

### Knockout Analysis

Specialized knockout implementations per model architecture:

- **GPT-2**: `src/experiments/knockout/gpt/`
- **Llama**: `src/experiments/knockout/llama/`
- **Mamba-1**: `src/experiments/knockout/mamba/mamba1/`
- **Mamba-2**: `src/experiments/knockout/mamba/mamba2/`

### Streamlit Application

Web interface in `src/app/`:

- **Entry Point**: `src/app/entry_point.py`
- **Pages**: `src/app/page/` (p01_home.py through p09_prompt_filteration_presets.py)
- **Components**: `src/app/components/`

## Development Rules

### Package Management

- **ALWAYS** use `uv` instead of `pip`
- **NEVER** install packages manually without updating `pyproject.toml`
- **ALWAYS** use the installation script for setup

### Core Module Updates

When adding new experiment types, datasets, or model architectures:

1. Add enum to `src/core/names.py`
2. Add type definition to `src/core/types.py` (if needed)
3. Add configuration to `src/core/consts.py`
4. Update corresponding experiment runners
5. Update data objects and result handlers

### Infrastructure Compliance

- **ALWAYS** inherit from `BaseRunner` for experiment runners
- **NEVER** create direct dependencies between runners
- **ALWAYS** use `ResultBank` for accessing experiment outputs
- **NEVER** hardcode paths or constants outside `src/core/`

### Knockout Analysis

- Model-specific knockout code must go in the appropriate architecture directory
- **NEVER** mix model-specific code in shared experiment files
- Use established patterns for attention/state knockout implementation

### Code Organization

- Utilities go in `src/utils/` hierarchy
- Analysis and plotting code in `src/analysis/`
- Data processing in `src/data_ingestion/`
- **NEVER** create utilities outside the established hierarchy

## Environment Requirements

### Required for All Installations
- **Python 3.12** (required - not compatible with other versions)
- **UV package manager** (required - pip not supported)

### GPU Installation (Recommended for Performance)
- **Linux** (required for causal-conv1d compilation)
- **NVIDIA GPU** with CUDA 11.6+
- **CUDA drivers and toolkit**

### CPU-Only Installation (Analysis/Visualization Mode)
- **Any OS** (Linux, macOS, Windows)
- **No GPU required**
- **Limited functionality**: Analysis, plotting, and development tools only
- **No mamba-ssm**: Mamba model experiments will not be available

## Troubleshooting

### Common Issues

1. **CUDA Compatibility**: If `causal_conv1d` fails, patch the `__init__.py` file as documented in README.md
2. **Streamlit-Pydantic**: Replace `BaseSettings` import as documented in README.md
3. **HuggingFace Authentication**: Set `HUGGINGFACE_TOKEN` environment variable if needed

### Installation Problems

- Always use `scripts/install.sh` for automated setup (detects CUDA automatically)
- Use `UV_TORCH_BACKEND=auto uv sync` for automatic PyTorch backend detection
- Check Python version compatibility (must be 3.12)
- For CPU-only systems: Use `uv sync --extra cpu` to avoid CUDA dependencies
- For GPU systems: Use `uv sync --extra gpu` to include causal-conv1d optimization

## Documentation

Comprehensive documentation in `docs/`:

- Read `docs/README.md` for complete navigation
- See `shrimp-rules.md` for high-level coordination rules
- Check model-specific documentation for architecture details
