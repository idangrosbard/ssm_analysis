---
inclusion: always
---
# Technology Stack and Build System

## Core Technologies

- **Python 3.12** (required - strict version requirement)
- **PyTorch 2.5.1** with CUDA support for GPU acceleration
- **UV Package Manager** (required - pip not supported)
- **Mamba-SSM 2.2.4** with causal-conv1d for state-space model operations

## Key Libraries and Frameworks

- **Transformers 4.50.3** - HuggingFace model loading and tokenization
- **Streamlit 1.44.1** - Interactive web interface
- **Pandas 2.2.2** - Data manipulation and analysis
- **NumPy 1.26.4** - Numerical computing
- **Matplotlib 3.9.2** + **Seaborn 0.13.2** - Plotting and visualization
- **Plotly 5.24.1** - Interactive visualizations
- **Datasets 2.20.0** - HuggingFace dataset loading
- **Pydantic 2.8.2** - Data validation and settings management

## Development Tools

- **MyPy 1.11.1** - Static type checking
- **Ruff 0.9.2** - Linting and code formatting
- **Pytest 8.3.4** - Testing framework
- **Pre-commit** - Git hooks for code quality
- **Beartype 0.19.0** + **JAXTyping 0.2.36** - Runtime type checking

## Build System and Package Management

### Installation

```bash
# Automated installation (recommended)
chmod +x scripts/install.sh
./scripts/install.sh

# Manual installation
uv venv --python 3.12
source .venv/bin/activate
UV_TORCH_BACKEND=auto uv sync --extra typing --extra streamlit --extra dev --extra gpu
```

### Common Commands

```bash
# Package management
uv add package_name                    # Add new dependency
uv sync                               # Install all dependencies
uv sync --extra gpu                   # Install with GPU support
uv sync --extra cpu                   # Install CPU-only mode

# Code quality
ruff check src/ tests/                # Linting
ruff check --fix src/ tests/          # Auto-fix linting issues
mypy src/ tests/                      # Type checking
pytest -vv                            # Run tests

# Development
streamlit run src/app/entry_point.py  # Start web interface
python -m src.experiments.runners.evaluate_model  # Run experiments
```

## Environment Requirements

### GPU Installation (Recommended)

- **Linux x86_64** (required for mamba-ssm compilation)
- **NVIDIA GPU** with CUDA 11.6+
- **CUDA drivers and toolkit**
- **Python 3.12** (not compatible with other versions)

### CPU-Only Installation (Limited Functionality)

- **Any OS** (Linux, macOS, Windows)
- **No GPU required**
- **Analysis and visualization only** (no mamba model experiments)
- **Python 3.12** required

## Configuration Management

- **pyproject.toml** - Main project configuration and dependencies
- **uv.cuda.lock** / **uv.cpu.lock** - Reproducible dependency locks
- **UV_TORCH_BACKEND=auto** - Automatic PyTorch backend detection
- **.ruff.toml** - Code formatting and linting rules
- **mypy configuration** in pyproject.toml

## Known Issues and Patches

- **streamlit-pydantic**: Requires BaseSettings import patch for Pydantic v2
- **causal_conv1d**: GPU compatibility patch for older CUDA versions
- **HuggingFace authentication**: Set HUGGINGFACE_TOKEN environment variable for private models

## Build Constraints

- **mamba-ssm**: Requires torch as build dependency, no-build-isolation needed
- **causal-conv1d**: Requires CUDA toolkit at build time
- **Python 3.12**: Strict version requirement for all components
