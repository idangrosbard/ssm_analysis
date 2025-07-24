---
description: Setup and Environment Management
inclusion: manual
---

# Setup and Environment Management

## Package Management

### UV Package Manager (Required)
- **ALWAYS use UV package manager** instead of pip
- **Installation**: `chmod +x scripts/install.sh && ./scripts/install.sh`
- **Sync**: `uv sync --extra typing --extra streamlit --extra dev --extra gpu`

### Environment Requirements
- **Python 3.12** (strict requirement - not compatible with other versions)
- **GPU**: Linux x86_64 with NVIDIA GPU and CUDA 11.6+
- **CPU-only**: Any OS (limited functionality - analysis only)

## Common Commands

### Package Management
```bash
uv add package_name                    # Add new dependency
uv sync                               # Install all dependencies
uv sync --extra gpu                   # Install with GPU support
uv sync --extra cpu                   # Install CPU-only mode
```

### Code Quality
```bash
ruff check src/ tests/                # Linting
ruff check --fix src/ tests/          # Auto-fix linting issues
mypy src/ tests/                      # Type checking
pytest -vv                            # Run tests
```

### Development
```bash
streamlit run src/app/entry_point.py  # Start web interface
python -m src.experiments.runners.evaluate_model  # Run experiments
```

## Configuration Files

### Primary Configuration
- **pyproject.toml** - Main project configuration and dependencies
- **uv.cuda.lock** / **uv.cpu.lock** - Reproducible dependency locks
- **UV_TORCH_BACKEND=auto** - Automatic PyTorch backend detection

### Build Constraints
- **mamba-ssm**: Requires torch as build dependency, no-build-isolation needed
- **causal-conv1d**: Requires CUDA toolkit at build time
- **Python 3.12**: Strict version requirement for all components

## Critical Rules

### Environment Rules
- **NEVER use pip** - UV package manager only
- **ALWAYS use Python 3.12** - strict version requirement
- **CHECK GPU compatibility** before GPU installation

### Reproducibility Requirements
- **USE lock files** for reproducible builds
- **SET environment variables** as specified
- **FOLLOW installation scripts** exactly

**Reference**: docs/setup-and-environment.md for complete setup procedures
