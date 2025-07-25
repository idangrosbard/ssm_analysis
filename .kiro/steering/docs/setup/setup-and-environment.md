---
description: Package Management and Environment Setup
inclusion: manual
---

# Setup and Environment

Environment setup rules and file references for reproducible package management and configuration.

## Package Management Rules

### UV Package Manager Requirements

**ALWAYS use `uv` command instead of `pip` for package operations**

```bash
# Correct package management commands
uv add transformers==4.50.3
uv add -e .[streamlit]
uv sync

# NEVER use pip directly
# WRONG: pip install transformers
```

### Key UV Commands

```bash
# Create virtual environment
uv venv --python 3.12

# Install dependencies
uv sync

# Add new dependencies
uv add package_name==version

# Install optional dependency groups
uv sync --extra typing --extra streamlit --extra dev

# Install in editable mode
uv add -e .
```

### Dependency Management Rules

#### Version Requirements

- **MUST maintain Python 3.12 compatibility requirement**
- **MUST ensure CUDA compatibility for mamba-ssm dependencies**
- **ALWAYS specify exact version numbers in `pyproject.toml`**

#### Adding Dependencies

**When adding dependencies**: Update `pyproject.toml` AND installation documentation in README.md

```toml
# Example: Adding to pyproject.toml
[project]
dependencies = [
    "new-package==1.2.3",  # Exact version required
    # ... existing dependencies
]

[project.optional-dependencies]
new_group = [
    "optional-package==4.5.6",
]
```

#### CUDA Compatibility

```bash
# Install PyTorch with CUDA support
UV_TORCH_BACKEND=auto uv sync

# Or specify CUDA version manually
uv add "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
```

## Configuration and Environment Standards

### Environment Setup Rules

#### Automated Installation

**ALWAYS use `scripts/install.sh` for automated installation**

```bash
# Make script executable
chmod +x scripts/install.sh

# Run automated installation
./scripts/install.sh
```

The installation script automatically:

1. **Checks Python 3.12 availability**
2. **Installs UV if not present**
3. **Creates virtual environment with Python 3.12**
4. **Installs all dependencies with automatic PyTorch backend detection**
5. **Applies required patches for compatibility**
6. **Tests the installation**

#### Manual Installation (Advanced)

If automated installation fails, follow manual steps:

```bash
# 1. Install UV
pip install uv

# 2. Create virtual environment
uv venv --python 3.12

# 3. Activate environment
source .venv/bin/activate

# 4. Install with automatic backend detection
UV_TORCH_BACKEND=auto uv sync

# 5. Install optional dependencies
uv sync --extra typing --extra streamlit --extra dev
```

#### System Requirements

- **MUST maintain Linux/CUDA requirements for mamba-ssm**
- **Python 3.12 required**
- **CUDA-compatible GPU recommended for full functionality**

## Reproducibility Requirements

### Documentation Coordination

#### Update Requirements

- **When modifying project structure**: MUST update README.md to reflect changes
- **When adding new installation steps**: Update both `scripts/install.sh` AND README.md
- **When changing dependencies**: Update installation documentation immediately
- **ALWAYS maintain README.md alignment with actual project structure**

#### Repository Maintenance Rules

- **CONTINUOUSLY update README.md to reflect structural changes**
- **MAINTAIN reproducible environment documentation**
- **NEVER leave documentation inconsistent with actual implementation**

### Version Control

#### Dependency Locking

```bash
# Generate lock file for reproducible builds
uv lock

# Install from lock file
uv sync --frozen
```

#### Environment Files

```bash
# Export environment for sharing
uv export --format requirements.txt > requirements.txt

# Install from requirements
uv pip install -r requirements.txt
```

## Critical Warnings

⚠️ **NEVER use pip instead of uv for package management**

⚠️ **NEVER bypass the installation script for dependency setup**

⚠️ **ALWAYS maintain Python 3.12 compatibility requirement**

⚠️ **ALWAYS update documentation when changing installation procedures**

## Cross-References

### Related Documentation

- **Core Modules**: Use #core-modules for constant definitions
- **Utilities**: Use #utilities-and-patterns for utility organization
- **Infrastructure**: Use #infrastructure for base class patterns
- **Experiment Runners**: Use #experiment-runners for experiment setup
- **Data Interfaces**: Use #data-relationships-interface for data file management
- **Main README**: See README.md for detailed installation instructions

### Dependencies

- **pyproject.toml**: Package configuration and dependency specifications
- **scripts/install.sh**: Automated installation script with patches
- **README.md**: Installation instructions and troubleshooting guide
