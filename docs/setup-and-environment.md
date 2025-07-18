# Setup and Environment

This document covers package management, configuration, and environment setup requirements for the project. These standards are critical for project reproducibility and ensure consistent environment setup across different development machines.

## Overview

The project uses sophisticated package management and configuration patterns to ensure reproducibility and maintainability. All setup procedures follow strict standards to guarantee consistent behavior across different development environments.

## Package Management Standards

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

#### Key UV Commands

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

### Configuration File Rules

**ALL configuration files MUST be symlinked from home directory (`~`) to repo files**

#### Symlink Pattern

```bash
# Create configuration in repository
mkdir -p configs/
echo "setting=value" > configs/my_config.conf

# Create symlink from home directory
ln -s ~/path/to/repo/configs/my_config.conf ~/.my_config.conf
```

#### Configuration Management

- **NEVER create direct configuration files in repository**
- **ALWAYS maintain symlink structure for reproducibility**
- **When adding new configuration**: Create in repo, symlink from home directory

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

## Installation Procedures

### Standard Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd ssm_analysis_public

# 2. Run automated installation
chmod +x scripts/install.sh
./scripts/install.sh

# 3. Activate environment
source .venv/bin/activate

# 4. Verify installation
python -c "
import torch
import mamba_ssm
from src.core.types import Float
print('✅ Installation successful!')
"
```

### Optional Dependencies

```bash
# Install type checking tools
uv sync --extra typing

# Install Streamlit web interface
uv sync --extra streamlit

# Install development tools
uv sync --extra dev

# Install all optional dependencies
uv sync --extra typing --extra streamlit --extra dev
```

### CUDA Setup

```bash
# Check CUDA availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'CUDA capability: {torch.cuda.get_device_capability()}')
"
```

## Troubleshooting

### Common Issues

#### Python Version Issues

```bash
# Check Python version
python --version

# Install Python 3.12 if needed
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12 python3.12-venv

# macOS
brew install python@3.12
```

#### CUDA Compatibility Issues

If you encounter CUDA errors:

```bash
# Check CUDA version
nvidia-smi

# Install appropriate PyTorch version
# For CUDA 11.8
uv add "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv add "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
```

#### Mamba-SSM Installation Issues

```bash
# Install with no build isolation
uv add "mamba-ssm[causal-conv1d]==2.2.4" --no-build-isolation

# Apply causal_conv1d patch if needed
# See scripts/install.sh for patch details
```

#### Streamlit-Pydantic Compatibility

```bash
# Apply Pydantic v2 compatibility patch
# The installation script handles this automatically
# Manual patch if needed:
sed -i 's/from pydantic import BaseSettings/from pydantic_settings import BaseSettings/' \
    .venv/lib/python3.12/site-packages/streamlit_pydantic/settings.py
```

### Advanced Troubleshooting

#### Manual Dependency Resolution

```bash
# Install dependencies one by one
uv add torch==2.5.1
uv add transformers==4.50.3
uv add mamba-ssm[causal-conv1d]==2.2.4 --no-build-isolation
uv add -e .
```

#### Environment Cleanup

```bash
# Remove virtual environment
rm -rf .venv

# Recreate environment
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

## Best Practices

### Package Management

1. **Always use UV**: Never use pip directly for package operations
2. **Specify exact versions**: Use exact version numbers in pyproject.toml
3. **Use dependency groups**: Organize optional dependencies into logical groups
4. **Test installations**: Verify installations work on clean environments

### Configuration Management

1. **Use symlinks**: Always symlink configuration files from home directory
2. **Version control configs**: Keep configuration templates in repository
3. **Document changes**: Update documentation when changing configuration patterns
4. **Test configurations**: Verify configurations work across different environments

### Reproducibility

1. **Lock dependencies**: Use lock files for reproducible builds
2. **Document requirements**: Keep installation documentation up to date
3. **Test on clean environments**: Verify installation works on fresh systems
4. **Maintain compatibility**: Ensure compatibility with specified Python version

## Critical Warnings

⚠️ **NEVER use pip instead of uv for package management**

⚠️ **NEVER create direct configuration files in repository - always use symlinks**

⚠️ **NEVER bypass the installation script for dependency setup**

⚠️ **ALWAYS maintain Python 3.12 compatibility requirement**

⚠️ **ALWAYS update documentation when changing installation procedures**

## Cross-References

### Related Documentation

- **Core Modules**: See [docs/core-modules.md](core-modules.md) for constant definitions
- **Utilities**: See [docs/utilities-and-patterns.md](utilities-and-patterns.md) for utility organization
- **Main README**: See README.md for detailed installation instructions

### Dependencies

- **pyproject.toml**: Package configuration and dependency specifications
- **scripts/install.sh**: Automated installation script with patches
- **README.md**: Installation instructions and troubleshooting guide 
