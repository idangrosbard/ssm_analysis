#!/bin/bash

# Auto-installation script for SSM Analysis
# This script sets up the complete environment with all required patches

set -e  # Exit on any error

echo "🚀 Starting SSM Analysis installation..."

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "❌ Python 3.12 is required but not found. Please install Python 3.12 first."
    exit 1
fi

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV..."
    pip install uv
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
uv venv --python 3.12

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install the project with automatic PyTorch backend detection
echo "📦 Installing the project with automatic PyTorch backend detection..."
UV_TORCH_BACKEND=auto uv sync

# Install optional dependencies
echo "📦 Installing optional dependencies..."
uv sync --extra typing --extra streamlit --extra dev

# Apply patches
echo "🔧 Applying required patches..."

# Patch streamlit-pydantic for Pydantic v2 compatibility
STREAMLIT_PYDANTIC_FILE=".venv/lib/python3.12/site-packages/streamlit_pydantic/settings.py"
if [ -f "$STREAMLIT_PYDANTIC_FILE" ]; then
    echo "🔧 Patching streamlit-pydantic..."
    sed -i 's/from pydantic import BaseSettings/from pydantic_settings import BaseSettings/' "$STREAMLIT_PYDANTIC_FILE"
    echo "✅ streamlit-pydantic patch applied"
else
    echo "⚠️  streamlit-pydantic not found, skipping patch"
fi

# Patch causal_conv1d for CUDA compatibility
CAUSAL_CONV1D_FILE=".venv/lib/python3.12/site-packages/causal_conv1d/__init__.py"
if [ -f "$CAUSAL_CONV1D_FILE" ]; then
    echo "🔧 Patching causal_conv1d..."
    cat > "$CAUSAL_CONV1D_FILE" << 'EOF'
__version__ = "1.5.0.post8"
import torch
try:
    if (torch.cuda.get_device_capability() > (6,1)):
        print("Using causal_conv1d")
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_update
    else:
        print("Not using causal_conv1d")
        causal_conv1d_fn, causal_conv1d_update = None, None
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
EOF
    echo "✅ causal_conv1d patch applied"
else
    echo "⚠️  causal_conv1d not found, skipping patch"
fi

# Test the installation
echo "🧪 Testing the installation..."
python -c "
import torch
import mamba_ssm
from src.core.types import Float
print('✅ All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'CUDA capability: {torch.cuda.get_device_capability()}')
"

echo "🎉 Installation completed successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "source .venv/bin/activate"
echo ""
echo "To run tests, use:"
echo "python tests/src/experiments/test_full_pipeline.py"
