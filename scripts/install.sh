#!/bin/bash

# Self-contained installation script for SSM Analysis
# This script sets up the complete environment with all required patches
# Requirements: Only bash and curl (no admin permissions needed)

set -euo pipefail  # Exit on any error, undefined variables, and pipe failures

# Trap for cleanup on script exit
trap cleanup EXIT

# Cleanup function
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "❌ Installation failed with exit code $exit_code"
        echo ""
        echo "🔧 Troubleshooting tips:"
        echo "• Check your internet connection"
        echo "• Ensure you have sufficient disk space (>2GB free)"
        echo "• Verify Python 3.12 is properly installed"
        echo "• Try running the script again (may be a temporary network issue)"
        echo ""
        echo "📞 For help:"
        echo "• Check README.md for detailed installation instructions"
        echo "• Review the error messages above"
        echo "• Consider manual installation steps"
        
        if [ -d ".venv" ]; then
            echo ""
            echo "🧹 Partial installation detected. To clean up:"
            echo "   rm -rf .venv"
        fi
    fi
}

# Enable debug mode if DEBUG=1 environment variable is set
if [ "${DEBUG:-}" = "1" ]; then
    set -x
    echo "🐛 Debug mode enabled"
fi

echo "🚀 Starting SSM Analysis self-contained installation..."
echo "📋 This script will:"
echo "   • Check system requirements"
echo "   • Install UV package manager (if needed)"
echo "   • Detect your system (OS, architecture, CUDA)"
echo "   • Create isolated Python environment"
echo "   • Install appropriate dependencies (GPU or CPU mode)"
echo "   • Apply necessary patches"
echo ""

# System requirements check
check_system_requirements() {
    echo "🔍 Checking system requirements..."
    
    # Check available disk space (in GB) - make it more robust
    local available_space="unknown"
    case "$OS_TYPE" in
        darwin)
            available_space=$(df -g . 2>/dev/null | awk 'NR==2 {print $4}' 2>/dev/null || echo "unknown")
            ;;
        linux)
            available_space=$(df --block-size=1G . 2>/dev/null | awk 'NR==2 {print $4}' 2>/dev/null || echo "unknown")
            ;;
        *)
            available_space="unknown"
            ;;
    esac
    
    if [ "$available_space" != "unknown" ] && [ "$available_space" -gt 0 ] && [ "$available_space" -lt 3 ]; then
        echo "⚠️  Low disk space detected: ${available_space}GB available"
        echo "   Installation requires at least 3GB free space"
        echo "   Consider freeing up space or continue at your own risk"
        echo "   (Press Enter to continue anyway, Ctrl+C to cancel)"
        read -r
    elif [ "$available_space" != "unknown" ] && [ "$available_space" -gt 0 ]; then
        echo "✅ Disk space: ${available_space}GB available"
    else
        echo "ℹ️  Disk space: Unable to check (proceeding anyway)"
    fi
    
    # Check internet connectivity
    if command -v curl &> /dev/null; then
        if ! curl -s --connect-timeout 5 https://pypi.org > /dev/null; then
            echo "❌ No internet connection to PyPI"
            echo "   Installation requires internet access"
            exit 1
        fi
        echo "✅ Internet connectivity: OK"
    fi
    
    echo ""
}

# Detect system information
OS_TYPE=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
echo "🔍 Detected system: $OS_TYPE-$ARCH"

# Run system requirements check
check_system_requirements

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "❌ Python 3.12 is required but not found."
    echo "Please install Python 3.12 first:"
    case "$OS_TYPE" in
        darwin)
            echo "  macOS: brew install python@3.12"
            ;;
        linux)
            echo "  Ubuntu/Debian: sudo apt install python3.12"
            echo "  CentOS/RHEL: sudo dnf install python3.12"
            ;;
        *)
            echo "  Visit: https://www.python.org/downloads/"
            ;;
    esac
    exit 1
fi

# Auto-install UV if not available (self-contained, no admin rights)
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV package manager..."
    if command -v curl &> /dev/null; then
        # Official UV installer (installs to ~/.cargo/bin or ~/.local/bin)
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Source the environment to get UV in PATH
        if [ -f "$HOME/.cargo/env" ]; then
            source "$HOME/.cargo/env"
        fi
        # Add to PATH for this session
        export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
        
        # Verify installation
        if ! command -v uv &> /dev/null; then
            echo "❌ UV installation failed. Trying alternative method..."
            # Fallback: try with python
            python3.12 -m pip install --user uv
            export PATH="$HOME/.local/bin:$PATH"
        fi
    else
        echo "❌ curl not found. Installing UV with pip..."
        python3.12 -m pip install --user uv
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Final verification
    if ! command -v uv &> /dev/null; then
        echo "❌ Failed to install UV. Please install manually:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo "✅ UV installed successfully"
else
    echo "✅ UV already available"
fi

echo "📋 UV version: $(uv --version)"

# Create virtual environment
echo "🔧 Creating virtual environment..."
uv venv --python 3.12

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Enhanced CUDA and platform detection
echo "🔍 Performing comprehensive system detection..."

# Initialize variables
CUDA_AVAILABLE=false
CUDA_VERSION=""
GPU_COUNT=0
INSTALL_MODE="cpu"

# Platform-specific CUDA detection
echo "🔍 Checking CUDA availability..."
if [[ "$OS_TYPE" == "linux" && "$ARCH" =~ ^(x86_64|amd64)$ ]]; then
    echo "   Platform supports CUDA: Linux x86_64"
    
    # Method 1: nvidia-smi (most reliable for runtime)
    if command -v nvidia-smi &> /dev/null; then
        echo "   ✅ nvidia-smi found"
        GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
        if [ "$GPU_COUNT" -gt 0 ]; then
            CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "unknown")
            CUDA_AVAILABLE=true
            echo "   ✅ Found $GPU_COUNT GPU(s), driver version: $CUDA_VERSION"
        fi
    fi
    
    # Method 2: CUDA toolkit (for development)
    if command -v nvcc &> /dev/null; then
        echo "   ✅ NVCC compiler found"
        NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' || echo "unknown")
        echo "   ✅ CUDA toolkit version: $NVCC_VERSION"
        CUDA_AVAILABLE=true
    fi
    
    # Method 3: CUDA installation directories
    for cuda_dir in "/usr/local/cuda" "/opt/cuda" "/usr/cuda" "/cuda"; do
        if [ -d "$cuda_dir" ]; then
            echo "   ✅ CUDA installation found: $cuda_dir"
            CUDA_AVAILABLE=true
            break
        fi
    done
    
    # Method 4: CUDA libraries in system
    if ldconfig -p 2>/dev/null | grep -q "libcuda\|libcudart"; then
        echo "   ✅ CUDA runtime libraries found in system"
        CUDA_AVAILABLE=true
    fi
    
    # Method 5: Check /proc/driver/nvidia (if nvidia module loaded)
    if [ -d "/proc/driver/nvidia" ]; then
        echo "   ✅ NVIDIA kernel module loaded"
        CUDA_AVAILABLE=true
    fi
    
elif [[ "$OS_TYPE" == "linux" ]]; then
    echo "   ⚠️  Linux detected but architecture ($ARCH) may not support CUDA"
    echo "   CUDA requires x86_64 architecture"
else
    echo "   ℹ️  Platform: $OS_TYPE-$ARCH (CUDA not supported)"
    case "$OS_TYPE" in
        darwin)
            echo "   macOS: CUDA support deprecated, use Metal or CPU"
            ;;
        cygwin*|mingw*|msys*)
            echo "   Windows: CUDA available but not supported by this script"
            ;;
        *)
            echo "   Other OS: CPU-only installation"
            ;;
    esac
fi

# Determine installation mode
if [ "$CUDA_AVAILABLE" = true ]; then
    INSTALL_MODE="gpu"
    echo "🚀 CUDA Detection Result: GPU mode enabled"
    echo "   • Will install mamba-ssm with CUDA optimization"
    echo "   • Will install causal-conv1d for performance"
    if [ -n "$GPU_COUNT" ] && [ "$GPU_COUNT" -gt 0 ]; then
        echo "   • $GPU_COUNT GPU(s) detected and ready"
    fi
else
    INSTALL_MODE="cpu"
    echo "💻 CUDA Detection Result: CPU-only mode"
    echo "   • Will skip mamba-ssm (requires CUDA)"
    echo "   • Analysis and visualization components will be available"
    echo "   • Use minimal mamba implementations for development"
fi

echo ""

# Install dependencies based on detected mode
echo "📦 Installing project dependencies in $INSTALL_MODE mode..."

# Set UV backend based on detection
if [ "$INSTALL_MODE" = "gpu" ]; then
    echo "🚀 Installing with GPU support..."
    export UV_TORCH_BACKEND=auto
    
    # Install base dependencies first
    if ! uv sync --extra typing --extra streamlit --extra dev --extra cpu; then
        echo "❌ Failed to install base dependencies"
        exit 1
    fi
    
    # Install GPU-specific dependencies with enhanced error handling
    echo "🚀 Adding GPU optimization packages..."
    if ! uv sync --extra gpu; then
        echo "⚠️  GPU package installation failed, falling back to CPU mode..."
        echo "   This might be due to:"
        echo "   • CUDA toolkit version mismatch"
        echo "   • Missing development headers"
        echo "   • Insufficient system resources"
        echo ""
        echo "   Continuing with CPU-only installation..."
        INSTALL_MODE="cpu"
        CUDA_AVAILABLE=false
    else
        echo "✅ GPU packages installed successfully"
    fi
else
    echo "💻 Installing CPU-only dependencies..."
    export UV_TORCH_BACKEND=cpu
    
    if ! uv sync --extra typing --extra streamlit --extra dev --extra cpu; then
        echo "❌ Failed to install CPU dependencies"
        echo "This is unexpected - CPU installation should always work"
        echo "Please check:"
        echo "• Internet connection"
        echo "• Disk space"
        echo "• Python 3.12 installation"
        exit 1
    fi
    echo "✅ CPU packages installed successfully"
fi

# Apply patches
echo "🔧 Applying required patches..."

# Patch streamlit-pydantic for Pydantic v2 compatibility
STREAMLIT_PYDANTIC_FILE=".venv/lib/python3.12/site-packages/streamlit_pydantic/settings.py"
if [ -f "$STREAMLIT_PYDANTIC_FILE" ]; then
    echo "🔧 Patching streamlit-pydantic..."
    sed -i '' 's/from pydantic import BaseSettings/from pydantic_settings import BaseSettings/' "$STREAMLIT_PYDANTIC_FILE"
    echo "✅ streamlit-pydantic patch applied"
else
    echo "⚠️  streamlit-pydantic not found, skipping patch"
fi

# Patch causal_conv1d for CUDA compatibility (only if installed)
CAUSAL_CONV1D_FILE=".venv/lib/python3.12/site-packages/causal_conv1d/__init__.py"
if [ -f "$CAUSAL_CONV1D_FILE" ]; then
    echo "🔧 Patching causal_conv1d for graceful fallback..."
    cat > "$CAUSAL_CONV1D_FILE" << 'EOF'
__version__ = "1.5.0.post8"
import torch
try:
    if torch.cuda.is_available() and (torch.cuda.get_device_capability() > (6,1)):
        print("Using causal_conv1d CUDA kernels")
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_update
    else:
        print("Not using causal_conv1d (CUDA not available or capability < 6.1)")
        causal_conv1d_fn, causal_conv1d_update = None, None
except (ImportError, RuntimeError):
    print("causal_conv1d not available, using fallback")
    causal_conv1d_fn, causal_conv1d_update = None, None
EOF
    echo "✅ causal_conv1d patch applied"
else
    echo "ℹ️  causal_conv1d not installed (CPU-only mode)"
fi

# Test the installation
echo "🧪 Testing the installation..."
python -c "
import torch
from src.core.types import Float

print('✅ Core imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Test mamba-ssm availability
try:
    import mamba_ssm
    print('✅ mamba-ssm available (GPU mode)')
    
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name()}')
        print(f'CUDA capability: {torch.cuda.get_device_capability()}')
        print('🚀 GPU mode: Fast CUDA kernels available')
        
        # Test causal_conv1d if available
        try:
            import causal_conv1d
            print('✅ causal_conv1d available for optimized performance')
        except ImportError:
            print('⚠️  causal_conv1d not available (will use slower fallback)')
    else:
        print('⚠️  mamba-ssm available but CUDA not detected')
        
except ImportError:
    print('ℹ️  mamba-ssm not available (CPU-only mode)')
    print('💻 CPU mode: Analysis and visualization components available')
    print('⚠️  Mamba model experiments will not work without GPU installation')

print('✅ Installation test completed successfully!')
"

echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Installation Summary:"
echo "   • System: $OS_TYPE-$ARCH"
echo "   • Mode: $INSTALL_MODE"
if [ "$INSTALL_MODE" = "gpu" ]; then
    echo "   • CUDA: Available"
    if [ -n "$GPU_COUNT" ] && [ "$GPU_COUNT" -gt 0 ]; then
        echo "   • GPUs: $GPU_COUNT detected"
    fi
    if [ -n "$CUDA_VERSION" ]; then
        echo "   • Driver: $CUDA_VERSION"
    fi
    echo "   • Components: Full functionality with mamba-ssm"
    echo "🚀 GPU-optimized installation complete!"
else
    echo "   • CUDA: Not available"
    echo "   • Components: Analysis, visualization, development tools"
    echo "   • Note: Mamba model experiments require GPU installation"
    echo "💻 CPU-only installation complete!"
fi

echo ""
echo "🔧 Next Steps:"
echo "1. Activate the environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Verify installation:"
if [ "$INSTALL_MODE" = "gpu" ]; then
    echo "   python -c \"import torch; import mamba_ssm; print('✅ GPU setup ready')\""
else
    echo "   python -c \"import torch; print('✅ CPU setup ready')\""
fi
echo ""
echo "3. Run tests:"
echo "   python tests/src/experiments/test_full_pipeline.py"
echo ""
if [ "$INSTALL_MODE" = "cpu" ]; then
    echo "ℹ️  To enable GPU support later:"
    echo "   1. Install CUDA toolkit on Linux x86_64"
    echo "   2. Run: uv sync --extra gpu"
    echo ""
fi
echo "📚 For more information, see README.md and docs/"
