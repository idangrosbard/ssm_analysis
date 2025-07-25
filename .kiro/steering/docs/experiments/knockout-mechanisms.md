---
description: Model Intervention and Knockout Implementations
inclusion: manual
---

# Knockout Mechanisms Documentation

## Overview

Knockout mechanisms provide a sophisticated system for selectively disabling or modifying specific components within language models to analyze their behavior and understand information flow. This system supports three major model architectures: GPT2, Llama (including Mistral and Qwen2 variants), and Mamba (both Mamba1 and Mamba2).

The knockout system operates through hook-based interventions that can:

- Block attention connections between specific tokens
- Disable specific SSM (State Space Model) features in Mamba architectures
- Provide fine-grained control over model information flow for analysis

## Directory Structure

```
src/experiments/knockout/
├── __init__.py
├── gpt/
│   └── gpt2/
│       └── gpt2_knockout_utils.py      # GPT2 attention masking
├── llama/
│   ├── llama_attn.py                   # Llama attention knockout
│   ├── llama_attention_forward.py      # Llama attention forward
│   ├── scaled_dot_product_attention.py # SDPA attention patterns
│   ├── sdpa_attention.py               # SDPA implementation
│   └── interfere_hook.py               # Llama interference hooks
└── mamba/
    ├── mamba1/
    │   ├── helpers/
    │   │   └── ssm_interfere.py        # SSM state interference
    │   ├── original_variant.py         # Original Mamba knockout
    │   └── falcon_variant.py           # Falcon-Mamba knockout
    └── mamba2/
        └── minimal_mamba2.py           # Mamba2 knockout (copied implementation with modifications)
```

## Model Interface Integration

Knockout system integrates through `ModelInterface` in `src/experiments/infrastructure/model_interface.py`:

```python
class ModelInterface(ABC):
    @abstractmethod
    def generate_logits(
        self,
        input_ids: torch.Tensor,
        num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]],
        feature_category: FeatureCategory,
    ) -> torch.Tensor:
        """Generate logits with optional attention masking"""
        pass
```

## num_to_masks Parameter

The `num_to_masks` parameter is a dictionary that maps layer numbers to lists of (source_index, target_index) tuples.

```python
num_to_masks = {
    0: [(1, 0), (2, 1)],  # Layer 0: block 1←0, 2←1
    1: [(3, 2)],           # Layer 1: block 3←2
    2: [(4, 3)]            # Layer 2: block 4←3
}
```

**Semantics**: `(source_index, target_index)` means that `source_index` won't receive information from `target_index`. This blocks the attention connection from target to source token.

## Implementation Methods

### PyTorch Forward Hooks (GPT2, Llama, Mamba1)

Most knockout implementations use PyTorch's `register_forward_hook()` to intercept and modify the forward pass:

```python
# Register hook on specific module
handle = module.register_forward_hook(hook_function)

# Hook function signature
def hook_function(module, input, output):
    # Modify input or output as needed
    return modified_output  # or None to keep original output

# Cleanup
handle.remove()
```

### Direct Implementation (Mamba2)

Mamba2 uses a copied implementation approach where the entire model code is copied and modified directly in `src/experiments/knockout/mamba/mamba2/minimal_mamba2.py`.

## Submodule Knockout Implementations

### GPT2 Attention Masking

**File**: `src/experiments/knockout/gpt/gpt2/gpt2_knockout_utils.py`

**Method**: PyTorch forward hooks on attention layers

**Original Flow** (from `.venv/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py`):
1. GPT2 attention computes query, key, value matrices
2. Calculates attention weights via `torch.matmul(query, key.transpose(-1, -2))`
3. Applies causal mask to enforce autoregressive structure
4. Applies softmax to get attention probabilities
5. Computes weighted sum of values

**Hook Modification**:
- **Interception Point**: Hooks are registered on `model.transformer.h[i].attn.forward`
- **Modification**: Creates binary attention masks that block specific token-to-token connections
- **Implementation**: Converts blocking patterns to large negative values in attention weights
- **Result**: Selected attention connections are effectively zeroed out, preventing information flow

**Key Changes**:
- Wraps the original forward function with a custom wrapper
- Injects attention masks with large negative values for blocked connections
- Maintains causal structure while allowing selective blocking

### Llama Attention Knockout

**File**: `src/experiments/knockout/llama/llama_attn.py`

**Method**: Custom attention module wrapper

**Original Flow** (from `.venv/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py`):
1. Llama attention projects input to Q, K, V matrices
2. Computes attention weights with scaling factor
3. Applies causal attention mask
4. Applies softmax to get attention probabilities
5. Computes weighted sum of values

**Hook Modification**:
- **Interception Point**: Replaces original attention modules with `LlamaAttentionKnockout` wrapper
- **Modification**: Creates boolean masks that block specific attention connections
- **Implementation**: Applies logical AND between original attention mask and knockout mask
- **Result**: Selected attention connections are blocked while preserving causal structure

**Key Changes**:
- Wraps the entire attention module instead of just the forward function
- Creates boolean masks for cleaner blocking semantics
- Supports multiple attention variants (Llama, Mistral, Qwen2)

### Mamba1 SSM Interference

**File**: `src/experiments/knockout/mamba/mamba1/helpers/ssm_interfere.py`

**Method**: PyTorch forward hooks on SSM layers

**Original Flow** (from `.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py`):
1. Mamba SSM processes input through linear projections
2. Applies convolution operation for local interactions
3. Computes SSM state updates using A, B, C, D parameters
4. Generates output through state-to-output mapping
5. Applies residual connections and normalization

**Hook Modification**:
- **Interception Point**: Hooks are registered on SSM mixer layers
- **Modification**: Intercepts SSM computation and applies feature masking
- **Implementation**: Uses custom forward functions that modify SSM state computation
- **Result**: Selected SSM features are disabled, affecting information processing

**Key Changes**:
- Intercepts the entire SSM forward pass
- Applies feature masks to disable specific SSM components
- Supports both original Mamba and Falcon-Mamba variants

### Mamba2 Direct Implementation

**File**: `src/experiments/knockout/mamba/mamba2/minimal_mamba2.py`

**Method**: Copied and modified implementation

**Original Flow** (from minimal Mamba2 implementation):
1. Mamba2 uses Structured State Space Duality (SSD)
2. Computes attention matrix from B and C projections
3. Applies exponential decay for state transitions
4. Combines intra-chunk and inter-chunk computations
5. Generates output through state-to-output mapping

**Hook Modification**:
- **Interception Point**: Direct modification of `ssd()` function
- **Modification**: Applies knockout masks directly to attention matrix computation
- **Implementation**: Zeroes out specific attention matrix entries
- **Result**: Selected attention connections are blocked at the SSD level

**Key Changes**:
- No hooks - direct code modification approach
- Modifies the core SSD algorithm directly
- Applies masks during attention matrix computation

## Mamba1 Feature Knockout

Mamba1 implements sophisticated feature masking based on SSM decay characteristics:

**File**: `src/experiments/infrastructure/model_interface.py` (Mamba1Interface._get_feature_mask)

**Original Flow**:
1. SSM uses decay matrices to control state transitions
2. Each feature has different decay characteristics
3. Fast-decay features process local information
4. Slow-decay features maintain long-range dependencies

**Hook Modification**:
- **Analysis**: Analyzes SSM decay matrices to determine feature importance
- **Classification**: Identifies fast/slow decay features based on norm calculations
- **Selection**: Creates feature masks based on `FeatureCategory` (SLOW_DECAY, FAST_DECAY, ALL)
- **Result**: Selectively disables features based on their temporal characteristics

**Key Changes**:
- Calculates feature importance using decay matrix norms
- Sorts features by decay characteristics
- Selects top/bottom third based on feature category
- Applies masks to disable selected features

## Third-Party Library Access

To understand or modify knockout implementations, you need to access the third-party library implementations in the virtual environment:

### Hugging Face Transformers
- **GPT2**: `.venv/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py`
- **Llama**: `.venv/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py`
- **Mistral**: `.venv/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py`
- **Qwen2**: `.venv/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py`
- **Mamba**: `.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py`

### PyTorch Hooks
- **Forward Hooks**: `torch.nn.Module.register_forward_hook()` - Registers forward hook called after `forward()`
- **Hook Management**: `torch.utils.hooks.RemovableHandle` - Provides capability to remove hooks with cleanup
- **Hook Cleanup**: Proper removal to prevent memory leaks using `RemovableHandle.remove()`

**Important**: When making changes or understanding implementations, always reference the third-party library code in `.venv/` to understand the original architecture and ensure compatibility.

## Usage Patterns

### Basic Knockout Usage

```python
from src.experiments.infrastructure.model_interface import get_model_interface

# Get model interface
interface = get_model_interface(MODEL_ARCH_AND_SIZE.GPT2_MEDIUM)

# Define knockout pattern: block token 5 from receiving info from token 3 in layer 2
num_to_masks = {2: [(5, 3)]}

# Generate logits with knockout
logits = interface.generate_logits(input_ids, num_to_masks=num_to_masks)
```

### Feature-Based Knockout (Mamba)

```python
from src.core.types import FeatureCategory

# Knockout slow-decay features in specific layers
interface.setup(layers=[0, 1, 2])
logits = interface.generate_logits(
    input_ids, 
    feature_category=FeatureCategory.SLOW_DECAY
)
```

### Multi-Layer Knockout

```python
# Block multiple connections across layers
num_to_masks = {
    0: [(1, 0), (2, 1)],  # Layer 0: block 1←0, 2←1
    1: [(3, 2)],           # Layer 1: block 3←2
    2: [(4, 3)]            # Layer 2: block 4←3
}

logits = interface.generate_logits(input_ids, num_to_masks=num_to_masks)
```

## Cross-References

- **Infrastructure Documentation**: Use #infrastructure for ModelInterface usage patterns
- **Core Types**: See `src/core/types.py` for `KnockoutMode` and `FeatureCategory` definitions
- **Experiment Runners**: Use #experiment-runners for how knockout is used in experiments
- **Core Modules**: Use #core-modules for type definitions and constants
- **Data Interfaces**: Use #data-relationships-interface for data object patterns
- **Setup and Environment**: Use #setup-and-environment for model setup

## Best Practices

1. **Hook Cleanup**: Always remove hooks after use to prevent memory leaks
2. **Device Compatibility**: Ensure knockout masks are on the correct device
3. **Causal Structure**: Maintain causal attention patterns when possible
4. **Feature Selection**: Use feature categories for systematic SSM analysis
5. **Error Handling**: Validate knockout indices to prevent out-of-bounds errors

## Troubleshooting

### Common Issues

1. **Memory Leaks**: Ensure hooks are properly removed using `remove_hooks()`
2. **Device Mismatch**: Move knockout masks to the same device as model tensors
3. **Index Errors**: Validate that knockout indices are within valid ranges
4. **Causal Violations**: Ensure knockout patterns don't violate causal structure

### Debugging Tips

1. **Hook Verification**: Check that hooks are properly registered/removed
2. **Mask Inspection**: Print attention masks to verify blocking patterns
3. **Feature Analysis**: Use feature masks to understand SSM behavior
4. **Gradient Flow**: Verify that gradients flow correctly through knockout layers
