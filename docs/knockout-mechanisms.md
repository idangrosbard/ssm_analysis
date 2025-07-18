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
│   ├── __init__.py
│   └── gpt2/
│       ├── __init__.py
│       └── gpt2_knockout_utils.py
├── llama/
│   ├── __init__.py
│   ├── llama_attn.py
│   ├── llama_attention_forward.py
│   ├── scaled_dot_product_attention.py
│   ├── sdpa_attention.py
│   ├── interfere_hook.py
│   └── ___init__.py
└── mamba/
    ├── __init__.py
    ├── mamba1/
    │   ├── __init__.py
    │   ├── helpers/
    │   │   └── ssm_interfere.py
    │   ├── original_variant.py
    │   └── falcon_variant.py
    └── mamba2/
        ├── __init__.py
        └── minimal_mamba2.py
```

## Model Interface Integration

The knockout system is integrated through the `ModelInterface` abstract base class and its concrete implementations in `src/experiments/infrastructure/model_interface.py`. Each model type has a specialized interface that handles knockout operations:

### Base ModelInterface

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

The `num_to_masks` parameter is a dictionary mapping layer numbers to lists of (source_index, target_index) tuples, where the source index won't receive information from the target index.

## GPT2 Knockout Implementation

### Architecture: Attention Masking

GPT2 knockout uses attention masking to block specific attention connections. The implementation is in `src/experiments/knockout/gpt/gpt2/gpt2_knockout_utils.py`.

### Key Components

1. **Hook Registration**: Uses `set_block_attn_hooks()` to register forward hooks on attention layers
2. **Attention Mask Creation**: Creates binary attention masks to block specific token-to-token connections
3. **Mask Application**: Applies masks during the forward pass to prevent information flow

### Implementation Details

Based on the actual Hugging Face transformers implementation in `.venv/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py`, the GPT2 attention mechanism works as follows:

**GPT2Attention Class Structure:**
```python
class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        # Register causal bias buffer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
```

**Attention Forward Function:**
```python
def eager_attention_forward(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    
    # Apply causal mask
    if not module.is_cross_attention:
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    
    # Apply custom attention mask (knockout)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
```

**Knockout Implementation:**
```python
def set_block_attn_hooks(model, from_to_index_per_layer, opposite=False):
    """Register hooks to block attention connections in GPT2"""
    
    def wrap_attn_forward(forward_fn, model_, from_to_index_, opposite_):
        @functools.wraps(forward_fn)
        def wrapper_fn(*args, **kwargs):
            # Create attention mask based on from_to_index pairs
            attn_mask = torch.tril(torch.ones((num_tokens, num_tokens), dtype=torch.uint8))
            for s, t in from_to_index_:
                attn_mask[s, t] = 0  # Block connection from source to target
            
            # Convert to float mask with large negative values
            attn_mask = (1.0 - attn_mask) * torch.finfo(model_.dtype).min
            new_kwargs["attention_mask"] = attn_mask
            return forward_fn(*new_args, **new_kwargs)
        
        return wrapper_fn
```

### Usage in ModelInterface

The `GPT2Interface` class uses the knockout utilities through the `_trace_with_attn_block()` method:

```python
def _trace_with_attn_block(self, model, inp, from_to_index_per_layer):
    """Apply attention blocking to GPT2 model"""
    hooks = gpt2_knockout_utils.set_block_attn_hooks(
        model, from_to_index_per_layer
    )
    # ... process with hooks
    gpt2_knockout_utils.remove_wrapper(model, hooks)
```

## Llama Knockout Implementation

### Architecture: Custom Attention Modules

Llama knockout uses custom attention modules that wrap the original attention layers. The implementation is in `src/experiments/knockout/llama/llama_attn.py`.

### Key Components

1. **LlamaAttentionKnockout**: Custom attention module that wraps original attention layers
2. **Knockout Mask Application**: Applies boolean masks to attention patterns
3. **Multi-Model Support**: Supports Llama, Mistral, and Qwen2 attention variants

### Implementation Details

Based on the actual Hugging Face transformers implementation in `.venv/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py`, the Llama attention mechanism works as follows:

**LlamaAttention Class Structure:**
```python
class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size)
```

**Attention Computation:**
```python
def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor], ...):
    # Project to Q, K, V
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    # Compute attention weights
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    # Apply attention mask (knockout)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
```

**Knockout Implementation:**
```python
class LlamaAttentionKnockout(nn.Module):
    def __init__(self, inner: T_LLAMA_ATTN, knockout_mask: Optional[Iterable[tuple[int, int]]] = None):
        super().__init__()
        self.inner = inner
        self.knockout_mask = knockout_mask

    def forward(self, hidden_states: Tensor, position_embeddings: Tuple[Tensor, Tensor], ...):
        t = hidden_states.shape[1]
        knockout_mask = torch.ones([1, 1, t, t], dtype=torch.bool, device=hidden_states.device)
        
        if self.knockout_mask is not None:
            for q, k in self.knockout_mask:
                knockout_mask[:, :, q, k] = False  # Block connection
        
        # Apply mask to attention computation
        if attention_mask is not None:
            attention_mask = attention_mask.logical_and(knockout_mask)
        else:
            attention_mask = knockout_mask
            
        return self.inner(hidden_states=hidden_states, attention_mask=attention_mask, ...)
```

### Usage in ModelInterface

The `LlamaInterface` class replaces attention modules with knockout versions:

```python
def setup(self, layers: Optional[Iterable[TLayerIndex]] = None):
    # Replace attention modules with knockout versions
    for layer_idx in layers:
        original_attn = self.model.model.layers[layer_idx].self_attn
        knockout_attn = LlamaAttentionKnockout(original_attn, knockout_mask)
        self.model.model.layers[layer_idx].self_attn = knockout_attn
```

## Mamba Knockout Implementation

### Architecture: SSM State Interference

Mamba knockout uses forward hooks to interfere with the State Space Model (SSM) computation. The implementation is in `src/experiments/knockout/mamba/mamba1/helpers/ssm_interfere.py`.

### Key Components

1. **SSMInterfereHook**: Forward hook that intercepts SSM computation
2. **Feature Masking**: Selective disabling of SSM features based on decay characteristics
3. **Multiple Variants**: Support for both original Mamba and Falcon-Mamba variants

### Implementation Details

Based on the actual Hugging Face transformers implementation in `.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py`, the Mamba SSM mechanism works as follows:

**MambaMixer Class Structure:**
```python
class MambaMixer(nn.Module):
    def __init__(self, config: MambaConfig, layer_idx: int):
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = int(config.time_step_rank)
        
        # SSM parameters (A, B, C, D)
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))  # State transition matrix
        self.D = nn.Parameter(torch.ones(self.intermediate_size))  # Output projection
        
        # Linear projections
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2)
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size)
```

**SSM Forward Function:**
```python
def forward(self, hidden_states, cache_params: Optional[MambaCache] = None, cache_position: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.LongTensor] = None):
    if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:
        return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
    return self.slow_forward(hidden_states, cache_params, cache_position, attention_mask)
```

**Knockout Implementation:**
```python
class SSMInterfereHook:
    def __init__(self, layer: int | str | nn.Module, knockout_type: KnockoutMode, is_falcon: bool, feature_mask: Optional[FloatTensor | Tensor] = None):
        self.layer = layer
        self.knockout_type = knockout_type
        self.knockout_indices: Iterable[int] = []
        self.affected_outputs: Iterable[int] = []
        self.feature_mask = feature_mask

    def hook(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        """Intercept SSM computation and apply knockout modifications"""
        slow_forward = (
            slow_forward_for_ssm_materializing_knockout_falcon
            if self.is_falcon
            else slow_forward_for_ssm_materializing_knockout
        )
        
        return slow_forward(
            module,
            inp[0],
            knockout_indices=self.knockout_indices,
            affected_outputs=self.affected_outputs,
            knockout_mode=self.knockout_type,
            knockout_feature_mask=self.feature_mask,
        )
```

### Feature Masking System

Mamba interfaces implement sophisticated feature masking based on SSM decay characteristics:

```python
def _get_feature_mask(self, layer: torch.nn.Module, feature_category: FeatureCategory) -> Tensor:
    decay_matrices = torch.exp(-torch.exp(layer.A_log))
    n_ssms = decay_matrices.shape[0]
    
    # Calculate norms to determine feature importance
    norms = torch.norm(decay_matrices, p=1, dim=1)
    sorted_indices = torch.argsort(norms, descending=(feature_category == FeatureCategory.SLOW_DECAY))
    
    # Select top/bottom third based on feature category
    mask = torch.zeros_like(norms, dtype=torch.bool)
    mask[sorted_indices[: n_ssms // 3]] = True
    return mask
```

### Usage in ModelInterface

The `Mamba1Interface` and `Mamba2Interface` classes register SSM interference hooks:

```python
def setup(self, layers: Optional[Iterable[TLayerIndex]] = None):
    # Register SSM interference hooks
    for i in range(len(self.model.backbone.layers)):
        if i in layers:
            self.hooks.append(SSMInterfereHook(i, self.knockout_mode, is_falcon=self.is_falcon))
            self.handles.append(self.get_layer_moi(i).register_forward_hook(self.hooks[-1]))
```

## Hook-Based Architecture Patterns

### Forward Hook Registration

All knockout implementations use PyTorch's forward hook system based on the actual implementation in `.venv/lib/python3.12/site-packages/torch/utils/hooks.py`:

1. **Registration**: Hooks are registered on specific modules using `register_forward_hook()`
2. **Interception**: Hooks intercept the forward pass and can modify inputs/outputs
3. **Cleanup**: Hooks must be properly removed to prevent memory leaks

**Hook Registration Pattern:**
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

**Hook Storage in PyTorch:**
- Hooks are stored in `module._forward_hooks` dictionary
- Each hook gets a unique ID via `RemovableHandle.next_id`
- Weak references prevent memory leaks when modules are deleted

### Attention Masking Techniques

1. **Binary Masks**: Boolean tensors that block specific attention connections
2. **Causal Masking**: Maintains causal structure while allowing selective blocking
3. **Device Compatibility**: Masks are moved to the correct device and dtype

### SSM State Manipulation

1. **State Interference**: Direct modification of SSM state during computation
2. **Feature Selection**: Selective disabling of SSM features based on characteristics
3. **Variant Support**: Different implementations for different Mamba variants

## External Library References

### Hugging Face Transformers
- **GPT2**: Uses `transformers.models.gpt2.modeling_gpt2.GPT2Attention`
- **Llama**: Uses `transformers.models.llama.modeling_llama.LlamaAttention`
- **Mistral**: Uses `transformers.models.mistral.modeling_mistral.MistralAttention`
- **Qwen2**: Uses `transformers.models.qwen2.modeling_qwen2.Qwen2Attention`
- **Mamba**: Uses `transformers.models.mamba.modeling_mamba.MambaMixer`

### PyTorch Hooks
- **Forward Hooks**: `torch.nn.Module.register_forward_hook()` - Registers a forward hook that is called every time after `forward()` has computed an output
- **Hook Management**: `torch.utils.hooks.RemovableHandle` - Provides capability to remove hooks with automatic cleanup
- **Hook Cleanup**: Proper removal to prevent memory leaks using `RemovableHandle.remove()`

**RemovableHandle Implementation:**
```python
class RemovableHandle:
    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]
```

### Model-Specific Architecture Papers
- **GPT2**: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- **Llama**: "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
- **Mamba**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)

## Code Examples

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

- **Infrastructure Documentation**: See [docs/infrastructure.md](infrastructure.md) for ModelInterface usage patterns
- **Core Types**: See `src/core/types.py` for `KnockoutMode` and `FeatureCategory` definitions
- **Experiment Runners**: See [docs/experiment-runners.md](experiment-runners.md) for how knockout is used in experiments

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
