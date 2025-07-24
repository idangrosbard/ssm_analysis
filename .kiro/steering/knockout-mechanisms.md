---
description: Model Knockout Mechanisms and Interventions
inclusion: manual
---

# Model Knockout Mechanisms and Interventions

## Knockout Implementation

### Model-Specific Implementations
- **GPT**: `src/experiments/knockout/gpt/` - GPT-2 knockout implementations
- **Llama**: `src/experiments/knockout/llama/` - Llama/Mistral/Qwen knockout implementations
- **Mamba**: `src/experiments/knockout/mamba/` - Mamba-1 and Mamba-2 specific knockout

### ModelInterface Pattern
- **File**: `src/experiments/infrastructure/model_interface.py`
- **Purpose**: Abstract interface for model intervention
- **Usage**: Consistent knockout operations across architectures

## Knockout Patterns

### Hook-Based Implementation
- **USE hook-based interventions** for model knockout
- **IMPLEMENT through ModelInterface** for consistency
- **SUPPORT different feature categories** (ALL, SLOW_DECAY, etc.)

### Feature Categories
- **FeatureCategory.ALL** - All features
- **FeatureCategory.SLOW_DECAY** - Slow decay features
- **Model-specific categories** as needed

## Implementation Methods

### Knockout Operations
```python
# Setup model interface
model_interface.setup(layers=[0, 1, 2, 3])

# Generate logits with knockout
logits = model_interface.generate_logits(
    input_ids=input_tokens,
    num_to_masks={0: [(1, 2), (3, 4)]},
    feature_category=FeatureCategory.SLOW_DECAY
)
```

### Model Interface Methods
- **generate_logits()** - Generate logits with optional knockout
- **n_layers()** - Get number of layers
- **setup()** - Setup model for knockout operations

## Critical Rules

### Knockout Rules
- **USE ModelInterface** for all knockout operations
- **IMPLEMENT model-specific knockout** in appropriate directories
- **SUPPORT feature categories** for different knockout types

### Implementation Standards
- **CONSISTENT interface** across all model types
- **PROPER error handling** for knockout failures
- **EFFICIENT implementation** for large-scale experiments

**Reference**: docs/knockout-mechanisms.md for complete implementation patterns
