---
description: Prompt Filtering Logic and Patterns
inclusion: manual
---

# Prompt Filtering Logic and Patterns

## Prompt Filteration System

### Core Implementation
- **File**: `src/analysis/prompt_filterations.py`
- **Purpose**: Prompt filtering logic and patterns
- **Usage**: Filter prompts for experiment analysis

### BasePromptFilteration Pattern
- **INHERIT from BasePromptFilteration** for all filtering logic
- **IMPLEMENT required methods**: `get_prompt_ids()`, `get_dependencies()`, `display_name()`
- **USE logical operations**: `&` (AND), `|` (OR), `~` (NOT)

## Filtering Patterns

### Filter Types
- **SelectivePromptFilteration** - Specific prompt selections
- **SamplePromptFilteration** - Sampling with seeds
- **LogicalPromptFilteration** - Complex logical combinations

### Logical Operations
```python
combined_filter = filter1 & filter2  # AND
combined_filter = filter1 | filter2  # OR
inverted_filter = ~filter1           # NOT
complex_filter = (filter1 & filter2) | (~filter3)
```

## Implementation Rules

### Filtering Rules
- **USE BasePromptFilteration** as base class
- **IMPLEMENT logical operations** for complex filtering
- **MAINTAIN filter reproducibility** with seeds

### Filter Coordination
- **COORDINATE with experiment runners** for prompt selection
- **USE consistent filtering patterns** across experiments
- **HANDLE filter dependencies** properly

## Critical Rules

### Prompt Filtering Rules
- **INHERIT from BasePromptFilteration**
- **USE logical operations** for filter combinations
- **MAINTAIN reproducibility** with proper seeding

**Reference**: docs/prompt-filteration.md for complete filtering patterns
