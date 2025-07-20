# Dataset Processing Flow

High-level data flow from raw datasets to experiment usage with persistent caching and consistency guarantees.

## Data Flow Overview

```
Raw Dataset (HuggingFace) 
    ↓ (download_dataset.py - persistent disk caching)
Split Datasets (splitting.py - seeded for consistency)
    ↓ Prompts Interface
Enriched Data (evaluate_model.py - adds model predictions)
    ↓ (Experiment Results - Prompts Interface - works with both raw and enriched)
    ...
```

## Key Components

### download_dataset.py
- **Persistent disk caching**: Split datasets are saved to disk and loaded from disk for performance
- **Split management**: Creates reproducible splits using `split_dataset()` with consistent seeds
- **Dataset support**: Currently uses counterfact dataset, but supports any dataset with column alignment

### splitting.py  
- **Seeded shuffling**: Uses consistent seed (42) for reproducible splits
- **Original ID preservation**: Maintains `original_idx` during shuffling for unambiguous reference
- **Split caching**: Split datasets are cached on disk for performance

### Prompts Interface
- **Universal access**: Works with both raw and enriched data
- **Original ID tracking**: Uses `COLS.ORIGINAL_IDX` to reference prompts unambiguously across splits
- **Consistent filtering**: Provides unified interface for data access regardless of enrichment state

### evaluate_model.py Enrichment

Adds model prediction columns to raw data:
- `COLS.EVALUATE_MODEL.MODEL_CORRECT`: Whether model's top prediction matches correct answer
- `COLS.EVALUATE_MODEL.TARGET_RANK`: Rank of correct answer in model's predictions (1 = top)
- `COLS.EVALUATE_MODEL.TARGET_PROBS`: Probability model assigns to correct answer
- `COLS.EVALUATE_MODEL.MODEL_OUTPUT`: Model's top prediction
- `COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUT_CONFIDENCE`: Confidence of top prediction
- `COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUTS`: JSON of top-k predictions with probabilities
- `COLS.EVALUATE_MODEL.MODEL_GENERATION`: Full generated sequence
- `COLS.EVALUATE_MODEL.TARGET_TOKENS`: JSON of tokenized correct answer

## Expected Dataset Columns

For counterfact dataset (can be adapted for other datasets):
- `COLS.COUNTER_FACT.PROMPT`: Input text
- `COLS.COUNTER_FACT.TARGET_TRUE`: Correct answer  
- `COLS.COUNTER_FACT.SUBJECT`: Subject entity
- `COLS.COUNTER_FACT.RELATION`: Relation type
- `COLS.ORIGINAL_IDX`: Original dataset index (preserved during splitting)

## Critical Rules

- **ALWAYS use original_idx** for unambiguous prompt reference
- **ALWAYS use Prompts interface** for data access

## Cross-References

- **Data Interfaces**: See [docs/data-relationships-interface.md](data-relationships-interface.md) for data object patterns
- **Core Modules**: See [docs/core-modules.md](core-modules.md) for column definitions and constants
- **Experiment Runners**: See [docs/experiment-runners.md](experiment-runners.md) for how data is used in experiments
- **Setup and Environment**: See [docs/setup-and-environment.md](setup-and-environment.md) for file management
- **Utilities**: See [docs/utilities-and-patterns.md](utilities-and-patterns.md) for data processing utilities 
