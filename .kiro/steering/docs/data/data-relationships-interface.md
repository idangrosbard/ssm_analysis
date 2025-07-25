---
description: Data Object Interfaces and Relationships
inclusion: manual
---

# Data Relationships Interface

Object-oriented interfaces for handling files, dataframes, and transitioning between different data models. All data access should go through interface objects rather than direct file access.

## Base Definitions

**File**: `src/utils/infra/data_object.py`

```python
class DataObject:  # Base class for all data objects
class IndexableDataObject[K, V](DataObject):  # Provides indexed access
class IterableDataObject[T](DataObject):  # Provides iteration over collections
```

## Core Data Interface Classes

**File**: `src/data_ingestion/data_defs/data_defs.py`

### DataRequirementCollection

- **Purpose**: Manages experiment data requirements across the project
- **Key Methods**: `add_data_req(data_req, prompt_filteration)`, `to_dict()`
- **Key Rules**: ALWAYS add requirements using `add_data_req(data_req, prompt_filteration)`

### DataReqs

- **Inheritance**: `IndexableDataObject[BaseVariantParams, BasePromptFilteration]`
- **Key Methods**: `to_fulfilled_reqs(result_bank)`, `from_data_reqs_collection(data_reqs)`
- **Usage**: Provides indexed access to experiment data requirements

### FulfilledReqs

- **Inheritance**: `IndexableDataObject[BaseVariantParams, tuple[BasePromptFilteration, tuple[BaseRunner, ...]]]`
- **Key Methods**: `summarize()`, `choose_latest_fulfilled()`, `get_config()`
- **Usage**: Tracks completed experiments and their availability

### SummarizedDataFulfilledReqs

- **Inheritance**: `IterableDataObject[dict[str, Any]]`
- **Key Methods**: `amount_missing()`, `to_data_reqs()`, `to_df()`
- **Usage**: Analysis and reporting of experiment availability

### ResultBank[T_RUNNER_TYPE]

- **Inheritance**: `IterableDataObject[T_RUNNER_TYPE]`
- **Key Methods**: `to_experiment_results_df()`, `to_info_flow_results()`, `to_evaluate_model_results()`, `unique_model_arch_and_sizes()`
- **Key Rules**: ALWAYS filter by experiment type when accessing specific results
- **Usage**: Central registry of all experiment results

### EvaluateModelResults

- **Inheritance**: `ResultBank[EvaluateModelRunner]`
- **Key Methods**: `get_hit_per_prompt(metric_name)`, `model_arch_and_sizes`
- **Usage**: Access to model evaluation specific data

### InfoFlowResults

- **Inheritance**: `ResultBank[InfoFlowRunner]`
- **Key Methods**: `get_common_indices()`, `max_layer()`, `min_layer()`, `subset_layers(layer_subset)`
- **Usage**: Access to information flow specific data

### PlotPlans

- **Inheritance**: `IndexableDataObject[TPlotID, PlotPlan]`
- **Key Methods**: `load()`, `save()`, `add_plan()`, `remove_plan()`, `is_plan_exists()`
- **Key Rules**: Storage in JSON files in `final_plots/` directory
- **Usage**: Manages plotting configurations and plans

### PromptFilterationsPresets

- **Inheritance**: `IndexableDataObject[TPresetID, BasePromptFilteration]`
- **Key Methods**: `load()`, `save()`, `add_preset()`, `get_default_presets()`
- **Key Rules**: ALWAYS use presets for consistent filtering across experiments
- **Usage**: Manages reusable prompt filter configurations

### ModelCombinationsPrompts

- **Inheritance**: `IterableDataObject[ModelCombination]`
- **Key Methods**: `sort_by_prompt_count()`, `change_chosen_prompt_by_seed()`, `to_display_df()`
- **Key Rules**: ALWAYS use consistent seed values for reproducible prompt selection
- **Usage**: Manages model-prompt combinations for analysis

### Tokenizers

- **Inheritance**: `IterableDataObject[UniqueTokenizerInfo]`
- **Key Methods**: `from_unique_tokenizers(model_arch_and_sizes)`
- **Key Rules**: ALWAYS use `from_unique_tokenizers()` for tokenizer creation
- **Usage**: Manages tokenizer instances and their configurations

### UniqueTokenizerInfo

- **Key Properties**: `tokenizer`, `model_arch_and_sizes`, `raw_config`, `display_name`, `_hash`
- **Key Rules**: ALWAYS maintain tokenizer configuration consistency
- **Usage**: Represents unique tokenizer configuration

### Prompts

- **Inheritance**: `IndexableDataObject[TPromptOriginalIndex, PromptNew]`
- **Key Methods**: `filter_by_prompt_ids()`, `filter_by_condition()`, `filter_by_prompt_filteration()`, `sample()`, `to_df()`
- **Key Rules**: ALWAYS use `filter_by_prompt_filteration()` for consistent filtering
- **Usage**: Central interface for accessing and filtering prompt datasets

### PromptNew

- **Inheritance**: `DataObject`
- **Key Methods**: `as_prompt()`
- **Usage**: Wrapper for prompt data

## Object Relationships and Interface Compatibility

### Complete Inheritance Hierarchy

```
DataObject (base)
├── IndexableDataObject[K, V]
│   ├── DataReqs[BaseVariantParams, BasePromptFilteration]
│   ├── FulfilledReqs[BaseVariantParams, tuple[BasePromptFilteration, tuple[BaseRunner, ...]]]
│   ├── PlotPlans[TPlotID, PlotPlan]
│   ├── PromptFilterationsPresets[TPresetID, BasePromptFilteration]
│   └── Prompts[TPromptOriginalIndex, PromptNew]
├── IterableDataObject[T]
│   ├── ResultBank[T_RUNNER_TYPE]
│   ├── SummarizedDataFulfilledReqs[dict[str, Any]]
│   ├── ModelCombinationsPrompts[ModelCombination]
│   ├── Tokenizers[UniqueTokenizerInfo]
│   ├── EvaluateModelResults[EvaluateModelRunner]
│   └── InfoFlowResults[InfoFlowRunner]
└── PromptNew (DataObject)
```

## Data Model Transition Patterns

### File to Object Transitions

```python
plot_plans = PlotPlans.load()  # From JSON
presets = PromptFilterationsPresets.load()  # From JSON
```

### DataFrame to Object Transitions

```python
data_reqs = DataReqs.from_data_reqs_collection(collection)
result_bank = ResultBank.from_experiment_results_df(df)
df = result_bank.to_experiment_results_df()
df = summarized_reqs.to_df()
df = prompts.to_df()
```

### Object to Object Transitions

```python
fulfilled_reqs = data_reqs.to_fulfilled_reqs(result_bank)
summarized_reqs = fulfilled_reqs.summarize()
data_reqs = summarized_reqs.to_data_reqs()
info_flow_results = result_bank.to_info_flow_results()
evaluate_results = result_bank.to_evaluate_model_results()
```

## Cross-References

- **Dataset Flow**: Use #dataset-processing-flow for raw data processing flow
- **Experiment Runners**: Use #experiment-runners for how runners interact with data objects
- **Analysis and Plotting**: Use #analysis-and-plotting for plotting configuration management
- **Infrastructure**: Use #infrastructure for base runner patterns
- **Core Modules**: Use #core-modules for type definitions and constants
- **Setup and Environment**: Use #setup-and-environment for data file management
- **Utilities**: Use #utilities-and-patterns for data object utilities

## Critical Warnings

⚠️ **Breaking the data object relationship patterns will cause cascading failures throughout the project**

⚠️ **Always test changes thoroughly - data interface errors propagate to all dependent experiments**
⚠️ **When in doubt, follow the established patterns exactly - don't improvise**

⚠️ **All data access must go through interface objects - never access files directly**
