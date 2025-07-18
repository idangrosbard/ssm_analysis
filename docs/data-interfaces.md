# Data Interfaces and Object-Oriented Entities

This document covers the object-oriented data interface patterns defined in `src/data_ingestion/data_defs/data_defs.py` and how experiment results objects interact with data across the application.

## Overview

The data interface system provides a sophisticated object-oriented approach to managing experiment data requirements, results, and interactions. All data access should go through these interface objects rather than direct file access.

## Core Data Objects

### DataRequirementCollection and DataReqs

**Purpose**: Manages experiment data requirements across the project
**Usage**: Tracks what data needs to be computed for experiments

```python
# Adding data requirements
data_reqs = DataRequirementCollection()
data_reqs.add_data_req(data_req, prompt_filteration)

# Converting to DataReqs object
data_reqs_obj = DataReqs.from_data_reqs_collection(data_reqs)
```

**Key Rules**:
- **ALWAYS add requirements using `add_data_req(data_req, prompt_filteration)`**
- **NEVER access experiment outputs directly** - use `ResultBank` and related objects
- **ALWAYS maintain object serialization compatibility** when modifying data structures

### FulfilledReqs and SummarizedDataFulfilledReqs

**Purpose**: Tracks completed experiments and their availability
**Usage**: Determines what experiments are ready for analysis

```python
# Converting DataReqs to FulfilledReqs
fulfilled_reqs = data_reqs.to_fulfilled_reqs(result_bank)

# Summarizing fulfilled requirements
summarized_reqs = fulfilled_reqs.summarize()
```

**Key Methods**:
- `choose_latest_fulfilled()`: Selects the most recent results for each requirement
- `get_config()`: Returns runner configurations for fulfilled requirements
- `amount_missing()`: Counts unfulfilled requirements

### ResultBank[T_RUNNER_TYPE]

**Purpose**: Central registry of all experiment results
**Usage**: Provides unified access to completed experiments

```python
# Converting to different result formats
experiment_df = result_bank.to_experiment_results_df()
info_flow_results = result_bank.to_info_flow_results()
evaluate_results = result_bank.to_evaluate_model_results()

# Filtering by experiment type
unique_models = result_bank.unique_model_arch_and_sizes()
```

**Key Methods**:
- `to_experiment_results_df()`: Converts to pandas DataFrame
- `to_info_flow_results()`: Converts to InfoFlowResults
- `to_evaluate_model_results()`: Converts to EvaluateModelResults
- `unique_model_arch_and_sizes()`: Gets unique model configurations

**Key Rules**:
- **ALWAYS filter by experiment type when accessing specific results**
- **NEVER access experiment outputs directly** - use ResultBank methods

### PlotPlans and PromptFilterationsPresets

**Purpose**: Manages plotting configurations and reusable prompt filters
**Storage**: JSON files in `final_plots/` directory

```python
# Loading plot plans
plot_plans = PlotPlans.load()

# Loading prompt filter presets
presets = PromptFilterationsPresets.load()
```

**Key Rules**:
- **ALWAYS use presets for consistent filtering across experiments**
- **Storage**: JSON files in `final_plots/` directory

## Object-Oriented Entity Interaction Patterns

### Data Object Coordination Rules

- **WHEN adding new experiment types**: Update corresponding data objects in `data_defs.py`
- **WHEN creating analysis workflows**: Use objects from `src/analysis/experiment_results/`
- **NEVER access experiment outputs directly** - use `ResultBank` and related objects
- **ALWAYS maintain object serialization compatibility** when modifying data structures

### Interface Compatibility Patterns

All data objects implement consistent interface patterns:

```python
# IndexableDataObject pattern
class DataReqs(IndexableDataObject[BaseVariantParams, BasePromptFilteration]):
    def to_fulfilled_reqs(self, result_bank: ResultBank[BaseRunner]) -> FulfilledReqs:
        # Implementation details...

# IterableDataObject pattern  
class ResultBank(IterableDataObject[T_RUNNER_TYPE]):
    def to_experiment_results_df(self) -> pd.DataFrame:
        # Implementation details...
```

## Experiment Results Object Integration

### Related Objects in `src/analysis/experiment_results/`

#### default_data_reqs.py

**Contains standard experiment configurations**

```python
# Getting default data requirements
data_reqs = get_default_data_reqs()

# Standard configurations include:
# - InfoFlow experiments with different token types
# - Heatmap experiments with sampling
# - Model evaluation experiments
```

**Key Rules**:
- **USE `get_default_data_reqs()` for typical experiment setups**
- **MODIFY when adding new standard experimental patterns**

#### helpers.py

**Provides utilities for result bank operations**

```python
# Getting model evaluations
model_evaluations = get_model_evaluations(code_version, model_arch_and_sizes)

# Initializing runners from parameters
runner = init_runner_from_params(variant_params, input_params, metadata_params)
```

**Key Functions**:
- `get_model_evaluations()`: Retrieves model performance data
- `init_runner_from_params()`: Creates runners programmatically
- `serialize_result_bank()`: Serializes result banks for storage

#### plot_plan.py and related

**Manages complex multi-dimensional plotting configurations**

```python
# Creating plot plans for publication-quality figures
plot_plan = PlotPlan(...)

# Coordinating with data_defs.py objects for data requirements
```

**Key Rules**:
- **USE `PlotPlan` for publication-quality figure generation**
- **COORDINATE with `data_defs.py` objects for data requirements**

## Data Loading Interface Utilities

### download_dataset.py

**Provides standardized data loading interfaces**

```python
# Loading split datasets
dataset = load_splitted_counter_fact(split=(SPLIT.TRAIN1,))

# Converting between flat and indexed formats
indexed_data = flat_to_indexed_prompt_data(flat_df)
flat_data = indexed_to_flat_prompt_data(indexed_df)
```

**Key Functions**:
- `load_splitted_counter_fact()`: Loads counterfactual dataset splits
- `flat_to_indexed_prompt_data()`: Converts flat to indexed format
- `indexed_to_flat_prompt_data()`: Converts indexed to flat format
- `get_prompt_ids()`: Retrieves prompt IDs for datasets

## Object Relationships and Interface Compatibility

### Inheritance Hierarchy

```
DataObject (base)
├── IndexableDataObject[K, V]
│   ├── DataReqs[BaseVariantParams, BasePromptFilteration]
│   ├── FulfilledReqs[BaseVariantParams, tuple[BasePromptFilteration, tuple[BaseRunner, ...]]]
│   ├── PlotPlans[TPlotID, PlotPlan]
│   └── PromptFilterationsPresets[TPresetID, BasePromptFilteration]
├── IterableDataObject[T]
│   ├── ResultBank[T_RUNNER_TYPE]
│   ├── SummarizedDataFulfilledReqs[dict[str, Any]]
│   ├── ModelCombinationsPrompts[ModelCombination]
│   └── Tokenizers[UniqueTokenizerInfo]
└── Prompts[TPromptOriginalIndex, PromptNew]
```

### Type Safety and Generic Patterns

All data objects use strong typing with generics:

```python
# Generic type parameters ensure type safety
class ResultBank[T_RUNNER_TYPE](IterableDataObject[T_RUNNER_TYPE]):
    def to_experiment_results_df(self) -> pd.DataFrame:
        # Type-safe implementation...

# Specialized result types
class EvaluateModelResults(ResultBank[EvaluateModelRunner]):
    def get_hit_per_prompt(self, metric_name: EvaluateModelMetricName) -> pd.DataFrame:
        # Specialized for evaluate model results...

class InfoFlowResults(ResultBank[InfoFlowRunner]):
    def get_common_indices(self) -> set[TPromptOriginalIndex]:
        # Specialized for info flow results...
```

## Usage Examples

### Creating and Managing Data Requirements

```python
# 1. Create data requirements collection
data_reqs = DataRequirementCollection()

# 2. Add experiment requirements
data_reqs.add_data_req(
    InfoFlowParams(model_arch="gpt2", model_size="small", ...),
    LogicalPromptFilteration.create_or([...])
)

# 3. Convert to DataReqs object
data_reqs_obj = DataReqs.from_data_reqs_collection(data_reqs)

# 4. Check fulfillment against result bank
fulfilled_reqs = data_reqs_obj.to_fulfilled_reqs(result_bank)
```

### Working with Result Banks

```python
# 1. Load existing result bank
result_bank = ResultBank.load_from_experiment_results_df(df)

# 2. Convert to specific result types
info_flow_results = result_bank.to_info_flow_results()
evaluate_results = result_bank.to_evaluate_model_results()

# 3. Access specialized methods
common_indices = info_flow_results.get_common_indices()
hit_per_prompt = evaluate_results.get_hit_per_prompt(metric_name)
```

### Managing Plot Plans and Presets

```python
# 1. Load existing configurations
plot_plans = PlotPlans.load()
presets = PromptFilterationsPresets.load()

# 2. Add new configurations
plot_plans.add_plan(new_plot_plan)
presets.add_preset("new_preset", new_filteration)

# 3. Save configurations
plot_plans.save()
presets.save()
```

## Cross-References

- **Experiment Runners**: See [docs/experiment-runners.md](experiment-runners.md) for how runners interact with data objects
- **Analysis and Plotting**: See [docs/analysis-and-plotting.md](analysis-and-plotting.md) for plotting configuration management
- **Infrastructure**: See [docs/infrastructure.md](infrastructure.md) for base runner patterns

## Best Practices

1. **Always use interface objects**: Never access experiment outputs directly
2. **Maintain type safety**: Use generic type parameters consistently
3. **Coordinate object updates**: When modifying data structures, update all related objects
4. **Use centralized configurations**: Leverage PlotPlans and PromptFilterationsPresets for consistency
5. **Check fulfillment**: Always verify data availability before starting dependent experiments 
