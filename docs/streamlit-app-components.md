# Streamlit App Components Documentation

## Overview

The Streamlit application components provide reusable, type-safe interfaces to the scientific computing infrastructure. Each component serves as a focused interface to different aspects of the project, enabling modular development and consistent user experience across pages.

The components follow a sophisticated component-based architecture that emphasizes type safety, reusability, and modular design. This documentation covers the major component categories and their integration patterns.

## Component Architecture

### Base Component System

All components inherit from the base `StreamlitComponent[OutputType]` system:

```python
from src.utils.streamlit.helpers.component import StreamlitComponent

class MyComponent(StreamlitComponent[OutputType]):
    def render(self) -> OutputType:
        # Component implementation
        pass
```

### Component Categories

Components are organized into logical categories based on their functionality:

1. **Data Display Components**: Results bank, data requirements, and data visualization
2. **Analysis Components**: Model analysis, tokenization, and specialized analysis
3. **Configuration Components**: Input widgets, parameter selection, and form handling
4. **Visualization Components**: Plot generation, heatmaps, and info flow analysis
5. **Management Components**: Prompt filtering, preset management, and job status

## Data Display Components

### ShowResultsBank Component

**Location**: `src/app/components/result_bank.py`

**Purpose**: Provides UI for exploring and managing `ResultBank` data objects and experiment results.

**Core Interface**:
```python
class ShowResultsBank(StreamlitComponent[T_RESULT_BANK_TYPE]):
    def __init__(
        self,
        results_bank: T_RESULT_BANK_TYPE,
        selection_mode: SelectionMode = SelectionMode.DISABLED,
        height: int = 1000,
        key: str = "results_bank",
        filters: Optional[dict[str, list]] = None,
        hide_columns: list[str] = [],
        hide_singular_columns: bool = False,
        pre_select_all_rows: bool = False,
    ):
        # Component initialization
```

**UI Interfaces Provided**:
- **Results Bank Interface**: UI for exploring experiment results through AgGrid integration
- **Model Evaluations Interface**: UI for selecting model evaluations using `select_model_evaluations`
- **Data Filtering Interface**: UI for filtering and exploring result data with default filters
- **Export Interface**: UI for exporting results and data

**Integration with Data Systems**:
- **ResultBank Objects**: Interfaces with `ResultBank` from `data_defs.py` for experiment results
- **DataFrame Conversion**: Converts ResultBank objects to DataFrames for display
- **AgGrid Integration**: Uses advanced AgGrid features for interactive data exploration
- **Type-Safe Return**: Returns filtered ResultBank objects with type safety

**Usage Examples**:
```python
# Basic usage
results_bank = ShowResultsBank(
    results_bank,
    selection_mode=SelectionMode.MULTIPLE,
    height=300,
    hide_singular_columns=True
).render()

# With filters
results_bank = ShowResultsBank(
    results_bank,
    filters={
        "model_arch": ["gpt2", "llama"],
        "model_size": ["small", "medium"]
    }
).render()
```

### RequirementsDisplay Component

**Location**: `src/app/components/data_requirements.py`

**Purpose**: Provides UI for displaying and selecting data requirements with fulfillment status.

**Core Interface**:
```python
class RequirementsDisplay(StreamlitComponent):
    def __init__(
        self,
        summarized_data_fulfilled_reqs: SummarizedDataFulfilledReqs,
        selection_mode: SelectionMode,
        height: int | None = None,
        key: str = "data_requirements",
        hide_columns: list[str] = [],
    ):
        # Component initialization
```

**UI Interfaces Provided**:
- **Requirements Display Interface**: UI for exploring data requirements with fulfillment status
- **Filtering Interface**: UI for filtering requirements by various criteria
- **Selection Interface**: UI for selecting requirements for execution
- **Status Tracking Interface**: UI for tracking requirement fulfillment status

**Integration with Data Systems**:
- **DataReqs Objects**: Interfaces with `DataReqs` from `data_defs.py` for experiment requirements
- **SummarizedDataFulfilledReqs**: Manages requirement fulfillment status
- **AgGrid Integration**: Uses AgGrid for interactive requirement exploration
- **Type-Safe Return**: Returns selected DataReqs objects for execution

**Usage Examples**:
```python
# Display requirements with selection
data_reqs_to_run = RequirementsDisplay(
    df,
    height=400,
    selection_mode=SelectionMode.MULTIPLE,
    hide_columns=[]
).render()

# Execute selected requirements
if data_reqs_to_run:
    RequirementExecution(data_reqs_to_run).render()
```

### RequirementExecution Component

**Location**: `src/app/components/data_requirements.py`

**Purpose**: Provides UI for executing selected data requirements through BaseRunner integration.

**Core Interface**:
```python
class RequirementExecution(StreamlitComponent):
    def __init__(self, data_reqs_to_run: DataReqs, key: str = "requirement_execution"):
        self.data_reqs_to_run = data_reqs_to_run
        self.key = key
```

**UI Interfaces Provided**:
- **Execution Configuration Interface**: UI for configuring execution parameters (GPU, code version)
- **SLURM Management Interface**: UI for managing SLURM job execution
- **Status Tracking Interface**: UI for tracking job status and completion
- **Progress Monitoring Interface**: UI for monitoring execution progress

**Integration with Infrastructure**:
- **BaseRunner Integration**: Creates runners from DataReqs for experiment execution
- **SLURM Coordination**: Manages job execution and status tracking
- **Configuration Management**: Handles execution parameters and settings
- **Status Monitoring**: Tracks job status and completion

**Usage Examples**:
```python
# Execute requirements with SLURM
RequirementExecution(data_reqs_to_run).render()

# The component handles:
# - Creating BaseRunner instances
# - Configuring SLURM jobs
# - Tracking execution status
# - Monitoring progress
```

## Analysis Components

### TokenizationComponent

**Location**: `src/app/components/tokenization.py`

**Purpose**: Provides UI for tokenization analysis and visualization across multiple models.

**Core Interface**:
```python
class TokenizationVisualizerComponent(StreamlitComponent):
    def __init__(
        self,
        prompts: Prompts,
        unique_tokenizers: Tokenizers,
        cache_key_name: str = "tokenization_component_cache",
    ):
        # Component initialization
```

**UI Interfaces Provided**:
- **Tokenization Analysis Interface**: UI for analyzing tokenization patterns across models
- **Background Task Interface**: UI for managing background tokenization tasks
- **Progress Tracking Interface**: UI for tracking analysis progress
- **Results Visualization Interface**: UI for visualizing tokenization results

**Integration with Analysis Systems**:
- **Background Task Management**: Uses `TasksManager` for long-running tokenization analysis
- **Model Integration**: Interfaces with multiple tokenizer models
- **Data Processing**: Handles large-scale tokenization analysis
- **Visualization**: Provides comprehensive result visualization

**Usage Examples**:
```python
# Create tokenization analysis
component = TokenizationVisualizerComponent(
    prompts=prompts,
    unique_tokenizers=unique_tokenizers
)
component.render()

# The component handles:
# - Background task management
# - Progress tracking
# - Result visualization
# - Error handling
```

### ModelAnalysisComponent

**Location**: `src/app/components/model_analysis.py`

**Purpose**: Provides UI for specialized model analysis including Mamba-specific analysis.

**Core Interface**:
```python
class AMatrixAnalysisComponent(StreamlitComponent):
    def __init__(self, selected_model: MODEL_ARCH_AND_SIZE):
        self.selected_model = selected_model
```

**UI Interfaces Provided**:
- **Model Selection Interface**: UI for selecting models for analysis
- **A Matrix Analysis Interface**: UI for analyzing A matrices through specialized components
- **Feature Dynamics Interface**: UI for exploring feature dynamics
- **Model-Specific Visualization Interface**: UI for model-specific visualizations

**Integration with Analysis Systems**:
- **Mamba-Specific Analysis**: Interfaces with Mamba model analysis tools
- **Architecture Analysis**: Provides architecture-specific insights
- **Visualization**: Creates model-specific visualizations
- **Data Processing**: Handles model-specific data processing

**Usage Examples**:
```python
# Mamba analysis
selected_model = select_mamba_model()
if selected_model:
    AMatrixAnalysisComponent(selected_model).render()
    FeatureDynamicsComponent(selected_model).render()
```

## Configuration Components

### Input Components

**Location**: `src/app/components/inputs.py`

**Purpose**: Provides reusable input widgets for configuration and parameter selection.

**Core Components**:
```python
def select_enum(label: str, enum_class: Type[T], session_key: SessionKey[T]) -> T:
    """Display a selection widget for a StrEnum."""
    return st.selectbox(
        label,
        options=enum_class,
        key=session_key.key_for_component,
    )

def select_gpu_type():
    """Select GPU type for model execution."""
    options = ["smart"] + [value for value in SLURM_GPU_TYPE]
    st.selectbox(
        AppGlobalText.gpu_type,
        options=options,
        key=AppSessionKeys._selected_gpu.key,
    )

def select_window_size():
    """Select window size for analysis parameters."""
    options = [1, 3, 5, 7, 9, 12, 15]
    st.selectbox(
        AppGlobalText.window_size,
        options=options,
        key=AppSessionKeys.window_size.key_for_component,
        index=options.index(AppSessionKeys.window_size.value),
    )
```

**UI Interfaces Provided**:
- **Enum Selection Interface**: UI for selecting from enumerated types
- **GPU Selection Interface**: UI for selecting GPU type (smart/manual)
- **Window Size Interface**: UI for configuring window size parameters
- **Token Type Interface**: UI for selecting token types

**Integration with Configuration Systems**:
- **Session State Management**: Uses `SessionKey` for type-safe state management
- **Global Configuration**: Integrates with global application configuration
- **Type Safety**: Provides type-safe parameter selection
- **Default Values**: Handles default value management

**Usage Examples**:
```python
# Select token type
select_token_type(session_key, label="Select Token Type")

# Select GPU type
select_gpu_type()

# Select window size
select_window_size()
```

## Visualization Components

### PlotGenerationComponent

**Location**: `src/app/components/plot_generation.py`

**Purpose**: Provides UI for generating and managing plots with automated plot generation capabilities.

**Core Interface**:
```python
class PlotGenerationComponent(StreamlitComponent):
    def __init__(self, plot_plan: PlotPlan):
        self.plot_plan = plot_plan
```

**UI Interfaces Provided**:
- **Plot Plan Interface**: UI for managing and configuring plot plans
- **Plot Generation Interface**: UI for generating plots with automated processes
- **Configuration Interface**: UI for configuring plot parameters and settings
- **Progress Tracking Interface**: UI for tracking plot generation progress

**Integration with Visualization Systems**:
- **Plot Plan Management**: Interfaces with plot plan system
- **Automated Generation**: Handles automated plot generation
- **Configuration Management**: Manages plot parameters and settings
- **Progress Tracking**: Tracks generation progress and status

**Usage Examples**:
```python
# Generate plots from plan
PlotGenerationComponent(plot_plan).render()

# The component handles:
# - Plot plan configuration
# - Automated generation
# - Progress tracking
# - Result management
```

### HeatmapPlotGenerationComponent

**Location**: `src/app/components/multi_plots.py`

**Purpose**: Provides UI for generating heatmap plots with specialized heatmap functionality.

**Core Interface**:
```python
class HeatmapPlotGenerationComponent(StreamlitComponent):
    def __init__(self, prompt_idx: TPromptOriginalIndex):
        self.prompt_idx = prompt_idx
```

**UI Interfaces Provided**:
- **Heatmap Generation Interface**: UI for generating heatmap plots
- **Prompt Selection Interface**: UI for selecting prompts for heatmap analysis
- **Configuration Interface**: UI for configuring heatmap parameters
- **Visualization Interface**: UI for displaying heatmap results

**Integration with Analysis Systems**:
- **Heatmap Analysis**: Interfaces with heatmap analysis pipeline
- **Prompt Processing**: Handles prompt selection and processing
- **Visualization**: Creates specialized heatmap visualizations
- **Data Integration**: Integrates with analysis data sources

**Usage Examples**:
```python
# Generate heatmap for specific prompt
HeatmapPlotGenerationComponent(prompt_idx).render()

# The component handles:
# - Prompt-specific heatmap generation
# - Configuration management
# - Result visualization
# - Data integration
```

## Management Components

### PromptFilterComponent

**Location**: `src/app/components/prompt_filter.py`

**Purpose**: Provides UI for prompt filtering and selection with complex filtering capabilities.

**Core Interface**:
```python
class FilterPromptsComponent(StreamlitComponent):
    def __init__(
        self,
        key: str,
        base_prompt_filteration: BasePromptFilteration,
    ):
        self.key = key
        self.base_prompt_filteration = base_prompt_filteration
```

**UI Interfaces Provided**:
- **Filter Configuration Interface**: UI for configuring prompt filters
- **Filter Composition Interface**: UI for composing complex filters
- **Filter Management Interface**: UI for managing filter presets
- **Selection Interface**: UI for selecting filtered prompts

**Integration with Filtering Systems**:
- **BasePromptFilteration**: Interfaces with prompt filtering system
- **Logical Operations**: Supports union, intersection, and sampling operations
- **Preset Management**: Manages reusable filter configurations
- **Type Safety**: Provides type-safe filter operations

**Usage Examples**:
```python
# Create filter component
filter_component = FilterPromptsComponent(
    key="my_filter",
    base_prompt_filteration=LogicalPromptFilteration.create_and([...])
)
filter_result = filter_component.render()

# Use filter result
if filter_result:
    # Process filtered prompts
    pass
```

### ShowRunnerStatus Component

**Location**: `src/app/components/result_bank.py`

**Purpose**: Provides UI for displaying and managing runner status and SLURM job information.

**Core Interface**:
```python
class ShowRunnerStatus(StreamlitComponent):
    def __init__(self, runner: BaseRunner):
        self.runner = runner
        self.sks = self.ShowRunnerStatusSks()
```

**UI Interfaces Provided**:
- **Job Status Interface**: UI for displaying SLURM job status
- **Output Display Interface**: UI for showing job output and error logs
- **Job Management Interface**: UI for managing job execution
- **Progress Tracking Interface**: UI for tracking job progress

**Integration with Infrastructure**:
- **BaseRunner Integration**: Interfaces with experiment runners
- **SLURM Management**: Manages SLURM job information and status
- **Log Display**: Shows job output and error logs
- **Status Tracking**: Tracks job execution status

**Usage Examples**:
```python
# Display runner status
ShowRunnerStatus(runner).render()

# The component handles:
# - Job status display
# - Output log viewing
# - Error log viewing
# - Job management
```

## Component Reusability Patterns

### Generic Component Pattern

Components use generics for type-safe interfaces:

```python
class ShowResultsBank(StreamlitComponent[T_RESULT_BANK_TYPE]):
    def render(self) -> T_RESULT_BANK_TYPE:
        # Type-safe implementation
        pass
```

### Session State Integration

Components integrate with session state management:

```python
class ComponentWithState(StreamlitComponent):
    def __init__(self):
        self.sks = self.ComponentSessionKeys()
    
    class ComponentSessionKeys(SessionKeysBase):
        state_key = SessionKeyDescriptor[str]("default_value")
```

### Background Task Integration

Components integrate with background task management:

```python
class ComponentWithBackgroundTasks(StreamlitComponent):
    def __init__(self):
        self.task_manager = TasksManager[InputType, ResultType]()
    
    def render(self):
        # Use background tasks for long operations
        show_tasks_manager_summary(
            get_manager=lambda: self.task_manager,
            auto_start=True
        )
```

## Integration with Core Systems

### Data Object Integration

Components interface with core data objects:

```python
# ResultBank integration
results_bank = ShowResultsBank(result_bank).render()

# DataReqs integration
data_reqs = RequirementsDisplay(df).render()

# PromptFilteration integration
filter_result = FilterPromptsComponent(base_filteration).render()
```

### Experiment Infrastructure Integration

Components integrate with experiment infrastructure:

```python
# BaseRunner integration
config = init_runner_from_params(req, input_params, metadata_params)

# SLURM integration
status = config.slurm_job_folder.get_latest_slurm_job_status()
```

### Visualization Integration

Components integrate with visualization systems:

```python
# Plot generation
PlotGenerationComponent(plot_plan).render()

# Heatmap generation
HeatmapPlotGenerationComponent(prompt_idx).render()
```

## Performance Considerations

### Component Optimization

1. **Lazy Loading**: Components load data only when needed
2. **Caching**: Components cache expensive operations
3. **Background Processing**: Components use background tasks for long operations
4. **Component Isolation**: Components isolate expensive operations

### Memory Management

1. **Session State Cleanup**: Components clean up session state when appropriate
2. **Resource Disposal**: Components properly dispose of resources
3. **Background Task Management**: Components cancel background tasks when appropriate
4. **Cache Management**: Components implement appropriate cache invalidation

## Extension Guidelines

### Adding New Components

1. **Inherit from Base Class**: Always inherit from `StreamlitComponent[OutputType]`
2. **Implement Render Method**: Provide a clear `render()` implementation
3. **Use Type Safety**: Define clear input/output types
4. **Handle Errors**: Implement graceful error handling
5. **Document Interface**: Provide clear documentation for component usage

### Component Customization

1. **Configuration Integration**: Use global configuration consistently
2. **Session State Management**: Use SessionKey for type-safe state management
3. **Error Handling**: Implement graceful error handling
4. **Performance**: Consider performance implications of component design

## Cross-References

- **Infrastructure Patterns**: See [docs/streamlit-infrastructure.md](streamlit-infrastructure.md) for base patterns
- **App Pages**: See [docs/streamlit-app-pages.md](streamlit-app-pages.md) for page architecture
- **Data Interfaces**: See [docs/data-interfaces.md](data-interfaces.md) for data management
- **Experiment Runners**: See [docs/experiment-runners.md](experiment-runners.md) for experiment infrastructure 
