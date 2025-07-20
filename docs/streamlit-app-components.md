# Streamlit App Components

Reusable UI components for scientific computing interface. Focus on what LLMs need to know about component architecture and patterns.

## Component Architecture

### Base Component System

All components inherit from `StreamlitComponent[OutputType]`:

**File**: `src/utils/streamlit/helpers/component.py`

```python
from src.utils.streamlit.helpers.component import StreamlitComponent

class MyComponent(StreamlitComponent[OutputType]):
    def render(self) -> OutputType:
        # Component implementation
        pass
```

## Component Categories

### 1. Data Display Components

#### ShowResultsBank Component

**File**: `src/app/components/result_bank.py`

**Purpose**: UI for exploring and managing `ResultBank` data objects and experiment results.

**Key Features**:
- **AgGrid Integration**: Advanced data table with filtering and selection
- **Type-Safe Return**: Returns filtered ResultBank objects
- **DataFrame Conversion**: Converts ResultBank objects to DataFrames
- **Export Interface**: UI for exporting results and data

**Usage Pattern**:
```python
results_bank = ShowResultsBank(
    results_bank,
    selection_mode=SelectionMode.MULTIPLE,
    height=300,
    hide_singular_columns=True
).render()
```

#### RequirementsDisplay Component

**File**: `src/app/components/data_requirements.py`

**Purpose**: UI for displaying and selecting data requirements with fulfillment status.

**Key Features**:
- **Fulfillment Tracking**: Shows requirement fulfillment status
- **Filtering Interface**: Advanced filtering and exploration
- **Data Availability**: Analyzes data availability patterns
- **AgGrid Integration**: Interactive data table with selection

### 2. Analysis Components

#### TokenizationComponent

**File**: `src/app/components/tokenization.py`

**Purpose**: Tokenization analysis and visualization interface.

**Key Features**:
- **Background Processing**: Uses background tasks for analysis
- **Multi-Model Support**: Analyzes tokenization across models
- **Progress Tracking**: Real-time progress updates
- **Result Visualization**: Interactive tokenization plots

#### ModelAnalysisComponent

**File**: `src/app/components/model_analysis.py`

**Purpose**: Model analysis and comparison interface.

**Key Features**:
- **Model Comparison**: Compare multiple models
- **Performance Metrics**: Display model performance
- **Configuration Interface**: Model parameter configuration
- **Result Export**: Export analysis results

#### InfoFlowComponent

**File**: `src/app/components/info_flow.py`

**Purpose**: Information flow analysis interface.

**Key Features**:
- **Flow Visualization**: Interactive flow diagrams
- **Path Analysis**: Analyze information paths
- **Filtering**: Filter flow data
- **Export Capabilities**: Export flow analysis

### 3. Configuration Components

#### PromptFilterComponent

**File**: `src/app/components/prompt_filter.py`

**Purpose**: Prompt filtering and selection interface.

**Key Features**:
- **Filter Management**: Create and manage filters
- **Preset System**: Save and load filter presets
- **Real-time Filtering**: Apply filters in real-time
- **Export Filters**: Export filter configurations

#### InputWidgetsComponent

**File**: `src/app/components/inputs.py`

**Purpose**: Reusable input widgets and forms.

**Key Features**:
- **Type-Safe Inputs**: Type-safe input widgets
- **Validation**: Input validation and error handling
- **Session State**: Session state integration
- **Reusable Patterns**: Common input patterns

### 4. Visualization Components

#### PlotGenerationComponent

**File**: `src/app/components/plot_generation.py`

**Purpose**: Plot generation and configuration interface.

**Key Features**:
- **Plot Configuration**: Configure plot parameters
- **Multiple Plot Types**: Support for various plot types
- **Export Options**: Export plots in multiple formats
- **Real-time Preview**: Live plot preview

#### MultiPlotsComponent

**File**: `src/app/components/multi_plots.py`

**Purpose**: Multi-plot layout and organization.

**Key Features**:
- **Grid Layout**: Configurable grid layouts
- **Plot Organization**: Organize multiple plots
- **Layout Management**: Manage plot arrangements
- **Export Capabilities**: Export multi-plot layouts

#### PlotPlansComponent

**File**: `src/app/components/plot_plans.py`

**Purpose**: Plot plan management and configuration.

**Key Features**:
- **Plan Management**: Create and manage plot plans
- **Template System**: Plot plan templates
- **Configuration Interface**: Plan configuration UI
- **Plan Export**: Export plot plans

## Component Patterns

### Type-Safe Component Design

All components follow type-safe patterns:

```python
# Component with complex return type
@dataclass
class AnalysisConfig:
    model_name: str
    parameters: Dict[str, Any]

class AnalysisComponent(StreamlitComponent[AnalysisConfig]):
    def render(self) -> AnalysisConfig:
        model_name = st.selectbox("Model", ["gpt2", "llama", "mamba"])
        parameters = self.get_parameters()
        return AnalysisConfig(model_name=model_name, parameters=parameters)
```

### Session State Integration

Components use session state for persistence:

```python
# Session state integration
class ConfigComponent(StreamlitComponent[Config]):
    def __init__(self, config_key: SessionKey[Config]):
        self.config_key = config_key
    
    def render(self) -> Config:
        # Use session state for persistence
        current_config = self.config_key.value
        # Update configuration
        return updated_config
```

### Background Task Integration

Components can integrate with background tasks:

```python
# Background task integration
class AnalysisComponent(StreamlitComponent[AnalysisResults]):
    def render(self) -> AnalysisResults:
        # Start background analysis
        analysis_job = AnalysisJob(data, models)
        analysis_job.start()
        
        # Display progress
        show_tasks_manager_summary(
            get_manager=lambda: analysis_job,
            auto_start=True
        )
        
        # Return results when complete
        if analysis_job.status == TaskStatus.COMPLETED:
            return analysis_job.get_results()
```

### AgGrid Integration Pattern

Components use AgGrid for advanced data tables:

```python
# AgGrid integration pattern
class DataTableComponent(StreamlitComponent[SelectedData]):
    def render(self) -> SelectedData:
        # Create grid builder
        df, grid_builder = base_grid_builder(
            df,
            selection_mode=SelectionMode.MULTIPLE,
            hide_columns=["internal_id"]
        )
        
        # Apply filters
        set_aagrid_apply_default_filters(
            grid_builder,
            {"status": ["active"]}
        )
        
        # Display grid
        grid_response = AgGrid(
            df,
            gridOptions=grid_builder.build(),
            height=400,
            key="data_grid"
        )
        
        return self.process_selection(grid_response)
```

## Component File Structure

```
src/app/components/
├── result_bank.py              # Results display and management
├── data_requirements.py        # Data requirements interface
├── model_analysis.py           # Model analysis components
├── tokenization.py             # Tokenization analysis
├── info_flow.py                # Information flow analysis
├── prompt_filter.py            # Prompt filtering interface
├── plot_generation.py          # Plot generation
├── multi_plots.py              # Multi-plot components
├── plot_plans.py               # Plot plan management
└── inputs.py                   # Input widgets and forms
```

## Component Integration Patterns

### Page-Component Integration

Pages compose components for complex UIs:

```python
class AnalysisPage(StreamlitPage):
    def render(self) -> None:
        # Compose multiple components
        config = ConfigComponent().render()
        data = DataComponent().render()
        
        if config and data:
            # Use component outputs
            results = AnalysisComponent(config=config, data=data).render()
            PlotComponent(results=results).render()
```


```

## Best Practices

1. **Type Safety**: Always use generic types for component return values
2. **Single Responsibility**: Each component should have one clear purpose
3. **Session State**: Use SessionKey for type-safe session state management
4. **Error Handling**: Implement graceful error handling
5. **Reusability**: Design components to be reusable across pages
6. **Performance**: Use background tasks for long-running operations
7. **Documentation**: Document component interfaces and usage patterns

## Cross-References

- **Streamlit Infrastructure**: See [docs/streamlit-infrastructure.md](streamlit-infrastructure.md) for base classes
- **Core Modules**: See [docs/core-modules.md](core-modules.md) for data object patterns
- **Utilities**: See [docs/utilities-and-patterns.md](utilities-and-patterns.md) for utility patterns
- **Data Interfaces**: See [docs/data-relationships-interface.md](data-relationships-interface.md) for data object patterns
- **Infrastructure**: See [docs/infrastructure.md](infrastructure.md) for base class patterns
- **Setup and Environment**: See [docs/setup-and-environment.md](setup-and-environment.md) for environment setup
