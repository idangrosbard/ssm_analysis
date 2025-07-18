# Streamlit App Navigation and Page Architecture

## Overview

The Streamlit application provides a sophisticated interface to the scientific computing infrastructure through 9 specialized pages. Each page serves as a focused interface to different aspects of the project, enabling users to explore, analyze, and manage complex scientific workflows.

The application uses a page factory pattern with centralized navigation management, ensuring consistent user experience and proper coordination between different analysis interfaces.

## Navigation Architecture

### Page Factory Pattern

The application uses a centralized page factory pattern in `src/app/entry_point.py`:

```python
def get_page(page_order: PAGE_ORDER) -> StreamlitPage:
    match page_order:
        case PAGE_ORDER.HOME:
            return p01_home.HomePage()
        case PAGE_ORDER.RESULTS_BANK:
            return p02_results_bank.ResultsBankPage()
        # ... other pages
```

### PAGE_ORDER Enum

The `PAGE_ORDER` enum in `src/app/app_consts.py` defines the navigation structure:

```python
class PAGE_ORDER(Enum):
    HOME = auto()
    RESULTS_BANK = auto()
    DATA_REQUIREMENTS = auto()
    HEATMAP = auto()
    INFO_FLOW_ANALYSIS = auto()
    FINAL_PLOTS = auto()
    PROMPTS_COMPARISON = auto()
    MAMBA_ANALYSIS = auto()
    PROMPT_FILTERATION_PRESETS = auto()
```

Each page has associated metadata including title and icon, accessed through the `page_details` property.

### Global Configuration

The application uses centralized global configuration through `AppSessionKeys`:

- **Code Version**: Controls which version of experiments to load
- **GPU Selection**: Smart or manual GPU type selection for different models
- **Window Size**: Configurable window size for analysis parameters

## Page Categorization

### 1. Configuration and Control Pages

#### Home Page (p01_home.py)
**Purpose**: Application entry point and global configuration
**UI Interfaces Provided**:
- **Global Configuration Interface**: UI for setting global application parameters
- **GPU Selection Interface**: UI for selecting GPU type (smart/manual) using `select_gpu_type`
- **Code Version Interface**: UI for selecting code version through `AppSessionKeys.code_version`
- **Window Size Interface**: UI for configuring window size using `select_window_size`
- **App Reset Interface**: UI for resetting application state

**Key Features**:
- Global app reset functionality
- GPU type selection (smart/manual)
- Code version configuration
- Window size configuration
- Navigation guidance

**Interface to Infrastructure**: Provides global configuration that affects all other pages and analysis components.

#### Data Requirements (p03_data_requirements.py)
**Purpose**: Manage and explore data requirements for experiments
**UI Interfaces Provided**:
- **Data Requirements Interface**: UI for exploring data requirements through `DataRequirementsComponent`
- **Requirement Filtering Interface**: UI for filtering and exploring data requirements
- **Fulfillment Status Interface**: UI for tracking requirement fulfillment status
- **Data Availability Interface**: UI for analyzing data availability

**Key Features**:
- Data requirement filtering and exploration
- Requirement fulfillment status tracking
- Data availability analysis

**Interface to Infrastructure**: Connects to the data management system and experiment infrastructure to ensure data availability.

### 2. Analysis and Visualization Pages

The analysis and visualization pages represent the core scientific functionality interfaces of the application. These pages provide sophisticated interfaces to complex experiment coordination while maintaining scientific rigor. Each page abstracts complex experimental workflows into intuitive user interfaces that coordinate with the underlying analysis infrastructure and plotting systems.

#### Heatmap Creation (p04_heatmap_creation.py)
**Purpose**: Interface to heatmap generation and model analysis through layer-by-layer probability visualization
**UI Interfaces Provided**:
- **Model Combinations Interface**: UI for exploring and selecting model combinations from `load_model_combinations_prompts`
- **Prompt Selection Interface**: UI for filtering and selecting prompts using `PromptSelectionForCombinationComponent`
- **Heatmap Generation Interface**: UI for executing heatmap generation through `HeatmapGenerationComponent`
- **Plot Generation Interface**: UI for creating heatmaps through `HeatmapPlotGenerationComponent`

**Key Features**:
- Model combination analysis with batch processing capabilities
- Prompt filtering and selection with advanced filtering options
- Heatmap generation and execution with SLURM integration
- Real-time visualization of layer-by-layer probability differences

**Interface to Infrastructure**: 
- **Experiment Runner Integration**: Connects to `HeatmapRunner` for layer-by-layer probability analysis
- **Model Evaluation Pipeline**: Integrates with `EvaluateModelRunner` results for baseline model performance
- **Prompt Filtering System**: Coordinates with `LogicalPromptFilteration` for sophisticated prompt selection
- **Plotting Infrastructure**: Integrates with `src/analysis/plots/heatmaps.py` for publication-quality visualization
- **Data Management**: Uses `load_model_combinations_prompts` and `load_model_evaluations_dict` for data coordination

**Analysis Workflow Pattern**:
1. **Model Selection**: Users select model combinations through the interface
2. **Prompt Filtering**: Advanced filtering capabilities using `PromptSelectionForCombinationComponent`
3. **Heatmap Generation**: Execution through `HeatmapGenerationComponent` with SLURM support
4. **Visualization**: Real-time generation of publication-quality heatmaps using `HeatmapPlotGenerationComponent`

**Scientific Computing Specialization**:
- Supports window-based analysis across model layers
- Provides incremental computation capabilities for large-scale experiments
- Integrates with HDF5 storage for efficient data management
- Coordinates with experiment completion status for reliable results

#### Info Flow Analysis (p05_info_flow_analysis.py)
**Purpose**: Interface to information flow experiments using knockout methodology for attention analysis
**UI Interfaces Provided**:
- **Info Flow Results Interface**: UI for selecting and exploring info flow results from `load_results_bank().to_info_flow_results()`
- **Prompt Filtering Interface**: UI for filtering prompts using `FilterPromptsComponent` with `LogicalPromptFilteration`
- **Info Flow Analysis Interface**: UI for analyzing info flow patterns through `InfoFlowAnalysisComponent`
- **Layer Range Interface**: UI for selecting layer ranges for analysis

**Key Features**:
- Interactive info flow visualization with confidence intervals
- Layer-specific analysis with customizable layer ranges
- Multi-model comparison capabilities
- Feature dynamics exploration across different token types

**Interface to Infrastructure**:
- **Experiment Runner Integration**: Connects to `InfoFlowRunner` for knockout-based information flow analysis
- **Statistical Analysis Pipeline**: Integrates with confidence interval calculations and statistical measures
- **Model Evaluation Results**: Coordinates with `EvaluateModelRunner` outputs for baseline comparisons
- **Visualization Systems**: Integrates with `src/analysis/plots/info_flow_confidence.py` for statistical plots
- **Data Coordination**: Uses `load_results_bank()` and `load_prompts()` for comprehensive data access

**Analysis Workflow Pattern**:
1. **Data Selection**: Users select info flow results from the results bank
2. **Prompt Filtering**: Advanced filtering using `FilterPromptsComponent` with logical operations
3. **Layer Configuration**: Customizable layer range selection for targeted analysis
4. **Statistical Analysis**: Real-time generation of confidence intervals and statistical measures
5. **Visualization**: Interactive plots showing information flow patterns across layers

**Scientific Computing Specialization**:
- Implements knockout methodology for attention analysis
- Provides statistical confidence measures for research rigor
- Supports incremental computation and partial recovery
- Coordinates with experiment completion for reliable statistical analysis

#### Final Plots (p06_final_plots.py)
**Purpose**: Interface to publication-quality plot generation with automated workflow management
**UI Interfaces Provided**:
- **Plot Plans Interface**: UI for managing and selecting plot plans through `PlotPlansComponent`
- **Plot Generation Interface**: UI for generating plots through `PlotGenerationComponent`
- **Image Combiner Interface**: UI for combining and arranging plot images
- **Plot Configuration Interface**: UI for configuring plot parameters and settings

**Key Features**:
- Plot plan management with creation, editing, and deletion capabilities
- Automated plot generation with experiment coordination
- Multi-panel figure generation using `ImageGridParams`
- Export capabilities for publication-ready figures

**Interface to Infrastructure**:
- **Plotting Infrastructure**: Connects to `src/analysis/plots/` for all visualization components
- **Experiment Coordination**: Integrates with experiment completion status through `is_computed()` checks
- **Configuration Management**: Uses Pydantic models for plot configuration validation
- **Data Requirements**: Coordinates with `PlotPlan` requirements and fulfillment tracking
- **File Management**: Integrates with experiment-specific plot directories and caching

**Analysis Workflow Pattern**:
1. **Plot Plan Management**: Users create, edit, and manage plot plans through the interface
2. **Data Requirement Detection**: Automatic detection and execution of missing data requirements
3. **Experiment Coordination**: Verification of experiment completion before plot generation
4. **Plot Generation**: Automated generation of publication-quality plots with caching
5. **Multi-Panel Assembly**: Combination of multiple plots into publication-ready figures

**Scientific Computing Specialization**:
- Maintains publication-quality standards with consistent styling
- Implements plot caching for expensive visualization operations
- Coordinates with experiment completion for reliable results
- Provides automated data requirement fulfillment for complex workflows

#### Prompts Comparison (p07_prompts_comparison.py)
**Purpose**: Interface to tokenization analysis and cross-model prompt comparison
**UI Interfaces Provided**:
- **Prompt Comparison Interface**: UI for comparing prompts across models using `PromptComparisonComponent`
- **Model Selection Interface**: UI for selecting models to compare
- **Performance Analysis Interface**: UI for analyzing prompt performance differences
- **Visualization Interface**: UI for visualizing comparison results

**Key Features**:
- Multi-model prompt comparison with statistical analysis
- Tokenization visualization using `TokenizationVisualizerComponent`
- Performance analysis across different evaluation metrics
- Statistical analysis of prompt differences

**Interface to Infrastructure**:
- **Tokenization Analysis**: Connects to `load_unique_tokenizers` for model-specific tokenization
- **Model Evaluation Pipeline**: Integrates with `EvaluateModelRunner` results for performance comparison
- **Prompt Analysis System**: Coordinates with prompt filtering and selection systems
- **Visualization Components**: Integrates with `TokenizationVisualizerComponent` for detailed token analysis
- **Data Coordination**: Uses `load_prompts()` and `load_results_bank()` for comprehensive data access

**Analysis Workflow Pattern**:
1. **Model Selection**: Users select models for comparison through the interface
2. **Prompt Filtering**: Advanced filtering using `FilterPromptsComponent` with preset configurations
3. **Tokenization Analysis**: Detailed tokenization visualization across different models
4. **Performance Comparison**: Statistical analysis of prompt performance differences
5. **Visualization**: Interactive comparison of tokenization patterns and performance metrics

**Scientific Computing Specialization**:
- Provides detailed tokenization analysis for model comparison
- Supports statistical analysis of prompt performance differences
- Implements cross-model comparison capabilities
- Coordinates with evaluation metrics for comprehensive analysis



#### Final Plots (p06_final_plots.py)
**Purpose**: Generate and display final analysis plots
**UI Interfaces Provided**:
- **Plot Plans Interface**: UI for managing and selecting plot plans through `PlotPlansComponent`
- **Plot Generation Interface**: UI for generating plots through `PlotGenerationComponent`
- **Image Combiner Interface**: UI for combining and arranging plot images
- **Plot Configuration Interface**: UI for configuring plot parameters and settings

**Key Features**:
- Plot plan management
- Automated plot generation
- Result visualization
- Export capabilities

**Interface to Infrastructure**:
- Connects to plotting infrastructure
- Integrates with analysis results
- Coordinates with visualization pipeline

### 3. Data Exploration Pages

#### Results Bank (p02_results_bank.py)
**Purpose**: Explore and manage experiment results
**UI Interfaces Provided**:
- **Results Bank Interface**: UI for exploring experiment results through `ShowResultsBank` component
- **Model Evaluations Interface**: UI for selecting model evaluations using `select_model_evaluations`
- **Data Filtering Interface**: UI for filtering and exploring result data
- **Export Interface**: UI for exporting results and data

**Key Features**:
- Results data exploration
- Model evaluation comparison
- Data filtering and selection
- Export functionality

**Interface to Infrastructure**:
- Connects to results storage system
- Integrates with model evaluation pipeline
- Provides data access for other pages

#### Prompts Comparison (p07_prompts_comparison.py)
**Purpose**: Compare prompts across different models and conditions
**UI Interfaces Provided**:
- **Prompt Comparison Interface**: UI for comparing prompts across models using `PromptComparisonComponent`
- **Model Selection Interface**: UI for selecting models to compare
- **Performance Analysis Interface**: UI for analyzing prompt performance differences
- **Visualization Interface**: UI for visualizing comparison results

**Key Features**:
- Multi-model prompt comparison
- Performance analysis
- Visualization of differences
- Statistical analysis

**Interface to Infrastructure**:
- Connects to prompt analysis system
- Integrates with model evaluation results
- Coordinates with comparison analysis pipeline

### 4. Specialized Analysis Pages

#### Mamba Analysis (p08_mamba_analysis.py)
**Purpose**: Specialized analysis for Mamba model architecture
**UI Interfaces Provided**:
- **Mamba Model Selection Interface**: UI for selecting Mamba models using `select_mamba_model`
- **A Matrix Analysis Interface**: UI for analyzing A matrices through `AMatrixAnalysisComponent`
- **Feature Dynamics Interface**: UI for exploring feature dynamics through `FeatureDynamicsComponent`
- **Model-Specific Visualization Interface**: UI for Mamba-specific visualizations

**Key Features**:
- A Matrix analysis
- Feature dynamics exploration
- Model-specific visualizations
- Architecture-specific insights

**Interface to Infrastructure**:
- Connects to Mamba-specific analysis tools
- Integrates with model architecture analysis
- Coordinates with specialized visualization systems

### 5. Management Pages

#### Prompt Filteration Presets (p09_prompt_filteration_presets.py)
**Purpose**: Manage and configure prompt filtering presets
**UI Interfaces Provided**:
- **Preset Management Interface**: UI for creating and managing prompt filtering presets
- **Filter Configuration Interface**: UI for configuring filter parameters and settings
- **Preset Sharing Interface**: UI for sharing and reusing presets
- **Template Management Interface**: UI for managing preset templates

**Key Features**:
- Preset creation and management
- Filter configuration
- Preset sharing and reuse
- Template management

**Interface to Infrastructure**:
- Connects to prompt filtering system
- Integrates with preset management
- Coordinates with filtering pipeline

## Page Relationships and Data Flow

### Data Flow Architecture

```
Home Page (Configuration)
    ↓
Results Bank (Data Access)
    ↓
Data Requirements (Data Validation)
    ↓
Analysis Pages (Processing)
    ↓
Final Plots (Visualization)
```

### Cross-Page Dependencies

1. **Configuration Dependencies**: All pages depend on global configuration from Home page
2. **Data Dependencies**: Analysis pages depend on data from Results Bank
3. **Filter Dependencies**: Analysis pages use filters from Prompt Filteration Presets

## Analysis Workflow Patterns and Infrastructure Integration

### Experiment Runner Coordination

The analysis pages coordinate with the experiment infrastructure through specialized runners:

- **HeatmapRunner Integration**: The Heatmap Creation page coordinates with `HeatmapRunner` for layer-by-layer probability analysis, supporting window-based analysis and incremental computation
- **InfoFlowRunner Integration**: The Info Flow Analysis page integrates with `InfoFlowRunner` for knockout-based attention analysis, providing statistical confidence measures
- **EvaluateModelRunner Foundation**: All analysis pages depend on `EvaluateModelRunner` results for baseline model performance data
- **FullPipelineRunner Orchestration**: Complex workflows can be coordinated through `FullPipelineRunner` for multi-experiment analysis

### Plotting Infrastructure Integration

The analysis pages integrate with the plotting infrastructure for publication-quality output:

- **Heatmap Visualization**: Uses `src/analysis/plots/heatmaps.py` with `HeatmapPlotConfig` for consistent styling
- **Statistical Plots**: Integrates with `src/analysis/plots/info_flow_confidence.py` for confidence interval visualization
- **Multi-Panel Assembly**: Uses `src/analysis/plots/image_combiner.py` with `ImageGridParams` for publication figures
- **Configuration Management**: All plots use Pydantic configuration models for validation and consistency

### Data Management and Coordination

The analysis pages coordinate with comprehensive data management systems:

- **Model Selection**: Advanced model combination analysis with batch processing capabilities
- **Prompt Filtering**: Sophisticated filtering using `LogicalPromptFilteration` with preset configurations
- **Data Requirements**: Automatic detection and execution of missing data requirements
- **Experiment Completion**: Verification of experiment completion before analysis and visualization

### Scientific Computing Specialization

Each analysis page provides specialized scientific computing capabilities:

- **Incremental Computation**: Support for partial computation and recovery for large-scale experiments
- **Statistical Rigor**: Confidence intervals, statistical measures, and research-grade analysis
- **Publication Quality**: Consistent styling, automated figure generation, and export capabilities
- **Workflow Automation**: Automated data requirement fulfillment and experiment coordination

### Typical Analysis Workflows

#### Heatmap Analysis Workflow
1. **Model Selection**: Users select model combinations through the interface
2. **Prompt Filtering**: Advanced filtering capabilities using `PromptSelectionForCombinationComponent`
3. **Heatmap Generation**: Execution through `HeatmapGenerationComponent` with SLURM support
4. **Visualization**: Real-time generation of publication-quality heatmaps using `HeatmapPlotGenerationComponent`

#### Information Flow Analysis Workflow
1. **Data Selection**: Users select info flow results from the results bank
2. **Prompt Filtering**: Advanced filtering using `FilterPromptsComponent` with logical operations
3. **Layer Configuration**: Customizable layer range selection for targeted analysis
4. **Statistical Analysis**: Real-time generation of confidence intervals and statistical measures
5. **Visualization**: Interactive plots showing information flow patterns across layers

#### Publication Plot Workflow
1. **Plot Plan Management**: Users create, edit, and manage plot plans through the interface
2. **Data Requirement Detection**: Automatic detection and execution of missing data requirements
3. **Experiment Coordination**: Verification of experiment completion before plot generation
4. **Plot Generation**: Automated generation of publication-quality plots with caching
5. **Multi-Panel Assembly**: Combination of multiple plots into publication-ready figures

#### Cross-Model Comparison Workflow
1. **Model Selection**: Users select models for comparison through the interface
2. **Prompt Filtering**: Advanced filtering using `FilterPromptsComponent` with preset configurations
3. **Tokenization Analysis**: Detailed tokenization visualization across different models
4. **Performance Comparison**: Statistical analysis of prompt performance differences
5. **Visualization**: Interactive comparison of tokenization patterns and performance metrics

## Component Reusability and Interface Patterns

The Streamlit application uses a sophisticated component-based architecture that enables high code reusability across pages while serving as interfaces between the UI and scientific computing infrastructure. This architecture provides type safety, modular design, and consistent patterns for building complex scientific workflows.

### Component Composition Patterns

The application follows a hierarchical component composition pattern that enables complex UI construction from simple, reusable components:

#### Base Component Architecture

All components inherit from `StreamlitComponent[OutputType]`, providing:
- **Type Safety**: Generic `OutputType` parameter ensures type-safe data flow
- **Consistent Interface**: Abstract `render()` method enforces uniform component behavior
- **Composition Support**: Components can be easily composed and reused
- **Profiling Capabilities**: Built-in performance analysis for optimization

#### Component Categories

The application organizes components into four main categories based on their functionality:

##### 1. Input Components (`src/app/components/inputs.py`)

**Purpose**: Provide type-safe input interfaces for user configuration
**Key Components**:
- `select_enum()`: Generic enum selection with type safety
- `select_token_type()`: Specialized token type selection
- `select_gpu_type()`: GPU type selection with smart/manual options
- `select_window_size()`: Window size configuration for analysis
- `choose_heatmap_parms()`: Heatmap parameter selection

**Interface Design Principles**:
- **Type Safety**: All inputs use `SessionKey[TSessionKey]` for type-safe state management
- **Configuration Externalization**: Parameters are externalized to configuration objects
- **Error Handling**: Built-in validation and error recovery mechanisms

**Usage Pattern**:
```python
# Type-safe enum selection
metric = select_enum(
    "Select Metric",
    EvaluateModelMetricName,
    PromptsComparisonSessionKeys.evaluate_model_metric_name
)

# Specialized input with validation
select_gpu_type()  # Integrates with SLURM infrastructure
```

##### 2. Filter Components (`src/app/components/prompt_filter.py`)

**Purpose**: Provide sophisticated filtering capabilities for scientific data
**Key Components**:
- `FilterPromptsComponent`: Advanced prompt filtering with logical operations
- `ShowModelCombinations`: Model combination selection and filtering
- `PromptSelectionForCombinationComponent`: Specialized prompt selection
- `SelectPromptsComponent`: Grid-based prompt selection with validation

**Interface Design Principles**:
- **Logical Operations**: Support for complex logical filtering operations
- **Preset Management**: Configurable filter presets for common use cases
- **Validation**: Built-in validation for filter consistency
- **Performance**: Optimized for large-scale data filtering

**Usage Pattern**:
```python
# Advanced filtering with logical operations
prompt_filteration = FilterPromptsComponent(
    key="info_flow_analysis_prompt_filteration",
    base_prompt_filteration=LogicalPromptFilteration.create_and([
        AnyExistingCompletePromptFilteration().contextualize(runner) 
        for runner in result_bank
    ])
).render()
```

##### 3. Display Components (`src/app/components/result_bank.py`, `src/app/components/info_flow.py`)

**Purpose**: Provide data visualization and exploration interfaces
**Key Components**:
- `ShowResultsBank`: Results bank exploration with selection capabilities
- `InfoFlowAnalysisComponent`: Interactive info flow visualization
- `TokenizationVisualizerComponent`: Tokenization analysis visualization
- `PlotPlanDetailsSummary`: Plot plan information display

**Interface Design Principles**:
- **Interactive Visualization**: Real-time data exploration and analysis
- **Selection Modes**: Multiple selection modes (single, multiple, disabled)
- **Grid Integration**: Advanced grid capabilities with filtering and sorting
- **Scientific Rigor**: Publication-quality visualization standards

**Usage Pattern**:
```python
# Interactive results exploration
results_bank = ShowResultsBank(
    results_bank,
    selection_mode=SelectionMode.MULTIPLE,
    height=300,
    filters={
        BaseVariantParamName.model_size: ["large", "medium"],
        WindowedVariantParam.window_size: ["9"]
    }
).render()
```

##### 4. Execution Components (`src/app/components/data_requirements.py`, `src/app/components/plot_generation.py`)

**Purpose**: Provide interfaces to scientific computing infrastructure
**Key Components**:
- `RequirementExecution`: Data requirement execution with SLURM integration
- `HeatmapGenerationComponent`: Heatmap generation with experiment coordination
- `PlotGenerator`: Publication-quality plot generation
- `InfoFlowAnalysisComponent`: Info flow analysis execution

**Interface Design Principles**:
- **Experiment Coordination**: Integration with experiment completion status
- **SLURM Integration**: HPC job management and monitoring
- **Error Recovery**: Robust error handling and recovery mechanisms
- **Progress Tracking**: Real-time progress monitoring and status updates

**Usage Pattern**:
```python
# Experiment execution with SLURM integration
RequirementExecution(data_reqs_to_run).render()

# Plot generation with experiment coordination
plot_path = PlotGenerator(selected_plan, result_bank).render()
```

### Interface Abstraction Between UI and Scientific Computing

The component system serves as a sophisticated interface layer that abstracts complex scientific operations into intuitive user interfaces:

#### Scientific Computing Abstraction

**Experiment Runner Integration**:
- Components abstract experiment runner complexity (`HeatmapRunner`, `InfoFlowRunner`)
- Type-safe interfaces to experiment parameters and results
- Automatic experiment completion verification before visualization

**Data Management Abstraction**:
- Components abstract complex data loading and filtering operations
- Type-safe data flow between components and infrastructure
- Automatic data requirement detection and fulfillment

**Visualization Abstraction**:
- Components abstract plotting infrastructure complexity
- Publication-quality output with consistent styling
- Automated figure generation and export capabilities

#### Type Safety and Error Handling Patterns

**Type-Safe Component Interfaces**:
```python
# Generic component with type-safe output
class FilterPromptsComponent(StreamlitComponent[BasePromptFilteration]):
    def render(self) -> BasePromptFilteration:
        # Implementation with type-safe return
        return filtered_filteration

# Type-safe composition
filtered_data = FilterPromptsComponent(...).render()
analysis_results = InfoFlowAnalysisComponent(filtered_data).render()
```

**Error Handling Patterns**:
- **Graceful Degradation**: Components handle missing data gracefully
- **User Feedback**: Clear error messages and recovery options
- **Validation**: Built-in validation for user inputs and data consistency
- **Recovery Mechanisms**: Automatic recovery from common error conditions

#### Configuration Externalization and Customization

**Configuration Management**:
- Components use external configuration objects for customization
- Pydantic models for configuration validation and consistency
- Default configurations with scientific visualization standards
- Configuration overrides through plot plans and user preferences

**Customization Patterns**:
```python
# Configuration-based customization
config = HeatmapPlotConfig(
    title="Custom Analysis",
    colormap="RdYlGn",
    figure_width=8.0,
    figure_height=6.0
)

# Component with configuration
HeatmapPlotGenerationComponent(prompt_idx, config).render()
```

### Extension Guidelines

#### Creating New Reusable Components

**Component Design Principles**:
1. **Inherit from StreamlitComponent**: Use the base class for consistency
2. **Define Clear OutputType**: Specify the exact return type for type safety
3. **Follow Naming Conventions**: Use descriptive names ending with "Component"
4. **Implement Error Handling**: Include graceful error handling and recovery
5. **Add Documentation**: Include comprehensive docstrings and usage examples

**Implementation Template**:
```python
from src.utils.streamlit.helpers.component import StreamlitComponent
from typing import Optional

class NewAnalysisComponent(StreamlitComponent[AnalysisResult]):
    def __init__(self, input_data: InputData, config: Optional[Config] = None):
        self.input_data = input_data
        self.config = config or Config()
    
    def render(self) -> AnalysisResult:
        # Implementation with type-safe return
        try:
            # Component logic
            return analysis_result
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return AnalysisResult.empty()
```

#### Extending Existing Component Functionality

**Extension Patterns**:
1. **Composition**: Combine existing components to create new functionality
2. **Configuration**: Extend through configuration objects rather than inheritance
3. **Wrapping**: Wrap existing components with additional functionality
4. **Specialization**: Create specialized versions for specific use cases

**Example Extension**:
```python
# Extending through composition
class AdvancedFilterComponent(StreamlitComponent[AdvancedFilterResult]):
    def __init__(self, base_filter: FilterPromptsComponent, advanced_config: AdvancedConfig):
        self.base_filter = base_filter
        self.advanced_config = advanced_config
    
    def render(self) -> AdvancedFilterResult:
        # Use base component
        base_result = self.base_filter.render()
        # Add advanced functionality
        advanced_result = self.apply_advanced_filtering(base_result)
        return advanced_result
```

#### Maintaining Architectural Consistency

**Consistency Guidelines**:
1. **Type Safety**: Always use generic types and type hints
2. **Error Handling**: Implement consistent error handling patterns
3. **Configuration**: Use external configuration objects for customization
4. **Documentation**: Maintain comprehensive documentation for all components
5. **Testing**: Include unit tests for component behavior and edge cases

**Architecture Compliance**:
- **Component Isolation**: Components should be independent and reusable
- **Data Flow**: Use type-safe data flow between components
- **State Management**: Use SessionKey system for state management
- **Performance**: Implement caching and optimization where appropriate

### Practical Examples of Component Reuse

#### Cross-Page Component Reuse

**FilterPromptsComponent Reuse**:
- **Heatmap Creation**: Used for prompt filtering in heatmap generation
- **Info Flow Analysis**: Used for prompt filtering in info flow analysis
- **Prompts Comparison**: Used for prompt filtering in comparison analysis
- **Final Plots**: Used for prompt filtering in plot generation

**ShowResultsBank Reuse**:
- **Results Bank Page**: Primary results exploration interface
- **Info Flow Analysis**: Used for selecting info flow results
- **Data Requirements**: Used for exploring data requirements
- **Final Plots**: Used for selecting data for plot generation

#### Component Composition Examples

**Complex Analysis Workflow**:
```python
# Component composition for complex analysis
class ComplexAnalysisWorkflow(StreamlitComponent[AnalysisResult]):
    def render(self) -> AnalysisResult:
        # 1. Data selection
        results_bank = ShowResultsBank(...).render()
        
        # 2. Prompt filtering
        filtered_data = FilterPromptsComponent(...).render()
        
        # 3. Analysis execution
        analysis_result = InfoFlowAnalysisComponent(filtered_data).render()
        
        # 4. Visualization
        PlotGenerator(analysis_result).render()
        
        return analysis_result
```

**Multi-Step Analysis Pipeline**:
```python
# Pipeline composition
def create_analysis_pipeline():
    return [
        ShowResultsBank(...),
        FilterPromptsComponent(...),
        InfoFlowAnalysisComponent(...),
        PlotGenerator(...)
    ]

# Execute pipeline
for component in pipeline:
    result = component.render()
    if result is None:
        break
```

The component system provides a robust foundation for building sophisticated scientific workflows while maintaining code reusability, type safety, and architectural consistency. This design enables both current functionality and future extensions while serving as a critical interface between the UI and scientific computing infrastructure.

### Component Reuse Patterns

Pages reuse components across different contexts:

- **ShowResultsBank**: Used in Results Bank, Info Flow Analysis, and other pages
- **PromptFilterComponent**: Used in Heatmap Creation, Info Flow Analysis, and other pages
- **PlotGenerationComponent**: Used in Heatmap Creation and Final Plots

## Navigation Flow and User Experience

### Entry Point Flow

1. **Home Page**: User starts here to configure global settings
2. **Data Exploration**: User explores available data in Results Bank
3. **Analysis Selection**: User chooses specific analysis type
4. **Configuration**: User configures analysis parameters
5. **Execution**: User runs analysis and views results
6. **Visualization**: User explores results through visualizations

### Page Transition Patterns

#### Configuration → Analysis Flow
```
Home → Results Bank → Data Requirements → Analysis Page → Final Plots
```

#### Quick Analysis Flow
```
Home → Analysis Page (with default settings) → Results
```

#### Comparison Flow
```
Home → Results Bank → Prompts Comparison → Analysis
```

## Integration with Scientific Computing Infrastructure

### Experiment Infrastructure Integration

Each page interfaces with the experiment infrastructure through:

1. **BaseRunner Integration**: Pages use BaseRunner for experiment execution
2. **Data Store Integration**: Pages access data through centralized data store
3. **Component Integration**: Pages compose functionality from reusable components

### Data Management Integration

Pages coordinate with the data management system:

1. **Results Bank**: Central data access point
2. **Data Requirements**: Data validation and availability checking
3. **Analysis Pages**: Data processing and transformation
4. **Visualization Pages**: Data presentation and export

### Analysis Pipeline Integration

Pages integrate with analysis pipelines:

1. **Model Evaluation**: Results Bank and analysis pages
2. **Prompt Analysis**: Prompt filtering and comparison pages
3. **Visualization**: Plot generation and display pages
4. **Specialized Analysis**: Mamba analysis and info flow pages

## Global Configuration Management

### Session State Coordination

The application uses centralized session state management:

```python
class AppSessionKeys(SessionKeysBase):
    code_version = SessionKeyDescriptor[TCodeVersionName](DEFAULT_CODE_VERSION)
    _selected_gpu = SessionKeyDescriptor[Union[SLURM_GPU_TYPE, Literal["smart"]]]("smart")
    window_size = SessionKeyDescriptor[TWindowSize](DEFAULT_WINDOW_SIZE)
```

### Configuration Propagation

Global configuration affects all pages:

1. **Code Version**: Controls which experiment results to load
2. **GPU Selection**: Affects model execution and analysis
3. **Window Size**: Affects analysis parameters across pages

## Extension and Customization

### Adding New Pages

To add a new page:

1. **Create Page Class**: Inherit from `StreamlitPage`
2. **Add to PAGE_ORDER**: Update the enum with new page
3. **Update Factory**: Add case to `get_page()` function
4. **Add Metadata**: Define title and icon in page details

### Page Customization Patterns

1. **Component Composition**: Build pages from reusable components
2. **Configuration Integration**: Use global configuration consistently
3. **Data Integration**: Connect to appropriate data sources
4. **Navigation Integration**: Follow established navigation patterns

## Performance Considerations

### Page Loading Optimization

1. **Lazy Loading**: Load data only when needed
2. **Caching**: Cache expensive operations and data
3. **Background Processing**: Use background tasks for long operations
4. **Component Isolation**: Isolate expensive components

### Memory Management

1. **Session State Cleanup**: Clean up unused session state
2. **Resource Disposal**: Properly dispose of resources
3. **Background Task Management**: Cancel background tasks when appropriate

## Data Exploration and Management Pages

The application includes three critical pages that provide interfaces to the data layer: Results Bank, Data Requirements, and Prompt Filteration Presets. These pages abstract complex data operations while maintaining type safety and providing intuitive user interfaces.

### Results Bank Page (p02_results_bank.py)

**Core Data Interface**: Provides UI for exploring and managing `ResultBank` data objects and experiment results.

**Data System Integration**:
- **ResultBank Objects**: Interfaces with `ResultBank` from `data_defs.py` for experiment results
- **Model Evaluations**: Connects to `EvaluateModelResults` for model performance data
- **Data Filtering**: Provides advanced filtering and selection capabilities
- **Export Functionality**: Enables data export and sharing

**Component Architecture**:
```python
# Core component for results display
class ShowResultsBank(StreamlitComponent[T_RESULT_BANK_TYPE]):
    def render(self) -> T_RESULT_BANK_TYPE:
        # Converts ResultBank to DataFrame for display
        df = self.results_bank.to_experiment_results_df()
        # Uses AgGrid for interactive display
        # Returns filtered ResultBank object
```

**Workflow Example**:
1. **Data Loading**: Loads experiment results from `load_results_bank()`
2. **Filtering**: User applies filters to focus on specific experiments
3. **Selection**: User selects relevant results for further analysis
4. **Export**: Selected data can be exported or passed to other pages

### Data Requirements Page (p03_data_requirements.py)

**Core Data Interface**: Provides UI for managing `DataReqs` objects and requirement execution.

**Data System Integration**:
- **DataReqs Objects**: Interfaces with `DataReqs` from `data_defs.py` for experiment requirements
- **SummarizedDataFulfilledReqs**: Manages requirement fulfillment status
- **BaseRunner Integration**: Connects to experiment infrastructure for requirement execution
- **SLURM Coordination**: Manages job execution and status tracking

**Component Architecture**:
```python
# Requirements display component
class RequirementsDisplay(StreamlitComponent):
    def render(self) -> DataReqs | None:
        # Displays requirements with filtering
        # Returns selected DataReqs for execution

# Requirement execution component
class RequirementExecution(StreamlitComponent):
    def render(self):
        # Creates runners from DataReqs
        # Manages SLURM job execution
        # Tracks execution status
```

**Workflow Example**:
1. **Requirements Display**: Shows available data requirements with fulfillment status
2. **Selection**: User selects requirements to execute
3. **Configuration**: User configures execution parameters (GPU, code version)
4. **Execution**: System creates and executes BaseRunner instances
5. **Monitoring**: Tracks job status and completion

### Prompt Filteration Presets Page (p09_prompt_filteration_presets.py)

**Core Data Interface**: Provides UI for managing `PromptFilterationsPresets` and creating complex filterations.

**Data System Integration**:
- **PromptFilterationsPresets**: Manages reusable prompt filtering configurations
- **BasePromptFilteration**: Creates and composes filteration objects
- **Logical Operations**: Supports union, intersection, and sampling operations
- **Preset Management**: Enables saving and reusing filter configurations

**Component Architecture**:
```python
# Filteration creator component
class FilterationCreator(StreamlitComponent[BasePromptFilteration]):
    def render(self) -> BasePromptFilteration:
        # Creates basic filterations (All, Selective, Model Correct)
        # Supports composition operations (Union, Intersection, Sample)
        # Returns configured BasePromptFilteration object
```

**Workflow Example**:
1. **Preset Management**: User browses existing presets
2. **Filteration Creation**: User creates new filterations using basic or composition methods
3. **Configuration**: User configures filter parameters and logic
4. **Saving**: User saves configurations as reusable presets
5. **Sharing**: Presets can be shared and reused across experiments

## Component Reusability Patterns

### ShowResultsBank Component

**Reusability**: Used across multiple pages for consistent results display
**Interface Pattern**: Generic component with type-safe return values
**Usage Examples**:
- Results Bank page: Primary results exploration
- Info Flow Analysis: Selecting info flow results
- Heatmap Creation: Selecting model evaluations

### RequirementsDisplay Component

**Reusability**: Provides consistent requirement display across contexts
**Interface Pattern**: Configurable display with selection capabilities
**Usage Examples**:
- Data Requirements page: Primary requirement management
- Experiment planning: Requirement validation
- Status tracking: Execution progress monitoring

### FilterPromptsComponent

**Reusability**: Used for prompt filtering across all analysis pages
**Interface Pattern**: Integrates with BasePromptFilteration system
**Usage Examples**:
- Heatmap Creation: Filtering prompts for analysis
- Info Flow Analysis: Selecting prompts for flow analysis
- Prompt Comparison: Filtering prompts for comparison

## Integration with Core Systems

### Data Objects Integration

**ResultBank Integration**:
```python
# Results Bank page interfaces with ResultBank objects
results_bank = load_results_bank().to_evaluate_model_results()
ShowResultsBank(results_bank, selection_mode=SelectionMode.MULTIPLE).render()
```

**DataReqs Integration**:
```python
# Data Requirements page interfaces with DataReqs objects
data_reqs_to_run = RequirementsDisplay(df, selection_mode=SelectionMode.MULTIPLE).render()
if data_reqs_to_run:
    RequirementExecution(data_reqs_to_run).render()
```

**PromptFilterationsPresets Integration**:
```python
# Prompt Filteration Presets page interfaces with preset objects
presets = PromptFilterationsPresets.load()
FilterationCreator(key="new_filteration").render()
```

### Experiment Infrastructure Integration

**BaseRunner Integration**:
```python
# Data Requirements page creates runners from requirements
config = init_runner_from_params(
    req,
    InputParams(filteration=filteration),
    MetadataParams(code_version=code_version, with_slurm=with_slurm)
)
```

**SLURM Job Management**:
```python
# Components track job status and execution
status = config.slurm_job_folder.get_latest_slurm_job_status()
if status.scheduled():
    # Handle scheduled jobs
```

### Data Management Integration

**Type-Safe Data Access**:
```python
# All components use type-safe data access patterns
class ShowResultsBank(StreamlitComponent[T_RESULT_BANK_TYPE]):
    def render(self) -> T_RESULT_BANK_TYPE:
        # Type-safe conversion and return
```

**Session State Management**:
```python
# Components use SessionKey for type-safe state management
self.current_filteration_sk = SessionKey[BasePromptFilteration](
    f"{key}_current_filteration", default_value=AllPromptFilteration()
)
```

## Practical Workflow Examples

### Complete Analysis Workflow

1. **Data Exploration** (Results Bank):
   - User explores available experiment results
   - Filters results by model, experiment type, or performance
   - Selects relevant results for analysis

2. **Requirement Validation** (Data Requirements):
   - User checks data requirements for planned analysis
   - Identifies missing data or incomplete experiments
   - Executes missing requirements if needed

3. **Filter Configuration** (Prompt Filteration Presets):
   - User creates or selects prompt filtering presets
   - Configures complex filtering logic
   - Saves configurations for reuse

4. **Analysis Execution** (Analysis Pages):
   - User applies configured filters to selected data
   - Executes analysis with validated data
   - Views results with proper data context

### Data Management Workflow

1. **Status Monitoring**:
   - User checks execution status of requirements
   - Monitors SLURM job progress
   - Identifies failed or stuck jobs

2. **Data Validation**:
   - User validates data completeness
   - Checks data quality and consistency
   - Identifies data issues or missing components

3. **Configuration Management**:
   - User manages experiment configurations
   - Updates parameters and settings
   - Maintains configuration consistency

## Cross-References

- **Infrastructure Patterns**: See [docs/streamlit-infrastructure.md](streamlit-infrastructure.md) for base patterns
- **Component Architecture**: See [docs/streamlit-infrastructure.md](streamlit-infrastructure.md) for component patterns
- **Data Interfaces**: See [docs/data-interfaces.md](data-interfaces.md) for data management
- **Experiment Runners**: See [docs/experiment-runners.md](experiment-runners.md) for experiment infrastructure 
