# Streamlit App Pages

Page architecture and navigation patterns for scientific computing interface. Focus on what LLMs need to know about page organization and navigation.

## Page File Structure

```
src/app/page/
├── p01_home.py                    # Home page and global configuration
├── p02_results_bank.py            # Results bank exploration
├── p03_data_requirements.py       # Data requirements management
├── p04_heatmap_creation.py        # Heatmap generation
├── p05_info_flow_analysis.py      # Information flow analysis
├── p06_final_plots.py             # Final plot generation
├── p07_prompts_comparison.py      # Prompts comparison
├── p08_mamba_analysis.py          # Mamba model analysis
└── p09_prompt_filteration_presets.py # Prompt filtering presets
```

## Page Implementation Patterns

### Basic Page Structure

All pages inherit from `StreamlitPage`:

```python
class HomePage(StreamlitPage):
    def render(self) -> None:
        st.title("Welcome to the Application")
        st.sidebar.success("Select a page above to explore different analyses.")
        
        # Global controls
        if st.button("Reset App"):
            st.session_state.clear()
        
        # Compose components
        select_gpu_type()
        AppSessionKeys.code_version.create_input_widget("CodeVersion")
        select_window_size()
```

## Navigation Architecture

### Page Factory Pattern

**File**: `src/app/entry_point.py`

**Purpose**: Centralized page factory with enum-based navigation.

### PAGE_ORDER Enum (in `src/app/app_consts.py`)

**Purpose**: Defines navigation structure and page metadata.

## Page Categories

### 1. Configuration and Control Pages

#### Home Page

**File**: `src/app/page/p01_home.py`

**Purpose**: Application entry point and global configuration.

**Key Features**:
- **Global Configuration**: Code version, GPU selection, window size
- **App Reset**: Global app reset functionality
- **Navigation Guidance**: User guidance and navigation help
- **Session Management**: Centralized session state management

**Component Integration**:
- `select_gpu_type()`: GPU type selection (smart/manual)
- `AppSessionKeys.code_version`: Code version configuration
- `select_window_size()`: Window size configuration

#### Data Requirements Page

**File**: `src/app/page/p03_data_requirements.py`

**Purpose**: Manage and explore data requirements for experiments.

**Key Features**:
- **Data Requirements Interface**: Explore data requirements
- **Requirement Filtering**: Filter and explore requirements
- **Fulfillment Status**: Track requirement fulfillment
- **Data Availability**: Analyze data availability patterns

**Component Integration**:
- `DataRequirementsComponent`: Data requirements display
- `RequirementExecution`: Execute selected requirements

### 2. Analysis and Visualization Pages

#### Heatmap Creation Page

**File**: `src/app/page/p04_heatmap_creation.py`

**Purpose**: Interface to heatmap generation and model analysis.

**Key Features**:
- **Model Combinations**: Explore and select model combinations
- **Prompt Selection**: Filter and select prompts
- **Heatmap Generation**: Execute heatmap generation
- **Plot Generation**: Create heatmaps through plot components

**Component Integration**:
- `load_model_combinations_prompts()`: Model combination loading
- `PromptSelectionForCombinationComponent`: Prompt selection
- `HeatmapGenerationComponent`: Heatmap generation
- `HeatmapPlotGenerationComponent`: Plot generation

#### Info Flow Analysis Page

**File**: `src/app/page/p05_info_flow_analysis.py`

**Purpose**: Information flow analysis and visualization.

**Key Features**:
- **Flow Analysis**: Analyze information flow patterns
- **Visualization**: Interactive flow diagrams
- **Path Analysis**: Analyze information paths
- **Export Capabilities**: Export flow analysis results

**Component Integration**:
- `InfoFlowComponent`: Information flow analysis
- Flow visualization components
- Export functionality

#### Final Plots Page

**File**: `src/app/page/p06_final_plots.py`

**Purpose**: Final plot generation and management.

**Key Features**:
- **Plot Management**: Manage and configure plots
- **Plot Generation**: Generate final plots
- **Configuration Interface**: Plot parameter configuration
- **Export Options**: Export plots in multiple formats

**Component Integration**:
- `PlotGenerationComponent`: Plot generation
- `PlotPlansComponent`: Plot plan management
- `MultiPlotsComponent`: Multi-plot layouts

#### Prompts Comparison Page

**File**: `src/app/page/p07_prompts_comparison.py`

**Purpose**: Compare prompts across models and configurations.

**Key Features**:
- **Prompt Comparison**: Compare prompts across models
- **Analysis Interface**: Prompt analysis tools
- **Visualization**: Comparative visualizations
- **Export Results**: Export comparison results

**Component Integration**:
- Prompt comparison components
- Analysis visualization
- Export functionality

#### Mamba Analysis Page

**File**: `src/app/page/p08_mamba_analysis.py`

**Purpose**: Specialized Mamba model analysis.

**Key Features**:
- **Mamba-Specific Analysis**: Mamba model analysis tools
- **A Matrix Analysis**: Analyze A matrices
- **Feature Dynamics**: Explore feature dynamics
- **Model-Specific Visualization**: Mamba-specific visualizations

**Component Integration**:
- `AMatrixAnalysisComponent`: A matrix analysis
- `FeatureDynamicsComponent`: Feature dynamics analysis
- Mamba-specific visualization components

### 3. Data Management Pages

#### Results Bank Page

**File**: `src/app/page/p02_results_bank.py`

**Purpose**: Explore and manage experiment results.

**Key Features**:
- **Results Exploration**: Explore experiment results
- **Data Filtering**: Filter and search results
- **Export Interface**: Export results and data
- **Status Tracking**: Track experiment status

**Component Integration**:
- `ShowResultsBank`: Results display and management
- `ShowRunnerStatus`: Runner status display
- Export functionality

#### Prompt Filtration Presets Page

**File**: `src/app/page/p09_prompt_filteration_presets.py`

**Purpose**: Manage prompt filtering presets and configurations.

**Key Features**:
- **Preset Management**: Create and manage filter presets
- **Filter Configuration**: Configure complex filters
- **Preset Export**: Export and import presets
- **Filter Testing**: Test filter configurations

**Component Integration**:
- `PromptFilterComponent`: Prompt filtering interface
- Preset management components
- Filter configuration tools

## Navigation Patterns

### Page Factory Integration

Pages are created through the page factory:

```python
# Get page instance
page = get_page(PAGE_ORDER.RESULTS_BANK)
page.render()
```

### Navigation State Management

Navigation state is managed through session state:

```python
# Navigation state
current_page = st.session_state.get("current_page", PAGE_ORDER.HOME)
page = get_page(current_page)
page.render()
```

## Global Configuration

### AppSessionKeys

**File**: `src/app/app_consts.py`

**Purpose**: Centralized session state management.

**Key Configuration**:
- **Code Version**: Controls which version of experiments to load
- **GPU Selection**: Smart or manual GPU type selection
- **Window Size**: Configurable window size for analysis parameters

### Global Configuration Pattern

```python
# Global configuration access
code_version = AppSessionKeys.code_version.value
gpu_type = AppSessionKeys._selected_gpu.value
window_size = AppSessionKeys.window_size.value

# Update configuration
AppSessionKeys.code_version.value = "v2.0"
```

## Page-Component Integration

### Component Selection Guidelines

**Use Data Display Components when:**
- Displaying large datasets
- Need interactive filtering
- Require export capabilities
- Show tabular data

**Use Analysis Components when:**
- Running complex analysis
- Need background processing
- Require progress tracking
- Generate visualizations

**Use Configuration Components when:**
- Need user input
- Require parameter selection
- Need form validation
- Manage application state

**Use Visualization Components when:**
- Generate plots and charts
- Need interactive visualizations
- Require plot configuration
- Export visual results

### Integration Patterns

```python
# Data flow pattern
class DataAnalysisPage(StreamlitPage):
    def render(self) -> None:
        # 1. Configuration
        config = ConfigComponent().render()
        
        # 2. Data loading
        data = DataComponent().render()
        
        # 3. Analysis
        if config and data:
            results = AnalysisComponent(config=config, data=data).render()
            
            # 4. Visualization
            if results:
                PlotComponent(results=results).render()
```

## Best Practices

1. **Page Composition**: Build pages from smaller, reusable components
2. **Session State**: Use centralized session state management
3. **Navigation**: Follow the page factory pattern for consistent navigation
4. **Error Handling**: Implement graceful error handling in pages
5. **Performance**: Use background tasks for long-running operations
6. **Documentation**: Document page interfaces and component usage

## Cross-References

- **Streamlit Infrastructure**: See [docs/streamlit-infrastructure.md](streamlit-infrastructure.md) for base classes
- **Streamlit Components**: See [docs/streamlit-app-components.md](streamlit-app-components.md) for component patterns
- **Core Modules**: See [docs/core-modules.md](core-modules.md) for data object patterns
- **Utilities**: See [docs/utilities-and-patterns.md](utilities-and-patterns.md) for utility patterns
- **Data Interfaces**: See [docs/data-relationships-interface.md](data-relationships-interface.md) for data object patterns
- **Infrastructure**: See [docs/infrastructure.md](infrastructure.md) for base class patterns
- **Setup and Environment**: See [docs/setup-and-environment.md](setup-and-environment.md) for environment setup
