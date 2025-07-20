# Analysis and Plotting

Plotting and visualization components in `src/analysis/plots/` for publication-quality research output and result presentation.

## Plot Types and File References

### `heatmaps.py` - Layer-by-Layer Probability Visualization
- **Purpose**: Creates diverging heatmaps for visualizing probability differences across model layers during token interventions
- **Input**: Probability matrices from `HeatmapRunner` showing layer-by-layer probability changes
- **Configuration**: `HeatmapPlotConfig` provides comprehensive styling control including colormap selection, normalization settings, and annotation positioning
- **Key Features**: 
  - Fixed difference scaling for consistent color interpretation
  - Base probability annotation and title customization
  - Percentage-based x-axis display for layer depth
  - Robust normalization options for outlier handling
- **File**: `src/analysis/plots/heatmaps.py`

### `info_flow_confidence.py` - Statistical Confidence Plots
- **Purpose**: Generates confidence interval plots for information flow analysis with statistical rigor
- **Input**: `TInfoFlowOutput` from `InfoFlowRunner` containing window-based probability data
- **Configuration**: `InfoFlowPlotConfig` supports multiple confidence calculation methods (CI, PI, bootstrap, SE) and extensive styling options
- **Key Features**:
  - Multiple confidence calculation methods for different statistical needs
  - Dual metric support (accuracy and probability difference)
  - Custom color and line style mapping for token types and feature categories
  - Publication-ready styling with configurable legends and annotations
- **File**: `src/analysis/plots/info_flow_confidence.py`

### `image_combiner.py` - Multi-Panel Figure Generation
- **Purpose**: Creates publication-ready multi-panel figures by combining individual plots into grid layouts
- **Configuration**: `ImageGridParams` manages grid organization, cropping, and label positioning
- **Features**: Automatic cropping, legend generation, label management, and flexible grid arrangements
- **Key Methods**: `combine_image_grid()`, `ImageGridParams`
- **File**: `src/analysis/plots/image_combiner.py`

## Plot Configuration Management

### BaseModel Configuration Pattern
All plot configurations use Pydantic BaseModel for validation and consistency:

```python
class HeatmapPlotConfig(BaseModel): # src/analysis/plots/heatmaps.py
class InfoFlowPlotConfig(BaseModel): # src/analysis/plots/info_flow_confidence.py
class ImageGridParams(BaseModel): # src/analysis/plots/image_combiner.py
```

### Configuration Organization
- **Grouped Fields**: Configuration parameters are organized into logical groups (basic_config, figure_settings, font_settings, etc.)
- **Validation**: Field constraints ensure valid parameter ranges and types
- **Defaults**: Sensible defaults provide reasonable scientific visualization standards
- **Extensibility**: Configuration objects can be extended through plot plan objects

### Plot Plan Integration
- **Cell Configuration**: Individual plot cells use specific configuration objects (`cell_plot_config`)
- **Combined Configuration**: Multi-panel layouts use `combine_plot_config` for grid organization
- **Parameter Mapping**: Plot plans map experiment parameters to visualization configurations
- **File**: `src/analysis/experiment_results/plot_plan.py`

## Plotting Coordination Rules



## Cross-References

- **Data Interfaces**: See [docs/data-relationships-interface.md](data-relationships-interface.md) for how plot plans coordinate with data objects
- **Core Modules**: See [docs/core-modules.md](core-modules.md) for color scheme and constant definitions
- **Experiment Runners**: See [docs/experiment-runners.md](experiment-runners.md) for how runners provide data for plotting
- **Infrastructure**: See [docs/infrastructure.md](infrastructure.md) for base class patterns
- **Setup and Environment**: See [docs/setup-and-environment.md](setup-and-environment.md) for file management
- **Utilities**: See [docs/utilities-and-patterns.md](utilities-and-patterns.md) for plotting utilities

## Critical Warnings

⚠️ **Plot generation must wait for experiment completion - check `is_computed()` first**

⚠️ **Always use configuration objects - never hardcode plot parameters**

⚠️ **Maintain consistent color schemes from core constants**

⚠️ **Cache expensive visualization operations for performance**
