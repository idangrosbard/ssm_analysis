# Analysis and Plotting

This document covers the plotting and visualization components in `src/analysis/plots/`, including plot types, configuration management, and visualization coordination rules. These components are critical for publication-quality research output and result presentation.

## Overview

The plotting system provides sophisticated visualization capabilities for scientific research, with emphasis on publication-quality output, consistent styling, and reproducible results. All plotting components use configuration objects for customization and maintain strict coordination with experiment completion.

## Plot Types and Usage

### `heatmaps.py` - Layer-by-Layer Probability Visualization

**Purpose**: Layer-by-layer probability difference visualization for model analysis
**Input**: Probability matrices from `HeatmapRunner`
**Configuration**: `HeatmapPlotConfig` for appearance customization

#### Key Features

```python
# Standard heatmap generation
from src.analysis.plots.heatmaps import simple_diff_fixed, HeatmapPlotConfig

config = HeatmapPlotConfig(
    title="Model Probability Differences",
    colormap="RdYlGn",
    figure_width=8.0,
    figure_height=6.0,
    with_fixed_diff=True,
    fixed_diff=0.3
)

# Generate heatmap
heatmap = simple_diff_fixed(
    model_id=model_id,
    window_size=window_size,
    last_tok=last_tok,
    base_prob=base_prob,
    target_rank=target_rank,
    true_word=true_word,
    toks=toks,
    config=config
)
```

#### Configuration Options

- **Basic Configuration**: Title, minimal title, base probability display
- **X-axis Options**: Percentage display, tick count customization
- **Figure Settings**: Width, height, layout, title positioning
- **Font Settings**: Title, axis, tick, and annotation font sizes
- **Color Settings**: Colormap selection, reverse direction
- **Normalization**: Fixed difference scaling, robust normalization

### `info_flow_confidence.py` - Statistical Confidence Plots

**Purpose**: Confidence interval plots for information flow analysis
**Input**: `TInfoFlowOutput` from `InfoFlowRunner`
**Configuration**: `InfoFlowPlotConfig` for line styles, colors, metrics

#### Key Features

```python
# Confidence plot generation
from src.analysis.plots.info_flow_confidence import create_confidence_plot, InfoFlowPlotConfig

config = InfoFlowPlotConfig(
    confidence_level=0.95,
    alpha=0.2,
    metrics_to_show=[TMetricType.DIFF, TMetricType.ACC],
    figure_width=8.0,
    figure_height=6.0
)

# Generate confidence plot
figure = create_confidence_plot(
    lines=window_outputs_dict,
    confidence_level=0.95,
    config=config
)
```

#### Configuration Options

- **Basic Config**: Confidence level, alpha transparency, metrics selection
- **Display Options**: X-axis percentage display, tick counts, point display format
- **Figure Settings**: Width, height, layout, label positioning
- **Font Settings**: Title, axis, legend, tick font sizes
- **Legend Settings**: Position, visibility, layout
- **Custom Colors**: Token type to color mapping
- **Custom Line Styles**: Feature category to style mapping

### `image_combiner.py` - Multi-Panel Figure Generation

**Purpose**: Multi-panel figure generation and layout for publication
**Configuration**: `ImageGridParams` for grid organization
**Features**: Automatic cropping, legend generation, label management

#### Key Features

```python
# Multi-panel figure generation
from src.analysis.plots.image_combiner import combine_image_grid, ImageGridParams

params = ImageGridParams(
    title="Model Comparison Results",
    padding=10,
    show_col_labels=True,
    show_row_labels=False,
    font_size=25
)

# Generate combined figure
combined_image = combine_image_grid(
    images_paths_grid=image_grid,
    params=params,
    legend_items=legend_items,
    row_labels=row_labels,
    col_labels=col_labels
)
```

#### Configuration Options

- **Grid Organization**: Row/column ordering, custom layouts
- **Image Processing**: Cropping parameters, size handling
- **Labeling**: Row/column labels, prefix styles, font sizes
- **Legend**: Position, styling, divider lines
- **Titles**: Main title, header styling

## Plotting Coordination Rules

### **ALWAYS Do These Actions**

- **ALWAYS use configuration objects** (`*PlotConfig`) instead of hardcoded parameters
- **COORDINATE plot generation with experiment completion** - check `is_computed()` first
- **USE consistent color schemes** defined in `src/core/consts.py`
- **MAINTAIN plot caching** for expensive visualization operations
- **SAVE plots in experiment-specific directories** under `variation_paths.plots_path`

### **NEVER Do These Actions**

- **NEVER hardcode plot parameters** - always use configuration objects
- **NEVER generate plots before experiments complete** - check computation status
- **NEVER use inconsistent color schemes** - use centralized color definitions
- **NEVER bypass plot caching** for expensive operations

## Plot Configuration Management

### Centralized Configuration

All plot configurations use Pydantic models for validation and consistency:

```python
# Example: HeatmapPlotConfig structure
class HeatmapPlotConfig(BaseModel):
    title: str = Field(default="", description="Custom plot title")
    figure_width: float = Field(default=4.0, ge=2.0, le=12.0)
    figure_height: float = Field(default=3.0, ge=2.0, le=10.0)
    colormap: str = Field(default="RdYlGn", description="Colormap name")
    # ... additional configuration options
```

### Configuration Best Practices

1. **CENTRALIZE color and style definitions** in core constants
2. **USE Pydantic models for plot configurations** to ensure validation
3. **PROVIDE default configurations** with reasonable scientific visualization standards
4. **ALLOW configuration overrides** through plot plan objects

### Color Scheme Management

```python
# Use centralized color definitions
from src.core.consts import TOKEN_TYPE_COLORS, CONVERT_TO_PLOTLY_LINE_STYLE

# Consistent color usage across plots
colors = TOKEN_TYPE_COLORS
line_styles = CONVERT_TO_PLOTLY_LINE_STYLE
```

## Plot Caching and Experiment Coordination

### Caching Strategy

```python
# Check experiment completion before plotting
if experiment_runner.is_computed():
    # Generate plot with caching
    plot_path = experiment_runner.variation_paths.plots_path / "heatmap.png"
    
    if not plot_path.exists():
        # Generate and cache plot
        plot = generate_heatmap(experiment_runner.get_outputs())
        plot.save(plot_path)
    else:
        # Load cached plot
        plot = load_cached_plot(plot_path)
```

### Experiment Coordination

```python
# Coordinate with experiment completion
def generate_experiment_plots(experiment_runner: BaseRunner):
    if not experiment_runner.is_computed():
        raise ValueError("Experiment must be completed before plotting")
    
    # Generate plots in experiment-specific directory
    plots_dir = experiment_runner.variation_paths.plots_path
    plots_dir.mkdir(exist_ok=True)
    
    # Generate different plot types
    generate_heatmap_plot(experiment_runner, plots_dir)
    generate_confidence_plot(experiment_runner, plots_dir)
    generate_combined_plot(experiment_runner, plots_dir)
```

## Visualization Workflow Examples

### Heatmap Generation Workflow

```python
# Complete heatmap generation workflow
def create_model_heatmap_analysis(model_runner: HeatmapRunner):
    # 1. Check experiment completion
    if not model_runner.is_computed():
        print("Experiment not completed yet")
        return
    
    # 2. Load experiment outputs
    outputs = model_runner.get_outputs()
    
    # 3. Configure plot settings
    config = HeatmapPlotConfig(
        title=f"Probability Differences - {model_runner.variant_params.model_arch}",
        colormap="RdYlGn",
        with_fixed_diff=True,
        fixed_diff=0.3
    )
    
    # 4. Generate heatmap
    heatmap = simple_diff_fixed(
        model_id=outputs["model_id"],
        window_size=outputs["window_size"],
        last_tok=outputs["last_tok"],
        base_prob=outputs["base_prob"],
        target_rank=outputs["target_rank"],
        true_word=outputs["true_word"],
        toks=outputs["toks"],
        config=config
    )
    
    # 5. Save to experiment directory
    plot_path = model_runner.variation_paths.plots_path / "heatmap.png"
    heatmap.savefig(plot_path, dpi=300, bbox_inches='tight')
```

### Confidence Plot Workflow

```python
# Complete confidence plot workflow
def create_info_flow_confidence_analysis(info_flow_runners: list[InfoFlowRunner]):
    # 1. Check all experiments are completed
    for runner in info_flow_runners:
        if not runner.is_computed():
            print(f"Experiment {runner.variant_params} not completed")
            return
    
    # 2. Collect outputs
    outputs_dict = {
        f"{runner.variant_params.model_arch}-{runner.variant_params.model_size}": 
        runner.get_outputs()
        for runner in info_flow_runners
    }
    
    # 3. Configure plot
    config = InfoFlowPlotConfig(
        confidence_level=0.95,
        metrics_to_show=[TMetricType.DIFF],
        figure_width=10.0,
        figure_height=6.0
    )
    
    # 4. Generate confidence plot
    figure = create_confidence_plot(
        lines=outputs_dict,
        confidence_level=0.95,
        config=config
    )
    
    # 5. Save plot
    plot_path = info_flow_runners[0].variation_paths.plots_path / "confidence_plot.png"
    figure.savefig(plot_path, dpi=300, bbox_inches='tight')
```

### Multi-Panel Figure Workflow

```python
# Complete multi-panel figure workflow
def create_publication_figure(experiment_runners: list[BaseRunner]):
    # 1. Collect individual plot paths
    plot_paths = []
    for runner in experiment_runners:
        if runner.is_computed():
            plot_path = runner.variation_paths.plots_path / "individual_plot.png"
            if plot_path.exists():
                plot_paths.append(plot_path)
    
    # 2. Organize into grid
    grid = organize_plots_to_grid(plot_paths)
    
    # 3. Configure multi-panel layout
    params = ImageGridParams(
        title="Model Comparison Analysis",
        padding=15,
        show_col_labels=True,
        show_row_labels=True,
        font_size=30
    )
    
    # 4. Generate combined figure
    combined_image = combine_image_grid(
        images_paths_grid=grid,
        params=params,
        legend_items=create_legend_items(),
        row_labels=row_labels,
        col_labels=col_labels
    )
    
    # 5. Save publication-ready figure
    final_path = Path("final_plots") / "publication_figure.png"
    combined_image.save(final_path, dpi=300)
```

## Cross-References

### Related Documentation

- **Data Interfaces**: See [docs/data-interfaces.md](data-interfaces.md) for how plot plans coordinate with data objects
- **Core Modules**: See [docs/core-modules.md](core-modules.md) for color scheme and constant definitions
- **Experiment Runners**: See [docs/experiment-runners.md](experiment-runners.md) for how runners provide data for plotting

### Dependencies

- **Core Constants**: Color schemes and styling constants from `src/core/consts.py`
- **Data Objects**: Plot plans and result banks from `src/data_ingestion/data_defs/`
- **Experiment Infrastructure**: Runner outputs and completion status

## Best Practices

1. **Always use configuration objects**: Never hardcode plot parameters
2. **Check experiment completion**: Verify `is_computed()` before plotting
3. **Use consistent styling**: Leverage centralized color and style definitions
4. **Implement caching**: Cache expensive visualization operations
5. **Coordinate with data**: Use proper data objects for plot generation
6. **Maintain publication quality**: Use high DPI and proper formatting
7. **Document configurations**: Keep plot configurations version-controlled
8. **Test thoroughly**: Verify plots render correctly across different environments

## Critical Warnings

⚠️ **Plot generation must wait for experiment completion - check `is_computed()` first**

⚠️ **Always use configuration objects - never hardcode plot parameters**

⚠️ **Maintain consistent color schemes from core constants**

⚠️ **Cache expensive visualization operations for performance**

⚠️ **Save plots in experiment-specific directories for organization** 
